"""Sliding-window debounce-and-merge plugin for AstrBot with agent-style
chain-of-thought.

When a message is about to hit the LLM, the request opens a merge window
(one per session+sender) and holds the reply. Instead of a blind random
timer, the leader runs a lightweight "inner thought" pass over everything
the user has sent so far: the thought judges whether the user has finished
talking and decides how much longer to wait. Every follow-up message
donated into the window wakes the leader and triggers another thought pass
that sees the previous thought, forming a real reasoning chain. When the
window finally goes quiet, the last thought is injected into the final
request so the persona call starts from that understanding and answers all
messages at once.

Failure isolation: if thinking is disabled, the thinking model call fails,
or its JSON verdict cannot be parsed, the window falls back to the
original behavior — a random quiet period (5-15s by default) that
mimics a human reading pause.

Window lifecycle (per merge key = unified_msg_origin:sender_id):

- The first request for a user becomes the leader and runs this loop
  inside the on_llm_request hook (each event runs in its own pipeline
  task, so other messages keep flowing meanwhile).
- Messages arriving while the window is open are stopped at the event
  handler stage; their text/images are donated to the window and the
  leader is woken up to think again.
- The loop exits when the thought-set quiet period passes with no new
  messages, or the hard cap (window start + max_window) is reached.
"""

from __future__ import annotations

import asyncio
import json
import random
import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star

FOLLOWUP_MARKER = (
    "[以下是这位用户在这条消息之后紧接着连发的后续内容，"
    "请把它们与上面的消息合并理解，并只针对全部内容回复一次]"
)

THOUGHT_MARKER = (
    "[以下是你在回复前对这位用户消息的内心思考，帮助你理解语境，"
    "回复时不要提及或复述这段思考]"
)

THINK_SYSTEM_PROMPT = (
    "你是一个聊天机器人的内心思考模块。用户正在给你的宿主机器人连发消息，"
    "机器人需要决定等多久再回复。请结合用户已发来的全部消息快速思考："
    "用户到底想表达或询问什么；用户是说完了，还是很可能继续连发。"
    '只输出一个 JSON 对象：{"thought": "一句话第一人称内心思考，不超过60字", '
    '"wait": 建议再安静等待的整数秒数}。'
    "wait 取值参考：用户已说完或是完整的问题给 3~8；不确定给 10~15；"
    "明显没说完（话说到一半、正在列举、刚发省略号或问号、正在连发表情包）给 15~25。"
    "不要输出 JSON 以外的任何内容。"
)

# Seconds to sleep after a donation before thinking, so a rapid sticker or
# typo burst folds into one thought pass instead of one pass per message.
THINK_SETTLE_SECONDS = 1.0
# Upper bound of thought passes per window, bounding LLM cost when a user
# keeps talking until the hard cap.
THINK_PASS_LIMIT = 10


def merge_key(event: AstrMessageEvent) -> str:
    """Build the debounce key: one merge window per user per session.

    Args:
        event: Incoming message event.

    Returns:
        The key combining the unified message origin and the sender id.
    """
    return f"{event.unified_msg_origin}:{event.get_sender_id()}"


class MergeWindow:
    """Buffered state of one open sliding window."""

    def __init__(self, quiet_seconds: float, max_window: float) -> None:
        """Create a window whose first quiet period is already running.

        Args:
            quiet_seconds: Initial quiet period from now, used as the
                fallback before the first thought pass runs.
            max_window: Total window lifetime cap from now; thought
                verdicts may never push the deadline past it.
        """
        self.buffer: list[str] = []
        self.images: list[str] = []
        self.thoughts: list[str] = []
        self.waits: list[float] = []
        now = time.monotonic()
        self.hard_cap = now + max(1.0, max_window)
        self.deadline = min(self.hard_cap, now + quiet_seconds)
        # Starts set so the leader immediately thinks about its own first
        # message; every donation sets it again to wake the leader.
        self.wakeup = asyncio.Event()
        self.wakeup.set()

    def donate(self, text: str, images: list[str]) -> None:
        """Fold one follow-up message into the buffer and wake the leader.

        Args:
            text: Stripped plain text of the message, "" for image-only.
            images: Image URLs/file paths from the message chain.
        """
        if text:
            self.buffer.append(f"- {text}")
        if images:
            self.images.extend(images)
            if not text:
                self.buffer.append("- [图片/表情包]")
        self.wakeup.set()


class ReplyMergePlugin(Star):
    """Hold LLM requests in a thought-driven window and merge follow-ups."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        """Initialize the plugin with an empty open-window table.

        Args:
            context: AstrBot plugin context.
            config: Plugin config rendered from _conf_schema.json.
        """
        super().__init__(context)
        self.config = config
        # merge_key -> MergeWindow, present only while a leader is waiting.
        self.pending: dict[str, MergeWindow] = {}
        # merge_key -> latest finished chain, kept for the show_thoughts
        # command; bounded so a long-running bot does not grow it forever.
        self.last_chain: dict[str, dict] = {}

    def _random_quiet(self) -> float:
        """Draw the random quiet window between min_wait and max_wait.

        Returns:
            Quiet period in seconds.
        """
        return random.uniform(
            float(self.config.get("min_wait", 5)),
            float(self.config.get("max_wait", 15)),
        )

    def _extract_images(self, event: AstrMessageEvent) -> list[str]:
        """Pull non-empty image sources from the event message chain.

        Args:
            event: Incoming message event.

        Returns:
            Image URLs or local file paths, in arrival order.
        """
        sources: list[str] = []
        for comp in event.get_messages():
            if isinstance(comp, Image):
                src = str(comp.url or comp.file or "").strip()
                if src:
                    sources.append(src)
        return sources

    async def _think(
        self, event: AstrMessageEvent, first_message: str, window: MergeWindow
    ) -> tuple[str, float] | None:
        """Run one inner-thought pass over everything sent so far.

        Args:
            event: The leader event, used to resolve the session provider.
            first_message: The leader's own prompt text.
            window: The open window holding follow-ups and past thoughts.

        Returns:
            (thought, quiet_seconds) from the verdict, or None when
            thinking is unavailable (no provider, call failure, or an
            unparsable verdict); the caller then falls back to a random
            quiet period.
        """
        provider_id = str(self.config.get("think_provider_id", "") or "").strip()
        provider = (
            self.context.get_provider_by_id(provider_id)
            if provider_id
            else await self.context.get_using_provider_async(
                umo=event.unified_msg_origin
            )
        )
        if provider is None:
            logger.warning("[merge] no provider for thinking; random quiet")
            return None

        lines = [f"用户发的第一条消息：{first_message or '（无文字）'}"]
        if window.buffer:
            lines.append("用户随后连发的消息：\n" + "\n".join(window.buffer))
        if window.thoughts and window.thoughts[-1]:
            lines.append(f"你上一轮的内心思考：{window.thoughts[-1]}")
        lines.append("请输出这一轮的 JSON。")

        try:
            resp = await provider.text_chat(
                prompt="\n".join(lines), system_prompt=THINK_SYSTEM_PROMPT
            )
            raw = (resp.completion_text or "").strip()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[merge] thinking call failed: {e}; random quiet")
            return None

        try:
            start, end = raw.find("{"), raw.rfind("}")
            verdict = json.loads(raw[start : end + 1])
            thought = str(verdict.get("thought", "")).strip()
            quiet = float(verdict["wait"])
            # json.loads accepts NaN/Infinity; reject absurd values so the
            # deadline stays a sane finite number.
            if not (0 < quiet < 3600):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            logger.warning(f"[merge] unparsable thought verdict: {raw[:200]}")
            return None
        quiet = min(
            max(quiet, float(self.config.get("min_wait", 5))),
            float(self.config.get("max_wait", 15)),
        )
        return thought, quiet

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def collect_followup(self, event: AstrMessageEvent) -> None:
        """Fold messages that arrive inside an open window into it.

        The message is stopped so the pipeline never builds its own reply;
        the donation wakes the leader, which thinks again and slides the
        deadline forward.

        Args:
            event: Incoming message event.
        """
        if not event.is_at_or_wake_command:
            return
        window = self.pending.get(merge_key(event))
        if window is None:
            return
        window.donate(
            (event.get_message_str() or "").strip(), self._extract_images(event)
        )
        event.stop_event()

    @filter.on_llm_request()
    async def debounce_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Hold the request in a thought-driven window, then merge in follow-ups.

        The leader loops: wake up on the first message or any donation,
        run a thought pass, and arm the quiet deadline from the verdict.
        It returns when the window has been quiet for a full period (or
        the hard cap). A request that arrives while a window is open
        loses the race, donates its own content to the winner and is
        stopped.

        Args:
            event: Incoming message event.
            req: The provider request about to be sent.
        """
        key = merge_key(event)
        window = self.pending.get(key)
        if window is not None:
            # Lost the creation race (both messages passed the event stage
            # within pipeline latency): donate to the open window and stop.
            window.donate(
                (event.get_message_str() or "").strip(), self._extract_images(event)
            )
            event.stop_event()
            return

        window = MergeWindow(
            self._random_quiet(), float(self.config.get("max_window", 60))
        )
        self.pending[key] = window
        thinking = bool(self.config.get("enable_thinking", True))
        passes = 0
        try:
            logger.info(f"[merge] leader opened a thinking window ({key})")
            while True:
                if time.monotonic() >= window.hard_cap:
                    break
                if window.wakeup.is_set():
                    window.wakeup.clear()
                    if thinking and passes < THINK_PASS_LIMIT:
                        # Let a rapid burst settle, then reason over all
                        # messages seen so far (chain: previous thought in).
                        await asyncio.sleep(THINK_SETTLE_SECONDS)
                        verdict = await self._think(event, req.prompt or "", window)
                        passes += 1
                        if verdict is not None:
                            window.thoughts.append(verdict[0])
                            quiet = verdict[1]
                            logger.info(
                                f"[merge] thought pass {passes} "
                                f"(wait {quiet:.0f}s, "
                                f"{len(window.buffer)} follow-ups): {verdict[0]}"
                            )
                        else:
                            window.thoughts.append("")
                            quiet = self._random_quiet()
                            logger.info(
                                f"[merge] thought pass {passes} failed; "
                                f"random quiet {quiet:.0f}s"
                            )
                        window.waits.append(quiet)
                        window.deadline = min(window.hard_cap, time.monotonic() + quiet)
                        continue
                    # Thinking disabled or exhausted: slide like the old
                    # random debounce so follow-ups still extend the wait.
                    window.deadline = min(
                        window.hard_cap, time.monotonic() + self._random_quiet()
                    )
                remaining = window.deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(window.wakeup.wait(), remaining)
                except asyncio.TimeoutError:
                    break
        except asyncio.CancelledError:
            if window.buffer:
                logger.warning(
                    f"[merge] leader cancelled with {len(window.buffer)} buffered "
                    f"follow-ups; they will not be answered ({key})"
                )
            raise
        finally:
            # Always close the window, even if the leader is cancelled.
            self.pending.pop(key, None)

        # Archive the finished chain for the show_thoughts command before
        # req.prompt gets the merge blocks appended below.
        self.last_chain[key] = {
            "time": time.strftime("%m-%d %H:%M:%S"),
            "messages": [(req.prompt or "").strip()]
            + [entry.lstrip("- ").strip() for entry in window.buffer],
            "thoughts": list(window.thoughts),
            "waits": list(window.waits),
        }
        if len(self.last_chain) > 100:
            self.last_chain.pop(next(iter(self.last_chain)))

        parts: list[str] = []
        if window.thoughts and window.thoughts[-1]:
            parts.append(f"{THOUGHT_MARKER}\n{window.thoughts[-1]}")
        if window.buffer:
            limit = max(1, int(self.config.get("max_followups", 10)))
            shown = window.buffer[:limit]
            omitted = len(window.buffer) - len(shown)
            if omitted > 0:
                shown.append(f"- ……（还有 {omitted} 条未展开）")
            parts.append(f"{FOLLOWUP_MARKER}\n" + "\n".join(shown))
            if self.config.get("forward_images", True) and window.images:
                # Allow a couple of images per merged message; one sticker
                # burst often carries several, and cutting by message count
                # drops them.
                req.image_urls.extend(window.images[: limit * 2])
            logger.info(f"[merge] merged {len(window.buffer)} follow-ups ({key})")
        if parts:
            req.prompt = f"{(req.prompt or '').rstrip()}\n\n" + "\n\n".join(parts)

    @filter.command("思维链", alias={"thoughts", "思考链"})
    async def show_thoughts(self, event: AstrMessageEvent):
        """Show the latest thought chain of the current chat.

        Renders the most recently finished merge window of this session:
        the messages that were merged and every thought pass with its
        wait verdict, so the bot's reasoning is inspectable from chat.

        Args:
            event: Command event.
        """
        chain = self.last_chain.get(merge_key(event))
        if not chain or not chain["thoughts"]:
            yield event.plain_result(
                "还没有思考记录（需开启思维链思考并完成过一次回复）"
            )
            return
        messages = [m[:20] for m in chain["messages"][:5]]
        if len(chain["messages"]) > 5:
            messages.append(f"…等{len(chain['messages']) - 5}条")
        lines = [f"🧠 最近一轮思考（{chain['time']}，共{len(chain['thoughts'])}轮）"]
        lines.append("📩 " + " / ".join(m for m in messages if m))
        for i, (thought, wait) in enumerate(zip(chain["thoughts"], chain["waits"]), 1):
            if thought:
                lines.append(f"{i}. {thought}（等{wait:.0f}秒）")
            else:
                lines.append(f"{i}. （思考失败，退回随机等待{wait:.0f}秒）")
        yield event.plain_result("\n".join(lines))
