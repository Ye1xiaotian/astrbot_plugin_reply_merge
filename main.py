"""Sliding-window debounce-and-merge plugin for AstrBot with agent-style
chain-of-thought.

When a message is about to hit the LLM, the request opens a merge window
(one per session+sender) and holds the reply. Instead of a blind random
timer, the leader runs a lightweight "inner thought" pass over everything
the user has sent so far. Each pass produces a structured understanding
(per-message points, an overall judgment, open questions) plus a decision:
reply now, or keep waiting for a suggested number of seconds. Every
follow-up message donated into the window wakes the leader and triggers
another thought pass that sees the previous structured thought, forming a
reasoning chain with low information loss. When the window finally goes
quiet (or the model says reply), the latest structured understanding is
injected into the final request so the persona call starts from it and
answers all messages at once.

The thought pass is told how long the window has been open and how much is
left before the hard cap, and every pass logs both the model-requested
wait and the effective (clamped) wait, so scheduling behavior is
measurable from logs.

Failure isolation: if thinking is disabled, the thinking model call fails,
or its JSON verdict cannot be parsed, the window falls back to the
original behavior — a random quiet period (5-15s by default) that mimics
a human reading pause.

Window lifecycle (per merge key = unified_msg_origin:sender_id):

- The first request for a user becomes the leader and runs this loop
  inside the on_llm_request hook (each event runs in its own pipeline
  task, so other messages keep flowing meanwhile).
- Messages arriving while the window is open are stopped at the event
  handler stage; their text/images are donated to the window and the
  leader is woken up to think again.
- The loop exits when the verdict-armed deadline passes with no new
  messages, the verdict says reply, or the hard cap (window start +
  max_window) is reached.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass

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
    "你需要判断用户是否说完，并决定现在回复还是继续等待。"
    '只输出一个 JSON 对象：{"points": ["对用户每条消息的逐条理解，一条一个短句"], '
    '"summary": "总体判断：用户想做什么、是否说完（第一人称，50字内）", '
    '"questions": ["尚未明确、需要用户补充的信息，没有则为空数组"], '
    '"action": "reply或wait", "wait": 建议再安静等待的整数秒数}。'
    "action 取值：用户明显说完或是完整问题给 reply；不确定给 wait 且 wait 取 10~15；"
    "明显没说完（话说到一半、正在列举、刚发省略号或问号、正在连发表情包）"
    "给 wait 且 wait 取 15~25。不要输出 JSON 以外的任何内容。"
)

# Seconds to sleep after a donation before thinking, so a rapid sticker or
# typo burst folds into one thought pass instead of one pass per message.
THINK_SETTLE_SECONDS = 1.0
# Upper bound of thought passes per window, bounding LLM cost when a user
# keeps talking until the hard cap.
THINK_PASS_LIMIT = 10
# Human-feel pause used when the verdict action is "reply": it skips the
# min_wait floor but still avoids an instant bot-like answer.
THINK_REPLY_PAUSE = 2.0


def merge_key(event: AstrMessageEvent) -> str:
    """Build the debounce key: one merge window per user per session.

    Args:
        event: Incoming message event.

    Returns:
        The key combining the unified message origin and the sender id.
    """
    return f"{event.unified_msg_origin}:{event.get_sender_id()}"


@dataclass
class Verdict:
    """One parsed inner-thought pass.

    Attributes:
        points: Per-message understanding, one short line per message.
        summary: Overall first-person judgment of intent and completeness.
        questions: Information still missing from the user, may be empty.
        action: "reply" or "wait" from the model; "fallback" marks a pass
            whose call/parse failed and whose wait is a random draw.
        wait: Model-requested quiet seconds; for fallback passes this is
            the random quiet that was actually used.
    """

    points: list[str]
    summary: str
    questions: list[str]
    action: str
    wait: float


def format_thought(verdict: Verdict) -> str:
    """Render one structured thought as compact plain lines.

    Args:
        verdict: The parsed thought to render.

    Returns:
        The block reused for the next pass's chain input, the final-prompt
        injection and the show_thoughts command.
    """
    lines = [f"- {p}" for p in verdict.points]
    if verdict.summary:
        lines.append(f"总体判断：{verdict.summary}")
    if verdict.questions:
        lines.append("未解决：" + "；".join(verdict.questions))
    return "\n".join(lines)


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
        self.thoughts: list[Verdict] = []
        self.waits: list[float] = []
        now = time.monotonic()
        self.started_at = now
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
        self,
        event: AstrMessageEvent,
        first_message: str,
        window: MergeWindow,
        elapsed: float,
        remaining: float,
    ) -> Verdict | None:
        """Run one inner-thought pass over everything sent so far.

        Args:
            event: The leader event, used to resolve the session provider.
            first_message: The leader's own prompt text.
            window: The open window holding follow-ups and past thoughts.
            elapsed: Seconds since the window opened.
            remaining: Seconds left before the hard cap closes it.

        Returns:
            The parsed Verdict, or None when thinking is unavailable (no
            provider, call failure, or an unparsable verdict); the caller
            then falls back to a random quiet period. Never raises except
            for cancellation: an exception escaping the on_llm_request hook
            is swallowed by the host and would skip the whole debounce.
        """
        try:
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

            lines = [
                f"[窗口状态] 已等待 {elapsed:.0f} 秒；"
                f"距强制收尾还剩 {max(0.0, remaining):.0f} 秒；"
                f"已收到 {1 + len(window.buffer)} 条消息",
                f"用户发的第一条消息：{first_message or '（无文字）'}",
            ]
            if window.buffer:
                lines.append("用户随后连发的消息：\n" + "\n".join(window.buffer))
            if window.thoughts and window.thoughts[-1].summary:
                lines.append(f"你上一轮的思考：\n{format_thought(window.thoughts[-1])}")
            lines.append("请输出这一轮的 JSON。")

            resp = await provider.text_chat(
                prompt="\n".join(lines), system_prompt=THINK_SYSTEM_PROMPT
            )
            raw = (resp.completion_text or "").strip()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Provider resolution itself can raise (e.g. a session bound to
            # a mistyped provider); every failure must degrade to a random
            # quiet instead of escaping this hook — the host swallows hook
            # exceptions and would skip the whole debounce.
            logger.warning(f"[merge] thinking unavailable: {e}; random quiet")
            return None

        try:
            start, end = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[start : end + 1])
            points = [str(p).strip() for p in data.get("points", []) if str(p).strip()]
            summary = str(data.get("summary", "")).strip()
            questions = [
                str(q).strip() for q in data.get("questions", []) if str(q).strip()
            ]
            action = str(data.get("action", "wait")).strip().lower()
            if action not in ("reply", "wait"):
                action = "wait"
            try:
                wait = float(data.get("wait"))
            except (TypeError, ValueError):
                wait = 0.0
            # json.loads accepts NaN/Infinity; reject absurd values so the
            # deadline stays a sane finite number. A missing wait is fatal
            # only when the model actually wants us to keep waiting.
            if wait != wait or not (0 < wait < 3600):
                if action != "reply":
                    raise ValueError
                wait = THINK_REPLY_PAUSE
        except (ValueError, KeyError, TypeError):
            logger.warning(f"[merge] unparsable thought verdict: {raw[:200]}")
            return None
        return Verdict(points, summary, questions, action, wait)

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
        run a thought pass, and arm the quiet deadline from the verdict
        (a "reply" verdict arms the short human-feel pause instead of the
        clamped wait). It returns when the window has been quiet for a
        full period (or the hard cap). A request that arrives while a
        window is open loses the race, donates its own content to the
        winner and is stopped.

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
                now = time.monotonic()
                if now >= window.hard_cap:
                    break
                if window.wakeup.is_set():
                    window.wakeup.clear()
                    if thinking and passes < THINK_PASS_LIMIT:
                        # Let a rapid burst settle, then reason over all
                        # messages seen so far (chain: previous thought in).
                        await asyncio.sleep(THINK_SETTLE_SECONDS)
                        now = time.monotonic()
                        try:
                            verdict = await self._think(
                                event,
                                req.prompt or "",
                                window,
                                elapsed=now - window.started_at,
                                remaining=window.hard_cap - now,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            # Safety net: any escape from the thinking
                            # layer must degrade to a fallback pass, never
                            # abort the window and skip the debounce.
                            logger.exception("[merge] thought pass crashed")
                            verdict = None
                        passes += 1
                        if verdict is not None:
                            if verdict.action == "reply":
                                quiet = THINK_REPLY_PAUSE
                            else:
                                quiet = min(
                                    max(
                                        verdict.wait,
                                        float(self.config.get("min_wait", 5)),
                                    ),
                                    float(self.config.get("max_wait", 15)),
                                )
                            window.thoughts.append(verdict)
                            logger.info(
                                f"[merge] thought pass {passes}: "
                                f"action={verdict.action} "
                                f"requested={verdict.wait:.0f}s "
                                f"effective={quiet:.0f}s | {verdict.summary}"
                            )
                        else:
                            quiet = self._random_quiet()
                            window.thoughts.append(
                                Verdict([], "", [], "fallback", quiet)
                            )
                            logger.info(
                                f"[merge] thought pass {passes}: fallback "
                                f"effective={quiet:.0f}s"
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
            "thoughts": [
                {
                    "points": v.points,
                    "summary": v.summary,
                    "questions": v.questions,
                    "action": v.action,
                    "wait": w,
                }
                for v, w in zip(window.thoughts, window.waits)
            ],
        }
        if len(self.last_chain) > 100:
            self.last_chain.pop(next(iter(self.last_chain)))

        parts: list[str] = []
        if window.thoughts and window.thoughts[-1].summary:
            parts.append(f"{THOUGHT_MARKER}\n{format_thought(window.thoughts[-1])}")
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
        the messages that were merged and every structured thought pass
        with its action and effective wait, so the bot's reasoning is
        inspectable from chat.

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
        for i, v in enumerate(chain["thoughts"], 1):
            if v["action"] == "fallback":
                lines.append(f"{i}. （思考失败，退回随机等待{v['wait']:.0f}秒）")
                continue
            lines.append(f"{i}. [{v['action']}·等{v['wait']:.0f}秒] {v['summary']}")
            for p in v["points"]:
                lines.append(f"   · {p}")
            if v["questions"]:
                lines.append("   ❓ " + "；".join(v["questions"]))
        yield event.plain_result("\n".join(lines))
