"""Logic tests for astrbot_plugin_reply_merge.

Stubs the astrbot host modules, imports the real plugin main.py and
checks the pure logic: window message grouping and ordering, truncation
direction, unseen counting for lazy evaluation, verdict parsing, the
scope gate, and the stale-reply discard. Run with:

    python tests/test_merge_logic.py

The leader loop itself is async and hook-driven, so its scheduling is
not covered here; everything it delegates to MergeWindow and the small
predicates is.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

PLUGIN = Path(__file__).resolve().parent.parent / "main.py"


def install_stubs() -> None:
    """Install minimal astrbot stand-ins so main.py imports standalone."""
    pkg = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    api.AstrBotConfig = dict

    event = types.ModuleType("astrbot.api.event")
    event.AstrMessageEvent = type("AstrMessageEvent", (), {})

    class _Filter:
        EventMessageType = type(
            "EventMessageType", (), {"ALL": "all", "GROUP_MESSAGE": "group"}
        )

        def event_message_type(self, *_a, **_k):
            return lambda f: f

        def on_waiting_llm_request(self, *_a, **_k):
            return lambda f: f

        def on_llm_request(self, *_a, **_k):
            return lambda f: f

        def on_llm_response(self, *_a, **_k):
            return lambda f: f

        def on_astrbot_loaded(self, *_a, **_k):
            return lambda f: f

        def command(self, *_a, **_k):
            return lambda f: f

    event.filter = _Filter()

    comps = types.ModuleType("astrbot.api.message_components")
    for name in ("At", "File", "Forward", "Image", "Record", "Video"):
        setattr(comps, name, type(name, (), {}))

    platform = types.ModuleType("astrbot.api.platform")
    platform.MessageType = type("MessageType", (), {"OTHER_MESSAGE": "other_message"})

    provider = types.ModuleType("astrbot.api.provider")
    provider.ProviderRequest = type("ProviderRequest", (), {})
    provider.LLMResponse = type("LLMResponse", (), {})

    star = types.ModuleType("astrbot.api.star")
    star.Context = type("Context", (), {})

    class _Star:
        def __init__(self, context):
            self.context = context

    star.Star = _Star

    pkg.api = api
    sys.modules["astrbot"] = pkg
    for mod in (api, event, comps, platform, provider, star):
        sys.modules[mod.__name__] = mod


def load_plugin():
    spec = importlib.util.spec_from_file_location("reply_merge_main", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules["reply_merge_main"] = module
    spec.loader.exec_module(module)
    return module


class FakeEvent:
    """Minimal event double for the key/extra/sender predicates."""

    def __init__(self, umo="aiocqhttp:GroupMessage:1", sender="u1", private=False):
        self.unified_msg_origin = umo
        self._sender = sender
        self._private = private
        self.message_obj = types.SimpleNamespace(session_id=umo.split(":")[-1])
        self._extra = {}

    def get_sender_id(self):
        return self._sender

    def is_private_chat(self):
        return self._private

    def get_extra(self, key, default=None):
        return self._extra.get(key, default)

    def set_extra(self, key, value):
        self._extra[key] = value


def test_window_grouping(m):
    w = m.MergeWindow(
        5.0,
        60.0,
        first_message="leader text",
        leader_order=(1, 0, 0),
        leader_images=["img_leader"],
    )
    assert w.unseen == 0 and w.close_reason == "" and w.rescued is False
    w.donate((0, 0, 9), "sent before leader", [], [])
    w.donate((2, 0, 1), "sent after", ["img_after"], ["语音消息"])
    w.donate((3, 0, 1), "", ["sticker"], [])
    before, after = w.ordered_messages()
    assert before == ["- sent before leader"], before
    assert after == [["- sent after", "- [语音消息]"], ["- [图片/表情包]"]], after
    # One buffer entry per message, and unseen counts donated messages.
    assert len(w.buffer) == 3 and w.unseen == 3
    w.donate((4, 0, 1), "newest", ["img_leader"], [])
    assert w.unseen == 4
    assert w.all_images() == ["img_leader", "img_after", "sticker"]
    # Truncation keeps whole messages and the newest ones.
    _, after = w.ordered_messages()
    shown = after[-2:]
    omitted = len(after) - len(shown)
    assert omitted == 1 and shown == [["- [图片/表情包]"], ["- newest"]], shown
    print("grouping/ordering OK")


def test_verdict_parsing(m):
    v = m.parse_verdict("想到什么算什么\n<<wait 12>>", True, 7.0)
    assert v is not None and v[0] == "想到什么算什么" and v[4:] == ("wait", 12.0), v
    v = m.parse_verdict("说完了 <<REPLY>>", True, 7.0)
    assert v[4] == "reply" and v[5] == m.THINK_REPLY_PAUSE, v
    v = m.parse_verdict("纯独白没有决定行", True, 7.0)
    assert v[4] == "wait" and v[5] == 7.0, v
    v = m.parse_verdict('{"thought":"心里话","action":"wait","wait":18}', True, 7.0)
    assert v[0] == "心里话" and v[5] == 18.0, v
    assert m.parse_verdict("", True, 7.0) is None
    v = m.parse_verdict(
        '{"points":"一句话","summary":"说完了","questions":[],'
        '"action":"reply","wait":0}',
        False,
        7.0,
    )
    assert v is not None and v[1] == ["一句话"] and v[4] == "reply", v
    print("verdict parsing OK")


def test_scope_gate(m):
    plugin = m.ReplyMergePlugin(None, {})
    group_ev = FakeEvent(umo="aiocqhttp:GroupMessage:1")
    private_ev = FakeEvent(umo="aiocqhttp:FriendMessage:2", private=True)
    assert plugin._debounce_enabled(group_ev)
    assert plugin._debounce_enabled(private_ev)

    both_off = m.ReplyMergePlugin(None, {"usage_scope": "group"})
    assert both_off._debounce_enabled(group_ev)
    assert not both_off._debounce_enabled(private_ev)
    priv_only = m.ReplyMergePlugin(None, {"usage_scope": "private"})
    assert not priv_only._debounce_enabled(group_ev)
    assert priv_only._debounce_enabled(private_ev)

    by_umo = m.ReplyMergePlugin(
        None, {"disabled_sessions": ["aiocqhttp:GroupMessage:1"]}
    )
    assert not by_umo._debounce_enabled(group_ev)
    assert by_umo._debounce_enabled(FakeEvent(umo="aiocqhttp:GroupMessage:9"))
    by_session = m.ReplyMergePlugin(None, {"disabled_sessions": ["1"]})
    assert not by_session._debounce_enabled(group_ev)

    by_user = m.ReplyMergePlugin(None, {"disabled_users": ["u1"]})
    assert not by_user._debounce_enabled(group_ev)
    assert by_user._debounce_enabled(FakeEvent(sender="u2"))
    print("scope gate OK")


def test_stale_reply_discard(m):
    plugin = m.ReplyMergePlugin(None, {})
    old_window = m.MergeWindow(5.0, 60.0, first_message="old")
    new_window = m.MergeWindow(5.0, 60.0, first_message="new")
    stale_event = FakeEvent()
    stale_event.set_extra("_reply_merge_from_window", old_window)
    plugin.inflight["aiocqhttp:GroupMessage:1:u1"] = (old_window, 0.0)
    plugin.stale["aiocqhttp:GroupMessage:1:u1"] = old_window

    class Resp:
        completion_text = "old reply"

    resp = Resp()
    asyncio.run(plugin.drop_stale_reply(stale_event, resp))
    assert resp.completion_text == "", resp.completion_text
    assert "aiocqhttp:GroupMessage:1:u1" not in plugin.inflight

    # A live (non-stale) reply passes through untouched.
    live_event = FakeEvent()
    live_event.set_extra("_reply_merge_from_window", new_window)
    resp = Resp()
    asyncio.run(plugin.drop_stale_reply(live_event, resp))
    assert resp.completion_text == "old reply"
    # Events without our marker are ignored entirely.
    resp = Resp()
    asyncio.run(plugin.drop_stale_reply(FakeEvent(), resp))
    assert resp.completion_text == "old reply"
    print("stale reply discard OK")


def test_lazy_loop(m):
    """Drive the real leader loop with a stubbed think pass.

    A burst of leader + one follow-up must cost exactly two passes: the
    eager first pass and the one final pass at the deadline — the stray
    follow-up must NOT trigger a pass of its own, and the window must
    still close on the quiet deadline with everything read.
    """
    settle, pause = m.THINK_SETTLE_SECONDS, m.THINK_REPLY_PAUSE
    m.THINK_SETTLE_SECONDS = 0.05
    m.THINK_REPLY_PAUSE = 0.05
    try:
        plugin = m.ReplyMergePlugin(
            None,
            {
                "min_wait": 1,
                "max_wait": 2,
                "max_window": 30,
                "enable_thinking": True,
                "think_context": False,
                "think_style": "",
            },
        )
        event = FakeEvent()
        key = m.merge_key(event)
        window = m.MergeWindow(1.0, 30.0, first_message="hello")
        plugin.pending[key] = window

        calls = []

        async def fake_think(
            event, first_message, window, elapsed=0.0, remaining=0.0, style_context=""
        ):
            calls.append(window.unseen)
            if len(calls) == 1:
                return m.Verdict([], "", [], "wait", 3.0)
            return m.Verdict([], "", [], "reply", 0.0)

        plugin._think = fake_think

        async def donate_later():
            await asyncio.sleep(0.4)
            window.donate((2, 0, 1), "follow-up", [], [])

        async def scenario():
            donator = asyncio.create_task(donate_later())
            await plugin._hold_window(key, event, window, True)
            await donator

        asyncio.run(scenario())
        # First pass read the leader; the lazy slide kept the follow-up
        # unread; the final pass at the deadline read it and said reply.
        assert calls == [0, 1], calls
        assert window.close_reason == "quiet", window.close_reason
        assert window.unseen == 0 and len(window.thoughts) == 2
        assert key not in plugin.pending
    finally:
        m.THINK_SETTLE_SECONDS = settle
        m.THINK_REPLY_PAUSE = pause
    print("lazy loop OK")


def main() -> None:
    install_stubs()
    m = load_plugin()
    test_window_grouping(m)
    test_verdict_parsing(m)
    test_scope_gate(m)
    test_stale_reply_discard(m)
    test_lazy_loop(m)
    print("ALL LOGIC TESTS PASSED")


if __name__ == "__main__":
    main()
