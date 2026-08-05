"""M5 exit tests: the vocabulary each adapter actually maps, pinned.

`note_unknown_native` covers one direction: an agent adds an event kind we do
not recognise, and the run says so in its data-quality panel. Nothing covered
the other direction — a kind we *used* to map quietly becoming unmapped, or
quietly changing what it means.

Both are silent by construction. An `elif` branch deleted in a refactor turns
into an `ADAPTER_WARNING` at the bottom of a panel nobody scrolls to, and the
tool call it used to produce simply stops appearing; the run still parses, the
numbers still add up, and the only evidence is a count that got smaller. A
mapping moved from `execute` to `inspect` is worse still: nothing anywhere is
missing, and every action histogram ever compared across that boundary is
wrong.

So the golden file records, per adapter, what each native kind produced the day
it was captured: the event types, and the action if there was one. The rule is
the same one the protocol gate uses — **fail closed on removal, open on
addition**. A new native kind is the agent moving, which is expected and is
what the golden is regenerated for. A kind that disappears, or that starts
meaning something else, is us breaking.

Regenerate deliberately, never to make a red test green:

    python tests/test_seer_vocabulary.py --write
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from nebulai.seer.adapters import ClaudeStreamAdapter, CodexExecAdapter
from nebulai.seer.adapters.base import ADAPTER_VERSION
from nebulai.seer.adapters.observed import HOOK_ADAPTERS, hook_events
from nebulai.seer.contract import EventType

FIXTURES = Path(__file__).parent / "fixtures" / "seer"
GOLDEN = FIXTURES / "vocabulary-golden.json"


def _mk(cls, **kw):
    kw.setdefault("run_id", "run_v")
    kw.setdefault("session_id", "ses_v")
    return cls(**kw)


def _replay_lines(adapter, path: Path) -> list:
    events = []
    for line in path.read_text().splitlines():
        events.extend(adapter.feed(line))
    events.extend(adapter.finish())
    return events


def _replay_hooks(agent: str) -> list:
    """Every hook event we register, fired once in a plausible order.

    Unlike the stream adapters there is no captured fixture to replay: the
    hook payloads come from the agent at runtime. The vocabulary is still
    ours to pin — the set of hook names we route is a promise the installer
    writes into the user's config, and a name that stops being handled means
    a hook firing into nothing.
    """
    a = HOOK_ADAPTERS[agent](run_id="run_v", session_id="ses_v")
    events = []
    for i, name in enumerate(hook_events(agent)):
        # a tool payload for every hook: the ones that ignore it are unaffected,
        # and the pre/post pair needs a matching name to close a span
        payload = {
            "tool_name": "Bash",
            "tool_input": {"command": "pytest -q"},
            "session_id": "ses_v",
        }
        events.extend(a.feed_hook(_hook_line(agent, name, payload, i)))
    return events


def _hook_line(agent: str, event: str, payload: dict, i: int = 0):
    from nebulai.seer.spool import SpoolLine

    return SpoolLine(
        agent=agent,
        event=event,
        payload=payload,
        shim_ts=1_700_000_000.0 + i,
        seen_at=1_700_000_000.0 + i,
        pid=1,
    )


#: adapter name → how to produce a full replay of its captured vocabulary.
SOURCES = {
    "codex_exec_json": lambda: _replay_lines(
        _mk(CodexExecAdapter), FIXTURES / "codex-exec.jsonl"
    )
    + _replay_lines(_mk(CodexExecAdapter), FIXTURES / "codex-tools.jsonl"),
    "claude_stream_json": lambda: _replay_lines(
        _mk(ClaudeStreamAdapter), FIXTURES / "claude-stream.jsonl"
    )
    + _replay_lines(_mk(ClaudeStreamAdapter), FIXTURES / "claude-tools.jsonl"),
    "claude_hooks": lambda: _replay_hooks("claude"),
    "codex_hooks": lambda: _replay_hooks("codex"),
    "hermes_hooks": lambda: _replay_hooks("hermes"),
}


def inventory() -> dict[str, dict[str, dict]]:
    """native kind → what it produced, for every adapter we can replay.

    Keyed by `native_type` because that is the agent's word for the thing;
    keying by our own event type would make the file agree with itself no
    matter how the mapping drifted.
    """
    out: dict[str, dict[str, dict]] = {}
    for name, produce in SOURCES.items():
        kinds: dict[str, dict] = {}
        for e in produce():
            if e.event_type is EventType.ADAPTER_WARNING:
                continue
            key = e.native_type or "(none)"
            slot = kinds.setdefault(key, {"events": set(), "actions": set()})
            slot["events"].add(e.event_type.value)
            if e.action is not None:
                slot["actions"].add(e.action.value)
        out[name] = {
            k: {"events": sorted(v["events"]), "actions": sorted(v["actions"])}
            for k, v in sorted(kinds.items())
        }
    return out


def unknowns() -> dict[str, list[str]]:
    """Native kinds each replay could not map, by adapter. Expected to be
    empty for our own captures — a fixture we recorded and cannot fully read
    is a bug, not a fact about the agent."""
    out: dict[str, list[str]] = {}
    for name, produce in SOURCES.items():
        seen = []
        for e in produce():
            if e.event_type is EventType.ADAPTER_WARNING:
                note = str(e.payload.get("note") or "")
                if "native" in note or "unknown" in note.lower():
                    seen.append(note)
        out[name] = sorted(seen)
    return out


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover - only before the first write
        pytest.fail(f"{GOLDEN} is missing; run `python {__file__} --write`")
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def live() -> dict:
    return inventory()


class TestVocabulary:
    def test_no_adapter_lost_a_native_kind(self, golden, live):
        """Fail closed on removal. A kind in the golden that no longer appears
        means the branch that handled it is gone, and the events it used to
        produce are simply absent from every run captured since."""
        lost = {
            name: sorted(set(kinds) - set(live.get(name, {})))
            for name, kinds in golden["vocabulary"].items()
        }
        lost = {k: v for k, v in lost.items() if v}
        assert not lost, f"native kinds that stopped being mapped: {lost}"

    def test_no_native_kind_quietly_changed_meaning(self, golden, live):
        """The dangerous half. A kind that still maps, but to a different event
        type or a different action, breaks every comparison that spans the
        change — and nothing is missing, so nothing looks wrong."""
        drift = []
        for name, kinds in golden["vocabulary"].items():
            for kind, was in kinds.items():
                now = live.get(name, {}).get(kind)
                if now is None:
                    continue  # covered by the removal test, with a better message
                if now["events"] != was["events"] or now["actions"] != was["actions"]:
                    drift.append(f"{name}:{kind} {was} → {now}")
        assert not drift, "native kinds whose meaning moved: " + "; ".join(drift)

    def test_a_new_native_kind_is_allowed(self, golden, live):
        """Fail open on addition — the same rule as the protocol gate. An agent
        adding an event is the agent moving, not us breaking; this test exists
        to say so out loud, and to print what is new so regenerating the golden
        is a decision rather than a reflex."""
        added = {
            name: sorted(set(kinds) - set(golden["vocabulary"].get(name, {})))
            for name, kinds in live.items()
        }
        added = {k: v for k, v in added.items() if v}
        if added:
            print(f"new native kinds since the golden: {added}")

    def test_every_adapter_in_the_golden_still_exists(self, golden, live):
        assert set(golden["vocabulary"]) <= set(live), (
            "an adapter in the golden has no replay source any more: "
            f"{sorted(set(golden['vocabulary']) - set(live))}"
        )

    def test_our_own_fixtures_replay_with_nothing_unmapped(self):
        """A capture we recorded ourselves and cannot fully read is a bug in
        the adapter, not news about the agent. This is the test that would have
        caught the mapping being deleted even with no golden file at all."""
        assert unknowns() == {name: [] for name in SOURCES}

    def test_the_golden_names_the_versions_it_was_taken_from(self, golden):
        """Without the version, "the vocabulary changed" is unactionable: you
        cannot tell an agent upgrade from a refactor of ours."""
        assert golden["agents"], "the golden records no agent versions"
        for agent, version in golden["agents"].items():
            assert version, f"no version recorded for {agent}"

    def test_the_gate_notices_a_removal_and_a_drift(self, golden, live):
        """A test of the test. A golden comparison that passes no matter what
        the code does is the most expensive kind of green, and this one is
        built entirely out of set differences — exactly the shape that quietly
        compares nothing when a key name changes.
        """
        pretend = {
            "vocabulary": {
                "codex_exec_json": {
                    "item.deleted_kind": {"events": ["tool.completed"], "actions": ["execute"]},
                    # a kind that still exists, recorded as meaning something else
                    "item.command_execution": {"events": ["tool.completed"], "actions": ["report"]},
                }
            }
        }
        lost = set(pretend["vocabulary"]["codex_exec_json"]) - set(live["codex_exec_json"])
        assert "item.deleted_kind" in lost

        was = pretend["vocabulary"]["codex_exec_json"]["item.command_execution"]
        now = live["codex_exec_json"]["item.command_execution"]
        assert now["actions"] != was["actions"], "drift detection compares nothing"

    @pytest.mark.parametrize(
        "line,want",
        [
            ("codex-cli 0.144.6", "0.144.6"),
            ("2.1.222 (Claude Code)", "2.1.222"),
            # the shape that was being parsed as `2026.6.5)` — a build date,
            # with the bracket still attached
            ("Hermes Agent v0.16.0 (2026.6.5) · upstream a41d280f", "0.16.0"),
            ("codex-cli 0.146.0-alpha.9.2", "0.146.0-alpha.9.2"),
            ("", "unknown"),
        ],
    )
    def test_the_version_recorded_is_the_agents_version(self, line, want):
        """`agent_version` is stamped on every event and is what a later
        comparison keys on, so a mangled parse quietly partitions runs of the
        same agent into versions that never match."""
        from nebulai.seer.runner import parse_version

        assert parse_version(line) == want

    def test_every_hook_the_installer_registers_is_routed(self):
        """The installer writes one config entry per hook event; each must
        reach a handler. A name registered but unrouted is a hook firing into
        `note_unknown_native` forever, and the thing pointing at it is the
        user's own config.

        Checked against `ROUTES` rather than against the replay: some handlers
        correctly produce nothing on their own — `StopFailure` closes a turn,
        and closing a turn that never opened is a no-op, not a gap.
        """
        for agent in ("claude", "codex", "hermes"):
            routes = HOOK_ADAPTERS[agent].ROUTES
            missing = [e for e in hook_events(agent) if e not in routes]
            assert not missing, f"{agent} registers unrouted hooks: {missing}"


def _write() -> None:
    """Regenerate the golden. Deliberate, and separate from the test run.

    The agent versions are read from the installed binaries at write time
    rather than typed in: a golden that claims a version nobody verified is
    worse than one that admits the binary was not on PATH.
    """
    from nebulai.seer.runner import agent_version

    doc = {
        "note": (
            "Recorded by `python tests/test_seer_vocabulary.py --write`. Fail "
            "closed on removal, open on addition: a native kind that disappears "
            "or changes meaning is a regression in our adapters; a new one is "
            "the agent moving. Regenerate when an agent adds vocabulary, never "
            "to make a red test green."
        ),
        "agents": {
            agent: agent_version(agent) or "not installed when this was written"
            for agent in ("codex", "claude", "hermes")
        },
        "adapter_version": ADAPTER_VERSION,
        "vocabulary": inventory(),
    }
    GOLDEN.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    if "--write" in sys.argv:
        _write()
    else:
        print(json.dumps(inventory(), indent=2, sort_keys=True))
