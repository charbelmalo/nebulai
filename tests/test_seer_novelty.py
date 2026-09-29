"""`Effect.NO_NEW_INFORMATION`: emitted where the payload decides it, absent
where it does not.

Every test here is really one of two assertions. Either *this shape is
decidable and the label says how it was decided*, or *this shape is not
decidable and nothing was labelled* — and the second kind matters more, because
the failure mode this whole module guards against is a plausible guess arriving
in a loop count that a researcher then reads as a measurement.

The last class is the honesty gate the task named: a run with no decidable
repetition reports `hits: None` / `fidelity: missing`, never `hits: 0`.
"""

from __future__ import annotations

import json

import pytest

from nebulai.seer.adapters import ClaudeStreamAdapter, CodexExecAdapter
from nebulai.seer.adapters.novelty import (
    NoveltyLedger,
    ZERO_RESULT_SENTINELS,
    is_zero_result,
    lookup_target,
)
from nebulai.seer.analysis import loop_rules
from nebulai.seer.contract import Action, Effect, EventType, Fidelity
from nebulai.seer.reducer import Reducer


def mk(cls, **kw):
    kw.setdefault("run_id", "run_t")
    kw.setdefault("session_id", "ses_t")
    return cls(**kw)


def tool_use(use_id: str, name: str, **inp) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {
                "id": f"msg_{use_id}",
                "content": [
                    {"type": "tool_use", "id": use_id, "name": name, "input": inp}
                ],
            },
        }
    )


def tool_result(use_id: str, text: str, *, is_error: bool = False) -> str:
    return json.dumps(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": use_id,
                        "content": text,
                        "is_error": is_error,
                    }
                ]
            },
        }
    )


def completions(events):
    return [
        e
        for e in events
        if e.event_type in (EventType.TOOL_COMPLETED, EventType.TOOL_FAILED)
    ]


# ── the ledger on its own ────────────────────────────────────────────────────


class TestLedger:
    def test_a_first_look_decides_nothing(self) -> None:
        """The baseline the rest of the file rests on: one action cannot be a
        repeat of anything, so the ledger returns `None` — which the adapter
        reads as *not decidable*, never as "there was new information"."""
        led = NoveltyLedger()
        assert led.decide(action=Action.INSPECT, target="a.py") is None

    def test_the_same_read_twice_with_no_edit_between_is_deterministic(self) -> None:
        led = NoveltyLedger()
        led.decide(action=Action.INSPECT, target="a.py")
        v = led.decide(action=Action.INSPECT, target="a.py")
        assert v is not None
        assert v.effect is Effect.NO_NEW_INFORMATION
        assert v.fidelity is Fidelity.DETERMINISTIC
        assert v.rule == "repeat_read"

    def test_an_edit_between_two_reads_makes_the_second_new_again(self) -> None:
        """The rule is "nothing wrote to it in between", not "we saw it once".
        Without this the ledger would label the read-after-write that every
        agent does to check its own edit — the single most informative read in
        a run — as surfacing nothing."""
        led = NoveltyLedger()
        led.decide(action=Action.INSPECT, target="a.py")
        led.note_edit("a.py")
        assert led.decide(action=Action.INSPECT, target="a.py") is None

    def test_an_edit_to_another_file_does_not_excuse_a_scoped_repeat(self) -> None:
        """When the caller names the file this lookup read, only edits to that
        file retire it."""
        led = NoveltyLedger()
        led.decide(action=Action.INSPECT, target="a.py", path="a.py")
        led.note_edit("b.py")
        v = led.decide(action=Action.INSPECT, target="a.py", path="a.py")
        assert v is not None and v.rule == "repeat_read"

    def test_an_unscoped_lookup_is_retired_by_any_edit(self) -> None:
        """A search ranges over a tree the payload does not bound, so any edit
        could have changed its answer. Conservative on purpose: the error that
        matters is calling a search that found something new a repeat."""
        led = NoveltyLedger()
        led.decide(action=Action.SEARCH, target="grep:TODO")
        led.note_edit("b.py")
        assert led.decide(action=Action.SEARCH, target="grep:TODO") is None

    def test_byte_identical_output_is_deterministic(self) -> None:
        led = NoveltyLedger()
        led.decide(action=Action.EXECUTE, command="pytest -q", output="1 passed")
        v = led.decide(action=Action.EXECUTE, command="pytest -q", output="1 passed")
        assert v is not None
        assert v.fidelity is Fidelity.DETERMINISTIC
        assert v.rule == "identical_output"

    def test_output_that_differs_by_one_byte_is_not_a_repeat(self) -> None:
        led = NoveltyLedger()
        led.decide(action=Action.EXECUTE, command="pytest -q", output="1 passed")
        assert (
            led.decide(action=Action.EXECUTE, command="pytest -q", output="2 passed")
            is None
        )

    def test_the_ledger_keeps_a_digest_and_never_the_output(self) -> None:
        """R-B costs the log nothing. If the text were retained the rule would
        drag every command event up to the `content` rung, and a metadata-level
        export would lose the effect label along with it."""
        led = NoveltyLedger()
        secret = "sk-live-not-a-real-key-0123456789"
        led.decide(action=Action.EXECUTE, command="printenv", output=secret)
        blob = json.dumps(
            {"seen": led._seen, "edited": led._edited, "output": led._output}
        )
        assert secret not in blob

    def test_a_zero_result_search_is_heuristic_not_deterministic(self) -> None:
        """R-C matches prose against a list we wrote. It earns `HEURISTIC` and
        must never be dressed up as an exact comparison."""
        led = NoveltyLedger()
        v = led.decide(action=Action.SEARCH, target="g:x", output="No matches found")
        assert v is not None
        assert v.effect is Effect.NO_NEW_INFORMATION
        assert v.fidelity is Fidelity.HEURISTIC
        assert v.rule == "zero_result_search"

    def test_a_search_result_we_do_not_recognise_gets_no_label(self) -> None:
        led = NoveltyLedger()
        assert (
            led.decide(action=Action.SEARCH, target="g:x", output="nada, zilch")
            is None
        )

    def test_zero_result_only_applies_to_searches(self) -> None:
        """An empty-bodied *read* is a real fact about the file — an empty file
        — not a failed lookup, and labelling it as one would be a guess."""
        led = NoveltyLedger()
        assert led.decide(action=Action.INSPECT, target="empty.py", output="") is None

    @pytest.mark.parametrize("s", ZERO_RESULT_SENTINELS)
    def test_every_sentinel_is_matched_case_insensitively(self, s: str) -> None:
        assert is_zero_result(s.upper() + " for pattern 'x'")

    def test_an_exact_repeat_outranks_the_heuristic(self) -> None:
        """When both R-B and R-C could fire, the exact comparison wins, so the
        reported fidelity is the strongest one actually available."""
        led = NoveltyLedger()
        led.decide(action=Action.SEARCH, target="g:x", output="No matches found")
        v = led.decide(action=Action.SEARCH, target="g:x", output="No matches found")
        assert v is not None and v.rule == "identical_output"
        assert v.fidelity is Fidelity.DETERMINISTIC


class TestLookupTarget:
    def test_a_different_window_of_the_same_file_is_a_different_read(self) -> None:
        a = lookup_target("Read", {"file_path": "/a.py", "offset": 1, "limit": 50})
        b = lookup_target("Read", {"file_path": "/a.py", "offset": 400, "limit": 50})
        assert a != b

    def test_a_tool_we_have_not_mapped_has_no_key(self) -> None:
        """Codex and Hermes name their tools differently. R-A simply does not
        fire for them, which is better than keying on a guessed name."""
        assert lookup_target("some_mcp_tool", {"path": "/a.py"}) is None


# ── through the live adapters ────────────────────────────────────────────────


class TestClaudeAdapter:
    def test_the_second_identical_read_carries_the_rule_and_its_fidelity(self) -> None:
        """A Claude `tool_result` carries the file body, so the two reads are
        compared byte for byte and the rule that fires is `identical_output` —
        the stronger evidence of the two, and the reason R-B is tried first.
        R-A is what decides a repeat when no body reached us."""
        a = mk(ClaudeStreamAdapter)
        ev: list = []
        for use_id in ("t1", "t2"):
            ev += a.feed(tool_use(use_id, "Read", file_path="/a.py"))
            ev += a.feed(tool_result(use_id, "line one\nline two"))
        first, second = completions(ev)
        assert first.effect is Effect.UNKNOWN
        assert "effect_rule" not in first.payload
        assert second.effect is Effect.NO_NEW_INFORMATION
        assert second.payload["effect_rule"] == "identical_output"
        assert second.payload["effect_fidelity"] == Fidelity.DETERMINISTIC.value

    def test_a_repeat_with_no_body_still_decides_by_r_a(self) -> None:
        """Codex's `web_search` item and Claude's block-list results with no
        text both land here: no body to compare, but the same lookup twice."""
        a = mk(ClaudeStreamAdapter)
        ev: list = []
        for use_id in ("t1", "t2"):
            ev += a.feed(tool_use(use_id, "Glob", pattern="**/*.py", path="/src"))
            ev += a.feed(
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": use_id,
                                    "content": [{"type": "image", "source": {}}],
                                }
                            ]
                        },
                    }
                )
            )
        second = completions(ev)[-1]
        assert second.payload["effect_rule"] == "repeat_read"

    def test_the_event_stays_native_while_its_effect_is_a_heuristic(self) -> None:
        """Two fidelities, two meanings. `source.fidelity` is about the event —
        Claude told us the call finished, which is native. `effect_fidelity` is
        about the label we attached to it. Collapsing them would either
        downgrade the event or launder the guess."""
        a = mk(ClaudeStreamAdapter)
        ev: list = []
        ev += a.feed(tool_use("t1", "Grep", pattern="zzz", path="/src"))
        ev += a.feed(tool_result("t1", "No matches found"))
        (done,) = completions(ev)
        assert done.source.fidelity is Fidelity.NATIVE
        assert done.payload["effect_fidelity"] == Fidelity.HEURISTIC.value
        assert done.effect is Effect.NO_NEW_INFORMATION

    def test_a_read_after_an_edit_to_that_file_is_not_labelled(self) -> None:
        a = mk(ClaudeStreamAdapter)
        ev: list = []
        ev += a.feed(tool_use("t1", "Read", file_path="/a.py"))
        ev += a.feed(tool_result("t1", "old"))
        ev += a.feed(tool_use("t2", "Write", file_path="/a.py", content="new\n"))
        ev += a.feed(tool_result("t2", "ok"))
        ev += a.feed(tool_use("t3", "Read", file_path="/a.py"))
        ev += a.feed(tool_result("t3", "new"))
        last = completions(ev)[-1]
        assert last.effect is Effect.UNKNOWN
        assert "effect_rule" not in last.payload

    def test_a_failed_repeat_keeps_failed(self) -> None:
        """`FAILED` is the agent's own word. A repeated failure is R3's
        business (`repeated_failure`), not a novelty label, and overwriting the
        failure would hide it from both."""
        a = mk(ClaudeStreamAdapter)
        ev: list = []
        for use_id in ("t1", "t2"):
            ev += a.feed(tool_use(use_id, "Read", file_path="/gone.py"))
            ev += a.feed(tool_result(use_id, "ENOENT", is_error=True))
        for e in completions(ev):
            assert e.effect is Effect.FAILED
            assert "effect_rule" not in e.payload

    def test_an_edit_result_keeps_state_changed(self) -> None:
        a = mk(ClaudeStreamAdapter)
        ev: list = []
        for use_id in ("t1", "t2"):
            ev += a.feed(tool_use(use_id, "Write", file_path="/a.py", content="x\n"))
            ev += a.feed(tool_result(use_id, "ok"))
        for e in completions(ev):
            assert e.effect is Effect.STATE_CHANGED


class TestCodexAdapter:
    @staticmethod
    def _cmd(item_id: str, command: str, output: str) -> str:
        return json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": item_id,
                    "type": "command_execution",
                    "command": command,
                    "exit_code": 0,
                    "status": "completed",
                    "aggregated_output": output,
                },
            }
        )

    def test_a_command_repeated_with_identical_output_is_labelled(self) -> None:
        a = mk(CodexExecAdapter)
        ev = a.feed(self._cmd("i1", "pytest -q", "1 passed"))
        ev += a.feed(self._cmd("i2", "pytest -q", "1 passed"))
        first, second = completions(ev)
        assert first.effect is Effect.STATE_CHANGED
        assert second.effect is Effect.NO_NEW_INFORMATION
        assert second.payload["effect_rule"] == "identical_output"

    def test_a_command_whose_output_changed_keeps_its_fallback(self) -> None:
        a = mk(CodexExecAdapter)
        ev = a.feed(self._cmd("i1", "pytest -q", "1 passed"))
        ev += a.feed(self._cmd("i2", "pytest -q", "1 failed"))
        assert completions(ev)[-1].effect is Effect.STATE_CHANGED

    def test_the_same_web_search_twice_is_a_repeat(self) -> None:
        a = mk(CodexExecAdapter)
        ev: list = []
        for i in ("i1", "i2"):
            ev += a.feed(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"id": i, "type": "web_search", "query": "pca null"},
                    }
                )
            )
        assert completions(ev)[-1].effect is Effect.NO_NEW_INFORMATION


# ── the honesty gate ─────────────────────────────────────────────────────────


def _reduce(events):
    r = Reducer("run_t")
    r.reduce(events)
    return r.finalize()


def _r1(view, events) -> dict:
    a = loop_rules(view, events)
    return next(r for r in a.rows if r["rule"] == "no_new_information_streak")


class TestMissingIsNotZero:
    def test_a_run_with_nothing_decidable_reports_missing_not_zero(self) -> None:
        """The rule the whole feature is built around. Four distinct reads
        decide nothing, so the streak count is *absent*. `0` would claim we
        looked and found no loops, which is a different and unearned claim."""
        a = mk(ClaudeStreamAdapter)
        ev: list = []
        for i, path in enumerate(["/a.py", "/b.py", "/c.py", "/d.py"]):
            ev += a.feed(tool_use(f"t{i}", "Read", file_path=path))
            ev += a.feed(tool_result(f"t{i}", f"contents of {path}"))
        row = _r1(_reduce(ev), ev)
        assert row["hits"] is None
        assert row["fidelity"] == Fidelity.MISSING.value
        assert row["hits"] != 0

    def test_a_decidable_run_reports_a_count_at_its_weakest_fidelity(self) -> None:
        """Three exact repeats plus one zero-result search: the streak is real,
        and it is `heuristic` because one of the spans in it was decided by a
        sentence list. Reporting `deterministic` would let the proxy travel to
        the reader labelled as a measurement."""
        a = mk(ClaudeStreamAdapter)
        ev: list = []
        ev += a.feed(tool_use("t0", "Read", file_path="/a.py"))
        ev += a.feed(tool_result("t0", "body"))
        for i in (1, 2, 3):
            ev += a.feed(tool_use(f"t{i}", "Read", file_path="/a.py"))
            ev += a.feed(tool_result(f"t{i}", "body"))
        ev += a.feed(tool_use("t9", "Grep", pattern="zzz", path="/src"))
        ev += a.feed(tool_result("t9", "No matches found"))
        row = _r1(_reduce(ev), ev)
        assert row["hits"] == 1
        assert row["fidelity"] == Fidelity.HEURISTIC.value
        assert "zero_result_search" in row["rules_used"]

    def test_a_search_is_retired_by_an_edit_anywhere(self) -> None:
        """A `Grep` ranges over a tree the payload does not bound, so any edit
        could have changed its answer. Scoping it to a path we guessed at is
        the one error that matters: calling a search that found something new
        a repeat."""
        a = mk(ClaudeStreamAdapter)
        ev: list = []
        ev += a.feed(tool_use("t1", "Grep", pattern="TODO", path="/src"))
        ev += a.feed(tool_result("t1", [{"type": "image", "source": {}}]))
        ev += a.feed(tool_use("t2", "Write", file_path="/elsewhere.py", content="x\n"))
        ev += a.feed(tool_result("t2", "ok"))
        ev += a.feed(tool_use("t3", "Grep", pattern="TODO", path="/src"))
        ev += a.feed(tool_result("t3", [{"type": "image", "source": {}}]))
        assert completions(ev)[-1].effect is Effect.UNKNOWN

    def test_an_all_deterministic_run_is_not_downgraded(self) -> None:
        a = mk(ClaudeStreamAdapter)
        ev: list = []
        ev += a.feed(tool_use("t0", "Read", file_path="/a.py"))
        ev += a.feed(tool_result("t0", "body"))
        for i in (1, 2, 3):
            ev += a.feed(tool_use(f"t{i}", "Read", file_path="/a.py"))
            ev += a.feed(tool_result(f"t{i}", "body"))
        row = _r1(_reduce(ev), ev)
        assert row["hits"] == 1
        assert row["fidelity"] == Fidelity.DETERMINISTIC.value
        assert row["rules_used"] == ["identical_output"]
