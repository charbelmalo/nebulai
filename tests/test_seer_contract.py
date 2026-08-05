"""M0 exit tests for the SessionSeer canonical contract.

The one that matters is `test_no_agent_specific_fields_leak`: the whole point of
the contract is that a Codex event and a Claude event are indistinguishable to
the analysis layer once normalized. If an adapter can smuggle `requestId` or
`thread_id` through, SessionSeer becomes three dashboards sharing a nav bar and
every cross-agent comparison silently stops being apples-to-apples.
"""

import pytest

from nebulai.seer.contract import (
    ANALYSIS_KEYS,
    Action,
    CaptureMode,
    Effect,
    Event,
    EventType,
    Fidelity,
    Outcome,
    SessionState,
    Source,
    fold_key,
    new_run_id,
    new_session_id,
)
from nebulai.seer.taxonomy import classify_command, classify_tool, unmatched_tools


def _src(agent: str = "codex", **kw) -> Source:
    return Source(
        agent=agent,
        agent_version="0.144.6",
        adapter=f"{agent}_exec_json",
        adapter_version="0.1.0",
        capture_mode=kw.get("capture_mode", CaptureMode.DRIVEN),
        fidelity=kw.get("fidelity", Fidelity.NATIVE),
        source_event_id=kw.get("source_event_id"),
    )


def _evt(**kw) -> Event:
    # ts/mono_ns/event_id are pinned so "the same logical event from three
    # agents" really is the same event; leaving the factories to fire would
    # make the leak test pass or fail on clock jitter instead of on schema.
    base = dict(
        event_type=EventType.TOOL_COMPLETED,
        source=_src(),
        run_id="run_1",
        session_id="ses_1",
        event_id="evt_fixed",
        ts=1_000_000.5,
        mono_ns=42,
    )
    base.update(kw)
    return Event(**base)


# ── the exit criterion ───────────────────────────────────────────────────────


def test_no_agent_specific_fields_leak():
    """Three agents, the same logical event, identical analysis-visible shape."""
    codex = _evt(
        source=_src("codex", source_event_id="item_123"),
        native_type="commandExecution",
        native={"item_id": "item_123", "thread_id": "th_9"},
        action=Action.VERIFY,
        effect=Effect.STATE_CHANGED,
        payload={"status": "completed", "duration_ms": 1200, "exit_code": 0},
    )
    claude = _evt(
        source=_src("claude", source_event_id="toolu_9"),
        native_type="Bash",
        native={"requestId": "req_a", "tool_use_id": "toolu_9"},
        action=Action.VERIFY,
        effect=Effect.STATE_CHANGED,
        payload={"status": "completed", "duration_ms": 1200, "exit_code": 0},
    )
    hermes = _evt(
        source=_src("hermes", source_event_id="tc_4"),
        native_type="shell",
        native={"tool_call_id": "tc_4", "spawn_tree": "st_1"},
        action=Action.VERIFY,
        effect=Effect.STATE_CHANGED,
        payload={"status": "completed", "duration_ms": 1200, "exit_code": 0},
    )

    def analysis_view(e: Event) -> dict:
        d = e.to_dict()
        return {k: v for k, v in d.items() if k in ANALYSIS_KEYS and k != "source"}

    a, b, c = analysis_view(codex), analysis_view(claude), analysis_view(hermes)
    assert a == b == c, "normalized events must be indistinguishable to analysis"

    # and the native truth is still there for display/audit
    assert codex.native["thread_id"] == "th_9"
    assert claude.native["requestId"] == "req_a"


def test_native_keys_are_not_analysis_keys():
    """`native`/`native_type` must stay out of the analysis surface — a metric
    reading them has re-introduced an agent-specific code path."""
    assert "native" not in ANALYSIS_KEYS
    assert "native_type" not in ANALYSIS_KEYS


def test_round_trip_is_lossless():
    e = _evt(
        turn_id="turn_1",
        span_id="span_1",
        parent_span_id="span_0",
        action=Action.EDIT,
        effect=Effect.STATE_CHANGED,
        native_type="Write",
        native={"x": 1},
        repo={"branch": "main", "head": "abc", "dirty": True},
        model={"provider": "openai", "model_id": "gpt-5"},
        payload={"status": "completed"},
    )
    again = Event.from_dict(e.to_dict())
    assert again.to_dict() == e.to_dict()


def test_from_dict_accepts_its_own_json():
    import json

    e = _evt(action=Action.SEARCH, effect=Effect.NO_NEW_INFORMATION)
    assert Event.from_dict(json.loads(e.to_json())).action is Action.SEARCH


# ── rule 1: provenance ───────────────────────────────────────────────────────


def test_missing_and_dropped_by_policy_are_distinct():
    """Codex streams reasoning; refusing it is a decision, not an absence.
    Collapsing the two would tell the researcher 'we don't know' when the truth
    is 'we chose not to look'."""
    assert Fidelity.MISSING is not Fidelity.DROPPED_BY_POLICY
    assert Fidelity.MISSING.value != Fidelity.DROPPED_BY_POLICY.value


def test_every_source_carries_a_fidelity():
    with pytest.raises(TypeError):
        Source(  # type: ignore[call-arg]
            agent="codex",
            agent_version="1",
            adapter="a",
            adapter_version="1",
            capture_mode=CaptureMode.DRIVEN,
        )


# ── rule 2: deltas never feed counters ───────────────────────────────────────


@pytest.mark.parametrize(
    "et,expected",
    [
        (EventType.MESSAGE_ASSISTANT_DELTA, EventType.MESSAGE_ASSISTANT_COMPLETED),
        (EventType.TOOL_OUTPUT_DELTA, EventType.TOOL_COMPLETED),
        (EventType.TOOL_COMPLETED, EventType.TOOL_COMPLETED),
        (EventType.TURN_COMPLETED, EventType.TURN_COMPLETED),
    ],
)
def test_fold_key(et, expected):
    assert fold_key(et) is expected


def test_is_delta_flags_exactly_the_streaming_families():
    deltas = {et for et in EventType if et.is_delta}
    assert deltas == {
        EventType.MESSAGE_ASSISTANT_DELTA,
        EventType.TOOL_OUTPUT_DELTA,
    }


def test_every_delta_has_a_distinct_fold_target():
    """A delta whose fold key is itself would let a stream feed counters."""
    for et in EventType:
        if et.is_delta:
            assert fold_key(et) is not et


# ── rule 3: 9 actions × effect ───────────────────────────────────────────────


def test_taxonomy_is_nine_actions():
    assert len(list(Action)) == 9


@pytest.mark.parametrize(
    "tool,expected",
    [
        ("Read", Action.INSPECT),
        ("mcp__workspace__Read", Action.INSPECT),
        ("Grep", Action.SEARCH),
        ("Glob", Action.SEARCH),
        ("Write", Action.EDIT),
        ("Edit", Action.EDIT),
        ("mcp__tokensave__str_replace", Action.EDIT),
        ("Bash", Action.EXECUTE),
        ("commandExecution", Action.EXECUTE),
        ("Task", Action.DELEGATE),
        ("Agent", Action.DELEGATE),
        ("Artifact", Action.REPORT),
        ("AskUserQuestion", Action.INTERACT),
        ("WebSearch", Action.SEARCH),
        ("WebFetch", Action.INSPECT),
    ],
)
def test_classify_tool(tool, expected):
    assert classify_tool(tool) is expected


def test_unknown_tool_falls_to_inspect_not_a_guess():
    """INSPECT is the least-committal label: it cannot fabricate a verification
    or an edit the run never performed."""
    assert classify_tool("ZzzUnknownTool") is Action.INSPECT
    assert unmatched_tools(["ZzzUnknownTool", "Read"]) == ["ZzzUnknownTool"]


@pytest.mark.parametrize(
    "cmd,expected",
    [
        ("pytest tests/ -q", Action.VERIFY),
        ("uv run pytest", Action.VERIFY),
        ("npm test", Action.VERIFY),
        ("npm run typecheck", Action.VERIFY),
        ("npm run lint", Action.VERIFY),
        ("cargo clippy", Action.VERIFY),
        ("cargo test --all", Action.VERIFY),
        ("go vet ./...", Action.VERIFY),
        ("npx tsc --noEmit", Action.VERIFY),
        ("CI=1 pytest -x", Action.VERIFY),
        ("git status", Action.VCS),
        ("gh pr list", Action.VCS),
        ("npm install", Action.EXECUTE),
        ("npm run dev", Action.EXECUTE),
        ("cargo run", Action.EXECUTE),
        ("python train.py", Action.EXECUTE),
        ("echo hi", Action.EXECUTE),
    ],
)
def test_classify_command(cmd, expected):
    assert classify_command(cmd) is expected


def test_verify_detection_is_anchored_to_the_program():
    """`rm -rf test/` contains 'test' and must never count as verification —
    a false verify fabricates coverage the run did not have."""
    assert classify_command("rm -rf test/") is Action.EXECUTE
    assert classify_command("cat tests/test_foo.py") is Action.EXECUTE
    assert classify_command("mkdir -p build") is Action.EXECUTE


# ── outcome and state ────────────────────────────────────────────────────────


def test_claimed_and_verified_never_collapse():
    assert Outcome.AGENT_CLAIMED_COMPLETE is not Outcome.VERIFIED_PASS
    assert Outcome.UNVERIFIED_COMPLETE is not Outcome.VERIFIED_PASS


def test_stalled_is_an_overlay_not_a_state():
    """Making 'stalled' a state would erase what it is stalled doing, which is
    the only actionable half."""
    assert "stalled" not in {s.value for s in SessionState}


def test_ids_are_prefixed_and_unique():
    assert new_run_id().startswith("run_")
    assert new_session_id().startswith("ses_")
    assert new_run_id() != new_run_id()
