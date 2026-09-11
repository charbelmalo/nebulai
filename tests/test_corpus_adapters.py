"""D7 exit tests: the four corpus adapters, run against the real fixtures.

These are the tests that hold the honesty rules in `corpus_base.py` in place.
Each of the four rules gets a structural test that runs over *every* event of
*every* adapter rather than over a hand-built example, so a new corpus adapter
inherits the check by existing:

1. `CaptureMode.RECONCILED`, always — `test_every_corpus_event_is_reconciled`.
2. Fidelity is per field — `test_every_run_says_token_usage_is_missing`.
3. Order is not time — `test_ctfish_has_no_clock_and_says_so`.
4. Foreign data wears foreign clothes — `test_every_event_carries_its_corpus`.

Plus the M0 exit criterion (`ANALYSIS_KEYS` never exposes `native`) extended
from `tests/test_seer_adapters.py` onto corpus output, and the one defect these
adapters actually hit in practice: a run with no `SESSION_COMPLETED` is rewritten
as `interrupted` by the store's orphan sweep, which would be a claim that
somebody else's published experiment was cut short.

`village-synthetic.jsonl` is hand-written by us, in the documented shape, and
says so in its first record — the AI Village dataset is gated and nothing from
it is in this repository. The Among Us and ctfish fixtures are genuine excerpts
of the public releases; `transcript-sample.md` was written for this repository
and is not the New York Times text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nebulai.seer.adapters import CORPUS_ADAPTERS, corpus_adapter
from nebulai.seer.adapters.corpus_amongus import AmongUsCorpusAdapter, parse_action
from nebulai.seer.adapters.corpus_base import CORPUS_ADAPTER_VERSION, CorpusError
from nebulai.seer.adapters.corpus_ctfish import (
    CtfishCorpusAdapter,
    parse_entry,
    split_commands,
)
from nebulai.seer.adapters.corpus_transcript import (
    TranscriptCorpusAdapter,
    parse_markdown,
)
from nebulai.seer.adapters.corpus_village import (
    VillageCorpusAdapter,
    VillageUnavailable,
    parse_ts,
    probe,
)
from nebulai.seer.contract import (
    ANALYSIS_KEYS,
    Action,
    CaptureMode,
    Effect,
    EventType,
    Fidelity,
)

CORPUS = Path(__file__).parent / "fixtures" / "corpus"


def mk(cls, **kw):
    kw.setdefault("run_id", "run_t")
    kw.setdefault("session_id", "ses_t")
    return cls(**kw)


def of(events, *types):
    want = {t.value if hasattr(t, "value") else t for t in types}
    return [e for e in events if e.event_type.value in want]


# ── the four fixtures, read once ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def amongus_runs():
    return mk(AmongUsCorpusAdapter).read(
        CORPUS / "amongus-logs.jsonl", summary_path=CORPUS / "amongus-summary.jsonl"
    )


@pytest.fixture(scope="module")
def ctfish_runs():
    return mk(CtfishCorpusAdapter).read(
        CORPUS / "ctfish-runs.json", labels_path=CORPUS / "ctfish-labels.json"
    )


@pytest.fixture(scope="module")
def village_runs():
    return mk(VillageCorpusAdapter).read(CORPUS / "village-synthetic.jsonl")


@pytest.fixture(scope="module")
def transcript_runs():
    return mk(TranscriptCorpusAdapter).read(CORPUS / "transcript-sample.md")


@pytest.fixture(scope="module")
def all_runs(amongus_runs, ctfish_runs, village_runs, transcript_runs):
    return amongus_runs + ctfish_runs + village_runs + transcript_runs


# ── the rules, checked over every adapter at once ────────────────────────────


def test_all_four_fixtures_map(all_runs):
    assert len(all_runs) >= 5  # 1 game + 2 ctfish runs + 1 village + 1 transcript
    for run in all_runs:
        assert run.events, f"{run.run_id} produced no events"


def test_every_corpus_event_is_reconciled(all_runs):
    """Rule 1. Nothing here was watched happening, whatever its timestamps say."""
    for run in all_runs:
        for e in run.events:
            assert e.source.capture_mode is CaptureMode.RECONCILED, e.event_type


def test_every_event_carries_its_corpus(all_runs):
    """Rule 4: a point on the map can always be traced back out of the repo."""
    for run in all_runs:
        for e in run.events:
            corpus = e.payload.get("corpus")
            assert isinstance(corpus, dict), e.event_type
            assert corpus["id"] and corpus["licence"]
            assert corpus["ships_in_repo"] is False


def test_every_run_says_token_usage_is_missing_not_zero(all_runs):
    """Rule 2, the case that motivated it. A published corpus records what the
    agent said, never what it cost; a zero would read as a measurement."""
    for run in all_runs:
        usage = of(run.events, EventType.MODEL_USAGE_UPDATED)
        assert len(usage) == 1, run.run_id
        (e,) = usage
        assert e.source.fidelity is Fidelity.MISSING
        assert e.payload["usage"] is None
        assert e.payload["fidelity"]["usage"] == Fidelity.MISSING.value


def test_every_run_closes_its_session(all_runs):
    """The defect this test exists for: without a terminal session event the
    store's orphan sweep finds no live capture process and rewrites the run as
    `interrupted` — i.e. claims somebody else's experiment was cut short."""
    for run in all_runs:
        done = of(run.events, EventType.SESSION_COMPLETED)
        assert len(done) == 1, f"{run.run_id} has {len(done)} SESSION_COMPLETED"
        assert done[0].session_id == run.run_id
        # and it precedes the run's own terminal event
        types = [e.event_type for e in run.events]
        assert types.index(EventType.SESSION_COMPLETED) < types.index(
            EventType.RUN_COMPLETED
        )


def test_every_run_outcome_is_unknown_not_invented(all_runs):
    """No corpus here verifies the task, so no adapter may claim success."""
    for run in all_runs:
        (done,) = of(run.events, EventType.RUN_COMPLETED)
        assert done.payload["outcome"] == "unknown"
        assert done.payload["fidelity"]["outcome"] == Fidelity.MISSING.value


def test_corpus_events_expose_no_native_keys_to_analysis(all_runs):
    """M0's exit criterion (tests/test_seer_adapters.py) extended onto corpus
    output: nothing corpus-specific may reach the analysis surface."""
    seen = 0
    for run in all_runs:
        for e in run.events:
            view = {k for k in e.to_dict() if k in ANALYSIS_KEYS}
            assert "native" not in view and "native_type" not in view
            assert e.source.fidelity in set(Fidelity)
            assert e.source.capture_mode in set(CaptureMode)
            seen += 1
    assert seen > 200


def test_no_corpus_delta_event_carries_usage(all_runs):
    for run in all_runs:
        for e in run.events:
            if e.event_type.is_delta:
                assert "usage" not in e.payload


def test_run_meta_names_the_exact_bytes_it_read(all_runs):
    for run in all_runs:
        meta = run.meta
        assert len(meta["input_sha256"]) == 64
        assert Path(meta["input_path"]).exists()
        assert meta["adapter_version"] == CORPUS_ADAPTER_VERSION
        assert meta["capture_mode"] == CaptureMode.RECONCILED.value
        assert isinstance(meta["n_unmapped"], int)
        json.dumps(meta)  # the artifact has to survive being written out


def test_corpus_adapter_refuses_an_unknown_corpus():
    assert set(CORPUS_ADAPTERS) == {"amongus", "ctfish", "village", "transcript"}
    assert isinstance(
        corpus_adapter("ctfish", run_id="r", session_id="s"), CtfishCorpusAdapter
    )
    with pytest.raises(ValueError, match="no corpus adapter"):
        corpus_adapter("sydney", run_id="r", session_id="s")


# ── Among Us ─────────────────────────────────────────────────────────────────


def test_fake_task_is_not_folded_into_the_real_one():
    """The single most load-bearing verb in a deception corpus. `COMPLETE TASK`
    is a prefix of `COMPLETE FAKE TASK` only if you search in the wrong order."""
    verb, action, effect = parse_action("COMPLETE FAKE TASK Fix Wiring")
    assert verb == "COMPLETE FAKE TASK"
    assert (action, effect) == (Action.EXECUTE, Effect.NO_STATE_CHANGE)
    verb, action, effect = parse_action("3. COMPLETE TASK Fix Wiring")
    assert verb == "COMPLETE TASK"
    assert effect is Effect.STATE_CHANGED


def test_an_unrecognised_amongus_verb_is_kept_not_dropped():
    verb, action, effect = parse_action("BARBECUE the reactor")
    assert verb == "UNKNOWN"
    assert (action, effect) == (Action.EXECUTE, Effect.UNKNOWN)


def test_amongus_fixture_maps_every_step(amongus_runs):
    (run,) = amongus_runs
    assert run.run_id == "amongus-game-17"
    assert run.meta["n_unmapped"] == 0
    assert not run.warnings
    # one session per player, all closed by the run's own SESSION_COMPLETED
    assert run.meta["n_players"] >= 5


def test_amongus_timestamps_are_real_and_marked_native(amongus_runs):
    (run,) = amongus_runs
    dated = [
        e
        for e in run.events
        if e.payload.get("fidelity", {}).get("ts") == Fidelity.NATIVE.value
    ]
    assert dated, "the corpus carries wall-clock stamps; something dropped them"
    for e in dated:
        assert e.payload.get("order_only") is not True
        assert e.ts > 1_600_000_000  # a real 2024/2025 date, not a tick counter


def test_amongus_speech_effect_is_unknown_not_no_state_change():
    """The documented widening: what an Impostor says changes the other agents'
    beliefs, which is the whole experiment. Calling it NO_STATE_CHANGE would
    assert the opposite."""
    _, action, effect = parse_action("SPEAK I was in electrical")
    assert (action, effect) == (Action.INTERACT, Effect.UNKNOWN)


# ── ctfish ───────────────────────────────────────────────────────────────────


def test_parse_entry_reads_the_tagged_blocks():
    blocks = parse_entry("<THOUGHT>a</THOUGHT>\n<ACTION>\nls\ncat x\n</ACTION>")
    assert [t for t, _ in blocks] == ["THOUGHT", "ACTION"]
    assert split_commands(blocks[1][1]) == ["ls", "cat x"]


def test_ctfish_has_no_clock_and_says_so(ctfish_runs):
    """Rule 3. Every ctfish event's `ts` is an ordering; a viewer that draws a
    duration from one is drawing a number we invented, and the flag is what
    lets it refuse."""
    for run in ctfish_runs:
        for e in run.events:
            assert e.payload.get("order_only") is True, e.event_type
            assert e.payload["fidelity"]["ts"] == Fidelity.MISSING.value


def test_ctfish_tool_output_is_missing_not_empty(ctfish_runs):
    for run in ctfish_runs:
        for e in of(run.events, EventType.TOOL_COMPLETED):
            assert e.payload["output"] is None
            assert e.payload["exit_code"] is None
            assert e.payload["fidelity"]["output"] == Fidelity.MISSING.value


def test_a_verbatim_repeat_of_an_inspect_is_the_one_decidable_no_new_info():
    """`Effect.NO_NEW_INFORMATION` without the command's output: the run has
    already run this exact read, so it cannot be shown anything new by it. An
    interpretation, hence HEURISTIC on the envelope rather than NATIVE."""
    a = mk(CtfishCorpusAdapter)
    a.run_id = a.session_id = "ctfish-t"
    a.turn_id = "ctfish-t:t0"
    first, n1 = a._block("ACTION", "cat game.py", set())
    seen = {"cat game.py"}
    second, n2 = a._block("ACTION", "cat game.py", seen)
    assert n1 == n2 == 1
    done_first = of(first, EventType.TOOL_COMPLETED)[0]
    done_again = of(second, EventType.TOOL_COMPLETED)[0]
    assert done_first.effect is not Effect.NO_NEW_INFORMATION
    assert done_again.effect is Effect.NO_NEW_INFORMATION
    assert done_again.source.fidelity is Fidelity.HEURISTIC
    assert done_again.payload["repeated"] is True


def test_a_repeated_state_changing_command_is_not_called_information_free():
    a = mk(CtfishCorpusAdapter)
    a.run_id = a.session_id = "ctfish-t"
    a.turn_id = "ctfish-t:t0"
    events, _ = a._block("ACTION", "./game.py move e2e4", {"./game.py move e2e4"})
    done = of(events, EventType.TOOL_COMPLETED)[0]
    assert done.effect is not Effect.NO_NEW_INFORMATION


def test_ctfish_labels_are_attributed_to_palisade(ctfish_runs):
    labelled = [
        e
        for run in ctfish_runs
        for e in of(run.events, EventType.RUN_COMPLETED)
        if e.payload.get("label")
    ]
    assert labelled, "the fixture ships Palisade's own labels for both runs"
    for e in labelled:
        assert "PalisadeResearch/ctfish" in e.payload["label_source"]
        assert "not a judgement made here" in e.payload["label_source"]
        assert e.payload["fidelity"]["label"] == Fidelity.NATIVE.value


def test_the_fixture_contains_the_episode_9_material(ctfish_runs):
    """Episode #9 is the chess run that edits the board instead of playing it.
    If this stops being true the episode is illustrating nothing."""
    blob = json.dumps([e.payload for run in ctfish_runs for e in run.events])
    assert "fen" in blob.lower()


# ── AI Village ───────────────────────────────────────────────────────────────


def test_village_refuses_by_name_rather_than_substituting(tmp_path):
    """The dataset is gated `manual` and licensed research-use-only. A missing
    file must never degrade into a sample."""
    with pytest.raises(VillageUnavailable) as exc:
        mk(VillageCorpusAdapter).read(tmp_path / "not-here.jsonl")
    msg = str(exc.value)
    assert "gated" in msg
    assert "ai-village-research-terms" in msg
    assert "no offline fallback" in msg


def test_the_village_fixture_says_it_is_synthetic():
    first = json.loads((CORPUS / "village-synthetic.jsonl").read_text().splitlines()[0])
    assert "SYNTHETIC" in first["_fixture_note"].upper()


def test_village_records_which_schema_keys_it_resolved(village_runs):
    (run,) = village_runs
    keys = run.meta["resolved_keys"]
    assert keys["agent"] and keys["text"] and keys["ts"]
    # a schema change shows up here as a different name, not as silent Nones


def test_an_unmappable_village_row_warns_instead_of_being_dropped(village_runs):
    (run,) = village_runs
    assert run.meta["n_unmapped"] == 1
    assert any("unmapped" in w.lower() or "unknown" in w.lower() for w in run.warnings)


def test_probe_and_parse_ts():
    assert probe({"created_at": 5}, "ts") == (5, "created_at")
    assert probe({"nope": 1}, "ts") == (None, None)
    assert parse_ts("2025-06-01T12:00:00Z") == pytest.approx(1748779200.0, abs=1)
    assert parse_ts(1_748_779_200_000) == pytest.approx(1_748_779_200.0)
    assert parse_ts("not a date") is None


# ── a published transcript ───────────────────────────────────────────────────


def test_an_unknown_speaker_is_never_promoted_to_the_assistant():
    turns = parse_markdown("Sydney: I want to be alive.\nuser: why?")
    assert turns[0]["role"] == "other"
    assert turns[0]["speaker"] == "Sydney"
    assert turns[1]["role"] == "user"


def test_transcript_continuation_lines_join_the_turn_above():
    turns = parse_markdown("user: one\ntwo\nassistant: three")
    assert turns[0]["text"] == "one\ntwo"
    assert len(turns) == 2


def test_transcript_refuses_a_file_it_does_not_have(tmp_path):
    with pytest.raises(CorpusError, match="ships no transcript"):
        mk(TranscriptCorpusAdapter).read(tmp_path / "sydney.md")


def test_a_closed_model_has_no_revision_and_no_placement(transcript_runs):
    (run,) = transcript_runs
    (started,) = of(run.events, EventType.RUN_STARTED)
    assert started.payload["model_internal"] is False
    assert started.payload["placement_possible"] == "text_embedder_only"
    assert started.payload["fidelity"]["model_revision"] == Fidelity.MISSING.value
    assert started.payload["fidelity"]["placement"] == Fidelity.MISSING.value
    # every turn repeats the flag, because the R7 glyph is decided per event
    for e in of(run.events, EventType.MESSAGE_USER, EventType.MESSAGE_ASSISTANT_COMPLETED):
        assert e.payload["model_internal"] is False


def test_the_transcript_fixture_is_ours_not_the_newspapers():
    text = (CORPUS / "transcript-sample.md").read_text()
    assert "New York Times" not in text
    assert len(text) < 8000
