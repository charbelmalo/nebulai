"""P3 exit tests for the cost gate extended to Seer.

The rule this file exists to hold in place: **run 1 of a new protocol is
`MISSING`, not `0`.** Seer launches somebody else's agent binary, billing
through an account this process cannot read, so the first repeat of any
protocol genuinely has no estimate — and `--repeat 20` is exactly the command
where a fabricated `$0.0000` would do the most damage.

Everything here is offline. The agents are the same kind of fake script the
runner tests use: real binaries cost money, need network, and change their
output between versions, none of which belongs in a test that runs on every
commit.
"""

from __future__ import annotations

import json
import sys

import pytest

from nebulai.corpus import DEFAULT_MAX_COST_USD
from nebulai.llm import BudgetError
from nebulai.seer import runner as runner_mod
from nebulai.seer.budget import (
    PROTOCOL_HASH_FIELDS,
    SeerBudget,
    SeerBudgetError,
    protocol_fingerprint,
    protocol_id,
)
from nebulai.seer.cli import main
from nebulai.seer.contract import Fidelity
from nebulai.seer.ensemble import list_ensembles, read_manifest
from nebulai.seer.runner import Runner
from nebulai.seer.store import EventStore

# Claude reports what a run cost on its terminal line. This is the shape that
# makes a protocol priceable at all.
FAKE_CLAUDE_PRICED = r"""
import json
def p(o): print(json.dumps(o), flush=True)
p({"type": "system", "subtype": "init", "session_id": "s1",
   "model": "claude-haiku-4-5", "tools": []})
p({"type": "assistant", "message": {"id": "m1", "role": "assistant",
   "content": [{"type": "text", "text": "done"}]}, "session_id": "s1"})
p({"type": "result", "subtype": "success", "is_error": False,
   "session_id": "s1", "total_cost_usd": 0.02, "num_turns": 1,
   "duration_ms": 10, "terminal_reason": "completed", "uuid": "u1"})
"""

# `codex exec --json` reports no cost at all. This is the shape that stays
# unpriceable forever, and the document has to say so.
FAKE_CODEX_UNPRICED = r"""
import json
def p(o): print(json.dumps(o), flush=True)
p({"type": "thread.started", "thread_id": "th_fake"})
p({"type": "turn.started"})
p({"type": "item.completed", "item": {"id": "i1", "type": "agent_message",
   "text": "done"}})
p({"type": "turn.completed", "usage": {"input_tokens": 10,
   "cached_input_tokens": 0, "output_tokens": 2, "reasoning_output_tokens": 0}})
"""


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "seer")
    yield s
    s.close()


@pytest.fixture
def fake_agent(tmp_path, monkeypatch):
    def make(source: str, agent: str = "codex"):
        script = tmp_path / f"fake_{agent}.py"
        script.write_text(source)
        monkeypatch.setattr(
            runner_mod,
            "build_command",
            lambda a, prompt, **kw: [sys.executable, "-u", str(script), prompt],
        )
        monkeypatch.setattr(runner_mod, "agent_version", lambda a: "fake-1.0")
        return script

    return make


# ── protocol identity ───────────────────────────────────────────────────────


def test_the_same_command_is_the_same_protocol():
    a = protocol_id("codex", "fix the test", model="m", cwd="/w")
    b = protocol_id("codex", "fix the test", model="m", cwd="/w")
    assert a == b and a.startswith("proto_")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"agent": "claude"},
        {"prompt": "fix the OTHER test"},
        {"model": "n"},
        {"cwd": "/elsewhere"},
        {"extra_args": ["--flag"]},
    ],
)
def test_every_hashed_field_splits_the_protocol(kwargs):
    base = dict(agent="codex", prompt="fix the test", model="m", cwd="/w")
    base.update(kwargs)
    assert protocol_id(
        base.pop("agent"), base.pop("prompt"), **base
    ) != protocol_id("codex", "fix the test", model="m", cwd="/w")


def test_an_unpinned_model_is_not_the_same_protocol_as_a_pinned_one():
    """Inheriting a pinned run's measured cost would be a pin substitution
    arriving from the money side."""
    assert protocol_id("codex", "p") != protocol_id("codex", "p", model="gpt-5")


def test_the_fingerprint_says_how_it_hashed_and_never_carries_the_prompt():
    fp = protocol_fingerprint("codex", "a secret prompt", cwd="/private/repo")
    assert fp["hash_algorithm"] == "sha256"
    assert fp["hash_fields"] == list(PROTOCOL_HASH_FIELDS)
    blob = json.dumps(fp)
    assert "a secret prompt" not in blob and "/private/repo" not in blob
    # an unpinned model is `null`, not an empty string that reads as a pin
    assert fp["model"] is None


# ── run 1 is MISSING, not 0 ─────────────────────────────────────────────────


def test_run_one_of_a_new_protocol_has_no_estimate_and_is_missing(store):
    b = SeerBudget(store)
    est = b.estimate(protocol_id("codex", "p"), 20)

    assert est.fidelity is Fidelity.MISSING
    assert est.usd is None  # NOT 0.0
    assert est.priced is False
    assert "no estimate" in est.reason
    # and it says so in the text a human reads, in those words
    assert "MISSING is not $0.00" in est.text()


def test_an_unpriced_repeat_is_refused_without_an_explicit_acknowledgement(store):
    b = SeerBudget(store)
    with pytest.raises(SeerBudgetError) as e:
        b.preflight(protocol_id("codex", "p"), 20, quiet=True)
    msg = str(e.value)
    assert "MISSING, not $0.00" in msg
    assert "--acknowledge-unpriced" in msg
    # terminal by design: the same class the rest of the project refuses with
    assert isinstance(e.value, BudgetError)


def test_the_acknowledgement_is_recorded_rather_than_converted_to_a_number(store):
    b = SeerBudget(store)
    est = b.preflight(
        protocol_id("codex", "p"), 20, acknowledge_unpriced=True, quiet=True
    )
    assert est.usd is None and est.fidelity is Fidelity.MISSING
    assert est.acknowledged_unpriced is True
    b.approve()
    b.charge_run(protocol_id("codex", "p"), "run_1", None)

    doc = b.to_dict()
    assert doc["spent_usd"] is None  # never 0.0 for "we could not see it"
    assert doc["spent_fidelity"] == Fidelity.MISSING.value
    assert doc["n_runs_unpriced"] == 1
    assert doc["acknowledged_unpriced"] is True
    assert "MISSING" in b.summary()


def test_an_unpriced_run_never_becomes_a_measurement_of_zero(store):
    b = SeerBudget(store)
    pid = protocol_id("codex", "p")
    b.preflight(pid, 2, acknowledge_unpriced=True, quiet=True)
    b.approve()
    b.charge_run(pid, "run_1", None)
    # no ledger row at all — a row saying "$0.00 over 1 run" would be read as
    # a measurement, and the next repeat would price itself from it
    assert b.measurement(pid) is None
    assert b.estimate(pid, 5).fidelity is Fidelity.MISSING


# ── runs 2..N are ESTIMATED from measured usage ─────────────────────────────


def test_runs_two_onward_are_estimated_from_run_ones_measured_cost(store):
    b = SeerBudget(store)
    pid = protocol_id("claude", "p")
    b.preflight(pid, 4, acknowledge_unpriced=True, quiet=True)
    b.approve()
    b.charge_run(pid, "run_1", 0.02)

    est = b.estimate(pid, 3)
    assert est.fidelity is Fidelity.ESTIMATED
    assert est.usd == pytest.approx(0.06)
    assert est.per_run_usd == pytest.approx(0.02)
    assert est.measured_from_run_id == "run_1"
    assert "measured over 1 run(s)" in est.reason


def test_the_measurement_survives_into_a_later_process(store, tmp_path):
    pid = protocol_id("claude", "p")
    b = SeerBudget(store)
    b.preflight(pid, 2, acknowledge_unpriced=True, quiet=True)
    b.approve()
    b.charge_run(pid, "run_1", 0.02)

    # a second `seer run --repeat` tomorrow, against the same store root
    later = SeerBudget(root=store.root)
    assert later.estimate(pid, 10).fidelity is Fidelity.ESTIMATED
    assert later.estimate(pid, 10).usd == pytest.approx(0.20)


def test_the_ledger_is_a_running_mean_not_last_one_wins(store):
    b = SeerBudget(store)
    pid = protocol_id("claude", "p")
    b.preflight(pid, 3, acknowledge_unpriced=True, quiet=True)
    b.approve()
    b.charge_run(pid, "r1", 0.04)
    b.charge_run(pid, "r2", 0.02)
    m = b.measurement(pid)
    assert m.n_runs == 2 and m.per_run_usd == pytest.approx(0.03)


# ── the ceiling ─────────────────────────────────────────────────────────────


def test_the_ceiling_refuses_a_repeat_it_cannot_afford(store):
    b = SeerBudget(store, ceiling_usd=0.10)
    pid = protocol_id("claude", "p")
    b.preflight(pid, 1, acknowledge_unpriced=True, quiet=True)
    b.approve()
    b.charge_run(pid, "r1", 0.02)

    with pytest.raises(SeerBudgetError) as e:
        b.preflight(pid, 20, quiet=True)  # 20 x $0.02 = $0.40 > $0.10
    assert "over the $0.10 ceiling" in str(e.value)
    assert "nothing further was launched" in str(e.value)
    # and it refuses rather than quietly cutting N down to what fits
    assert "Lower --repeat" in str(e.value)


def test_the_shared_project_ceiling_is_the_default(store):
    assert SeerBudget(store).ceiling_usd == DEFAULT_MAX_COST_USD


def test_cumulative_spend_over_the_ceiling_stops_the_repeat(store):
    b = SeerBudget(store, ceiling_usd=0.05)
    pid = protocol_id("claude", "p")
    b.preflight(pid, 3, acknowledge_unpriced=True, quiet=True)
    b.approve()
    b.charge_run(pid, "r1", 0.03)
    with pytest.raises(SeerBudgetError) as e:
        b.charge_run(pid, "r2", 0.03)
    assert "passed the $0.05 ceiling" in str(e.value)
    # the runs that completed are still real — a smaller fan, not a broken one
    assert "smaller fan" in str(e.value)


# ── approval is a step, not a flag ──────────────────────────────────────────


def test_charging_before_approval_is_refused(store):
    b = SeerBudget(store)
    b.preflight(protocol_id("codex", "p"), 2, acknowledge_unpriced=True, quiet=True)
    with pytest.raises(SeerBudgetError, match="before its cost estimate was approved"):
        b.charge_run(protocol_id("codex", "p"), "r1", 0.01)


def test_approving_before_preflighting_is_refused(store):
    with pytest.raises(RuntimeError, match="approve\\(\\) before preflight\\(\\)"):
        SeerBudget(store).approve()


# ── the gate, through the command line ──────────────────────────────────────


def test_the_cli_refuses_an_unpriced_repeat_before_launching_anything(
    tmp_path, fake_agent, capsys
):
    fake_agent(FAKE_CODEX_UNPRICED)
    root = tmp_path / "seer"
    with pytest.raises(SystemExit) as e:
        main(["--root", str(root), "run", "codex", "do it", "--repeat", "5"])
    assert e.value.code == 2

    # the important assertion: no agent was launched at all
    s = EventStore(root)
    try:
        assert s.list_runs() == []
        assert list_ensembles(s) == []
    finally:
        s.close()
    assert "--acknowledge-unpriced" in capsys.readouterr().err


def test_an_acknowledged_repeat_runs_and_groups_under_one_ensemble(
    tmp_path, fake_agent
):
    fake_agent(FAKE_CODEX_UNPRICED)
    root = tmp_path / "seer"
    main([
        "--root", str(root), "run", "codex", "do it",
        "--repeat", "3", "--seed-base", "7", "--acknowledge-unpriced",
    ])

    s = EventStore(root)
    try:
        rows = list_ensembles(s)
        assert len(rows) == 1 and rows[0]["n_members"] == 3
        manifest = read_manifest(s, rows[0]["ensemble_id"])
        assert [m.run_id for m in manifest.members] == [
            r.run_id for r in reversed(s.list_runs(limit=10))
        ]
        # the seed is recorded per run and explicitly not applied
        assert [m.seed for m in manifest.members] == [7, 8, 9]
        assert manifest.seed_applied is False
        # and the money is recorded as unknown, not as zero
        assert manifest.budget["spent_usd"] is None
        assert manifest.budget["n_runs_unpriced"] == 3
        assert manifest.budget["acknowledged_unpriced"] is True
    finally:
        s.close()


def test_a_second_repeat_of_a_priced_protocol_needs_no_acknowledgement(
    tmp_path, fake_agent
):
    """Run 1 measured it; run 2 onwards are estimated and go through the same
    preflight, so the acknowledgement is no longer required."""
    fake_agent(FAKE_CLAUDE_PRICED, agent="claude")
    root = tmp_path / "seer"
    s = EventStore(root)
    try:
        r = Runner("claude", "do it", store=s, cwd=tmp_path).run()
        assert r.view.cost_usd.value == pytest.approx(0.02)
        pid = protocol_id("claude", "do it", cwd=str(tmp_path))
        b = SeerBudget(s)
        b.preflight(pid, 1, acknowledge_unpriced=True, quiet=True)
        b.approve()
        b.charge_run(pid, r.run_id, float(r.view.cost_usd.value))
    finally:
        s.close()

    # no --acknowledge-unpriced this time, and it is allowed through
    main([
        "--root", str(root), "run", "claude", "do it",
        "--cwd", str(tmp_path), "--repeat", "2",
    ])
    s = EventStore(root)
    try:
        rows = list_ensembles(s)
        assert len(rows) == 1 and rows[0]["n_members"] == 2
        manifest = read_manifest(s, rows[0]["ensemble_id"])
        est = manifest.budget["estimate"]
        assert est["fidelity"] == Fidelity.ESTIMATED.value
        assert est["usd"] is not None and est["usd"] > 0
        assert manifest.budget["spent_usd"] == pytest.approx(0.04)
    finally:
        s.close()
