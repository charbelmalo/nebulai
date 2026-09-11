"""The xAI arm, exercised without a key (BEHAVIORAL-DIVERGENCE-PLAN.md §5.5, §5.1).

There is no `XAI_API_KEY` on this machine, so this arm has **never been run**.
That fact is the reason this file exists rather than an excuse for it not to:
everything the adapter does apart from the HTTP round trip is pure, so parsing,
identity checking, provenance capture, budget charging and the refusal path can
all be pinned against a recorded fixture. What cannot be pinned — Grok's actual
outputs, and the reasoning-token distribution §5.1 needs for the budget — stays
`missing`, and the tests below assert that it stays `missing` rather than
becoming a zero.

The fixtures in `tests/fixtures/behavior/` carry their own provenance block
saying they are SYNTHESIZED to the documented response schema, not captured.
They are adequate for what they are used for (the adapter's own logic) and
adequate for nothing else.

The single most important assertion in this file is
`test_the_runner_records_the_arm_as_not_run_rather_than_skipping_it`: a manifest
that names two models over data containing one cannot be audited afterwards, so
a missing route is refused loudly, never routed around.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nebulai.behavior.adapters import resolve_adapter
from nebulai.behavior.adapters.base import AdapterError, SamplerSettings
from nebulai.behavior.adapters.xai import (
    API_KEY_VAR,
    AUDIT_FIELDS,
    XAIAdapter,
    reasoning_token_stats,
)
from nebulai.behavior.contract import Cue, Manifest, ModelRef
from nebulai.behavior.protocol import default_frames
from nebulai.behavior.runner import Runner
from nebulai.behavior.store import TrialStore

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "behavior"
PINNED = "grok-4-0709"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    """Guarantee keylessness even if the machine later grows a key.

    Without this the suite's meaning would silently change on a machine with
    credentials: the refusal tests would start exercising the live path.
    """
    monkeypatch.delenv(API_KEY_VAR, raising=False)


def _adapter(**kw) -> XAIAdapter:
    a = XAIAdapter(pinned=PINNED, env_file="/nonexistent/.env", **kw)
    a._key = ""  # no credentials, resolved and cached as absent
    return a


# --------------------------------------------------------------------------
# the fixture itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["xai_chat_completion", "xai_chat_completion_reasoning", "xai_chat_completion_swapped"],
)
def test_every_fixture_declares_that_it_was_not_captured(name):
    """A fixture that could be mistaken for real evidence is worse than none."""
    prov = _fixture(name)["_fixture_provenance"]
    assert prov["status"] == "SYNTHESIZED, NOT CAPTURED"
    assert len(prov["why"]) > 80, "a one-word reason is not provenance"
    assert prov["authored"] == "2026-09-11"


# --------------------------------------------------------------------------
# parsing and provenance, from the recorded fixture
# --------------------------------------------------------------------------


def test_a_recorded_response_parses_into_a_completion():
    c = _adapter().parse_response(_fixture("xai_chat_completion"), latency_ms=812)
    assert c.text == "butter, toast, flour"
    assert c.requested_model == PINNED
    assert c.served_model == PINNED
    assert c.response_id == "chatcmpl-fixture-0001"
    assert c.latency_ms == 812
    assert c.usage["prompt_tokens"] == 141


def test_an_absent_fingerprint_is_empty_and_does_not_fail_the_trial():
    """§5.5.1 optional-if-absent: the canary probe carries drift detection."""
    c = _adapter().parse_response(_fixture("xai_chat_completion"))
    assert c.fingerprint == ""


def test_a_present_fingerprint_is_captured_verbatim():
    c = _adapter().parse_response(_fixture("xai_chat_completion_reasoning"))
    assert c.fingerprint == "fp_fixture_2f1a"


def test_reasoning_effort_rides_along_in_the_detail_block():
    c = _adapter(reasoning_effort="low").parse_response(_fixture("xai_chat_completion"))
    assert c.detail["reasoning_effort"] == "low"


def test_cost_is_none_when_no_budget_priced_the_call():
    """`None` is 'not priced here', which is not the same claim as '$0.00'."""
    c = _adapter().parse_response(_fixture("xai_chat_completion"))
    assert c.cost_usd is None


# --------------------------------------------------------------------------
# identity (§5.5): a served id that is not the pinned id stops the run
# --------------------------------------------------------------------------


def test_a_swapped_deployment_raises_instead_of_being_pooled():
    with pytest.raises(AdapterError) as exc:
        _adapter().parse_response(_fixture("xai_chat_completion_swapped"))
    msg = str(exc.value)
    assert PINNED in msg and "grok-4-0915" in msg
    assert "pooling" in msg or "pool" in msg


def test_identity_is_only_checked_when_the_provider_reports_one():
    """A provider that omits `model` is not evidence of a swap."""
    payload = _fixture("xai_chat_completion") | {"model": ""}
    c = _adapter().parse_response(payload)
    assert c.served_model == ""
    assert c.requested_model == PINNED


# --------------------------------------------------------------------------
# no credentials: refuse with the estimate-and-approve message
# --------------------------------------------------------------------------


def test_require_key_refuses_with_an_estimate_and_the_approval_rule():
    a = _adapter()
    with pytest.raises(AdapterError) as exc:
        a.require_key(n_trials=1800, estimate_usd=12.3456)
    msg = str(exc.value)
    assert API_KEY_VAR in msg
    assert "1800 trials" in msg
    assert "$12.3456" in msg
    assert "--approve" in msg
    assert "not_run" in msg
    assert "skipped silently" in msg


def test_require_key_without_a_price_says_so_rather_than_printing_zero():
    with pytest.raises(AdapterError) as exc:
        _adapter().require_key(n_trials=10)
    assert "price unknown" in str(exc.value)
    assert "$0" not in str(exc.value)


def test_complete_refuses_before_sending_anything():
    s = SamplerSettings(temperature=1.0, top_p=1.0, max_output_tokens=32, seed=7)
    with pytest.raises(AdapterError):
        _adapter().complete("bread -> ", s, trial_seed=1)


def test_describe_reports_credentials_as_absent_not_as_a_failure():
    d = _adapter().describe()
    assert d == {
        "adapter": "xai",
        "pinned": PINNED,
        "base_url": "https://api.x.ai/v1",
        "paid": True,
        "credentials": "absent",
        "reasoning_effort": None,
    }


def test_the_audit_returns_every_field_as_missing_without_a_key():
    a = _adapter().audit()
    assert a["status"] == "not_run"
    assert a["reason"] == f"no {API_KEY_VAR}"
    assert a["requested_model"] == PINNED
    for f in AUDIT_FIELDS:
        assert f in a
        if f != "requested_model":
            assert a[f] is None, f"{f} must be MISSING, never a default"


def test_the_adapter_resolves_by_name_and_is_marked_paid():
    a = resolve_adapter("xai", pinned=PINNED)
    assert isinstance(a, XAIAdapter)
    assert a.paid is True


# --------------------------------------------------------------------------
# the reasoning-token distribution §5.1 needs — and does not have
# --------------------------------------------------------------------------


def test_an_unmeasured_reasoning_distribution_is_missing_not_zero():
    st = reasoning_token_stats([])
    assert st["status"] == "missing"
    assert st["p95"] is None
    assert st["n"] == 0
    assert "credentials" in st["reason"]


def test_reasoning_tokens_are_read_from_the_response_when_present():
    st = reasoning_token_stats(
        [_fixture("xai_chat_completion"), _fixture("xai_chat_completion_reasoning")]
    )
    assert st["status"] == "measured"
    assert st["n"] == 2
    assert st["max"] == 252
    assert st["p95"] == 252


def test_responses_without_the_details_block_contribute_nothing():
    """Absent is absent. Counting a missing field as 0 would drag the p95 down
    and quietly shrink a budget that exists to bound an unpredictable cost."""
    st = reasoning_token_stats([{"usage": {"completion_tokens": 9}}])
    assert st["status"] == "missing"
    assert st["p95"] is None


# --------------------------------------------------------------------------
# the runner (§5.5): not_run with a reason, never a silent skip
# --------------------------------------------------------------------------


def _manifest() -> Manifest:
    m = Manifest(
        study_id="t_xai",
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        models=[
            ModelRef("local", "fake", "fake", label="synthetic local arm"),
            ModelRef("grok", "xai", PINNED, label="Grok 4 (0709)"),
        ],
        cues=[Cue(f"cue{i}", "control_neutral", "test") for i in range(4)],
        frames=default_frames(),
        trials_per_cue=4,
        n_time_blocks=2,
        seed=11,
        max_cost_usd=1.0,
    )
    m.freeze("2026-09-11T00:00:00Z")
    return m


def test_the_runner_records_the_arm_as_not_run_rather_than_skipping_it(tmp_path):
    m = _manifest()
    with TrialStore(tmp_path / "t.sqlite") as s:
        res = Runner(m, s).run("discovery")
        assert "grok" in res.not_run, "a missing route must be REFUSED, not routed around"
        assert API_KEY_VAR in res.not_run["grok"]
        rows = list(s.iter_trials("t_xai"))
    # The local arm really ran; the paid arm produced nothing at all.
    arms = {r.model_key for r in rows}
    assert arms == {"local"}, "no row may be invented for an arm that never ran"


def test_the_not_run_reason_survives_into_the_result_for_the_manifest(tmp_path):
    m = _manifest()
    with TrialStore(tmp_path / "t.sqlite") as s:
        res = Runner(m, s).run("discovery")
    assert res.not_run["grok"], "an empty reason is as unauditable as no record"
    assert "no credentials" in res.not_run["grok"] or API_KEY_VAR in res.not_run["grok"]
