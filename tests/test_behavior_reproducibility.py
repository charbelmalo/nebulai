"""Phase 1's reproducibility claim, end to end (§11 Phase 1).

The claim is narrow and worth stating exactly: **the same frozen manifest and
the same raw trial database produce the same compact metrics.** Not "the same
model returns the same text" — a paid endpoint at temperature 0.8 gives no such
guarantee and the study never asks for one. What must be reproducible is
everything *downstream* of the raw text: the schedule, the per-trial seeds, the
parse, the embedding, Δ̂, the permutation p, the BY correction, the landscape.

That boundary is why this file re-analyzes one store twice rather than running
the study twice against a live model. The one place a fresh collection IS
pinned is the schedule and the seeds, and those are asserted separately (a real
defect: they were salted by `hash()` and differed between processes).

`analyze` is not free of nondeterminism by construction — permutations draw
from an RNG and the landscape fits a PCA — so if any of it were seeded from the
clock or from `id()`, the second run would differ and nothing else in the suite
would notice.
"""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from nebulai.behavior.cli import run_analyze, run_plan, run_run
from nebulai.behavior.contract import load_manifest
from nebulai.behavior.store import TrialStore

# fields that are *supposed* to differ between two runs of the same analysis
VOLATILE = {"generated", "created", "analyzed_at", "wrote", "elapsed_s", "started", "finished"}


def _strip_volatile(obj):
    """Drop wall-clock fields recursively; everything else must match."""
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> tuple[Path, str]:
    """A small fake-adapter study, planned and collected once."""
    out = tmp_path_factory.mktemp("behavior-repro")
    run_plan(
        Namespace(
            study_id="repro",
            preset="fake",
            cues="control",
            trials=8,
            blocks=2,
            min_valid=4,
            min_within_block=2,
            seed=7,
            pin_b="",
            exploratory=False,
            max_cost_usd=0.0,
            notes="reproducibility test",
            out=str(out),
        )
    )
    manifest = out / "repro" / "manifest.json"
    run_run(
        Namespace(
            manifest=str(manifest),
            arm="discovery",
            limit=None,
            approve=False,
            verbose=False,
            out=str(out),
        )
    )
    return out, str(manifest)


def _analyze(out: Path, manifest: str) -> dict:
    run_analyze(
        Namespace(
            manifest=manifest,
            embedder="hash",  # deterministic and offline; not a semantic space
            permutations=200,
            capability_reference=None,
            out=str(out),
        )
    )
    return json.loads((out / "repro" / "behavior.json").read_text(encoding="utf-8"))


def test_the_raw_store_holds_what_the_manifest_asked_for(study):
    out, manifest = study
    m = load_manifest(manifest)
    store = TrialStore(out / "repro" / "trials.sqlite")
    trials = [t for t in store.iter_trials(m.study_id) if t.arm == "discovery"]
    store.close()
    assert trials, "the fake adapter collected nothing"
    assert len({t.model_key for t in trials}) == 2
    # every trial records the exact prompt it was issued (by hash) and the
    # parser version that read the reply back — without both, a re-analysis is
    # of a different experiment than the one that ran
    assert all(t.prompt_sha for t in trials)
    assert len({t.parser_version for t in trials}) == 1
    assert {t.block for t in trials} == set(range(m.n_time_blocks))


def test_analyzing_the_same_store_twice_gives_the_same_compact_metrics(study):
    out, manifest = study
    first = _strip_volatile(_analyze(out, manifest))
    second = _strip_volatile(_analyze(out, manifest))
    assert first == second, "a second analysis of the same raw trials disagreed"


def test_the_reproduced_export_is_not_trivially_empty(study):
    """A test that compares two empty dicts passes forever. Pin that the thing
    being compared actually carries the numbers the page reads."""
    out, manifest = study
    payload = _analyze(out, manifest)
    assert payload["cues"], "no cues in the export"
    c = payload["cues"][0]
    for key in ("cue", "delta_hat", "p_value", "q_value", "status", "arms"):
        assert key in c, key
    assert payload.get("landscape"), "no landscape in the export"


def test_a_fake_adapter_can_never_produce_a_confirmed_cue(study):
    """The strict-source rule (§6.6): synthetic text is not evidence about any
    model, so every status is downgraded regardless of how small p gets."""
    out, manifest = study
    payload = _analyze(out, manifest)
    assert all(c["status"] != "confirmed" for c in payload["cues"])
    assert payload["diagnostics"]["strict_source"] is False
