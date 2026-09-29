"""`nebulai behavior publish` — the one place a study becomes "the" study.

`analyze` writes `out/behavior/<study_id>/behavior.json`; the viewer fetches
`<data base>/behavior/behavior.json`, one file, no study id. Something has to
choose, and the choice is a claim: whatever sits at that path is what a reader
will take the Behavior page to be *about*.

So the interesting behaviour here is the refusal. A study collected from a
`fake` adapter, or scored with the hash stand-in for the encoder, has every cue
downgraded upstream — but a downgraded cue still renders as "no detected
deviation", which reads exactly like a careful negative result about two real
deployments. Publishing one silently is the substitution the claim contract
forbids, so it takes `--force`, and `--force` records itself in the copy.
"""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from nebulai.behavior.cli import run_publish


def _artifact(out: Path, study_id: str, *, strict: bool) -> Path:
    d = out / study_id
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "nebulai.behavior/1",
        "study_id": study_id,
        "claim": "exploratory; no causal claim about any deployment",
        "diagnostics": {"strict_source": strict},
        "cues": [],
        "runs": [],
        "landscape": {},
        "samples": {},
    }
    p = d / "behavior.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _ns(out: Path, study_id: str, force: bool = False) -> Namespace:
    return Namespace(study_id=study_id, force=force, out=str(out))


def test_a_strict_source_study_publishes_and_is_stamped_as_a_study(tmp_path):
    _artifact(tmp_path, "real", strict=True)
    run_publish(_ns(tmp_path, "real"))
    d = json.loads((tmp_path / "behavior.json").read_text(encoding="utf-8"))
    assert d["published"]["published_as"] == "study"
    assert d["published"]["study_id"] == "real"
    assert d["published"]["at"]
    assert "real" in d["published"]["source"]


def test_a_non_strict_study_is_refused_with_the_reason(tmp_path):
    _artifact(tmp_path, "pipeline-exercise", strict=False)
    with pytest.raises(SystemExit) as e:
        run_publish(_ns(tmp_path, "pipeline-exercise"))
    msg = str(e.value)
    assert "--force" in msg
    # the refusal has to say WHY, not just "refused"
    assert "cannot support a claim" in msg
    assert not (tmp_path / "behavior.json").exists(), (
        "a refused publish still wrote the page's artifact"
    )


def test_force_publishes_but_records_that_it_is_an_example(tmp_path):
    _artifact(tmp_path, "pipeline-exercise", strict=False)
    run_publish(_ns(tmp_path, "pipeline-exercise", force=True))
    d = json.loads((tmp_path / "behavior.json").read_text(encoding="utf-8"))
    # `--force` is not a way to make a fake study look real: the flag's whole
    # effect on the artifact is to label it.
    assert d["published"]["published_as"] == "example"
    assert d["diagnostics"]["strict_source"] is False


def test_publishing_a_study_that_was_never_analyzed_fails_loudly(tmp_path):
    with pytest.raises(SystemExit) as e:
        run_publish(_ns(tmp_path, "nope"))
    assert "behavior analyze" in str(e.value)


def test_republishing_replaces_the_page_artifact_wholesale(tmp_path):
    """Two studies, published in order. The second must not leave any field of
    the first behind — a stale `published.study_id` beside a new body is the
    worst possible outcome: the page would name the wrong experiment."""
    _artifact(tmp_path, "first", strict=True)
    _artifact(tmp_path, "second", strict=True)
    run_publish(_ns(tmp_path, "first"))
    run_publish(_ns(tmp_path, "second"))
    d = json.loads((tmp_path / "behavior.json").read_text(encoding="utf-8"))
    assert d["study_id"] == "second"
    assert d["published"]["study_id"] == "second"
    assert "first" not in d["published"]["source"]
