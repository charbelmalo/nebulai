"""Build-server pure-function tests — the stage parser and argv builder that
the whole /build progress model hangs off. No HTTP, no subprocess."""

import sys

import pytest

from nebulai.backend.build_server import (
    _STAGE_AFTER,
    build_cmd,
    dataset_id_for,
    parse_stage_line,
    parse_substage_line,
)

# The exact prints from cli._run_tokens (plus realistic noise lines).
STAGE_CASES = [
    (
        "[1/5] loaded 49857 token units from gpt2 (vocab 50257, curated to 49857) [12.3s]",
        (1, "loaded 49857 token units from gpt2 (vocab 50257, curated to 49857) [12.3s]"),
    ),
    ("[2/5] UMAP -> 10d/3d/2d [901.4s]", (2, "UMAP -> 10d/3d/2d [901.4s]")),
    (
        "[2/5] reused cached reductions from out/gpt2/reduced.npz [0.1s]",
        (2, "reused cached reductions from out/gpt2/reduced.npz [0.1s]"),
    ),
    ("[3/5] HDBSCAN: 208 clusters, 55% noise [4.2s]", (3, "HDBSCAN: 208 clusters, 55% noise [4.2s]")),
    ("[4/5] named 208 clusters via 'ollama:lfm2.5' [120.0s]", (4, "named 208 clusters via 'ollama:lfm2.5' [120.0s]")),
    ("[5/5] exported [8.1s]", (5, "exported [8.1s]")),
    ("  [5/5] exported [8.1s]  ", (5, "exported [8.1s]")),  # stray whitespace
    # non-stage lines pass through as None (kept in the log untouched)
    ("  out/gpt2/nebulai.json", None),
    ("Downloading model.safetensors: 100%", None),
    ("[1/2] built comparison: …", None),  # compare's 2-stage prints must not match
    ("[6/5] impossible", None),
    ("", None),
    ("garbage [3/5] not at start", None),
]


@pytest.mark.parametrize("line,expected", STAGE_CASES)
def test_parse_stage_line(line, expected):
    assert parse_stage_line(line) == expected


def test_stage_after_covers_every_index():
    # stage k finished -> viewer ProbeStage name of the stage now running
    assert _STAGE_AFTER[0] == "loading"
    assert [_STAGE_AFTER[k] for k in (1, 2, 3, 4)] == [
        "reducing",
        "clustering",
        "naming",
        "exporting",
    ]


def test_build_cmd_minimal():
    cmd = build_cmd("gpt2", "hf", {})
    assert cmd[:5] == [sys.executable, "-u", "-m", "nebulai", "tokens"]
    assert cmd[5:] == ["--model", "gpt2"]
    assert "--anthropic-model" not in cmd  # no namer_model -> no flag


def test_build_cmd_full_params():
    cmd = build_cmd(
        "EleutherAI/pythia-70m",
        "hf",
        {
            "max_tokens": 2000,
            "n_neighbors": 15,
            "seed": 7,
            "min_cluster_size": 20,
            "min_samples": 5,
            "cluster_method": "eom",
            "namer": "none",
            "namer_model": "claude-haiku-4-5",
            "edges": "cluster",
            "force": True,
        },
    )
    for flag, val in [
        ("--max-tokens", "2000"),
        ("--n-neighbors", "15"),
        ("--seed", "7"),
        ("--min-cluster-size", "20"),
        ("--min-samples", "5"),
        ("--cluster-method", "eom"),
        ("--namer", "none"),
        ("--anthropic-model", "claude-haiku-4-5"),
        ("--edges", "cluster"),
    ]:
        i = cmd.index(flag)
        assert cmd[i + 1] == val
    assert "--force" in cmd


def test_build_cmd_seed_zero_is_kept_and_api_flags():
    cmd = build_cmd("gpt2", "api", {"seed": 0, "embed_host": "http://localhost:11434", "embed_model": "mxbai-embed-large"})
    assert cmd[cmd.index("--seed") + 1] == "0"  # 0 is a valid seed, not falsy-dropped
    assert cmd[cmd.index("--source") + 1] == "api"
    assert cmd[cmd.index("--embed-model") + 1] == "mxbai-embed-large"


def test_build_cmd_rejects_bad_input():
    with pytest.raises(ValueError):
        build_cmd("gpt2; rm -rf /", "hf", {})
    with pytest.raises(ValueError):
        build_cmd("gpt2", "carrier-pigeon", {})
    with pytest.raises(ValueError):
        build_cmd("gpt2", "hf", {"cluster_method": "banana"})
    with pytest.raises(ValueError):
        build_cmd("gpt2", "hf", {"namer": "gpt9"})
    with pytest.raises(ValueError):
        build_cmd("gpt2", "hf", {"edges": "all"})


# --- within-stage progress ------------------------------------------------
# Stage 1 of a probe runs for minutes before `[1/5]` prints. These indented
# lines are the only thing moving during it.

SUBSTAGE_CASES = [
    ("  depth 1: +12 terms (total 13)", "depth 1: +12 terms (total 13)"),
    ("  depth 2: expanding 3/12 'bereavement'…", "depth 2: expanding 3/12 'bereavement'…"),
    ("  depth 2: +105 terms (total 118)", "depth 2: +105 terms (total 118)"),
    ("  sensitivity 0.35: kept 53/118 (dropped 65, 55%)", "sensitivity 0.35: kept 53/118 (dropped 65, 55%)"),
    # a failed expansion is progress the user must see live, not only in the log
    ("  expansion failed for 'counseling' (RuntimeError: truncated)", "expansion failed for 'counseling' (RuntimeError: truncated)"),
    # top-level lines belong to parse_stage_line, not here
    ("[1/5] grew 53 concepts from 'grief'", None),
    ("Downloading model.safetensors: 100%", None),
    # library noise merged in from stderr
    ("  warn(", None),
    ("  /Users/x/.venv/lib/python3.12/site-packages/umap/umap_.py:19: UserWarning: n_jobs", None),
    # the artifact paths printed after [5/5] must not clobber the done message
    ("  /Users/charbelmalo/Developer/nebulai/out/probe__grief/nebulai.json", None),
    ("", None),
    ("   ", None),
]


@pytest.mark.parametrize("line,expected", SUBSTAGE_CASES)
def test_parse_substage_line(line, expected):
    assert parse_substage_line(line) == expected


def test_substage_and_stage_parsers_never_both_fire():
    for line, _ in STAGE_CASES + SUBSTAGE_CASES:
        assert not (parse_stage_line(line) and parse_substage_line(line))


def test_parse_substage_line_truncates_a_runaway_line():
    out = parse_substage_line("  " + "x" * 500)
    assert out is not None and len(out) == 160


def test_dataset_id_for():
    assert dataset_id_for("gpt2", "hf") == "gpt2"
    assert dataset_id_for("EleutherAI/pythia-70m", "hf") == "EleutherAI__pythia-70m"
    assert dataset_id_for("gpt2", "api", "mxbai-embed-large") == "gpt2__api-mxbai-embed-large"
    assert dataset_id_for("gpt2", "api", None) == "gpt2__api-embed"


# --- probe source ---------------------------------------------------------
# The probe front-end has no model: `model` carries the seed PHRASE, the
# subcommand is `probe` not `tokens`, and its RNG knob is --seed-rng because
# the positional arg is already the seed word.


def test_build_cmd_probe_uses_the_probe_subcommand_with_the_seed_positional():
    cmd = build_cmd("grief", "probe", {})
    assert cmd == [sys.executable, "-u", "-m", "nebulai", "probe", "grief"]


def test_build_cmd_probe_full_params():
    cmd = build_cmd(
        "medieval siege warfare",
        "probe",
        {
            "depth": 3,
            "breadth": 16,
            "sensitivity": 0.42,
            "generator": "openai",
            "namer": "openai",
            "seed": 7,
            "n_neighbors": 15,
            "llm_host": "http://llm.test:8050",
            "llm_model": "Qwen3.6-35B",
            "embed_host": "http://embed.test:8040",
            "embed_api": "openai",
            "embed_model": "all-MiniLM-L6-v2",
        },
    )
    assert cmd[5] == "medieval siege warfare"
    for flag, val in [
        ("--depth", "3"),
        ("--breadth", "16"),
        ("--sensitivity", "0.42"),
        ("--generator", "openai"),
        ("--namer", "openai"),
        ("--seed-rng", "7"),
        ("--n-neighbors", "15"),
        ("--llm-host", "http://llm.test:8050"),
        ("--llm-model", "Qwen3.6-35B"),
        ("--embed-host", "http://embed.test:8040"),
        ("--embed-api", "openai"),
        ("--embed-model", "all-MiniLM-L6-v2"),
    ]:
        assert cmd[cmd.index(flag) + 1] == val
    # `--seed` would be parsed as a SECOND positional seed phrase
    assert "--seed" not in cmd
    # probe grows a fresh cloud every time; there is no reduction cache to skip
    assert "--source" not in cmd


def test_build_cmd_probe_never_sends_force():
    cmd = build_cmd("grief", "probe", {"force": True})
    assert "--force" not in cmd


def test_build_cmd_probe_rejects_a_shell_ish_seed():
    for bad in ["grief; rm -rf /", "a`whoami`", 'x"y', "a|b", "a$(id)", "a\nb", ""]:
        with pytest.raises(ValueError):
            build_cmd(bad, "probe", {})


def test_build_cmd_probe_allows_ordinary_phrases():
    for good in ["grief", "medieval siege warfare", "co-operative game theory", "K2"]:
        assert build_cmd(good, "probe", {})[5] == good


def test_build_cmd_probe_rejects_a_bad_generator():
    with pytest.raises(ValueError):
        build_cmd("grief", "probe", {"generator": "telepathy"})


def test_dataset_id_for_probe_matches_the_cli_slug():
    from nebulai.frontends.probe import probe_dataset_id

    for seed in ["grief", "Medieval Siege Warfare", "  ...  "]:
        assert dataset_id_for(seed, "probe") == probe_dataset_id(seed)
    assert dataset_id_for("Medieval Siege Warfare", "probe") == "probe__medieval-siege-warfare"
