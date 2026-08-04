"""CLI-transport namers — `--namer claude-cli` and `--namer codex-cli`.

These shell out to agent CLIs so the naming runs against an existing
subscription instead of per-token API billing. The subprocess is mocked here:
the point of these tests is the envelope handling and the failure accounting,
which is where a CLI transport differs from an HTTP one. A real invocation is a
network+billing event and has no place in the default suite.
"""

import json
import subprocess

import pytest

from nebulai.backend.name import (
    _cli_prompt,
    _ingest_titles,
    _name_with_claude_cli,
    _name_with_codex_cli,
)

REPS = {0: ["cat", "dog"], 1: ["red", "blue"]}


def _claude_ok(titles):
    return json.dumps({"is_error": False, "result": json.dumps({"titles": titles})})


# --- the shared contract --------------------------------------------------


def test_both_backends_send_the_identical_prompt():
    """A claude-vs-codex comparison only means something if the prompt is held
    fixed — otherwise it measures prompt drift, not the models."""
    sent = []

    def fake_run(cmd, **kw):
        sent.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, _claude_ok([{"id": 0, "title": "a"}]), "")

    import nebulai.backend.name as N

    orig = subprocess.run
    try:
        subprocess.run = fake_run
        _name_with_claude_cli(REPS, "sonnet", batch_size=2)
    finally:
        subprocess.run = orig
    assert _cli_prompt(REPS, [0, 1]) in sent[0]
    # the JSON contract has to be in the prompt: CLIs have no schema parameter
    assert '"titles"' in _cli_prompt(REPS, [0, 1])


def test_ingest_titles_reports_whether_it_contributed():
    titles = {}
    assert _ingest_titles('{"titles":[{"id":3,"title":"x"}]}', titles) is True
    assert titles == {3: "x"}
    assert _ingest_titles("not json at all", titles) is False
    assert _ingest_titles('{"titles":[]}', titles) is False
    # a blank title is not a title — it would export as an unnamed cluster
    assert _ingest_titles('{"titles":[{"id":9,"title":"  "}]}', titles) is False


# --- claude-cli -----------------------------------------------------------


def test_claude_cli_parses_the_envelope(monkeypatch):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 0, _claude_ok([{"id": 0, "title": "animals"}, {"id": 1, "title": "colours"}]), ""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _name_with_claude_cli(REPS, "sonnet", batch_size=2) == {
        0: "animals",
        1: "colours",
    }


def test_claude_cli_strips_the_harness(monkeypatch):
    """A bare `claude -p` loads ~41k tokens of system prompt, tools and MCP
    schemas for a ~1.2k prompt. The stripping flags are load-bearing, not
    cosmetic, so pin them."""
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, _claude_ok([{"id": 0, "title": "a"}]), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _name_with_claude_cli(REPS, "sonnet", batch_size=2)
    c = seen["cmd"]
    assert "--strict-mcp-config" in c
    assert c[c.index("--setting-sources") + 1] == ""
    assert c[c.index("--allowed-tools") + 1] == ""
    assert c[c.index("--model") + 1] == "sonnet"


def test_claude_cli_reports_an_error_envelope_as_a_failed_chunk(monkeypatch):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 0, json.dumps({"is_error": True, "subtype": "max_turns"}), ""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="every claude-cli chunk failed"):
        _name_with_claude_cli(REPS, "sonnet", batch_size=2)


def test_claude_cli_missing_binary_is_a_run_level_failure(monkeypatch):
    """A missing binary must not degrade to centroid titles chunk by chunk —
    it is a setup problem for the whole run and has to say so."""

    def fake_run(cmd, **kw):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="not on PATH"):
        _name_with_claude_cli(REPS, "sonnet", batch_size=2)


def test_partial_results_survive_a_failed_chunk(monkeypatch):
    """One bad chunk must not discard the good ones — the caller stamps the
    shortfall into the namer string so the map discloses it."""
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return subprocess.CompletedProcess(cmd, 0, _claude_ok([{"id": 0, "title": "animals"}]), "")
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _name_with_claude_cli(REPS, "sonnet", batch_size=1) == {0: "animals"}


# --- codex-cli ------------------------------------------------------------


def test_codex_cli_reads_the_output_file_not_the_event_stream(monkeypatch):
    """Codex's JSONL stream interleaves reasoning with the answer;
    --output-last-message is the stable surface."""
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        out = cmd[cmd.index("-o") + 1]
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"titles": [{"id": 0, "title": "animals"}]}))
        return subprocess.CompletedProcess(cmd, 0, "noise on stdout", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _name_with_codex_cli(REPS, "gpt-5.6-sol", batch_size=2) == {0: "animals"}
    c = seen["cmd"]
    assert c[1] == "exec"
    # sandboxed, config-free, repo-free — none of which naming needs
    assert "--ignore-user-config" in c and "--ephemeral" in c
    assert c[c.index("-s") + 1] == "read-only"
    assert c[c.index("-m") + 1] == "gpt-5.6-sol"


def test_codex_cli_missing_output_file_is_a_failed_chunk(monkeypatch):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, "", "")  # writes nothing

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="every codex-cli chunk failed"):
        _name_with_codex_cli(REPS, "gpt-5.6-sol", batch_size=2)
