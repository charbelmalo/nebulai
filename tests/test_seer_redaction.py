"""M5 — what is in the log, and what comes back out of it.

Three separable claims, tested separately:

1. **Classification is complete.** Every payload key the whole seer suite can
   produce is registered in `FIELD_POLICY`. The registry-coverage test is the
   one that earns its keep on the day an agent version starts handing us a new
   field: it fails then, not when someone reads an export six months later.
2. **Scrubbing cannot be bypassed.** A credential is removed in the `Event`
   constructor, so no producer — adapter, runner, reconciler, server — can
   write one to disk, whatever route it takes.
3. **Redaction is honest.** What comes out at a level contains nothing above
   it, says what it dropped, and keeps the *length* of what it dropped, so a
   redacted absence is never read as a real one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nebulai.seer import redaction as R  # noqa: E402
from nebulai.seer.adapters.claude import ClaudeStreamAdapter  # noqa: E402
from nebulai.seer.adapters.codex import CodexExecAdapter  # noqa: E402
from nebulai.seer.contract import (  # noqa: E402
    CaptureMode,
    Event,
    EventType,
    Fidelity,
    Source,
)
from nebulai.seer.export import export, redact_events  # noqa: E402
from nebulai.seer.reducer import reduce_run  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "seer"


def _event(**kw) -> Event:
    kw.setdefault("event_type", EventType.TOOL_STARTED)
    kw.setdefault("run_id", "r")
    kw.setdefault("session_id", "s")
    kw.setdefault(
        "source",
        Source(
            agent="codex", agent_version="1", adapter="t", adapter_version="1",
            capture_mode=CaptureMode.DRIVEN, fidelity=Fidelity.NATIVE,
        ),
    )
    return Event(**kw)


def _replay(cls, fixture: str) -> list[Event]:
    a = cls(run_id="r", session_id="s")
    out: list[Event] = []
    for line in (FIXTURES / fixture).read_text().splitlines():
        if line.strip():
            out += a.feed(line)
    return out + a.finish()


# ── 1. classification ────────────────────────────────────────────────────────


class TestRegistry:
    def test_every_key_the_suite_can_emit_is_registered(self, tmp_path: Path) -> None:
        """Run the whole seer suite with `Event.to_dict` instrumented, and
        assert the union of payload keys is a subset of the registry.

        This is the gate the module exists for. An unregistered key defaults to
        `content`, so nothing leaks — but it also means the key is invisible to
        anyone reading `FIELD_POLICY` to find out what a log holds, and a field
        that is really metadata would be redacted away for no reason. Either
        way someone has to look at it.
        """
        out = tmp_path / "keys.json"
        plugin = tmp_path / "collect_payload_keys.py"
        plugin.write_text(
            "import atexit, json\n"
            "from nebulai.seer.contract import Event\n"
            "SEEN = set()\n"
            "_orig = Event.to_dict\n"
            "def _patched(self):\n"
            "    SEEN.update(self.payload)\n"
            "    return _orig(self)\n"
            "Event.to_dict = _patched\n"
            "@atexit.register\n"
            "def _dump():\n"
            f"    open({str(out)!r}, 'w').write(json.dumps(sorted(SEEN)))\n"
        )
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            # This file is excluded: it deliberately builds an event with an
            # unregistered key, to prove such a key is treated as content.
            [sys.executable, "-m", "pytest", "tests/", "-q", "-x", "-k", "seer",
             "--ignore", str(Path(__file__)), "-p", "collect_payload_keys"],
            cwd=root,
            env={"PYTHONPATH": f"src:{tmp_path}", "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, proc.stdout[-3000:]
        keys = set(json.loads(out.read_text()))
        assert keys, "the instrumented run produced no events at all"
        missing = sorted(keys - set(R.FIELD_POLICY) - R._ADDED_BY_REDACTION)
        assert not missing, (
            "payload keys with no content-level policy — classify them in "
            f"redaction.FIELD_POLICY before they ship: {missing}"
        )

    def test_the_registry_has_nothing_the_suite_cannot_produce(self) -> None:
        """The other direction is a warning, not a failure: a key can be real
        and simply unexercised. Kept as an explicit allowlist so the registry
        cannot quietly accumulate keys nobody emits any more."""
        stale = set(R.FIELD_POLICY) - _EXPECTED_KEYS
        assert not stale, f"registry entries not in the expected inventory: {sorted(stale)}"

    def test_an_unknown_key_is_treated_as_content(self) -> None:
        assert R.payload_level({"something_new": "x"}) is R.ContentLevel.CONTENT
        assert R.unregistered({"something_new": "x", "tool": "Bash"}) == ["something_new"]

    def test_an_absent_field_reveals_nothing(self) -> None:
        """`text: None` is the shape an adapter uses to say the agent said
        nothing. Letting it pin the event at `content` would redact events that
        hold no content."""
        assert R.payload_level({"text": None, "tool": "Bash"}) is R.ContentLevel.METADATA
        assert R.payload_level({"stderr_tail": [], "tool": "Bash"}) is R.ContentLevel.METADATA


class TestLabels:
    def test_an_event_carrying_prose_is_not_labelled_metadata(self) -> None:
        """The bug M5 fixes. Before this, every event said `metadata` — the one
        carrying the model's reply included."""
        e = _event(event_type=EventType.MESSAGE_ASSISTANT_COMPLETED, payload={"text": "hello"})
        assert e.privacy["content_level"] == "content"

    def test_a_command_sits_between_metadata_and_prose(self) -> None:
        e = _event(payload={"tool": "Bash", "command": "pytest -q"})
        assert e.privacy["content_level"] == "command"

    def test_a_counted_only_event_is_metadata(self) -> None:
        e = _event(payload={"tool": "Read", "path": "/x/y.py", "chars": 12})
        assert e.privacy["content_level"] == "metadata"

    def test_the_native_passthrough_pins_an_event_at_content(self) -> None:
        """`native` is the agent's own message kept for audit. Whatever else is
        in it, it is the agent's words."""
        e = _event(payload={"tool": "Read"}, native={"type": "item.started"})
        assert e.privacy["content_level"] == "content"

    def test_an_adapter_warning_survives_a_metadata_export(self) -> None:
        """Our own text is `note`, not `message`. A data-quality panel that
        blanks on redaction would report a clean capture."""
        a = CodexExecAdapter(run_id="r", session_id="s")
        w = a.warn("unmapped native event kind: 'x'")
        assert w.privacy["content_level"] == "metadata"
        kept = R.redact_event(w.to_dict(), R.ContentLevel.METADATA)
        assert kept["payload"]["note"] == "unmapped native event kind: 'x'"

    def test_an_explicit_label_is_not_overwritten(self) -> None:
        """The annotation route sets its own, because text a person typed into
        SessionSeer knowing it would be recorded is not the agent's content."""
        e = _event(
            payload={"text": "this run is the baseline"},
            privacy={"content_level": "content", "author_supplied": True},
        )
        assert e.privacy["author_supplied"] is True

    def test_reading_a_log_keeps_the_label_the_log_was_written_with(self) -> None:
        """A run captured under an older ruleset is described by that ruleset.
        Re-labelling on read would erase the difference between 'this build
        thinks it is metadata' and 'it was written as metadata'."""
        d = _event(payload={"text": "hi"}).to_dict()
        d["privacy"] = {"content_level": "metadata", "ruleset": "r0"}
        assert Event.from_dict(d).privacy == {"content_level": "metadata", "ruleset": "r0"}


# ── 2. scrubbing ─────────────────────────────────────────────────────────────


SECRETS = [
    ("sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA", "anthropic key"),
    ("ghp_0123456789abcdefghijABCDEF", "github token"),
    ("xoxb-1234567890-abcdefghij", "slack bot token"),
    ("AKIAIOSFODNN7EXAMPLE", "aws access key id"),
    ("AIzaSyA1234567890abcdefghijklmnopqrstu", "google api key"),
]


class TestScrub:
    @pytest.mark.parametrize("secret,what", SECRETS)
    def test_a_shaped_token_is_removed_wherever_it_appears(
        self, secret: str, what: str
    ) -> None:
        text, n = R.scrub_secrets(f"curl -H 'X: {secret}' https://api.example.com")
        assert secret not in text, what
        assert n == 1
        assert "https://api.example.com" in text, "scrubbing ate the rest of the command"

    def test_a_labelled_value_is_removed_even_when_the_token_is_shapeless(self) -> None:
        for cmd in (
            "export ANTHROPIC_API_KEY=hunter2",
            "psql --password hunter2",
            "curl -H 'Authorization: Bearer hunter2xxxxxx'",
            'deploy --token="hunter2"',
        ):
            text, n = R.scrub_secrets(cmd)
            assert "hunter2" not in text, cmd
            assert n >= 1

    def test_it_keeps_the_label_so_the_command_still_reads(self) -> None:
        text, _ = R.scrub_secrets("export ANTHROPIC_API_KEY=hunter2")
        assert text.startswith("export ANTHROPIC_API_KEY=")

    def test_an_ordinary_command_is_left_exactly_alone(self) -> None:
        for cmd in (
            "pytest tests/ -q",
            "git commit -m 'fix the token bucket'",
            "rg --files-with-matches password src/",
        ):
            assert R.scrub_secrets(cmd) == (cmd, 0)

    def test_scrubbing_is_idempotent(self) -> None:
        once, n1 = R.scrub_secrets("export API_TOKEN=sk-ant-api03-AAAAAAAAAAAAAAAAAAAA")
        twice, n2 = R.scrub_secrets(once)
        assert twice == once and n2 == 0

    def test_no_producer_can_write_a_secret_because_it_happens_in_the_constructor(
        self,
    ) -> None:
        """Not in the adapter, not in the runner, not at export: in `Event`.
        A scrub that can be sidestepped by building an event directly is not a
        scrub."""
        e = _event(payload={"tool": "Bash", "command": "export TOKEN=ghp_0123456789abcdefghijABCDEF"})
        assert "ghp_" not in e.to_json()
        assert e.privacy["scrubbed"] == 1
        assert e.privacy["ruleset"] == R.RULESET

    def test_program_output_is_scrubbed_too(self) -> None:
        """A tool that echoes the environment back is the other half of the
        problem: the user never typed the secret into the command."""
        e = _event(payload={"stderr_tail": ["using key sk-ant-api03-BBBBBBBBBBBBBBBBBBBB"]})
        assert "sk-ant" not in e.to_json()

    def test_a_clean_event_says_nothing_about_scrubbing(self) -> None:
        e = _event(payload={"tool": "Bash", "command": "pytest -q"})
        assert "scrubbed" not in e.privacy


# ── 3. redaction ─────────────────────────────────────────────────────────────


class TestRedact:
    def test_metadata_level_drops_prose_and_commands_and_native(self) -> None:
        e = _event(
            payload={"tool": "Bash", "command": "pytest -q", "text": "running tests"},
            native={"raw": "everything the agent said"},
        )
        out = R.redact_event(e.to_dict(), R.ContentLevel.METADATA)
        assert out["payload"]["command"] is None
        assert out["payload"]["text"] is None
        assert "native" not in out
        assert out["payload"]["tool"] == "Bash"

    def test_command_level_keeps_the_command_and_drops_the_prose(self) -> None:
        e = _event(payload={"command": "pytest -q", "text": "running tests"})
        out = R.redact_event(e.to_dict(), R.ContentLevel.COMMAND)
        assert out["payload"]["command"] == "pytest -q"
        assert out["payload"]["text"] is None

    def test_what_went_missing_leaves_its_length_behind(self) -> None:
        """A redacted absence must not read as a real one. Several analyses
        read only the length anyway, so this is not merely a courtesy."""
        e = _event(payload={"text": "x" * 4000})
        out = R.redact_event(e.to_dict(), R.ContentLevel.METADATA)
        assert out["payload"]["redacted_chars"]["text"] == 4000

    def test_the_event_says_which_fields_went_and_under_which_rules(self) -> None:
        e = _event(payload={"command": "ls", "text": "hi"})
        out = R.redact_event(e.to_dict(), R.ContentLevel.METADATA)
        assert out["privacy"]["redacted"] == ["command", "text"]
        assert out["privacy"]["ruleset"] == R.RULESET
        assert out["privacy"]["content_level"] == "metadata"

    def test_an_event_already_below_the_level_is_returned_untouched(self) -> None:
        e = _event(payload={"tool": "Read", "path": "/x"})
        d = e.to_dict()
        assert R.redact_event(d, R.ContentLevel.METADATA) == d

    def test_an_unregistered_key_is_redacted_rather_than_passed_through(self) -> None:
        e = _event(payload={"tool": "Read", "surprise_from_a_new_version": "content?"})
        out = R.redact_event(e.to_dict(), R.ContentLevel.METADATA)
        assert out["payload"]["surprise_from_a_new_version"] is None

    def test_a_redacted_event_survives_a_round_trip_through_the_log(self) -> None:
        e = _event(payload={"command": "ls", "text": "hi"})
        back = Event.from_dict(R.redact_event(e.to_dict(), R.ContentLevel.METADATA))
        assert back.payload["text"] is None
        assert back.privacy["redacted"] == ["command", "text"]


class TestExport:
    @pytest.fixture
    def run(self):
        events = _replay(ClaudeStreamAdapter, "claude-tools.jsonl")
        return reduce_run("r", events), events

    def test_a_metadata_jsonl_export_contains_no_prose(self, run) -> None:
        view, events = run
        prose = {
            e.payload["text"] for e in events
            if isinstance(e.payload.get("text"), str) and len(e.payload["text"]) > 20
        }
        assert prose, "the fixture has no prose to redact — the test proves nothing"
        body, _ct, name = export("jsonl", view, events, R.ContentLevel.METADATA)
        text = body.decode()
        for p in prose:
            assert p not in text
        assert name.endswith("-metadata.jsonl"), "a redacted export must say so in its name"

    def test_an_unredacted_export_is_unchanged(self, run) -> None:
        view, events = run
        assert export("jsonl", view, events)[0] == export("jsonl", view, events, None)[0]

    def test_the_csv_blanks_the_command_column_and_says_it_did(self, run) -> None:
        view, events = run
        body = export("csv", view, events, R.ContentLevel.METADATA)[0].decode()
        assert "Redacted to content level 'metadata'" in body
        assert "wc -c" not in body

    def test_the_csv_keeps_commands_at_command_level(self, run) -> None:
        view, events = run
        plain = export("csv", view, events)[0].decode()
        kept = export("csv", view, events, R.ContentLevel.COMMAND)[0].decode()
        assert "wc -c" in plain and "wc -c" in kept

    def test_the_analysis_export_redacts_the_view_it_carries(self, run) -> None:
        view, events = run
        doc = json.loads(export("analysis", view, events, R.ContentLevel.METADATA)[0])
        assert doc["redaction"] == {"content_level": "metadata", "ruleset": R.RULESET}
        for s in doc["view"]["spans"]:
            assert s["detail"] is None or "detail_chars" not in s

    def test_the_analyses_themselves_are_computed_before_redaction(self, run) -> None:
        """They read counts, actions and timing, never prose. Redacting first
        would change nothing except to make a shared export disagree with the
        run it came from."""
        view, events = run
        plain = json.loads(export("analysis", view, events)[0])
        cut = json.loads(export("analysis", view, events, R.ContentLevel.METADATA)[0])
        assert plain["analyses"] == cut["analyses"]

    def test_redacting_events_does_not_mutate_the_originals(self, run) -> None:
        _view, events = run
        before = [e.to_dict() for e in events]
        redact_events(events, R.ContentLevel.METADATA)
        assert [e.to_dict() for e in events] == before

    def test_an_unknown_level_is_refused_by_listing_the_options(self) -> None:
        with pytest.raises(ValueError, match="metadata, command, content"):
            R.parse_level("everything")


class TestCanaries:
    """Salt a fixture with unique markers and prove where each one can end up.

    Structural: this walks a *real* adapter over a *real* stream shape, so it
    catches a leak through a field nobody thought to classify — the failure
    mode a per-field unit test cannot see.
    """

    def _claude_line(self, **msg) -> str:
        return json.dumps(msg)

    def test_a_prompt_never_reaches_the_log_at_metadata_level(self) -> None:
        a = ClaudeStreamAdapter(run_id="r", session_id="s")
        events = a.feed(self._claude_line(
            type="user",
            message={"role": "user", "content": "CANARY_PROMPT please refactor"},
        ))
        blob = json.dumps([
            R.redact_event(e.to_dict(), R.ContentLevel.METADATA) for e in events
        ])
        assert "CANARY_PROMPT" not in blob

    def test_the_prompt_length_is_kept_even_when_the_prompt_is_not(self) -> None:
        """The user message adapter already keeps only `chars` — this pins that
        the count survives, because prompt length is an input to more than one
        analysis."""
        a = ClaudeStreamAdapter(run_id="r", session_id="s")
        events = a.feed(self._claude_line(
            type="user", message={"role": "user", "content": "x" * 137},
        ))
        assert any(e.payload.get("chars") == 137 for e in events)

    def test_file_content_never_enters_the_log_at_any_level(self) -> None:
        """Not a redaction property — an ingress one. `Write` inputs carry the
        whole file; the adapter counts newlines and keeps the integer."""
        a = ClaudeStreamAdapter(run_id="r", session_id="s")
        events = a.feed(self._claude_line(
            type="assistant",
            message={"role": "assistant", "content": [{
                "type": "tool_use", "id": "t1", "name": "Write",
                "input": {"file_path": "/x/y.py", "content": "CANARY_FILE_BODY\nline2\n"},
            }]},
        ))
        assert events
        assert "CANARY_FILE_BODY" not in json.dumps([e.to_dict() for e in events])

    def test_reasoning_is_dropped_by_policy_and_says_so(self) -> None:
        a = ClaudeStreamAdapter(run_id="r", session_id="s")
        events = a.feed(self._claude_line(
            type="assistant",
            message={"role": "assistant", "content": [
                {"type": "thinking", "thinking": "CANARY_REASONING"},
            ]},
        ))
        blob = json.dumps([e.to_dict() for e in events])
        assert "CANARY_REASONING" not in blob
        assert any(e.source.fidelity is Fidelity.DROPPED_BY_POLICY for e in events)

    def test_keeping_reasoning_is_opt_in_and_labelled_native(self) -> None:
        a = ClaudeStreamAdapter(run_id="r", session_id="s", keep_reasoning=True)
        events = a.feed(self._claude_line(
            type="assistant",
            message={"role": "assistant", "content": [
                {"type": "thinking", "thinking": "CANARY_REASONING"},
            ]},
        ))
        blob = json.dumps([e.to_dict() for e in events])
        assert "CANARY_REASONING" in blob
        cut = json.dumps([
            R.redact_event(e.to_dict(), R.ContentLevel.METADATA) for e in events
        ])
        assert "CANARY_REASONING" not in cut, "opting in must not opt out of redaction"


#: What `FIELD_POLICY` is expected to hold. Kept beside the tests rather than
#: derived from them, so adding a key to the registry is a deliberate act with
#: a diff, not a side effect of a passing run.
_EXPECTED_KEYS = {
    # content
    "text", "message", "status_detail", "stderr_tail", "reason", "error",
    # command
    "command",
    # metadata
    "agent", "author", "authoritative", "bytes", "cache_write",
    "cache_write_fidelity", "capture_gaps", "chars", "cli_version",
    "clock_resolution_s", "codex_bin", "compatible", "context_window",
    "cost_usd", "counted", "cumulative", "cwd", "decision", "delta",
    "duration_api_ms", "duration_ms", "duration_s", "effort", "exit_code",
    "first_hook", "golden_version", "gone_since_golden", "has_name",
    "history_mode", "is_error",
    "joined_midstream", "kind", "label", "limit_type", "lines_added",
    "lines_removed", "mcp_failed", "model_requested", "n_changes", "n_events",
    "n_parts", "n_tools", "n_turns", "native_categories", "native_session_id",
    "missing_notifications", "missing_requests", "needs_action",
    "new_since_golden", "note", "num_turns", "outcome", "output_chars", "path",
    "paths", "permission_mode", "provisional", "reasoning",
    "reasoning_fidelity", "reasoning_tokens_estimated", "recovered",
    "request_id", "resets_at", "source", "state", "status", "status_category",
    "stop_reason", "tags",
    "terminal_reason", "text_retained", "tool", "tools", "total_lines",
    "transport", "ttft_ms", "unmapped_notifications", "unmapped_requests",
    "usage", "using_overage",
}
