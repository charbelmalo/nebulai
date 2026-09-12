"""`edit_churn` on Codex: counting an `apply_patch` envelope, and refusing to.

Codex's `exec --json` `file_change` item carries `{path, kind}` and no line
counts, which is why `edit_churn` reported "this agent's file-change events
carry no line counts" for every Codex run. The counts do exist — one level up,
in the `apply_patch` argument the model wrote — and this file is the two halves
of getting them out: the shapes we can count exactly, and the shapes we refuse
to count at all.

The refusals are the point. A `*** Delete File:` directive says a file is gone
and never says how big it was; reporting `lines_removed: 0` for it would be a
confident lie about a whole file, and `edit_churn` would divide by it.
"""

from __future__ import annotations

import json

import pytest

from nebulai.seer.adapters import CodexExecAdapter
from nebulai.seer.adapters.codex_app_server import diff_extent
from nebulai.seer.adapters.patches import extract_patch, parse_apply_patch
from nebulai.seer.analysis import edit_churn
from nebulai.seer.contract import EventType, Fidelity
from nebulai.seer.reducer import Reducer
from nebulai.seer.taxonomy import edit_extent

UPDATE = (
    "*** Begin Patch\n"
    "*** Update File: src/a.py\n"
    "@@\n"
    " context line\n"
    "+added one\n"
    "+added two\n"
    "-removed one\n"
    "*** End Patch\n"
)

ADD = (
    "*** Begin Patch\n"
    "*** Add File: docs/new.md\n"
    "+# Title\n"
    "+\n"
    "+body\n"
    "*** End Patch\n"
)

DELETE = "*** Begin Patch\n*** Delete File: src/gone.py\n*** End Patch\n"


def mk(**kw):
    kw.setdefault("run_id", "run_t")
    kw.setdefault("session_id", "ses_t")
    return CodexExecAdapter(**kw)


def command_item(item_id: str, command: str, *, exit_code: int = 0) -> str:
    return json.dumps(
        {
            "type": "item.completed",
            "item": {
                "id": item_id,
                "type": "command_execution",
                "command": command,
                "exit_code": exit_code,
                "status": "completed",
                "aggregated_output": "",
            },
        }
    )


def file_change_item(item_id: str, *changes: dict) -> str:
    return json.dumps(
        {
            "type": "item.completed",
            "item": {"id": item_id, "type": "file_change", "changes": list(changes)},
        }
    )


def changed(events):
    return [e for e in events if e.event_type is EventType.FILE_CHANGED]


# ── the parser ───────────────────────────────────────────────────────────────


class TestParser:
    def test_an_update_hunk_is_counted_exactly(self) -> None:
        (e,) = parse_apply_patch(UPDATE)
        assert (e.path, e.kind) == ("src/a.py", "update")
        assert (e.lines_added, e.lines_removed) == (2, 1)
        assert e.payload()["lines_added"] == 2

    def test_an_added_file_also_reports_its_final_length(self) -> None:
        """`Add File` carries the whole new body, so its length is known rather
        than accumulated — which is what `edit_churn`'s denominator needs."""
        (e,) = parse_apply_patch(ADD)
        assert (e.lines_added, e.lines_removed, e.total_lines) == (3, 0, 3)

    def test_the_envelope_is_found_inside_a_shell_heredoc(self) -> None:
        cmd = f"apply_patch <<'EOF'\n{UPDATE}EOF\n"
        assert extract_patch(cmd) is not None
        assert parse_apply_patch(cmd)[0].lines_added == 2

    def test_a_unified_diff_is_not_an_apply_patch_envelope(self) -> None:
        """`[]` means "not a patch we recognise", so the caller emits nothing —
        never "a patch that changed nothing"."""
        assert parse_apply_patch("@@ -1 +1 @@\n+x\n-y\n") == []
        assert parse_apply_patch(None) == []

    def test_taxonomy_still_declines_to_count_a_patch_tool(self) -> None:
        """`edit_extent` reads a tool *input* and has no envelope parser. It
        must keep returning `None` for one, or two layers would count the same
        edit."""
        assert edit_extent("apply_patch", {"patch": "@@ -1 +1 @@"}) is None


class TestRefusals:
    def test_a_deleted_file_is_missing_not_zero(self) -> None:
        """The task's rule, at its sharpest. The patch says the file is gone
        and never says how big it was."""
        (e,) = parse_apply_patch(DELETE)
        assert e.kind == "delete"
        assert e.lines_added is None and e.lines_removed is None
        p = e.payload()
        assert p["lines_fidelity"] == Fidelity.MISSING.value
        assert "lines_added" not in p and "lines_removed" not in p
        assert "without stating its length" in p["note"]

    def test_an_unparseable_hunk_line_makes_that_file_missing(self) -> None:
        bad = (
            "*** Begin Patch\n"
            "*** Update File: src/a.py\n"
            "@@\n"
            "+added\n"
            "zzz not a hunk line\n"
            "*** End Patch\n"
        )
        (e,) = parse_apply_patch(bad)
        assert not e.counted
        assert e.payload()["lines_fidelity"] == Fidelity.MISSING.value

    def test_a_truncated_patch_makes_every_file_missing(self) -> None:
        """No `*** End Patch` means the last file's hunks may be cut off
        mid-count, and we cannot tell which file that was."""
        edits = parse_apply_patch(UPDATE.replace("*** End Patch\n", ""))
        assert edits and not any(e.counted for e in edits)
        assert "truncated" in edits[0].payload()["note"]

    def test_one_uncountable_file_does_not_poison_its_siblings(self) -> None:
        multi = (
            "*** Begin Patch\n"
            "*** Update File: src/a.py\n"
            "@@\n"
            "+one\n"
            "*** Delete File: src/gone.py\n"
            "*** End Patch\n"
        )
        a, gone = parse_apply_patch(multi)
        assert a.lines_added == 1
        assert not gone.counted


class TestAppServerDiff:
    def test_a_unified_diff_is_counted_as_before(self) -> None:
        assert diff_extent("--- a\n+++ b\n+x\n+y\n-z\n") == {
            "lines_added": 2,
            "lines_removed": 1,
        }

    def test_an_envelope_in_the_diff_field_goes_through_the_parser(self) -> None:
        assert diff_extent(UPDATE) == {"lines_added": 2, "lines_removed": 1}

    def test_a_whole_file_deletion_is_not_counted_as_zero(self) -> None:
        """Counted as a unified diff, `*** Delete File:` has no `+`/`-` markers
        and would read as a file that changed by nothing."""
        assert diff_extent(DELETE) is None


# ── through the adapter ──────────────────────────────────────────────────────


class TestCodexAdapter:
    def test_a_shell_apply_patch_emits_counted_file_changes(self) -> None:
        a = mk()
        ev = a.feed(command_item("i1", f"apply_patch <<'EOF'\n{UPDATE}EOF"))
        (fc,) = changed(ev)
        assert fc.payload["path"] == "src/a.py"
        assert fc.payload["lines_added"] == 2
        assert fc.payload["via"] == "apply_patch"
        assert fc.source.fidelity is Fidelity.DETERMINISTIC

    def test_a_native_file_change_item_says_its_counts_are_missing(self) -> None:
        """`exec --json` states which file changed and never by how much. The
        event has to say so, or the reducer cannot tell it apart from a file
        that changed by zero lines."""
        a = mk()
        ev = a.feed(file_change_item("i1", {"path": "src/b.py", "kind": "update"}))
        (fc,) = changed(ev)
        assert fc.payload["lines_fidelity"] == Fidelity.MISSING.value
        assert "lines_added" not in fc.payload

    def test_a_patch_reported_twice_is_counted_once(self) -> None:
        a = mk()
        ev = a.feed(command_item("i1", f"apply_patch <<'EOF'\n{UPDATE}EOF"))
        ev += a.feed(file_change_item("i2", {"path": "src/a.py", "kind": "update"}))
        assert [e.payload["path"] for e in changed(ev)] == ["src/a.py"]
        done = [e for e in ev if e.event_type is EventType.TOOL_COMPLETED][-1]
        assert done.payload["n_deduped"] == 1

    def test_dedup_is_consume_once(self) -> None:
        """A genuinely later edit to the same file still counts. Suppressing
        every future `file_change` for a path would lose real edits."""
        a = mk()
        ev = a.feed(command_item("i1", f"apply_patch <<'EOF'\n{UPDATE}EOF"))
        ev += a.feed(file_change_item("i2", {"path": "src/a.py", "kind": "update"}))
        ev += a.feed(file_change_item("i3", {"path": "src/a.py", "kind": "update"}))
        assert [e.payload["path"] for e in changed(ev)] == ["src/a.py", "src/a.py"]

    def test_a_failed_apply_patch_changes_nothing(self) -> None:
        a = mk()
        ev = a.feed(
            command_item("i1", f"apply_patch <<'EOF'\n{UPDATE}EOF", exit_code=1)
        )
        assert changed(ev) == []


# ── what churn does with it ──────────────────────────────────────────────────


def _churn(events):
    r = Reducer("run_t")
    r.reduce(events)
    return edit_churn(r.finalize(), list(events))


class TestEditChurn:
    def test_codex_now_reports_a_number(self) -> None:
        a = mk()
        ev = a.feed(command_item("i1", f"apply_patch <<'EOF'\n{ADD}EOF"))
        ev += a.feed(command_item("i2", f"apply_patch <<'EOF'\n{UPDATE}EOF"))
        rows = {r["path"]: r for r in _churn(ev).rows}
        assert rows["docs/new.md"]["lines_written"] == 3
        assert rows["src/a.py"]["lines_written"] == 3

    def test_a_deleted_file_reports_a_gap_not_a_zero(self) -> None:
        """The absent case, end to end: the row for a file we could not count
        carries a note and no `lines_written`, rather than a 0 that would make
        the file look untouched next to its counted siblings."""
        a = mk()
        ev = a.feed(command_item("i1", f"apply_patch <<'EOF'\n{UPDATE}EOF"))
        ev += a.feed(command_item("i2", f"apply_patch <<'EOF'\n{DELETE}EOF"))
        rows = {r["path"]: r for r in _churn(ev).rows}
        gone = rows["src/gone.py"]
        assert "lines_written" not in gone
        assert "lines_added" not in gone
        assert "note" in gone
        # and the counted sibling is unaffected
        assert rows["src/a.py"]["lines_written"] == 3

    @pytest.mark.parametrize("patch", [DELETE, UPDATE.replace("*** End Patch\n", "")])
    def test_an_uncountable_patch_never_produces_a_zero_count(self, patch: str) -> None:
        a = mk()
        ev = a.feed(command_item("i1", f"apply_patch <<'EOF'\n{patch}EOF"))
        for e in changed(ev):
            assert e.payload.get("lines_added") != 0
            assert e.payload["lines_fidelity"] == Fidelity.MISSING.value
