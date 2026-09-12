"""Codex's `apply_patch` envelope → per-file line counts.

`taxonomy.edit_extent` counts an edit from the tool *input*, which is what lets
`edit_churn` report a number without any file content reaching the log. It
returns `None` for patch-shaped tools, and that `None` is why `edit_churn` said
"this agent's file-change events carry no line counts" for every Codex run:
`codex exec --json`'s `file_change` item carries `{path, kind}` and nothing
else, and the line counts live one level up, in the `apply_patch` argument the
model wrote.

That argument is not a unified diff. It is the `*** Begin Patch` envelope, and a
real one (captured from `~/.codex/sessions/**/rollout-*.jsonl`, a
`custom_tool_call` with `name: "apply_patch"`) looks like this:

    *** Begin Patch
    *** Update File: components/neurograph/neurograph-edge.tsx
    @@
     const CONNECTION_LANE_SAMPLE_COUNT = 24
    +const EFFECT_OPACITY_EPSILON = 0.005
    -const OLD = 1
    *** End Patch

The counts are therefore *in the payload*, which is what makes this
`Fidelity.DETERMINISTIC` rather than an estimate: `+` lines are added, `-` lines
are removed, and nothing is inferred.

**A file whose hunks cannot be counted exactly contributes `MISSING`, not 0.**
Three shapes reach that outcome and each of them is a real case, not a
defensive branch:

* `*** Delete File: p` — the file is gone and the patch never says how big it
  was. "0 lines removed" would be a confident lie about a whole file.
* a body line in an update hunk that is neither `+`, `-` nor context — a shape
  this parser does not understand, where counting the lines it *does* recognise
  would silently under-report the edit.
* a patch with `*** Begin Patch` and no `*** End Patch` — truncated in transit,
  so its last file's hunks may be cut off mid-count.

A `FileEdit` with `lines_added is None` carries no line keys into its
`FILE_CHANGED` payload at all, which is exactly what `Reducer._file_stat`
already treats as "this agent told us the file changed but not by how much".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

BEGIN = "*** Begin Patch"
END = "*** End Patch"

_FILE_HEADER = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+?)\s*$")
_MOVE = re.compile(r"^\*\*\* Move to: (.+?)\s*$")
#: `\ No newline at end of file` is diff bookkeeping, not a line of the file.
_NO_NEWLINE = "\\ No newline at end of file"

_KIND = {"Add": "add", "Update": "update", "Delete": "delete"}


@dataclass(frozen=True, slots=True)
class FileEdit:
    """One file's share of a patch. `lines_added is None` means *unknown*."""

    path: str
    kind: str  # "add" | "update" | "delete"
    lines_added: int | None = None
    lines_removed: int | None = None
    #: only set for `Add File`, where the patch body *is* the whole file and so
    #: the file's final length is known rather than accumulated
    total_lines: int | None = None
    move_to: str | None = None
    #: why the counts are absent, when they are
    note: str | None = None

    @property
    def counted(self) -> bool:
        return self.lines_added is not None or self.lines_removed is not None

    def payload(self) -> dict[str, Any]:
        """`FILE_CHANGED` payload keys for this file.

        Emits no line keys when the extent is unknown, so the reducer leaves
        `line_data` false and `edit_churn` reports a gap for this file instead
        of a ratio built from a zero it invented.
        """
        p: dict[str, Any] = {"path": self.path, "kind": self.kind}
        if self.counted:
            p["lines_added"] = self.lines_added or 0
            p["lines_removed"] = self.lines_removed or 0
            if self.total_lines is not None:
                p["total_lines"] = self.total_lines
        else:
            p["lines_fidelity"] = "missing"
            if self.note:
                p["note"] = self.note
        return p


def extract_patch(text: str | None) -> str | None:
    """The `*** Begin Patch …` envelope inside a larger string, if there is one.

    Handles the shell form (`apply_patch <<'EOF' … EOF`) and the bare argument
    form with the same code, because both are located by the envelope's own
    markers rather than by shell parsing — quoting varies, the markers do not.
    Returns the text from `*** Begin Patch` onwards, including a missing
    `*** End Patch`, so the truncation case reaches `parse_apply_patch` and is
    reported there rather than being silently dropped here.
    """
    if not isinstance(text, str):
        return None
    i = text.find(BEGIN)
    if i < 0:
        return None
    j = text.find(END, i)
    return text[i:] if j < 0 else text[i : j + len(END)]


def parse_apply_patch(text: str | None) -> list[FileEdit]:
    """Per-file extents for an `apply_patch` envelope. `[]` if it is not one.

    An empty list means "this is not a patch we recognise" and never "a patch
    that changed nothing" — the caller emits nothing at all for it, so an
    unrecognised shape leaves the run exactly as blind as it was before,
    which is the honest outcome.
    """
    patch = extract_patch(text)
    if patch is None:
        return []
    truncated = END not in patch
    lines = patch.splitlines()

    out: list[FileEdit] = []
    path: str | None = None
    kind = ""
    added = removed = 0
    move_to: str | None = None
    broke: str | None = None

    def flush() -> None:
        nonlocal path, kind, added, removed, move_to, broke
        if path is None:
            return
        if kind == "delete":
            out.append(
                FileEdit(
                    path, kind, move_to=move_to,
                    note="apply_patch deletes a file without stating its length",
                )
            )
        elif broke is not None:
            out.append(FileEdit(path, kind, move_to=move_to, note=broke))
        else:
            out.append(
                FileEdit(
                    path, kind, lines_added=added, lines_removed=removed,
                    # Only `Add File` carries the file's whole new body, so it
                    # is the only shape that establishes a final length.
                    total_lines=added if kind == "add" else None,
                    move_to=move_to,
                )
            )
        path, kind, added, removed, move_to, broke = None, "", 0, 0, None, None

    for raw in lines:
        if raw.startswith(BEGIN) or raw.startswith(END):
            continue
        header = _FILE_HEADER.match(raw)
        if header:
            flush()
            kind = _KIND[header.group(1)]
            path = header.group(2)
            continue
        if path is None:
            continue
        mv = _MOVE.match(raw)
        if mv:
            move_to = mv.group(1)
            continue
        if raw.startswith("@@"):
            continue
        if raw.startswith("*** "):
            # Another envelope directive — `*** End of File` is the one that
            # actually occurs. Envelope bookkeeping, not a line of any file.
            continue
        if raw.startswith("+"):
            added += 1
        elif raw.startswith("-"):
            removed += 1
        elif raw.startswith(" ") or raw == "" or raw == _NO_NEWLINE:
            continue
        elif broke is None:
            broke = (
                f"unparseable line in this file's hunks ({raw[:24]!r}…): the "
                f"counts would be wrong, so they are absent"
            )
    flush()

    if truncated:
        out = [
            FileEdit(
                f.path, f.kind, move_to=f.move_to,
                note="patch has no *** End Patch marker — it may be truncated",
            )
            for f in out
        ]
    return out


__all__ = ["BEGIN", "END", "FileEdit", "extract_patch", "parse_apply_patch"]
