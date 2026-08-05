"""Native tool name → `Action`, and command line → `Action.VERIFY` detection.

Rule-based on purpose. A model classifier would be more flexible and would also
make every downstream metric non-deterministic and unversionable, which is the
opposite of what a research instrument needs. If a name is unrecognised we say
so (`Action.EXECUTE` for commands, `Action.INSPECT` for tools) rather than
guessing a specific category — an honest coarse label beats a confident wrong
one, and `unmatched_tools()` surfaces what the rules are missing so the rules
can be extended deliberately.

The tool-name rules are ported from `viewer/src/chrome/sessionlog.ts`, which
already learned that MCP names must be reduced to their leaf verb
(`mcp__workspace__bash` → `bash`) before matching.
"""

from __future__ import annotations

import re
import shlex

from .contract import Action

# ── tool names ───────────────────────────────────────────────────────────────


def tool_leaf(name: str) -> str:
    """`mcp__server__do_thing` → `do_thing`. Keys matching on the verb so an
    MCP-provided Read behaves like a native Read."""
    return name.split("__")[-1] or name


_TOOL_RULES: tuple[tuple[re.Pattern[str], Action], ...] = (
    # order matters: the first match wins, so specific before general
    (re.compile(r"^(Task(Create|Update|Stop|Get|List|Output)?|Agent|SendMessage|"
                r"subagent|spawn|delegate)", re.I), Action.DELEGATE),
    (re.compile(r"^(Write|Edit|MultiEdit|NotebookEdit|create_file|create_new_file|"
                r"str_replace|multi_str_replace|replace_symbol|insert_at|apply_patch|"
                r"applyPatch|fileChange)", re.I), Action.EDIT),
    (re.compile(r"^(Grep|Glob|Search|search|find|ToolSearch|WebSearch|fuzzy|"
                r"signature_search|similar)", re.I), Action.SEARCH),
    (re.compile(r"^(Bash|bash|shell|exec|run_|command|commandExecution|"
                r"unified_exec|process)", re.I), Action.EXECUTE),
    (re.compile(r"^(git|vcs|commit|branch|diff)", re.I), Action.VCS),
    (re.compile(r"^(Artifact|present_files|PushNotification|export_|upload_|"
                r"gif_creator|report)", re.I), Action.REPORT),
    (re.compile(r"^(Ask(UserQuestion)?|approval|clarif|permission|elicit|"
                r"requestUserInput)", re.I), Action.INTERACT),
    (re.compile(r"^(Read|LS|ls|WebFetch|fetch|get_|list_|read_|context|query|"
                r"snapshot|view)", re.I), Action.INSPECT),
)


def classify_tool(name: str) -> Action:
    """Normalized action for a native tool name.

    Unknown tools fall to `INSPECT` — the least-committal label, and the one
    that cannot fake a verification or an edit that never happened.
    """
    leaf = tool_leaf(name)
    for pat, action in _TOOL_RULES:
        if pat.match(leaf):
            return action
    return Action.INSPECT


def unmatched_tools(names: list[str]) -> list[str]:
    """Tool names no rule matched. Surfaced in the data-quality panel so the
    taxonomy's blind spots are visible instead of silently collapsing to
    INSPECT."""
    out = []
    for n in names:
        leaf = tool_leaf(n)
        if not any(p.match(leaf) for p, _ in _TOOL_RULES):
            out.append(n)
    return out


# ── commands ─────────────────────────────────────────────────────────────────

#: A command is VERIFY when its *intent* is to check work. Matched on the
#: program and its subcommand, never on a substring of the whole line — `rm
#: -rf test/` contains "test" and is emphatically not a verification.
_VERIFY_PROGRAMS = frozenset(
    {
        "pytest", "tox", "nose2", "unittest", "jest", "vitest", "mocha", "ava",
        "phpunit", "rspec", "ctest", "gotestsum", "cargo-test", "tsc", "mypy",
        "pyright", "ruff", "flake8", "pylint", "eslint", "biome", "clippy",
        "shellcheck", "hadolint", "stylelint",
    }
)

#: `<program> <subcommand>` pairs that verify. `cargo build` verifies; `cargo
#: run` does not. `npm test` verifies; `npm install` does not.
_VERIFY_PAIRS = frozenset(
    {
        ("cargo", "test"), ("cargo", "check"), ("cargo", "clippy"), ("cargo", "build"),
        ("go", "test"), ("go", "vet"), ("go", "build"),
        ("npm", "test"), ("npm", "run"), ("pnpm", "test"), ("yarn", "test"),
        ("bun", "test"), ("uv", "run"), ("dotnet", "test"), ("mvn", "test"),
        ("gradle", "test"), ("make", "test"), ("make", "check"), ("make", "lint"),
    }
)

#: For `<program> run <script>` forms, the script names that mean verification.
_VERIFY_SCRIPTS = re.compile(
    r"^(test|tests|check|lint|typecheck|type-check|tsc|build|ci|verify|audit)\b", re.I
)

_VCS_PROGRAMS = frozenset({"git", "jj", "hg", "svn", "gh", "glab"})

_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "fish", "ksh"})


def unwrap_shell(cmdline: str) -> str:
    """Strip a `<shell> -lc '<real command>'` wrapper.

    Codex wraps every `command_execution` this way — a real capture shows
    `/bin/zsh -lc 'wc -c hello.txt'`, and Claude Code's Bash tool passes the
    same command bare. Without unwrapping, every Codex command classifies as
    `zsh` → EXECUTE, and a Codex run that ran the full test suite would show
    zero verification while the identical Claude run showed one. That is not a
    finding about the agents; it is an artifact of who quotes what.
    """
    for _ in range(3):  # `sudo bash -lc "..."` nests at most a couple deep
        try:
            parts = shlex.split(cmdline)
        except ValueError:
            return cmdline
        if len(parts) < 3:
            return cmdline
        prog = parts[0].rsplit("/", 1)[-1]
        if prog not in _SHELLS:
            return cmdline
        # the flag bundle before the script: -c, -lc, -ec, --login -c, …
        i = 1
        while i < len(parts) and parts[i].startswith("-"):
            if parts[i].lstrip("-").endswith("c"):
                i += 1
                break
            i += 1
        if i >= len(parts):
            return cmdline
        cmdline = parts[i]
    return cmdline


def classify_command(cmdline: str) -> Action:
    """Normalized action for a shell command line.

    Returns `EXECUTE` for anything unrecognised. Getting this wrong in the
    permissive direction would fabricate verification coverage the run never
    had, so every rule here is anchored to the program position.
    """
    cmdline = unwrap_shell(cmdline)
    try:
        parts = shlex.split(cmdline)
    except ValueError:  # unbalanced quotes — still worth a coarse answer
        parts = cmdline.split()
    # step over env assignments and common prefixes: `FOO=1 uv run pytest`
    while parts and ("=" in parts[0] and not parts[0].startswith("-")):
        parts = parts[1:]
    while parts and parts[0] in ("sudo", "time", "nice", "env", "npx", "pnpx", "poetry"):
        parts = parts[1:]
    if not parts:
        return Action.EXECUTE

    prog = parts[0].rsplit("/", 1)[-1]
    sub = parts[1] if len(parts) > 1 else ""

    if prog in _VCS_PROGRAMS:
        return Action.VCS
    if prog in _VERIFY_PROGRAMS:
        return Action.VERIFY
    if (prog, sub) in _VERIFY_PAIRS:
        # `npm run <script>` / `uv run <thing>` only verify for some scripts
        if sub == "run":
            tail = parts[2] if len(parts) > 2 else ""
            tail_prog = tail.rsplit("/", 1)[-1]
            if tail_prog in _VERIFY_PROGRAMS:
                return Action.VERIFY
            return Action.VERIFY if _VERIFY_SCRIPTS.match(tail) else Action.EXECUTE
        return Action.VERIFY
    return Action.EXECUTE


# ── edit extent ──────────────────────────────────────────────────────────────


def _nlines(s: object) -> int | None:
    """Lines in a string, counted without keeping the string."""
    if not isinstance(s, str):
        return None
    if s == "":
        return 0
    return s.count("\n") + 1


def edit_extent(tool: str, tool_input: dict | None) -> dict[str, int] | None:
    """How many lines an edit tool's *input* says it will touch.

    Counting newlines is metadata: the numbers survive, the text does not, so
    this runs at ingress under the `metadata` privacy tier without leaking file
    contents into the log. Churn cannot be computed at all without it —
    `FILE_CHANGED` otherwise carries only a path, and a path tells you a file
    was rewritten seventeen times but not by how much.

    Returns `None` when the tool's shape gives no line information, which is
    the common case for patch-style and MCP edit tools. That `None` is what
    makes `edit_churn` say "missing" instead of "0" for a Codex run.

    `total_lines` is only set when the input contains the file's *whole* new
    body (a `Write`), because that is the only case where the log knows the
    file's length rather than the size of a change to it.
    """
    if not isinstance(tool_input, dict):
        return None
    leaf = tool_leaf(tool)

    if leaf in ("Write", "create_file", "create_new_file"):
        n = _nlines(tool_input.get("content") or tool_input.get("file_text"))
        if n is None:
            return None
        return {"lines_added": n, "lines_removed": 0, "total_lines": n}

    if leaf in ("Edit", "str_replace", "replace_symbol"):
        old = _nlines(tool_input.get("old_string") or tool_input.get("old_str"))
        new = _nlines(tool_input.get("new_string") or tool_input.get("new_str"))
        if old is None and new is None:
            return None
        # `replace_all` multiplies the extent by a count we do not know. Report
        # the single-occurrence extent and mark it, rather than guessing.
        d = {"lines_added": new or 0, "lines_removed": old or 0}
        if tool_input.get("replace_all"):
            d["at_least"] = 1
        return d

    if leaf in ("MultiEdit", "multi_str_replace"):
        edits = tool_input.get("edits")
        if not isinstance(edits, list):
            return None
        added = removed = 0
        for ed in edits:
            if not isinstance(ed, dict):
                continue
            added += _nlines(ed.get("new_string") or ed.get("new_str")) or 0
            removed += _nlines(ed.get("old_string") or ed.get("old_str")) or 0
        return {"lines_added": added, "lines_removed": removed}

    if leaf in ("NotebookEdit",):
        n = _nlines(tool_input.get("new_source"))
        if n is None:
            return None
        mode = tool_input.get("edit_mode") or "replace"
        if mode == "delete":
            return {"lines_added": 0, "lines_removed": n}
        return {"lines_added": n, "lines_removed": 0 if mode == "insert" else n}

    return None
