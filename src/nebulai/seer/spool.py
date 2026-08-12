"""Observed mode: the hook shim, the spool it appends to, and the reader.

The shim is the whole ingress for observed capture, and it is not a client. It
appends one JSON line to a file and exits 0. There is no socket, no daemon
dependency, no retry, and nothing to fall back to — which is why it survives the
collector being down, restarted, or never installed at all.

Two properties are non-negotiable, because the shim runs inside somebody's real
coding session:

* **It always exits 0.** A hook that exits non-zero can block a tool call or
  abort a turn. An observability tool that can break the thing it observes is
  not one. Every failure path here — no spool directory, unwritable file, absurd
  payload — ends in a successful exit and a missing line.
* **It forks nothing.** Process spawn is the entire cost. A script that does
  nothing but `exit 0` measures 4.7–6.0 ms on this machine depending on load,
  and the shim measures 0.3–0.8 ms more than whatever that floor is on the run.
  Every `$(...)` would add ~3.5 ms, so the timestamp comes from a shell builtin
  or not at all (`pick_clock`).

The plan's "< 5 ms p95" target is therefore unreachable for *any* external hook
program here — the floor alone can exceed it — so the criterion this codebase
holds itself to is the one `test_the_shim_costs_about_what_an_empty_script_costs`
asserts instead: the shim costs a fraction of a millisecond more than the empty
script the agent was going to spawn anyway. That difference is the only part we
control, and it is the only part measured.

Timestamps carry their resolution. `zsh`'s `$EPOCHREALTIME` is microseconds for
+0.7 ms; the fallbacks are whole seconds. A whole-second clock cannot support a
tool-span duration, so events built from one say `estimated` and the reducer's
durations inherit it, rather than reporting `0.0s` for everything short.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

#: Payload bytes the shim will pass through. Beyond this it writes a marker with
#: the true size instead: a `Write` tool call carries the whole file body and a
#: `Read` result carries the whole file — neither belongs in a metadata-first
#: capture, and a 200 KB single `write()` is also where append atomicity stops
#: being something we can rely on.
MAX_PAYLOAD_BYTES = 16384

#: How often the reader looks for new lines. Also the bound on how stale a live
#: timestamp can be when the shim had no sub-second clock, which is why it is
#: recorded on the events rather than left as folklore.
POLL_INTERVAL_S = 0.1

SHIM_NAME = "hook.sh"
#: Written next to the shim so the reader knows what clock produced the
#: timestamps it is reading, without having to parse the shim back.
MANIFEST_NAME = "shim.json"


# ── clock selection ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Clock:
    """A way for the shim to learn the time without forking."""

    name: str
    interpreter: str
    #: shell prologue, if the clock needs one
    setup: str
    #: shell expansion yielding a JSON number
    expr: str
    #: seconds; the granularity a duration computed from this clock can claim
    resolution_s: float

    @property
    def sub_second(self) -> bool:
        return self.resolution_s < 1.0


_ZSH = Clock(
    name="zsh:EPOCHREALTIME",
    interpreter="/bin/zsh -f",
    setup="zmodload zsh/datetime 2>/dev/null",
    expr='"${EPOCHREALTIME:-0}"',
    resolution_s=1e-6,
)
_BASH42 = Clock(
    name="bash:printf-%(%s)T",
    interpreter="/bin/bash",
    setup='t=""',
    expr='"$(printf "%(%s)T" -1)"',  # a builtin in bash >= 4.2: no fork
    resolution_s=1.0,
)
_DATE = Clock(
    name="sh:date",
    interpreter="/bin/sh",
    setup="",
    expr='"$(date +%s)"',  # the one fallback that forks
    resolution_s=1.0,
)
#: Last resort: no clock at all. The reader stamps at ingest, which is accurate
#: while it is running and unknowable for anything it did not see appear.
_NONE = Clock(
    name="none",
    interpreter="/bin/sh",
    setup="",
    expr="0",
    resolution_s=float("inf"),
)

CLOCKS = (_ZSH, _BASH42, _DATE, _NONE)


def _works(clock: Clock) -> bool:
    """Actually run the clock expression. Version sniffing gets this wrong —
    macOS ships bash 3.2 as both /bin/sh and /bin/bash, and zsh needs a module
    load that can be compiled out."""
    exe = clock.interpreter.split()[0]
    if not Path(exe).exists():
        return False
    script = f"{clock.setup}\nprintf '%s' {clock.expr}\n"
    try:
        out = subprocess.run(
            [*clock.interpreter.split(), "-c", script],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return False
    try:
        return float(out.stdout.strip()) > 1e9
    except ValueError:
        return False


def pick_clock() -> Clock:
    """The best clock this machine actually provides, verified by running it."""
    for c in CLOCKS[:-1]:
        if _works(c):
            return c
    return _NONE


# ── the shim ─────────────────────────────────────────────────────────────────

_SHIM_TEMPLATE = """#!{interpreter}
# nebulai SessionSeer hook shim — generated, do not edit.
#
# usage: {shim} <agent> <event-name>   (the hook payload arrives on stdin)
#
# Appends one JSON line to the spool and exits 0, always. Deleting the spool
# directory disables it without touching any agent's configuration, which is
# what `seer uninstall` relies on.
#
# clock: {clock}   resolution: {resolution}s
d="${{NEBULAI_SEER_SPOOL:-{spool}}}"
[ -d "$d" ] || exit 0
{setup}
p=''
while IFS= read -r l || [ -n "$l" ]; do p="$p$l"; done
[ -n "$p" ] || p='null'
# a tool payload can carry a whole file; past the cap we keep the size and drop
# the body, and the collector reports it as dropped rather than as absent
if [ ${{#p}} -gt {cap} ]; then p="{{\\"seer_oversized\\":${{#p}}}}"; fi
# `$$` is this shim, a fresh process per hook, so it identifies a *firing* and
# never a session. `$PPID` is whoever ran us — usually a shell the agent spawned,
# so it is a weak grouping hint and not a correlation key. Both are recorded
# under their own names; neither is presented as "the agent's pid".
# The braces matter: `printf … >> f 2>/dev/null` redirects *printf's* stderr,
# but it is the shell that reports a failed redirection, so an unwritable spool
# would print "permission denied" into the agent's hook output. Grouping puts
# the shell's own complaint inside the redirection too. The semicolon before the
# closing brace is what makes the group POSIX sh as well as zsh.
{{ printf '{{"v":1,"agent":"%s","event":"%s","t":%s,"pid":%s,"ppid":%s,"payload":%s}}\\n' \\
  "$1" "$2" {expr} "$$" "${{PPID:-0}}" "$p" >> "$d/$1.jsonl"; }} 2>/dev/null
exit 0
"""


def render_shim(spool: Path, clock: Clock) -> str:
    return _SHIM_TEMPLATE.format(
        interpreter=clock.interpreter,
        shim=SHIM_NAME,
        clock=clock.name,
        resolution=clock.resolution_s,
        spool=spool,
        setup=clock.setup,
        cap=MAX_PAYLOAD_BYTES,
        expr=clock.expr,
    )


def install_shim(root: Path, clock: Clock | None = None) -> tuple[Path, Clock]:
    """Write the shim and its spool directory. Idempotent.

    Returns the shim path and the clock it was built against. Nothing here
    touches an agent's configuration — that is `install.py`, and keeping the two
    apart is what lets the shim be reinstalled after an OS upgrade changes which
    shells exist without going near `settings.json`.
    """
    c = clock or pick_clock()
    d = spool_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    shim = shim_path(root)
    shim.write_text(render_shim(d, c), encoding="utf-8")
    shim.chmod(0o755)
    (root / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "clock": c.name,
                "resolution_s": c.resolution_s,
                "interpreter": c.interpreter,
                "max_payload_bytes": MAX_PAYLOAD_BYTES,
                "written_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return shim, c


def spool_dir(root: Path) -> Path:
    return Path(root) / "spool"


def shim_path(root: Path) -> Path:
    return Path(root) / SHIM_NAME


def read_manifest(root: Path) -> dict[str, Any]:
    p = Path(root) / MANIFEST_NAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def remove_shim(root: Path, *, keep_spool: bool = True) -> None:
    """Disable observed capture without editing any agent config.

    The shim's first act is to check for the spool directory, so removing it is
    a complete off switch even while hooks remain registered. The captured lines
    are left alone unless asked otherwise: uninstalling a collector is not a
    request to delete data.
    """
    shim_path(root).unlink(missing_ok=True)
    (Path(root) / MANIFEST_NAME).unlink(missing_ok=True)
    if not keep_spool:
        shutil.rmtree(spool_dir(root), ignore_errors=True)
    else:
        d = spool_dir(root)
        if d.exists():
            (d / ".disabled").write_text(
                "The shim was removed. Lines already here are still importable;"
                " delete this directory to make the hooks inert.\n",
                encoding="utf-8",
            )


# ── the spool ────────────────────────────────────────────────────────────────


@dataclass
class SpoolLine:
    """One hook firing, as the shim wrote it."""

    agent: str
    event: str
    payload: dict[str, Any]
    #: the shim's clock, or None when it had none / could not be trusted
    shim_ts: float | None
    #: when the reader saw the line
    seen_at: float
    #: the shim's own pid — identifies one hook firing, never a session
    pid: int | None = None
    #: whoever ran the shim. Usually a shell the agent spawned per hook, so it
    #: is a grouping hint of last resort, not an identity.
    ppid: int | None = None
    #: payload dropped for size; the byte count the shim measured
    oversized: int | None = None

    @property
    def ts(self) -> float:
        return self.shim_ts if self.shim_ts else self.seen_at


@dataclass
class SpoolStats:
    """What the reader could not use. Surfaced, never swallowed — a spool that
    is quietly dropping half its lines looks exactly like a quiet session."""

    torn: int = 0
    unparsable: int = 0
    skipped_backlog: int = 0
    files: dict[str, int] = field(default_factory=dict)


class SpoolReader:
    """Follows the agent spool files, yielding whole lines as they land.

    Reads by byte offset rather than line number so a partially-flushed tail is
    simply not consumed yet: the offset only advances past a complete,
    newline-terminated line. That, plus the shim's single `printf`, is why a
    torn line is a rarity rather than the normal case — and `stats.torn` counts
    the ones that happen anyway.
    """

    def __init__(self, root: Path, *, from_start: bool = False) -> None:
        self.dir = spool_dir(root)
        self.manifest = read_manifest(root)
        self.stats = SpoolStats()
        self._offsets: dict[Path, int] = {}
        self._from_start = from_start
        if not from_start:
            for p in self._files():
                size = p.stat().st_size
                self._offsets[p] = size
                if size:
                    self.stats.skipped_backlog += 1

    @property
    def clock_resolution_s(self) -> float:
        return float(self.manifest.get("resolution_s") or float("inf"))

    def _files(self) -> list[Path]:
        if not self.dir.is_dir():
            return []
        return sorted(p for p in self.dir.glob("*.jsonl") if p.is_file())

    def poll(self) -> list[SpoolLine]:
        """Every complete line appended since the last call."""
        out: list[SpoolLine] = []
        now = time.time()
        for path in self._files():
            start = self._offsets.get(path, 0)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size < start:
                # truncated or rotated under us; start over rather than read
                # from a meaningless offset
                start = 0
            if size == start:
                continue
            with path.open("rb") as fh:
                fh.seek(start)
                buf = fh.read(size - start)
            cut = buf.rfind(b"\n")
            if cut < 0:
                continue  # no complete line yet; leave the offset where it was
            self._offsets[path] = start + cut + 1
            n = 0
            for raw in buf[: cut + 1].splitlines():
                line = self._parse(raw, now)
                if line is not None:
                    out.append(line)
                    n += 1
            self.stats.files[path.name] = self.stats.files.get(path.name, 0) + n
        return out

    def _parse(self, raw: bytes, now: float) -> SpoolLine | None:
        if not raw.strip():
            return None
        try:
            d = json.loads(raw)
        except ValueError:
            # a concurrent append landed inside ours; the bytes are unrecoverable
            self.stats.torn += 1
            return None
        if not isinstance(d, dict) or "agent" not in d:
            self.stats.unparsable += 1
            return None
        payload = d.get("payload")
        oversized = None
        if isinstance(payload, dict) and "seer_oversized" in payload:
            oversized = int(payload["seer_oversized"])
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        t = d.get("t")
        try:
            shim_ts = float(t) if t else None
        except (TypeError, ValueError):
            shim_ts = None
        if shim_ts is not None and shim_ts < 1e9:
            shim_ts = None  # a clock that returned 0 is not a time
        return SpoolLine(
            agent=str(d.get("agent") or "unknown"),
            event=str(d.get("event") or ""),
            payload=payload,
            shim_ts=shim_ts,
            seen_at=now,
            pid=int(d["pid"]) if str(d.get("pid") or "").isdigit() else None,
            ppid=int(d["ppid"]) if str(d.get("ppid") or "").isdigit() else None,
            oversized=oversized,
        )

    def follow(self, *, stop: Any = None) -> Iterator[SpoolLine]:
        """Poll forever, yielding lines as they arrive. `stop` is anything with
        an `is_set()` — a `threading.Event` in the collector, nothing in tests."""
        while stop is None or not stop.is_set():
            for line in self.poll():
                yield line
            time.sleep(POLL_INTERVAL_S)


def spool_files(root: Path) -> list[Path]:
    d = spool_dir(root)
    return sorted(d.glob("*.jsonl")) if d.is_dir() else []


def spool_size(root: Path) -> int:
    return sum(p.stat().st_size for p in spool_files(root))


def clear_spool(root: Path) -> int:
    """Delete captured spool lines. Returns bytes freed."""
    n = spool_size(root)
    for p in spool_files(root):
        p.unlink(missing_ok=True)
    return n


__all__ = [
    "CLOCKS",
    "MANIFEST_NAME",
    "MAX_PAYLOAD_BYTES",
    "POLL_INTERVAL_S",
    "SHIM_NAME",
    "Clock",
    "SpoolLine",
    "SpoolReader",
    "SpoolStats",
    "clear_spool",
    "install_shim",
    "pick_clock",
    "read_manifest",
    "remove_shim",
    "render_shim",
    "shim_path",
    "spool_dir",
    "spool_files",
    "spool_size",
]
