"""The observed-mode collector: spool lines in, runs in the store out.

One `SpoolCollector` owns the mapping from *sessions the user is driving* to
*runs SessionSeer knows about*. Everything difficult here is that mapping:

* **Routing.** A hook line names its agent and (usually) its session. Two agents
  running side by side, or two Claude windows in two repos, must not fold into
  one trajectory — so the key is `(agent, session_id)` and a line with no
  session id gets one derived from the writing process, which is the narrowest
  correct grouping available.

* **Ending.** Sessions do not reliably end. A closed terminal fires no
  `SessionEnd`, so a run left silent past `idle_timeout_s` is closed as
  `interrupted` — never `completed`, which is a verdict no hook ever gives us.

* **Not inventing time.** The collector stamps nothing. Timestamps come from the
  shim's clock, and where that clock is coarse the adapters mark the derived
  durations `estimated`. The one exception is the interrupted-close event, whose
  timestamp genuinely is "when we gave up", and says so.

The collector is deliberately dumb about restarts: it starts reading at the end
of the spool, and reports the backlog it skipped rather than importing lines
whose timestamps it cannot place relative to its own clock. `import_spool`
exists for the explicit, opt-in version of that.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .adapters.observed import HOOK_ADAPTERS, HookAdapter
from .contract import CaptureMode, Event, EventType, Fidelity, Source, new_run_id
from .spool import POLL_INTERVAL_S, SpoolLine, SpoolReader
from .store import EventStore

#: A session with no hook activity for this long is closed as interrupted. Long
#: enough that a coffee break does not end a run; short enough that a closed
#: terminal does not leave a run "running" forever in the UI.
IDLE_TIMEOUT_S = 30 * 60


@dataclass
class ObservedRun:
    run_id: str
    agent: str
    session_id: str
    adapter: HookAdapter
    started_at: float
    last_seen: float
    n_events: int = 0
    cwd: str | None = None


@dataclass
class CollectorStats:
    lines: int = 0
    events: int = 0
    runs_opened: int = 0
    runs_closed_idle: int = 0
    unknown_agents: dict[str, int] = field(default_factory=dict)


class SpoolCollector:
    """Turns spool lines into store-backed runs. Thread-safe for one writer."""

    def __init__(
        self,
        store: EventStore,
        root: Path,
        *,
        from_start: bool = False,
        idle_timeout_s: float = IDLE_TIMEOUT_S,
        on_events: Callable[[list[Event]], None] | None = None,
    ) -> None:
        self.store = store
        self.root = Path(root)
        self.reader = SpoolReader(self.root, from_start=from_start)
        self.idle_timeout_s = idle_timeout_s
        self.on_events = on_events
        self.runs: dict[tuple[str, str], ObservedRun] = {}
        self.stats = CollectorStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── routing ──────────────────────────────────────────────────────────

    def _session_key(self, line: SpoolLine) -> tuple[str, str]:
        p = line.payload
        sid = p.get("session_id") or p.get("sessionId") or p.get("thread_id") or p.get("id")
        if not sid:
            # No session id. NOT `line.pid`: the shim is a new process on every
            # hook, so keying on it would open a fresh run per firing and bury
            # the real ones. The working directory is coarse — two windows in
            # one repo fold together — but coarse beats one-run-per-hook, and
            # `ppid` at least separates two agents in the same directory when
            # the hook is not run through a per-firing shell.
            sid = str(p.get("cwd") or (f"ppid-{line.ppid}" if line.ppid else "unattributed"))
        return (line.agent, str(sid))

    def _run_for(self, line: SpoolLine) -> ObservedRun | None:
        key = self._session_key(line)
        run = self.runs.get(key)
        if run is not None:
            return run
        cls = HOOK_ADAPTERS.get(line.agent)
        if cls is None:
            self.stats.unknown_agents[line.agent] = (
                self.stats.unknown_agents.get(line.agent, 0) + 1
            )
            return None
        run_id = new_run_id()
        adapter = cls(
            run_id=run_id,
            session_id=key[1],
            agent_version=str(line.payload.get("version") or "unknown"),
            capture_mode=CaptureMode.OBSERVED,
            clock_resolution_s=self.reader.clock_resolution_s,
        )
        run = ObservedRun(
            run_id=run_id,
            agent=line.agent,
            session_id=key[1],
            adapter=adapter,
            started_at=line.ts,
            last_seen=line.ts,
            cwd=line.payload.get("cwd"),
        )
        self.runs[key] = run
        self.stats.runs_opened += 1
        self.store.register_run(
            run_id,
            agent=line.agent,
            capture_mode=CaptureMode.OBSERVED.value,
            label=_label_for(run),
            started_at=line.ts,
        )
        # `register_run` opens a row as `starting`, which is what a run we
        # launched is doing between the id being issued and the process saying
        # anything. An observed run has already said something — that is how we
        # learned it exists — so it is running, and saying otherwise would put
        # every live session in the list under the wrong state.
        self.store.set_state(run_id, "running")
        self._emit(run, [_run_started(run, line)])
        return run

    # ── ingest ───────────────────────────────────────────────────────────

    def feed(self, line: SpoolLine) -> int:
        """One spool line. Returns how many canonical events it produced."""
        self.stats.lines += 1
        run = self._run_for(line)
        if run is None:
            return 0
        events = run.adapter.feed_hook(line)
        run.last_seen = max(run.last_seen, line.ts)
        self._emit(run, events)
        if run.adapter.ended:
            self.runs.pop(self._session_key(line), None)
            self.store.set_state(run.run_id, "completed", _label_for(run))
        return len(events)

    def poll(self) -> int:
        """Drain the spool once and close anything that has gone quiet."""
        n = 0
        for line in self.reader.poll():
            n += self.feed(line)
        n += self.reap(time.time())
        return n

    def reap(self, now: float) -> int:
        """Close runs that have been silent past the idle timeout."""
        n = 0
        for key, run in list(self.runs.items()):
            if now - run.last_seen < self.idle_timeout_s:
                continue
            events = run.adapter.close(
                run.last_seen,
                reason=f"no hook activity for {self.idle_timeout_s:.0f}s",
            )
            self._emit(run, events)
            self.runs.pop(key, None)
            self.stats.runs_closed_idle += 1
            self.store.set_state(run.run_id, "interrupted", _label_for(run))
            n += len(events)
        return n

    def close_all(self, reason: str = "collector stopped") -> int:
        """End every open run. Called on shutdown so nothing stays `running`
        in the UI after the process that was watching it is gone."""
        n = 0
        for key, run in list(self.runs.items()):
            # the last hook we saw, not now: "when we stopped watching" is not
            # when the session stopped, and only one of those is a fact
            events = run.adapter.close(run.last_seen, reason=reason)
            self._emit(run, events)
            self.runs.pop(key, None)
            self.store.set_state(run.run_id, "interrupted", _label_for(run))
            n += len(events)
        return n

    def _emit(self, run: ObservedRun, events: list[Event]) -> None:
        if not events:
            return
        self.store.append_many(events)
        run.n_events += len(events)
        self.stats.events += len(events)
        if self.on_events:
            self.on_events(events)

    # ── background loop ──────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="seer-spool", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:  # noqa: BLE001
                # A collector that dies takes observed capture down silently.
                # Losing one poll is recoverable; the next one re-reads from the
                # same offset.
                pass
            self._stop.wait(POLL_INTERVAL_S)

    def stop(self, *, close_runs: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if close_runs:
            self.close_all()

    def status(self) -> dict[str, Any]:
        res = self.reader.clock_resolution_s
        return {
            "watching": self.reader.dir.is_dir(),
            "spool_dir": str(self.reader.dir),
            # `inf` internally means "no manifest, so no clock we can trust",
            # and over the wire that is `null` — not because JSON cannot hold
            # infinity (it cannot), but because the fact is that we do not know.
            "clock_resolution_s": res if math.isfinite(res) else None,
            "open_runs": [
                {
                    "run_id": r.run_id,
                    "agent": r.agent,
                    "session_id": r.session_id,
                    "n_events": r.n_events,
                    "idle_s": round(time.time() - r.last_seen, 1),
                }
                for r in self.runs.values()
            ],
            "lines": self.stats.lines,
            "events": self.stats.events,
            "runs_opened": self.stats.runs_opened,
            "runs_closed_idle": self.stats.runs_closed_idle,
            "unknown_agents": dict(self.stats.unknown_agents),
            "spool_torn": self.reader.stats.torn,
            "spool_unparsable": self.reader.stats.unparsable,
            "spool_backlog_files_skipped": self.reader.stats.skipped_backlog,
        }


def _label_for(run: ObservedRun) -> str:
    where = Path(run.cwd).name if run.cwd else run.session_id[:8]
    return f"{where} (observed)"


def _run_started(run: ObservedRun, line: SpoolLine) -> Event:
    """Our own bookkeeping event, marked as ours.

    `DETERMINISTIC` and not `NATIVE`: the agent did not tell us a run began, we
    decided one had.
    """
    return Event(
        event_type=EventType.RUN_STARTED,
        source=Source(
            agent=run.agent,
            agent_version=run.adapter.agent_version,
            adapter=run.adapter.adapter_name,
            adapter_version="0.1.0",
            capture_mode=CaptureMode.OBSERVED,
            fidelity=Fidelity.DETERMINISTIC,
        ),
        run_id=run.run_id,
        session_id=run.session_id,
        ts=line.ts,
        native_type="collector.open",
        payload={"cwd": run.cwd, "first_hook": line.event},
    )


def import_spool(store: EventStore, root: Path, *, idle_timeout_s: float = 60.0) -> dict:
    """Import the whole spool from the beginning, once, and return a summary.

    The opposite trade from the live collector: this reads lines whose
    timestamps come entirely from the shim, so it is only as good as the shim's
    clock — with a whole-second clock every span in a session collapses. The
    adapters already mark those `estimated`; this exists so a session captured
    while nothing was watching is recoverable rather than lost.
    """
    c = SpoolCollector(store, root, from_start=True, idle_timeout_s=idle_timeout_s)
    for line in c.reader.poll():
        c.feed(line)
    c.close_all(reason="imported from spool after the fact")
    return c.status()


__all__ = [
    "IDLE_TIMEOUT_S",
    "CollectorStats",
    "ObservedRun",
    "SpoolCollector",
    "import_spool",
]
