"""Close runs whose capture process died without saying so.

A clean shutdown ends every open run: the runner writes its verdict, the
collector's `close_all` writes an interruption per observed session. A crash
writes nothing. What is left is a row in `running` with no terminal event and
no process behind it, and from the outside that is indistinguishable from a run
that is going fine — the viewer draws it as live, `seer list` reports it as
live, and it stays that way forever.

The fix has to be written into the **log**, not just the index, or the next
`reindex` puts the run back in `running`: the index is a cache and the log is
the record, so a recovery that only touches SQLite is a recovery that undoes
itself.

Two rules shape the event it writes:

* **It is our observation, not the agent's.** The source says `seer_recovery`,
  not the adapter that was capturing, and the fidelity is `deterministic`: we
  did not watch the session end, we deduced from a dead pid that we stopped
  watching it. Attributing that to the agent would put words in its mouth.
* **Interrupted, never completed or failed.** We do not know what the agent
  did after we stopped looking. It may have finished the task perfectly. The
  only thing we can state is that our record of it stops here.
"""

from __future__ import annotations

import time
from typing import Any

from .contract import CaptureMode, Event, EventType, Fidelity, Outcome, Source
from .store import EventStore

ADAPTER = "seer_recovery"
ADAPTER_VERSION = "0.1.0"


def recover_orphans(store: EventStore, *, reason: str | None = None) -> list[dict[str, Any]]:
    """End every run left mid-capture by a process that is gone.

    Called when a store is opened for serving. Returns one dict per run
    recovered, so the caller can say what it did rather than silently rewriting
    history — a run that changed state while nobody was looking is exactly the
    kind of thing that should appear in a log line.
    """
    recovered: list[dict[str, Any]] = []
    for run in store.orphans():
        last = store.tail(run.run_id, n=1)
        # The last thing we actually saw, not `now`. The gap between them is
        # however long the machine was off, and dating the interruption from
        # the clock would stretch every crashed run's wall time by it.
        ts = last[0].ts if last else (run.started_at or time.time())
        ev = Event(
            event_type=EventType.SESSION_INTERRUPTED,
            source=Source(
                agent=run.agent,
                agent_version=run.agent_version or "unknown",
                adapter=ADAPTER,
                adapter_version=ADAPTER_VERSION,
                capture_mode=CaptureMode(run.capture_mode or "driven"),
                fidelity=Fidelity.DETERMINISTIC,
            ),
            run_id=run.run_id,
            session_id=run.native_session_id or run.run_id,
            ts=ts,
            payload={
                "outcome": Outcome.INTERRUPTED.value,
                "note": reason
                or "the process capturing this run ended without recording an "
                "end; what the agent did next was not observed",
                "recovered": True,
                "n_events": run.n_events,
            },
        )
        store.append(ev)
        store.set_state(run.run_id, "interrupted", run.label)
        recovered.append(
            {"run_id": run.run_id, "agent": run.agent, "n_events": run.n_events,
             "was": run.state, "ts": ts}
        )
    return recovered
