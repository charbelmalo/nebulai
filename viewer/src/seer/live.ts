/** live.ts — the leading edge of a run, and nothing else.
 *
 *  Every projection in the live view (lanes, fleet strip, span tree, particle
 *  field) reads this model, and none of them parse SSE. That is what keeps the
 *  views from disagreeing with each other.
 *
 *  ## What this is not
 *
 *  It is not a reducer. `SeerPage` already refuses to fold — "the Python
 *  reducer owns every derived number… a second implementation of the fold in TS
 *  would drift from the first, and the drift would be invisible" — and drawing
 *  a moving picture is not an exemption from that. `reducer._state` makes the
 *  point sharper: the state machine runs off a `_TRANSITIONS` table and refuses
 *  to read `payload["state"]`, because an agent's own words deciding our states
 *  is the one thing the contract forbids. Any TS copy of that table would rot.
 *
 *  So the split is:
 *
 *  · **Python owns the record.** Totals, closed spans, the current state, time
 *    in state. It arrives through the 200ms coalesced refetch the page already
 *    performs, and lands here through `adopt()`.
 *  · **This owns the leading edge.** How far the open spans have grown since
 *    that snapshot, the marks inside the visible window, and which spans are
 *    actively producing output right now. Positions, not figures.
 *
 *  `figures()` hands back the adopted `RunView` untouched. There is no code
 *  path in this file that writes to it, which is the property the test replays
 *  a thousand events to assert.
 *
 *  ## Why a delta contributes only a timestamp
 *
 *  It is tempting to accumulate `payload.chars` off the streaming deltas and
 *  call it a live output size. It cannot be done honestly: `claude.py` puts the
 *  *fragment* length in `chars` while `codex.py` puts the *cumulative* item
 *  length in the same key, so summing is wrong for one agent and taking the
 *  latest is wrong for the other. There is no third option that is right for
 *  both, which is the contract's "deltas never feed counters" rule showing up
 *  as an outright impossibility rather than a policy. A delta therefore moves
 *  `producingUntil` and does nothing else — an activity pulse, no magnitude,
 *  identical in meaning across all three agents.
 */

import {
  isDelta,
  type Action,
  type Effect,
  type Fidelity,
  type SeerEvent,
} from "./contract";
import type { RunView } from "./client";

/** One event, placed. A direct projection of fields already on the event —
 *  never an accumulation, which is what keeps marks on the safe side of the
 *  fold line. */
export interface Mark {
  runId: string;
  eventId: string;
  ts: number;
  eventType: string;
  action: Action | null;
  effect: Effect | null;
  fidelity: Fidelity;
  spanId: string | null;
  /** A reasoning fragment. `payload.kind` is ours, set by our own adapters —
   *  unlike `native_type`, which nothing here may read. */
  reasoning: boolean;
}

/** A span we have seen begin and not yet seen end.
 *
 *  Its extent is *not* a duration. Whoever draws it must cap it live and must
 *  not let the eye compare it against a closed span's measured length — see
 *  `isProvisional` in encoding.ts. */
export interface OpenSpan {
  runId: string;
  spanId: string;
  action: Action | null;
  startedAt: number;
  reasoning: boolean;
  /** Last time a streaming delta arrived for this span, or null if none has.
   *  A timestamp, deliberately not a count — see the module note. */
  producingUntil: number | null;
}

export interface LiveModelOptions {
  /** Seconds of marks to retain, measured back from the newest mark seen. */
  windowS?: number;
  /** Hard ceiling on retained marks per run, so a pathological burst cannot
   *  grow the page's memory without bound. Trimming is reported by
   *  `droppedMarks` rather than done silently. */
  maxMarks?: number;
}

const DEFAULT_WINDOW_S = 120;
const DEFAULT_MAX_MARKS = 4000;

/** Event types that open and close a span. Spelled out rather than pattern
 *  matched on `.started` / `.completed`: `run.started` and `session.started`
 *  end in the same word and are not spans. */
const OPENS = new Set(["tool.started", "subagent.started"]);
const CLOSES = new Set(["tool.completed", "tool.failed", "subagent.completed"]);

interface RunLive {
  marks: Mark[];
  seen: Set<string>;
  open: Map<string, OpenSpan>;
  view: RunView | null;
  lastEventAt: number | null;
  droppedMarks: number;
}

function emptyRun(): RunLive {
  return {
    marks: [],
    seen: new Set(),
    open: new Map(),
    view: null,
    lastEventAt: null,
    droppedMarks: 0,
  };
}

function isReasoning(e: SeerEvent): boolean {
  return (e.payload as { kind?: unknown } | undefined)?.kind === "reasoning";
}

export class LiveModel {
  private readonly runs = new Map<string, RunLive>();
  private readonly windowS: number;
  private readonly maxMarks: number;

  constructor(opts: LiveModelOptions = {}) {
    this.windowS = opts.windowS ?? DEFAULT_WINDOW_S;
    this.maxMarks = opts.maxMarks ?? DEFAULT_MAX_MARKS;
  }

  /** Install the authoritative snapshot for a run. Nothing in this file ever
   *  writes to it; `figures()` hands back exactly this object. */
  adopt(runId: string, view: RunView): void {
    this.run(runId).view = view;
  }

  /** The record, as Python computed it. `null` until the first `adopt`, which
   *  is honest: a run whose events we have seen but whose view we have not
   *  fetched has no figures we are entitled to state. */
  figures(runId: string): RunView | null {
    return this.runs.get(runId)?.view ?? null;
  }

  /** Fold a live event into the leading edge.
   *
   *  Ignores events it has already seen: a reconnect replays, and a particle
   *  drawn twice for one action would misrepresent how busy the run was. */
  ingest(e: SeerEvent): void {
    const r = this.run(e.run_id);
    if (r.seen.has(e.event_id)) return;
    r.seen.add(e.event_id);
    r.lastEventAt = r.lastEventAt == null ? e.ts : Math.max(r.lastEventAt, e.ts);

    if (isDelta(e.event_type)) {
      // A delta animates and never counts. It moves the span's activity pulse
      // and produces no mark, so nothing downstream can tally it by accident.
      const span = e.span_id ? r.open.get(e.span_id) : undefined;
      if (span) span.producingUntil = Math.max(span.producingUntil ?? e.ts, e.ts);
      return;
    }

    if (e.span_id && OPENS.has(e.event_type)) {
      // Re-opening an already-open span keeps the original start: the first
      // beginning is the one the bar should grow from.
      if (!r.open.has(e.span_id)) {
        r.open.set(e.span_id, {
          runId: e.run_id,
          spanId: e.span_id,
          action: e.action ?? null,
          startedAt: e.ts,
          reasoning: isReasoning(e),
          producingUntil: null,
        });
      }
    } else if (e.span_id && CLOSES.has(e.event_type)) {
      r.open.delete(e.span_id);
    }

    // Older than the window we are keeping. Its span bookkeeping above still
    // ran — a late `tool.completed` must close its span whenever it lands — but
    // appending the mark would put the list out of order, and `trim` reads the
    // newest timestamp off the end. Dedup is pruned with the window, so this is
    // also what stops a replayed stale event from cycling back in.
    if (r.marks.length > 0 && e.ts < r.marks[r.marks.length - 1]!.ts - this.windowS) {
      r.droppedMarks++;
      return;
    }

    r.marks.push({
      runId: e.run_id,
      eventId: e.event_id,
      ts: e.ts,
      eventType: e.event_type,
      action: e.action ?? null,
      effect: e.effect ?? null,
      fidelity: e.source.fidelity,
      spanId: e.span_id ?? null,
      reasoning: isReasoning(e),
    });
    this.trim(r);
  }

  /** Every run the model has heard anything about, in first-seen order. */
  runIds(): string[] {
    return [...this.runs.keys()];
  }

  /** Marks retained for a run, oldest first. Bounded by the window and the
   *  mark ceiling — this is the visible tail, not the run's history. */
  marks(runId: string): readonly Mark[] {
    return this.runs.get(runId)?.marks ?? [];
  }

  openSpans(runId: string): readonly OpenSpan[] {
    const r = this.runs.get(runId);
    return r ? [...r.open.values()] : [];
  }

  /** The newest event timestamp seen on the stream, or null. Read straight off
   *  an event; not accumulated, and not a duration. */
  lastEventAt(runId: string): number | null {
    return this.runs.get(runId)?.lastEventAt ?? null;
  }

  /** How many marks fell out of the window or off the ceiling. A view that
   *  bounds what it shows should say so rather than imply it drew everything. */
  droppedMarks(runId: string): number {
    return this.runs.get(runId)?.droppedMarks ?? 0;
  }

  /** A run ended: nothing it had open is open any more.
   *
   *  The marks stay. A run that finishes two seconds after you looked at it
   *  should not have its last two seconds blink out of the picture. */
  finish(runId: string): void {
    const r = this.runs.get(runId);
    if (r) r.open.clear();
  }

  /** Drop every trace of a run, for a delete arriving from any window. */
  forget(runId: string): void {
    this.runs.delete(runId);
  }

  private run(runId: string): RunLive {
    let r = this.runs.get(runId);
    if (!r) {
      r = emptyRun();
      this.runs.set(runId, r);
    }
    return r;
  }

  /** Trim to the window, then to the ceiling.
   *
   *  The window is measured from the newest mark rather than from the wall
   *  clock: a reconciled import lands with month-old timestamps all at once,
   *  and trimming that against `now` would throw away every mark it just
   *  received. */
  private trim(r: RunLive): void {
    const newest = r.marks[r.marks.length - 1]!.ts;
    const floor = newest - this.windowS;
    let cut = 0;
    while (cut < r.marks.length && r.marks[cut]!.ts < floor) cut++;
    if (r.marks.length - cut > this.maxMarks) cut = r.marks.length - this.maxMarks;
    if (cut === 0) return;
    for (let i = 0; i < cut; i++) r.seen.delete(r.marks[i]!.eventId);
    r.marks.splice(0, cut);
    r.droppedMarks += cut;
  }
}
