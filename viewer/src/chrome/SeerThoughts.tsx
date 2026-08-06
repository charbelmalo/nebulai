/** The thought rail — what the model was doing between the tool calls.
 *
 *  The chart above it is made of things the agent *did*. This is the other
 *  half: the stretches where it was thinking and nothing was happening on any
 *  lane. On a Codex run those are real spans with real wall clock, and they are
 *  frequently the largest single block of time in a session — the pale
 *  remainder that structure mode draws as `outside_spans_s` is mostly this.
 *
 *  ## Two states, and neither of them is an empty box
 *
 *  Reasoning text is `dropped_by_policy` unless the run was captured with
 *  `--keep-reasoning`, and those are different facts that must not look alike:
 *
 *  · **kept** — the words, streamed as they arrive.
 *  · **dropped** — the *activity*: when the model started thinking, how long it
 *    thought, and how many characters the adapter measured before discarding
 *    them. Drawn in policy ink, which is visibly ours rather than the agent's.
 *
 *  Building this turned up a third, which the first two had been swallowing:
 *  **the agent never carried the text**. Codex thread history records that the
 *  model reasoned and keeps none of what it said, so `--keep-reasoning` changes
 *  nothing for a reconciled run. Every one of those fragments was arriving
 *  labelled `dropped_by_policy` with `chars: 0` — a decision nobody made, and a
 *  size of zero for text that was never there. `reasoning_payload` now returns
 *  `missing` for it, and this rail draws it in missing ink rather than policy
 *  ink, because "we declined" and "there was nothing" are not the same
 *  sentence.
 *
 *  A fourth case is not a thought at all: **no reasoning events in the record**.
 *  That is not "the agent wasn't thinking" either — Claude emits a thinking
 *  block only when extended thinking is on, so its absence is a statement about
 *  the request. The rail says which of the four it is in every case, and never
 *  renders blank.
 *
 *  ## Why it reads the log back
 *
 *  `LiveModel` holds only what it ingested over SSE, so a run that finished
 *  before this tab connected has no thoughts in it. Leaving that empty would be
 *  the same lie the field hit in L4 — a historical run sitting dark beside a
 *  live one reads as a run that did nothing. So the rail asks the events
 *  endpoint for the run's log once and folds the reasoning events out of it.
 *  Once per run: that endpoint hands over everything, which is not a thing to
 *  do on a timer.
 */

import { useEffect, useRef, useState } from "preact/hooks";
import { fetchEvents, type RunView } from "../seer/client";
import type { LiveModel, Thought } from "../seer/live";
import { ABSENT_INK, stateInk } from "../seer/encoding";
import { TERMINAL_STATES, type SessionState } from "../seer/contract";

/** How many thoughts the rail shows. The model retains more; this is what fits
 *  without the card becoming the page. */
const VISIBLE = 24;

/** How often the rail re-reads the model. Thoughts arrive over SSE between the
 *  page's coalesced refetches, so a prop change is not enough to see one land —
 *  but nothing here needs a frame, either. */
const POLL_MS = 300;

interface Row extends Thought {
  label: string;
  state: SessionState;
}

export function SeerThoughts(props: { views: RunView[]; live: LiveModel }) {
  const { views, live } = props;
  const [rows, setRows] = useState<Row[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState<string[]>([]);
  const [failed, setFailed] = useState<string[]>([]);
  const sig = useRef("");

  const latest = useRef({ views, live });
  latest.current = { views, live };

  // Backfill: once per run, and only for runs currently on screen.
  useEffect(() => {
    let cancelled = false;
    for (const v of views) {
      if (live.isBackfilled(v.run_id)) continue;
      setLoading((l) => (l.includes(v.run_id) ? l : [...l, v.run_id]));
      fetchEvents(v.run_id)
        .then((r) => {
          if (!cancelled) live.ingestHistory(v.run_id, r.events);
        })
        .catch(() => {
          // A failed read is not an absence of thoughts. Say so rather than
          // letting the rail fall through to its "nothing in the record" text.
          if (!cancelled) setFailed((f) => (f.includes(v.run_id) ? f : [...f, v.run_id]));
        })
        .finally(() => {
          if (!cancelled) setLoading((l) => l.filter((id) => id !== v.run_id));
        });
    }
    return () => {
      cancelled = true;
    };
  }, [views.map((v) => v.run_id).join(","), live]);

  useEffect(() => {
    const read = () => {
      const cur = latest.current;
      const all: Row[] = [];
      for (const v of cur.views) {
        const label = v.summary?.label || v.run_id.replace(/^run_/, "");
        for (const t of cur.live.thoughts(v.run_id)) {
          all.push({ ...t, label, state: v.state });
        }
      }
      all.sort((a, b) => b.startedAt - a.startedAt);
      const shown = all.slice(0, VISIBLE);
      // Re-render only when something a reader could see has changed. The text
      // of an open thought grows, so its length is part of the signature.
      const next =
        `${all.length}|` +
        shown
          .map((t) => `${t.key}:${t.endedAt ?? 0}:${t.text?.length ?? t.chars ?? -1}`)
          .join(",");
      if (next === sig.current) return;
      sig.current = next;
      setRows(shown);
      // What the rail is not showing. A capped list that reported only its own
      // length reads as the whole of the run's thinking.
      let dropped = 0;
      for (const v of cur.views) dropped += cur.live.droppedThoughts(v.run_id);
      setTotal(all.length + dropped);
    };
    read();
    const id = window.setInterval(read, POLL_MS);
    return () => window.clearInterval(id);
  }, []);

  const kept = rows.some((t) => t.fidelity === "native");
  const dropped = rows.some((t) => t.fidelity === "dropped_by_policy");
  const absent = rows.some((t) => t.fidelity === "missing");
  const showRun = views.length > 1;

  const states = [
    kept && "text kept",
    dropped && "text dropped by policy",
    absent && "text never in the record",
  ].filter(Boolean) as string[];

  return (
    <div class="seer-card seer-thoughts">
      <h3>
        Thoughts
        <span class="seer-dim">
          {" "}
          {rows.length === 0
            ? "the model between the tool calls"
            : states.length === 1 && kept
              ? "text kept — captured with --keep-reasoning"
              : states.join(" · ")}
        </span>
        {total > rows.length && (
          <span class="seer-dim tnum">
            newest {rows.length} of {total.toLocaleString()}
          </span>
        )}
      </h3>

      {rows.length === 0 ? (
        <Empty views={views} loading={loading} failed={failed} />
      ) : (
        <ol class="seer-thought-list">
          {rows.map((t) => (
            <ThoughtRow key={`${t.runId}/${t.key}`} t={t} showRun={showRun} />
          ))}
        </ol>
      )}

      {rows.length > 0 && (
        <p class="seer-note">
          {kept && (
            <>
              The words are the model's own, kept because the run was captured
              with <code>--keep-reasoning</code>. They are the agent's account of
              itself, not a measurement — nothing on this page is derived from
              them.{" "}
            </>
          )}
          {dropped && (
            <>
              Where the size is shown, the agent sent the text and we declined to
              keep it: the number is what the adapter measured before discarding,
              and the interval is real wall clock the model spent. Capture with{" "}
              <code>--keep-reasoning</code> to see the words.{" "}
            </>
          )}
          {absent && (
            <>
              <span class="seer-warn">Where the text is marked absent</span>, the
              agent never carried it — Codex thread history records that the model
              reasoned and keeps none of what it said, so re-importing with{" "}
              <code>--keep-reasoning</code> would change nothing. The interval is
              still real; there is simply no size to state, and a{" "}
              <code>0</code> there would be a measurement of something that was
              never in the record.
            </>
          )}
        </p>
      )}
    </div>
  );
}

/** One reasoning stream. Prints a duration only when both ends were observed. */
function ThoughtRow(props: { t: Row; showRun: boolean }) {
  const t = props.t;
  const policy = t.fidelity === "dropped_by_policy";
  const absent = t.fidelity === "missing";
  // An open stream on a finished run was never closed — the run ended first.
  // "still thinking" would be a claim about now; this is a claim about capture.
  const abandoned = t.endedAt == null && TERMINAL_STATES.has(t.state);
  const seconds =
    t.endedAt != null && t.observedStart ? Math.max(0, t.endedAt - t.startedAt) : null;

  return (
    <li class={`seer-thought${policy ? " is-policy" : absent ? " is-absent" : ""}`}>
      <span class="seer-thought-meta tnum">
        {props.showRun && (
          <span class="seer-thought-run">
            <i style={{ background: stateInk(t.state) }} />
            {t.label}
          </span>
        )}
        <span class="seer-dim">{clock(t.startedAt)}</span>
        {seconds != null ? (
          <strong>{seconds < 1 ? `${seconds.toFixed(2)}s` : `${seconds.toFixed(1)}s`}</strong>
        ) : abandoned ? (
          <strong
            class="seer-warn"
            title="the run ended before this stream was closed; its end was never reported"
          >
            no end
          </strong>
        ) : t.endedAt == null ? (
          <strong class="seer-thought-live">thinking…</strong>
        ) : (
          <strong
            class="seer-dim"
            title="this agent reports a thought only once it is finished, so there is no interval to measure"
          >
            —
          </strong>
        )}
      </span>
      {absent ? (
        <span class="seer-thought-body is-absent" style={{ color: ABSENT_INK.missing }}>
          no text in the record — this capture never carried it
        </span>
      ) : policy ? (
        <span class="seer-thought-body is-policy" style={{ color: ABSENT_INK.dropped_by_policy }}>
          {t.chars == null
            ? "text dropped by policy"
            : `${t.chars.toLocaleString()} character${t.chars === 1 ? "" : "s"} dropped by policy`}
        </span>
      ) : t.text ? (
        <span class="seer-thought-body">{t.text}</span>
      ) : (
        <span class="seer-thought-body seer-dim">
          {t.text === "" ? "kept, and empty — the agent emitted a thought with no words" : "…"}
        </span>
      )}
    </li>
  );
}

/** The empty states, all three of which mean different things. */
function Empty(props: { views: RunView[]; loading: string[]; failed: string[] }) {
  if (props.views.length === 0) {
    return <p class="seer-note">Select a run to see what its model was thinking about.</p>;
  }
  if (props.loading.length > 0) {
    return <p class="seer-note">Reading the log…</p>;
  }
  if (props.failed.length > 0) {
    return (
      <p class="seer-note">
        <span class="seer-warn">The log could not be read</span>, so this rail has
        nothing to show — which is not the same as the run having thought nothing.
      </p>
    );
  }
  return (
    <p class="seer-note">
      No reasoning events in the record for {props.views.length === 1 ? "this run" : "these runs"}.
      That is a statement about the capture, not about the model: Claude emits a
      thinking block only when extended thinking is on, and an agent that never
      streams one leaves nothing here to keep or to drop.
    </p>
  );
}

function clock(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString(undefined, { hour12: false });
}
