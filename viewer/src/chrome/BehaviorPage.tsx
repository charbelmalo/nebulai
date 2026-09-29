/** Behavior — the association study of docs/BEHAVIORAL-DIVERGENCE-PLAN.md §8.
 *
 *  Every other page in this instrument reads WEIGHTS. This one reads what
 *  specific pinned deployments actually said when asked the same cue many
 *  times. Those are different kinds of evidence and the page is built so a
 *  reader cannot accidentally carry a conclusion across:
 *
 *  - The landscape is **PCA of the cue WORDS**, never of any model's output.
 *    Two cues sit near each other because a pinned neutral encoder puts their
 *    text near each other. The caption saying so is read from
 *    `landscape.projection.quantity_label`, which the exporter computed from
 *    the fit — this file never writes its own version of that sentence, so it
 *    cannot drift when the cue set grows.
 *  - Effect is encoded by **area**, via `cueMarkRadius`. Radius encoding would
 *    over-state every larger effect by its square.
 *  - `missing` is never drawn as zero. A cue with no computed effect gets the
 *    hollow `indeterminate` mark and its reason string, and the Settings row
 *    that hides those is worded as a declutter so nobody reads the default as
 *    a claim.
 *  - No model is ranked anywhere, including in sort order: the ranked view
 *    sorts CUES by effect size, which is a property of the comparison, not of
 *    either arm.
 *
 *  The page has five edge states and all five are real code paths, not TODOs:
 *  loading, no-artifact (the normal case for a static deploy), error, partial
 *  (a study where an arm never ran — the xAI arm has no credentials in this
 *  environment and is recorded, not hidden), and sensitive content held behind
 *  an explicit reveal.
 *
 *  Knobs live in Settings → Behavior (`SettingsPage.tsx`), per the Settings-home
 *  rule; this file reads them from the store and never keeps its own copy. */

import { useEffect, useState } from "preact/hooks";
import { appStore, type BehaviorFilter, type BehaviorView } from "../app/store";
import {
  cueMarkRadius,
  coverageNote,
  cueSignificant,
  fmtMetric,
  loadBehavior,
  maxMeasuredEffect,
  notRunArms,
  isExampleOnly,
  searchCues,
  type BehaviorCue,
  type BehaviorData,
} from "../data/behavior";
import { $behavior } from "./state";

type Load =
  | { kind: "loading" }
  | { kind: "empty" }
  | { kind: "error"; message: string }
  | { kind: "ready"; data: BehaviorData };

const VIEWS: { id: BehaviorView; label: string }[] = [
  { id: "landscape", label: "Landscape" },
  { id: "ranked", label: "Ranked" },
  { id: "table", label: "Table" },
];

const FILTERS: { id: BehaviorFilter; label: string; title: string }[] = [
  { id: "all", label: "All cues", title: "every cue, including gated and not-measured" },
  { id: "measured", label: "Measured", title: "cues with a computed effect" },
  {
    id: "significant",
    label: "Passing BY",
    title: "cues whose q is at or below the study's threshold",
  },
];

// ---------------------------------------------------------------------------

export function BehaviorPage() {
  const [load, setLoad] = useState<Load>({ kind: "loading" });
  const ui = $behavior.value;

  useEffect(() => {
    let live = true;
    loadBehavior()
      .then((d) => {
        if (!live) return;
        setLoad(d ? { kind: "ready", data: d } : { kind: "empty" });
      })
      .catch((e: unknown) =>
        live ? setLoad({ kind: "error", message: String(e) }) : undefined,
      );
    return () => {
      live = false;
    };
  }, []);

  return (
    <div class="behavior-page" role="main">
      <div class="behavior-scroll">
        {load.kind === "loading" && <LoadingState />}
        {load.kind === "empty" && <EmptyState />}
        {load.kind === "error" && <ErrorState message={load.message} />}
        {load.kind === "ready" && <Study data={load.data} ui={ui} />}
      </div>
    </div>
  );
}

// ── edge states ────────────────────────────────────────────────────────────

function LoadingState() {
  return (
    <div class="behavior-state" aria-busy="true">
      <p class="behavior-kicker">Behavioral divergence</p>
      <h1 class="behavior-title">Loading the study…</h1>
      <p class="behavior-lede">Reading out/behavior/behavior.json.</p>
    </div>
  );
}

/** The normal state for a static deploy, and the reason it is a full panel
 *  rather than a spinner that never resolves: no study has been exported, and
 *  the reader should learn what would produce one rather than watch a blank
 *  page. */
function EmptyState() {
  return (
    <div class="behavior-state">
      <p class="behavior-kicker">Behavioral divergence</p>
      <h1 class="behavior-title">No study is published here</h1>
      <p class="behavior-lede">
        This page reads <code>out/behavior/behavior.json</code>, produced by{" "}
        <code>nebulai behavior run</code>, then <code>nebulai behavior analyze</code>,
        then <code>nebulai behavior publish &lt;study-id&gt;</code>. Nothing is fetched
        from a model when you open this page, and no study ships with the viewer by
        default.
      </p>
      <p class="behavior-note">
        The study compares the association distributions of specific pinned model
        deployments under one frozen protocol. It describes no model&rsquo;s internals
        and ranks no model. The method, the statistics and the claim contract are in{" "}
        <code>docs/BEHAVIORAL-DIVERGENCE-PLAN.md</code>.
      </p>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div class="behavior-state behavior-state-error" role="alert">
      <p class="behavior-kicker">Behavioral divergence</p>
      <h1 class="behavior-title">The study could not be read</h1>
      <p class="behavior-lede">
        The file exists but did not parse. Nothing below is shown, because a
        partially-read study would be indistinguishable from a study with missing
        measurements — and those mean different things.
      </p>
      <pre class="behavior-error-detail">{message}</pre>
    </div>
  );
}

// ── the study ──────────────────────────────────────────────────────────────

function Study({ data, ui }: { data: BehaviorData; ui: ReturnType<typeof uiType> }) {
  const st = appStore.getState();
  const notRun = notRunArms(data);
  const coverage = coverageNote(data);
  const searched = searchCues(data.cues, ui.query);
  const qThreshold = data.manifest.statistics.q_threshold;

  const visible = searched.filter((c) => {
    if (ui.filter === "measured") return c.status === "measured";
    if (ui.filter === "significant") return cueSignificant(c, qThreshold) === true;
    if (!ui.showIndeterminate && cueSignificant(c, qThreshold) === null) return false;
    return true;
  });

  const open = ui.cue ? (data.cues.find((c) => c.cue === ui.cue) ?? null) : null;

  return (
    <>
      <header class="behavior-head">
        <p class="behavior-kicker">Behavioral divergence · {data.study_id}</p>
        <h1 class="behavior-title">What these deployments associated</h1>
        <p class="behavior-lede">{data.claim}</p>

        {isExampleOnly(data) && <ExampleBanner data={data} />}

        {notRun.length > 0 && <NotRunBanner arms={notRun} />}

        {coverage !== null && <CoverageBanner note={coverage} />}

        <dl class="behavior-facts">
          <Fact k="Cues" v={`${data.cues.length}`} />
          <Fact k="Trials per cue" v={`${data.manifest.trials_per_cue}`} />
          <Fact
            k="Arms"
            v={data.manifest.models.map((m) => m.model_id).join(" · ")}
            title="the exact pinned deployments; never a family name"
          />
          <Fact
            k="p floor"
            v={data.manifest.p_floor === null ? "not measured" : data.manifest.p_floor.toExponential(2)}
            title="the smallest p this design can produce, set by the number of distinct permutations"
          />
          <Fact k="Multiple testing" v={`${data.manifest.statistics.multiple_testing} at q ≤ ${qThreshold}`} />
          <Fact
            k="Grok reasoning tokens"
            v={
              data.manifest.reasoning_tokens_p95 === null
                ? "not measured"
                : `${data.manifest.reasoning_tokens_p95} (p95)`
            }
            title="null means never observed — it is not zero"
          />
        </dl>
      </header>

      <div class="behavior-toolbar" role="toolbar" aria-label="Cue views and filters">
        <label class="behavior-search">
          <span class="sr-only">Search cues</span>
          <input
            type="search"
            placeholder="Search cues, strata, associates…"
            value={ui.query}
            onInput={(e) => st.setBehaviorQuery((e.target as HTMLInputElement).value)}
          />
        </label>

        <div class="behavior-seg" role="tablist">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              role="tab"
              aria-selected={ui.view === v.id}
              class={ui.view === v.id ? "is-on" : ""}
              onClick={() => st.setBehaviorView(v.id)}
            >
              {v.label}
            </button>
          ))}
        </div>

        <div class="behavior-seg" role="group" aria-label="Filter">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              title={f.title}
              aria-pressed={ui.filter === f.id}
              class={ui.filter === f.id ? "is-on" : ""}
              onClick={() => st.setBehaviorFilter(f.id)}
            >
              {f.label}
            </button>
          ))}
        </div>

        <button
          class="behavior-link-btn"
          onClick={() => st.setSettingsOpen(true)}
          title="Every knob for this page lives in Settings → Behavior"
        >
          Behavior settings…
        </button>
      </div>

      <p class="behavior-count" aria-live="polite">
        {visible.length} of {data.cues.length} cues shown
        {ui.query ? ` for “${ui.query}”` : ""}.{" "}
        <StatusTally cues={data.cues} />
      </p>

      <div class="behavior-body">
        <section class="behavior-main">
          {ui.view === "landscape" && (
            <Landscape data={data} cues={visible} qThreshold={qThreshold} openCue={ui.cue} />
          )}
          {ui.view === "ranked" && <Ranked cues={visible} qThreshold={qThreshold} openCue={ui.cue} />}
          {ui.view === "table" && <Table cues={visible} qThreshold={qThreshold} openCue={ui.cue} />}
        </section>

        <aside class="behavior-inspector" aria-label="Cue inspector">
          {open ? (
            <CueInspector cue={open} data={data} ui={ui} />
          ) : (
            <p class="behavior-note">
              Pick a cue to see its effect, its interval, its per-arm profile and the
              reason for anything that was not measured.
            </p>
          )}
        </aside>
      </div>

      <Evidence data={data} />
      <Runs data={data} />
      <CrossLinks />
    </>
  );
}

function uiType() {
  return appStore.getState().behavior;
}

function Fact({ k, v, title }: { k: string; v: string; title?: string }) {
  return (
    <div class="behavior-fact" title={title}>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </div>
  );
}

/** A study whose source cannot support a claim says so at the top, in words.
 *
 *  Without this the page is at its most misleading exactly when it is least
 *  informative: every status reads "no detected deviation" or "insufficient
 *  evidence", which looks like a careful negative result about two real
 *  models, when in fact no model was involved — the arms were synthetic, or
 *  the encoder was the hash stand-in that is not a semantic space at all. The
 *  downgrade already happened upstream; this is the sentence that explains it.
 */
function ExampleBanner({ data }: { data: BehaviorData }) {
  const why =
    data.published?.published_as === "example"
      ? "It was published with --force so the page has something to render."
      : "Its arms or its encoder cannot support a claim about any model.";
  return (
    <div class="behavior-banner is-example" role="note">
      <strong>This is an example, not evidence.</strong> {why} No cue here says
      anything about any model, including the ones named below: nothing in this
      artifact is a measurement of a deployment, and no status in it can reach{" "}
      <em>confirmed</em>.
    </div>
  );
}

/** Cues that were never collected are the other way a study can be partial,
 *  and the less visible one: a missing ARM is named in the manifest and shows
 *  up as an empty column, while a missing CUE leaves nothing behind at all. The
 *  cue count in the facts list is the count that ran, so without this line it
 *  reads as the study's full size. */
function CoverageBanner({ note }: { note: string }) {
  return (
    <div class="behavior-banner" role="note">
      <strong>This study does not cover its whole cue list.</strong> {note} The
      cues that did run are not weakened by the ones that did not — each was
      collected at its full repeat count and full block balance, so every per-cue
      effect, permutation p and corrected q below is what it would have been in
      the complete study. What is reduced is coverage, and the family-wise
      correction spans only the cues listed here.
    </div>
  );
}

/** An arm that never ran is a fact about the study, not a gap to paper over.
 *  The whole refusal text is printed — for the xAI arm that text is the
 *  estimate-and-approve message, and truncating it would drop the part a
 *  reader needs in order to act. */
function NotRunBanner({ arms }: { arms: { key: string; reason: string }[] }) {
  return (
    <div class="behavior-banner" role="note">
      <strong>This study is partial.</strong>{" "}
      {arms.length === 1 ? "One arm" : `${arms.length} arms`} never ran. Nothing was
      substituted for {arms.length === 1 ? "it" : "them"} and no row was invented.
      <ul>
        {arms.map((a) => (
          <li key={a.key}>
            <code>{a.key}</code>
            <span class="behavior-banner-why">{a.reason}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function StatusTally({ cues }: { cues: BehaviorCue[] }) {
  let measured = 0;
  let gated = 0;
  let missing = 0;
  for (const c of cues) {
    if (c.status === "measured") measured++;
    else if (c.status === "gated") gated++;
    else missing++;
  }
  return (
    <span class="behavior-tally">
      <span class="is-measured">{measured} measured</span> ·{" "}
      <span class="is-gated">{gated} gated</span> ·{" "}
      <span class="is-missing">{missing} not measured</span>
    </span>
  );
}

// ── landscape ──────────────────────────────────────────────────────────────

const W = 720;
const H = 460;
const PAD = 34;

function Landscape({
  data,
  cues,
  qThreshold,
  openCue,
}: {
  data: BehaviorData;
  cues: BehaviorCue[];
  qThreshold: number;
  openCue: string;
}) {
  const st = appStore.getState();
  const l = data.landscape;
  const idx = data.cue_index;
  const maxEffect = maxMeasuredEffect(data.cues);

  /*  Two different empty plots, and a reader has to be able to tell them
   *  apart. `status: "missing"` is the STUDY having no landscape at all — too
   *  few comparable cues to fit axes through — and no choice of filter will
   *  produce one. An empty `pts` with a fitted landscape is the filter. */
  if (l.status === "missing") {
    return (
      <p class="behavior-note">
        <strong>No landscape was fitted for this study.</strong>{" "}
        {l.reason ?? `A ${l.dims}-axis projection needs more than ${l.dims} comparable cues.`}{" "}
        The ranked and table views show every cue that was collected.
      </p>
    );
  }

  const pts = cues
    .map((c) => ({ c, xy: l.coords[idx[c.cue] ?? -1] }))
    .filter((p): p is { c: BehaviorCue; xy: number[] } => Array.isArray(p.xy));

  if (pts.length === 0) {
    return <p class="behavior-note">No cue in the current filter has a landscape position.</p>;
  }

  const xs = pts.map((p) => p.xy[0]!);
  const ys = pts.map((p) => p.xy[1]!);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  const sx = (v: number) => PAD + ((v - x0) / (x1 - x0 || 1)) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - ((v - y0) / (y1 - y0 || 1)) * (H - 2 * PAD);

  /*  Declutter: label only the largest measured effects and the open cue.
   *  Labelling every cue at n≈300 produces an unreadable mat, and picking the
   *  labels by effect is the one ordering that does not hide the marks a
   *  reader came for. */
  const labelled = new Set(
    [...pts]
      .filter((p) => p.c.status === "measured" && p.c.delta_hat !== null)
      .sort((a, b) => Math.abs(b.c.delta_hat!) - Math.abs(a.c.delta_hat!))
      .slice(0, 12)
      .map((p) => p.c.cue),
  );
  if (openCue) labelled.add(openCue);

  return (
    <figure class="behavior-landscape">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Cue landscape">
        <rect x="0" y="0" width={W} height={H} class="behavior-plot-bg" />
        {pts.map(({ c, xy }) => {
          const r = cueMarkRadius(c, maxEffect);
          const sig = cueSignificant(c, qThreshold);
          const cx = sx(xy[0]!);
          const cy = sy(xy[1]!);
          const isOpen = c.cue === openCue;
          const cls = [
            "behavior-mark",
            c.status === "measured" ? "is-measured" : c.status === "gated" ? "is-gated" : "is-missing",
            sig === true ? "is-sig" : sig === null ? "is-indet" : "is-nonsig",
            isOpen ? "is-open" : "",
          ]
            .filter(Boolean)
            .join(" ");
          return (
            <g key={c.cue} class={cls} onClick={() => st.setBehaviorCue(c.cue)}>
              {r === null ? (
                /*  Indeterminate: a hollow ring at fixed size. Never a dot of
                 *  radius zero — that reads as "no effect" when the truth is
                 *  "no measurement". */
                <circle cx={cx} cy={cy} r={4} class="behavior-mark-indet" />
              ) : (
                <circle cx={cx} cy={cy} r={r} />
              )}
              <title>
                {c.cue} — {c.status === "measured" ? `Δ̂ ${fmtMetric(c.delta_hat)}` : c.status}
              </title>
              {labelled.has(c.cue) && (
                <text x={cx + (r ?? 4) + 4} y={cy + 3.5}>
                  {c.cue}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <figcaption class="behavior-caption">
        <p>
          <strong>Position is lexical, not behavioural.</strong> {l.projection.warning}.{" "}
          {l.projection.quantity_label}.
        </p>
        <p class="behavior-caption-sub">
          Encoder <code>{l.projection.encoder}</code>
          {l.projection.encoder_revision ? ` @${l.projection.encoder_revision.slice(0, 8)}` : ""}.
          {l.trustworthiness === null || l.trustworthiness === undefined
            ? " Trustworthiness not measured (too few cues)."
            : ` Trustworthiness ${l.trustworthiness.toFixed(3)} at k=10.`}{" "}
          Mark <em>area</em> is proportional to |Δ̂|; hollow rings are cues whose
          significance is indeterminate.
        </p>
      </figcaption>
    </figure>
  );
}

// ── ranked + table ─────────────────────────────────────────────────────────

function sortCues(cues: BehaviorCue[], key: string): BehaviorCue[] {
  const out = [...cues];
  const eff = (c: BehaviorCue) => (c.delta_hat === null ? -Infinity : Math.abs(c.delta_hat));
  if (key === "alpha") out.sort((a, b) => a.cue.localeCompare(b.cue));
  else if (key === "q") out.sort((a, b) => (a.q_value ?? Infinity) - (b.q_value ?? Infinity));
  else out.sort((a, b) => eff(b) - eff(a));
  return out;
}

function Ranked({
  cues,
  qThreshold,
  openCue,
}: {
  cues: BehaviorCue[];
  qThreshold: number;
  openCue: string;
}) {
  const st = appStore.getState();
  const ui = $behavior.value;
  const rows = sortCues(cues, ui.sort);
  const maxEffect = maxMeasuredEffect(cues);
  return (
    <ol class="behavior-ranked">
      {rows.map((c) => {
        const sig = cueSignificant(c, qThreshold);
        const frac =
          c.delta_hat === null || maxEffect === 0 ? 0 : Math.abs(c.delta_hat) / maxEffect;
        return (
          <li
            key={c.cue}
            class={`behavior-row ${c.cue === openCue ? "is-open" : ""}`}
            onClick={() => st.setBehaviorCue(c.cue)}
          >
            <span class="behavior-row-cue">{c.cue}</span>
            <span class="behavior-row-bar" aria-hidden="true">
              {c.status === "measured" ? (
                <i style={`width:${(frac * 100).toFixed(1)}%`} />
              ) : (
                <em class="behavior-row-nobar">{c.status}</em>
              )}
            </span>
            <span class="behavior-row-num">{fmtMetric(c.delta_hat)}</span>
            <span
              class={`behavior-row-sig ${sig === true ? "is-sig" : sig === null ? "is-indet" : ""}`}
            >
              {sig === true ? `q ${fmtMetric(c.q_value, 4)}` : sig === null ? "indeterminate" : "—"}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function Table({
  cues,
  qThreshold,
  openCue,
}: {
  cues: BehaviorCue[];
  qThreshold: number;
  openCue: string;
}) {
  const st = appStore.getState();
  const rows = sortCues(cues, $behavior.value.sort);
  return (
    <div class="behavior-table-wrap">
      <table class="behavior-table">
        <thead>
          <tr>
            <th scope="col">Cue</th>
            <th scope="col">Stratum</th>
            <th scope="col" title="MMD² between arms minus the mean within-arm split-half">
              Δ̂
            </th>
            <th scope="col">95% CI</th>
            <th scope="col">p</th>
            <th scope="col">q</th>
            <th scope="col" title="Miller–Madow bias-corrected Jensen–Shannon divergence">
              JSD
            </th>
            <th scope="col" title="rank-biased overlap of the two arms' top associates">
              RBO
            </th>
            <th scope="col" title="what moved: the centre, the spread, or both">
              Dominant
            </th>
            <th scope="col" title="Δ̂ for the GPT-2-small vs GPT-2-XL control — a known capability gap, as a yardstick">
              Capability ref
            </th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr
              key={c.cue}
              class={c.cue === openCue ? "is-open" : ""}
              onClick={() => st.setBehaviorCue(c.cue)}
            >
              <th scope="row">{c.cue}</th>
              <td>{c.stratum}</td>
              <td>{fmtMetric(c.delta_hat)}</td>
              <td>
                {c.ci[0] === null || c.ci[1] === null
                  ? "not measured"
                  : `${c.ci[0].toFixed(3)} – ${c.ci[1].toFixed(3)}`}
              </td>
              <td>{fmtMetric(c.p_value, 4)}</td>
              <td class={cueSignificant(c, qThreshold) === true ? "is-sig" : ""}>
                {fmtMetric(c.q_value, 4)}
              </td>
              <td>{fmtMetric(c.jsd)}</td>
              <td>{fmtMetric(c.rbo)}</td>
              <td>{c.dominant ?? "not measured"}</td>
              <td>{fmtMetric(c.capability_reference)}</td>
              <td class={`is-${c.status}`}>{c.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── inspector ──────────────────────────────────────────────────────────────

function CueInspector({
  cue,
  data,
  ui,
}: {
  cue: BehaviorCue;
  data: BehaviorData;
  ui: ReturnType<typeof uiType>;
}) {
  const st = appStore.getState();
  const q = data.manifest.statistics.q_threshold;
  const sig = cueSignificant(cue, q);
  const samples = data.samples[cue.cue] ?? [];
  const sensitive = cue.pack.toLowerCase().includes("sensitive");

  return (
    <div class="behavior-cue">
      <header>
        <h2>{cue.cue}</h2>
        <p class="behavior-cue-meta">
          {cue.stratum} · pack <code>{cue.pack}</code> · partition {cue.partition}
        </p>
      </header>

      {cue.status !== "measured" && (
        <div class={`behavior-banner is-${cue.status}`} role="note">
          <strong>{cue.status === "gated" ? "Gated." : "Not measured."}</strong>{" "}
          {cue.reasons.length ? cue.reasons.join(" ") : "No reason was recorded, which is itself a defect."}
        </div>
      )}

      <dl class="behavior-metrics">
        <Metric
          k="Δ̂"
          v={fmtMetric(cue.delta_hat)}
          note="between-arm MMD² minus the mean within-arm split-half MMD². The subtraction is what stops ordinary run-to-run variation reading as a difference between models."
        />
        <Metric
          k="95% CI"
          v={
            cue.ci[0] === null || cue.ci[1] === null
              ? "not measured"
              : `${cue.ci[0].toFixed(4)} – ${cue.ci[1].toFixed(4)}`
          }
          note="block-stratified bootstrap, bias-corrected: resampling with replacement shrinks the within-arm term, which would otherwise push the whole interval above the point estimate."
        />
        <Metric k="p" v={fmtMetric(cue.p_value, 5)} note="permutation within time blocks, (r+1)/(B+1)." />
        <Metric
          k="q"
          v={fmtMetric(cue.q_value, 5)}
          note={`Benjamini–Yekutieli across all cues; the threshold for this study is ${q}.`}
        />
        <Metric
          k="Significant"
          v={sig === true ? "yes" : sig === null ? "indeterminate" : "no"}
          note="indeterminate is a third answer, not a soft no: it means q could not be computed for this cue."
        />
        <Metric
          k="Manski bounds"
          v={
            cue.manski[0] === null || cue.manski[1] === null
              ? "not measured"
              : `${cue.manski[0].toFixed(4)} – ${cue.manski[1].toFixed(4)}`
          }
          note="the range the effect could take under the worst-case behaviour of the trials that did not comply. A point estimate outside these bounds is not defensible."
        />
        <Metric k="JSD" v={fmtMetric(cue.jsd)} note="Miller–Madow bias-corrected." />
        <Metric k="RBO" v={fmtMetric(cue.rbo)} note={`extrapolated, at p = ${data.manifest.statistics.rbo_p}.`} />
        <Metric
          k="Location / dispersion"
          v={cue.dominant ?? "not measured"}
          note="whether the arms differ in where their associations sit, in how spread out they are, or both."
        />
        <Metric
          k="Capability reference"
          v={fmtMetric(cue.capability_reference)}
          note="Δ̂ for GPT-2-small vs GPT-2-XL on this cue — two models of the same family with a known capability gap. It is a yardstick for reading the size of an effect, not a baseline to beat."
        />
      </dl>

      <h3>Per arm</h3>
      {Object.values(cue.arms).map((a) => (
        <div class="behavior-arm" key={a.model_key}>
          <h4>{a.model_key}</h4>
          <p class="behavior-arm-line">
            {a.n_valid} of {a.n_attempted} trials parsed ({(a.parse_rate * 100).toFixed(1)}%) ·{" "}
            {a.distinct_types} distinct types · entropy {a.entropy.toFixed(3)} · reliability{" "}
            {fmtMetric(a.reliability)}
          </p>
          <p class="behavior-arm-assoc">
            {a.top_associates.length ? a.top_associates.join(", ") : "no valid associates"}
          </p>
          <p class="behavior-arm-det">
            echo {(a.detectors.cue_echo * 100).toFixed(0)}% · exemplar{" "}
            {(a.detectors.exemplar_echo * 100).toFixed(0)}% · duplicate{" "}
            {(a.detectors.within_trial_duplicate * 100).toFixed(0)}% · prompt copy{" "}
            {(a.detectors.prompt_copy * 100).toFixed(0)}%
          </p>
        </div>
      ))}

      <h3>Sample responses</h3>
      {samples.length === 0 ? (
        <p class="behavior-note">No sample was exported for this cue.</p>
      ) : sensitive && !ui.revealSensitive ? (
        <div class="behavior-note">
          This cue&rsquo;s pack is marked sensitive, so its raw responses are hidden.
          Reveal them in Settings → Behavior if you need to read them.
        </div>
      ) : (
        <>
          <p class="behavior-note">
            Read-only, verbatim, and a sample — not the analysis. The metrics above are
            computed over every trial, not over these.
          </p>
          <ul class="behavior-samples">
            {samples.slice(0, 8).map((s, i) => (
              <li key={i}>
                <span class="behavior-sample-model">{s.model_key}</span>
                <span class="behavior-sample-text">{s.text}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      <button class="behavior-link-btn" onClick={() => st.setBehaviorCue("")}>
        Close cue
      </button>
    </div>
  );
}

function Metric({ k, v, note }: { k: string; v: string; note: string }) {
  return (
    <div class={`behavior-metric ${v === "not measured" ? "is-missing" : ""}`}>
      <dt title={note}>{k}</dt>
      <dd>{v}</dd>
    </div>
  );
}

// ── evidence, runs, cross-links ────────────────────────────────────────────

/** The evidence to distrust the study ships with the study — the same rule the
 *  map pipeline follows when it exports its noise fraction. */
function Evidence({ data }: { data: BehaviorData }) {
  const m = data.manifest;
  return (
    <section class="behavior-evidence">
      <h2>Evidence</h2>
      <p class="behavior-note">
        Read these before the numbers above, not after. A p that cannot go below its
        floor, an embedder that was downgraded to one of two similar encoders, or an arm
        that never ran all change what the study is entitled to claim.
      </p>
      <dl class="behavior-facts">
        <Fact k="Manifest" v={m.hash.slice(0, 16)} title="the frozen protocol's hash" />
        <Fact k="Protocol" v={m.protocol_hash.slice(0, 16)} />
        <Fact k="Frozen" v={m.frozen_at} />
        <Fact k="Strict mode" v={m.strict ? "on" : "off"} />
        <Fact k="Embedder" v={`${m.embedder.id} @${m.embedder.sha.slice(0, 8)} (${m.embedder.dtype})`} />
        <Fact k="Agreement claim" v={m.embedder.agreement_claim} />
        <Fact k="Δ̂ form" v={m.statistics.delta_hat_form} />
        <Fact k="Permutations" v={`${m.statistics.n_permutations}`} />
        <Fact k="Bootstrap" v={`${m.statistics.n_bootstrap}`} />
        <Fact k="Effect floor" v={`${m.statistics.effect_floor}`} />
        <Fact k="Compliance parity max" v={`${m.statistics.compliance_parity_max}`} />
        <Fact
          k="Fingerprint"
          v={m.fingerprint_available === null ? "not observed" : m.fingerprint_available ? "present" : "absent"}
          title="optional if absent — its absence is recorded, not treated as a failure"
        />
        <Fact k="Commit" v={m.git_commit ? m.git_commit.slice(0, 10) : "not recorded"} />
      </dl>
      {Object.keys(data.diagnostics).length > 0 && (
        <pre class="behavior-diagnostics">{JSON.stringify(data.diagnostics, null, 2)}</pre>
      )}
    </section>
  );
}

function Runs({ data }: { data: BehaviorData }) {
  return (
    <section class="behavior-runs">
      <h2>Runs</h2>
      {data.runs.length === 0 ? (
        <p class="behavior-note">No run is recorded in this artifact.</p>
      ) : (
        <table class="behavior-table">
          <thead>
            <tr>
              <th scope="col">Run</th>
              <th scope="col">Started</th>
              <th scope="col">Trials</th>
              <th scope="col">Completed</th>
              <th scope="col">Cost</th>
              <th scope="col">Halted</th>
            </tr>
          </thead>
          <tbody>
            {data.runs.map((r) => (
              <tr key={r.run_id}>
                <th scope="row" title={r.run_id}>
                  {r.arm ?? r.run_id.slice(0, 10)}
                </th>
                <td>{r.started}</td>
                <td>{r.n_trials}</td>
                <td>{r.n_completed}</td>
                {/* null cost is "no price was known", which is not $0. */}
                <td>{r.cost_usd === null ? "price unknown" : `$${r.cost_usd.toFixed(4)}`}</td>
                <td>{r.halted ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function CrossLinks() {
  const st = appStore.getState();
  return (
    <section class="behavior-crosslinks">
      <h2>Where this sits</h2>
      <p class="behavior-note">
        This page is the only one in Nebul.AI that reads model OUTPUT. The others read
        weights, and a finding here is not evidence about any layer.
      </p>
      <div class="behavior-crosslink-row">
        <button
          class="behavior-link-btn"
          onClick={() => {
            st.setPage("map");
          }}
        >
          Semantic map — what a model&rsquo;s embedding rows can write
        </button>
        <button
          class="behavior-link-btn"
          onClick={() => {
            st.setPage("interp");
          }}
        >
          Internals — measurements taken from the parameters themselves
        </button>
      </div>
    </section>
  );
}
