/** LessonPanel — the Learn experience's map page while a guided lesson or a
 *  narrated map episode is active.
 *
 *  A lesson is the SAME tour runner as every episode (chrome/tours.ts), shown
 *  with the SAME evidence components as Atlas: the evidence header, the
 *  results table and the inspector. What the lesson adds is a task per step
 *  and completion judged only from what the reader actually did — the unit
 *  they chose and the answers they gave, on the verified map. Nothing is
 *  marked done because an animation ran or a step was visited.
 *
 *  Layout: desktop — lesson on the left, the map in the middle, the unit on
 *  the right. Narrow — one panel at a time behind a Lesson / Map switch, with
 *  the unit inline in the lesson. Without a renderer the lesson is the whole
 *  interface and every step can still be completed from the list. */

import { signal } from "@preact/signals";
import { useEffect, useRef, useState } from "preact/hooks";
import { requestFlyToPoint, runEpisodeStep } from "../../app/actions";
import { appStore } from "../../app/store";
import { sourceUnit } from "../../data/columns";
import { activeManifest, findArtifact } from "../../data/experience";
import type { Dataset } from "../../data/loader";
import { AxisRail } from "../AxisRail";
import { LegendCard } from "../LegendCard";
import { SearchPanel } from "../SearchPanel";
import {
  $dataset,
  $datasetId,
  $dims,
  $loading,
  $pendingDatasetId,
  $renderer,
  $selection,
  $tour,
} from "../state";
import { applyTourStep, findTour, type LessonOption, type Tour } from "../tours";
import { EvidenceHeader, ResultsTable } from "../atlas/AtlasWorkspace";
import { isUnlabelled, metaOf } from "../atlas/evidence";
import { Inspector, unitLink } from "../atlas/Inspector";
import { $narrow } from "../atlas/layout";
import {
  findStepOf,
  incompleteSteps,
  isLessonTour,
  labelMatches,
  lessonIdOf,
  loadProgress,
  markCompleted,
  needsUnit,
  noteLessonExit,
  saveProgress,
  stepDone,
  type LessonProgress,
} from "./lesson";

/** narrow screens: which panel is showing */
const $learnPanel = signal<"lesson" | "map">("lesson");

function gotoStep(tour: Tour, step: number): void {
  if (step < 0 || step >= tour.steps.length) return;
  appStore.getState().setTour({ id: tour.id, step });
  applyTourStep(tour, step);
}

const exitTour = () => {
  noteLessonExit();
  appStore.getState().setTour(null);
};

/** The raw label of a row, or null on an unlabelled map (its placeholder is
 *  not evidence and must never satisfy a find task). */
function labelOf(ds: Dataset, row: number): string | null {
  if (isUnlabelled(metaOf(ds))) return null;
  return sourceUnit(ds.columns, row)?.label ?? null;
}

/** The lesson's map is on screen only when it is the manifest's exact bytes. */
function lessonArtifact(tour: Tour) {
  const m = activeManifest();
  const li = m?.learn_intro;
  if (!li || li.lesson_id !== lessonIdOf(tour.id)) return null;
  return findArtifact(m, li.dataset_id, li.sha256);
}

/* ── step markers ───────────────────────────────────────────────────── */

function StepMarkers({ tour, step, done }: { tour: Tour; step: number; done: (i: number) => boolean }) {
  return (
    <ol class="lesson-steps" aria-label="Lesson steps">
      {tour.steps.map((s, i) => {
        // a step with a task is done when its task is; one without is done
        // once the reader has moved past it
        const ok = s.task ? done(i) : i < step;
        const cls = ["lesson-step", i === step && "is-current", ok && "is-done"].filter(Boolean).join(" ");
        return (
          <li key={i} class={cls} aria-current={i === step ? "step" : undefined}>
            <span aria-hidden="true">{ok ? "✓" : i + 1}</span>
            <span class="sr-only">
              Step {i + 1}: {s.title}
              {ok ? " (done)" : ""}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

/* ── tasks ──────────────────────────────────────────────────────────── */

function CheckTask({
  step,
  question,
  options,
  chosen,
  onChoose,
}: {
  step: number;
  question: string;
  options: LessonOption[];
  chosen: string | undefined;
  onChoose: (id: string) => void;
}) {
  const pick = options.find((o) => o.id === chosen);
  const name = `lesson-check-${step}`;
  return (
    <div class="lesson-task">
      <fieldset class="lesson-check">
        <legend class="lesson-question">{question}</legend>
        {options.map((o) => (
          <label key={o.id} class={o.id === chosen ? "lesson-option is-chosen" : "lesson-option"}>
            <input
              type="radio"
              name={name}
              value={o.id}
              checked={o.id === chosen}
              onChange={() => onChoose(o.id)}
            />
            <span>{o.text}</span>
          </label>
        ))}
      </fieldset>
      <div class="lesson-feedback" role="status" aria-live="polite">
        {pick && (
          <p class={pick.correct ? "lesson-verdict is-right" : "lesson-verdict is-wrong"}>
            <strong>{pick.correct ? "Yes. " : "Not quite. "}</strong>
            {pick.why}
            {!pick.correct && " Choose another answer."}
          </p>
        )}
      </div>
    </div>
  );
}

function FindTask({ ds, row, ok, hint }: { ds: Dataset; row: number | null; ok: boolean; hint: string }) {
  const label = row !== null ? (labelOf(ds, row) ?? "an unlabelled unit") : null;
  return (
    <div class="lesson-task">
      <ResultsTable ds={ds} heading="Search the map" />
      <div class="lesson-feedback" role="status" aria-live="polite">
        {row === null ? (
          <p class="lesson-hint">{hint}</p>
        ) : ok ? (
          <p class="lesson-verdict is-right">
            <strong>Found. </strong>You chose “{label}”. Its details are open{$narrow.value ? " below" : " beside the lesson"}.
          </p>
        ) : (
          <p class="lesson-verdict is-wrong">
            <strong>Not this one. </strong>“{label}” is not a label about numbers. Choose another unit.
          </p>
        )}
      </div>
    </div>
  );
}

/* ── the lesson ─────────────────────────────────────────────────────── */

function LessonBody({ tour, step, ds, datasetId }: { tour: Tour; step: number; ds: Dataset; datasetId: string }) {
  const lessonId = lessonIdOf(tour.id)!;
  const sha = ds.sha256!;
  const [p, setP] = useState<LessonProgress>(() => loadProgress(lessonId, sha));
  const update = (next: LessonProgress) => {
    setP(next);
    saveProgress(lessonId, next);
  };
  const label = (row: number) => labelOf(ds, row);
  const done = (i: number) => stepDone(tour, i, p, label);
  const spec = tour.steps[step]!;
  const task = spec.task;
  const findStep = findStepOf(tour);
  const narrow = $narrow.value;
  const noGpu = $renderer.value === "unavailable";
  const sel = $selection.value;
  const selRow = sel?.kind === "point" ? sel.id : null;
  const [complete, setComplete] = useState(false);

  // the find step records whichever unit the reader chooses — from the list
  // or on the map — and judges it; later steps are about that unit
  useEffect(() => {
    if (step === findStep && selRow !== null && selRow !== p.unitRow && selRow < ds.columns.count) {
      update({ ...p, unitRow: selRow });
    }
  }, [step, selRow]);

  // on a later step (and after a reload) the map shows the lesson's unit
  useEffect(() => {
    if (step > findStep && p.unitRow !== null && selRow !== p.unitRow) {
      appStore.getState().setSelection({ kind: "point", id: p.unitRow });
      requestFlyToPoint(p.unitRow);
    }
  }, [step]);

  // each step's heading takes focus so its instructions are read next
  const headRef = useRef<HTMLHeadingElement>(null);
  const first = useRef(true);
  useEffect(() => {
    // on first mount the unit inspector may have claimed focus already (its
    // own rule); the lesson's instructions come first
    const active = document.activeElement;
    if (!first.current || !active || active === document.body || active.closest(".inspector"))
      headRef.current?.focus({ preventScroll: true });
    first.current = false;
    setComplete(false);
  }, [step]);

  const missingUnit = needsUnit(tour, step) && (p.unitRow === null || !done(findStep));
  const stepOk = !missingUnit && done(step);
  const last = step === tour.steps.length - 1;
  const pending = incompleteSteps(tour, p, label).filter((i) => i !== step);
  const inspectRowNow = step === findStep ? selRow : p.unitRow;
  const showInspector = inspectRowNow !== null && !missingUnit && step >= findStep;
  const link = p.unitRow !== null ? unitLink(ds, datasetId, p.unitRow, $dims.value) : null;

  // finishing removes the Finish button, so focus moves to the result
  const completeRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (complete) completeRef.current?.focus({ preventScroll: true });
  }, [complete]);

  const finish = () => {
    if (pending.length > 0) return;
    update({ ...p, finished: p.finished.includes(step) ? p.finished : [...p.finished, step] });
    markCompleted(lessonId);
    setComplete(true);
  };

  const reason = missingUnit
    ? `Choose a unit in step ${findStep + 1} first.`
    : task?.kind === "find"
      ? "Choose a unit whose label is about numbers to continue."
      : task?.kind === "check"
        ? "Choose the correct answer to continue."
        : "";

  const inspector = showInspector ? (
    <Inspector row={inspectRowNow!} actions={task?.kind === "finish"} closable={false} />
  ) : null;

  return (
    <>
      <aside
        class="lesson-panel"
        aria-labelledby="lesson-step-title"
        hidden={narrow && !noGpu && $learnPanel.value === "map"}
      >
        <div class="lesson-top">
          <p class="lesson-kicker">
            Lesson · <span>{tour.label}</span>
          </p>
          <button type="button" class="aw-btn aw-btn-quiet lesson-exit" onClick={exitTour}>
            Exit lesson
          </button>
        </div>
        <StepMarkers tour={tour} step={step} done={done} />
        <h2 id="lesson-step-title" class="lesson-title" tabIndex={-1} ref={headRef}>
          <span class="sr-only">
            Step {step + 1} of {tour.steps.length}:{" "}
          </span>
          {spec.title}
        </h2>
        <p class="lesson-caption">{spec.caption}</p>

        {step === 0 && <EvidenceHeader ds={ds} datasetId={datasetId} />}

        {missingUnit ? (
          <div class="lesson-task lesson-need" role="note">
            <p>This step is about the unit you choose in step {findStep + 1}, and none is chosen yet.</p>
            <button type="button" class="aw-btn" onClick={() => gotoStep(tour, findStep)}>
              Go to step {findStep + 1}
            </button>
          </div>
        ) : task?.kind === "find" ? (
          <FindTask ds={ds} row={p.unitRow} ok={done(step)} hint={task.hint} />
        ) : task?.kind === "check" ? (
          <CheckTask
            step={step}
            question={task.question}
            options={task.options}
            chosen={p.answers[step]}
            onChoose={(id) => update({ ...p, answers: { ...p.answers, [step]: id } })}
          />
        ) : null}

        {task?.kind === "finish" && !missingUnit && (
          <div class="lesson-task lesson-finish">
            {complete || done(step) ? (
              <div class="lesson-complete" role="status" tabIndex={-1} ref={completeRef}>
                <p class="lesson-verdict is-right">
                  <strong>Lesson complete. </strong>
                  You found one direction, identified it by its index and the map's digest, and read its place as
                  geometry — not as an activation or a cause.
                </p>
              </div>
            ) : pending.length > 0 ? (
              <div class="lesson-need" role="note">
                <p>
                  To finish, complete step{pending.length > 1 ? "s" : ""} {pending.map((i) => i + 1).join(" and ")}.
                </p>
                <button type="button" class="aw-btn" onClick={() => gotoStep(tour, pending[0]!)}>
                  Go to step {pending[0]! + 1}
                </button>
              </div>
            ) : null}
            <div class="lesson-row">
              {!(complete || done(step)) && (
                <button type="button" class="aw-btn aw-btn-primary" disabled={pending.length > 0} onClick={finish}>
                  Finish lesson
                </button>
              )}
              {link && (
                <a class="aw-btn" href={link}>
                  Continue in Atlas
                </a>
              )}
              {(complete || done(step)) && (
                <button type="button" class="aw-btn" onClick={exitTour}>
                  Back to lessons
                </button>
              )}
            </div>
            <p class="aw-dim">
              Continue in Atlas opens this exact unit in the Atlas explorer. Your answers stay in this browser tab and
              are not put in the link.
            </p>
          </div>
        )}

        {narrow && inspector && <div class="lesson-inline-unit">{inspector}</div>}

        <nav class="lesson-nav" aria-label="Lesson navigation">
          <button type="button" class="aw-btn" disabled={step === 0} onClick={() => gotoStep(tour, step - 1)}>
            <span aria-hidden="true">‹ </span>Previous step
          </button>
          {!last && (
            <button
              type="button"
              class="aw-btn aw-btn-primary"
              disabled={!stepOk}
              aria-describedby={stepOk ? undefined : "lesson-next-reason"}
              onClick={() => gotoStep(tour, step + 1)}
            >
              Next step<span aria-hidden="true"> ›</span>
            </button>
          )}
          {!last && !stepOk && reason && (
            <p id="lesson-next-reason" class="aw-dim lesson-reason">
              {reason}
            </p>
          )}
        </nav>
        {noGpu && (
          <p class="aw-note">
            The map needs WebGL or WebGPU, which this browser does not provide. Every step works from the list, with
            the same evidence.
          </p>
        )}
      </aside>
      {!narrow && inspector}
      {narrow && !noGpu && <LearnSwitch />}
    </>
  );
}

function LearnSwitch() {
  const panel = $learnPanel.value;
  const set = (p: "lesson" | "map") => ($learnPanel.value = p);
  return (
    <section class="atlas-switch" aria-label="Panel switch">
      <div role="group" aria-label="Show">
        <button type="button" class={panel === "lesson" ? "aw-seg is-on" : "aw-seg"} aria-pressed={panel === "lesson"} onClick={() => set("lesson")}>
          Lesson
        </button>
        <button type="button" class={panel === "map" ? "aw-seg is-on" : "aw-seg"} aria-pressed={panel === "map"} onClick={() => set("map")}>
          Map
        </button>
      </div>
    </section>
  );
}

/** Not ready: the verified map is still loading, or the lesson refused it. */
function LessonState({ tour, step, blocked }: { tour: Tour; step: number; blocked?: string }) {
  const loading = $loading.value;
  const pending = $pendingDatasetId.value;
  const headRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (blocked) headRef.current?.focus({ preventScroll: true });
  }, [blocked]);
  const pct = loading.total > 0 ? ` — ${Math.round((loading.loaded / loading.total) * 100)}%` : "";
  return (
    <aside class="lesson-panel" aria-labelledby="lesson-state-title">
      <div class="lesson-top">
        <p class="lesson-kicker">
          Lesson · <span>{tour.label}</span>
        </p>
      </div>
      {blocked ? (
        <div role="alert" class="lesson-blocked">
          <h2 id="lesson-state-title" class="lesson-title" tabIndex={-1} ref={headRef}>
            This lesson's map is not available
          </h2>
          <p>{blocked}</p>
          <p class="aw-dim">The lesson is not marked as done while its map cannot be checked.</p>
          <div class="lesson-row">
            <button type="button" class="aw-btn aw-btn-primary" onClick={() => void runEpisodeStep(tour.id, step)}>
              Try again
            </button>
            <button type="button" class="aw-btn" onClick={exitTour}>
              Back to lessons
            </button>
          </div>
        </div>
      ) : (
        <>
          <h2 id="lesson-state-title" class="lesson-title">
            Opening the lesson's map
          </h2>
          <p class="aw-state" role="status">
            {pending ? `Loading and checking the published file${pct}…` : "Checking the published file…"}
          </p>
        </>
      )}
    </aside>
  );
}

/* ── narrated map episodes ──────────────────────────────────────────── */

/** Step controls for an existing narrated episode whose step is on the map
 *  page. The Internals page has its own tour bar; without this one a map step
 *  had no way forward or out. */
export function EpisodeBar({ tour, step }: { tour: Tour; step: number }) {
  const spec = tour.steps[step];
  if (!spec) return null;
  const last = step + 1 >= tour.steps.length;
  return (
    <div class="interp-tourbar episode-bar" role="group" aria-label={`Guided episode: ${tour.label}`}>
      <div class="interp-tourbar-head">
        <span class="interp-tourbar-tour">⚑ {tour.label}</span>
        <span class="interp-tourbar-count">
          {step + 1} / {tour.steps.length}
        </span>
        <button type="button" class="interp-tourbar-exit" aria-label="Exit episode" onClick={exitTour}>
          ×
        </button>
      </div>
      <h2 class="interp-tourbar-title">{spec.title}</h2>
      <p class="interp-tourbar-caption">{spec.caption}</p>
      <div class="interp-tourbar-nav">
        <button type="button" disabled={step === 0} onClick={() => void runEpisodeStep(tour.id, step - 1)}>
          ‹ back
        </button>
        <button
          type="button"
          class="is-primary"
          onClick={() => (last ? exitTour() : void runEpisodeStep(tour.id, step + 1))}
        >
          {last ? "finish ✓" : "next ›"}
        </button>
      </div>
    </div>
  );
}

/** Learn's map page with a tour active. */
export function LearnMapPanels() {
  const ref = $tour.value;
  const tour = ref ? findTour(ref.id) : undefined;
  if (!ref || !tour) return null;
  if (!isLessonTour(tour)) {
    return (
      <>
        <LegendCard />
        <SearchPanel />
        <AxisRail />
        <EpisodeBar tour={tour} step={ref.step} />
      </>
    );
  }
  const ds = $dataset.value;
  const id = $datasetId.value;
  const art = lessonArtifact(tour);
  const verified = !!(ds && id && art && id === art.dataset_id && ds.sha256 === art.sha256);
  if (ref.blocked || !verified) return <LessonState tour={tour} step={ref.step} blocked={ref.blocked} />;
  // keyed by digest: progress never carries across different bytes
  return <LessonBody key={ds!.sha256!} tour={tour} step={ref.step} ds={ds!} datasetId={id!} />;
}
