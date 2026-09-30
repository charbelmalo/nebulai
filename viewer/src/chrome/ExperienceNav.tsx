/** ExperienceNav.tsx — NebulAI's top navigation, one per experience.
 *
 *  Imported only by apps/nebulai.tsx (through the AppShell `renderTopNav`
 *  hook), so Seer's bundle never sees the experience registry or the tours.
 *  It draws, for the active experience:
 *
 *    - that experience's own pills over the unchanged wire pages (collapsed
 *      into one disclosure menu on compact viewports when there is more than
 *      one destination);
 *    - Learn's current-lesson progress while a guided episode runs;
 *    - Help: what this experience is for, and (Atlas/Research) a way into
 *      Learn that remembers where to come back to;
 *    - Other experiences: the two sibling entries with explicit handoffs, then
 *      Other tools (Seer, psychiX).
 *
 *  Every menu is a disclosure button (`aria-expanded` + `aria-controls`) over
 *  plain links: Escape closes and returns focus to its button, and so do a
 *  click outside and focus leaving the menu. Switching experience is a normal
 *  navigation to that entry in the same tab — the next document resets
 *  transient UI by construction. */

import { useSignal } from "@preact/signals";
import type { ComponentChildren } from "preact";
import { useEffect, useRef } from "preact/hooks";
import { appStore, type Experience, type Page } from "../app/store";
import { EXPERIENCES } from "../app/experience";
import { APP_ROOT } from "../data/base";
import type { ExperienceChrome, SiblingLink } from "./apps/nav";
import { $behaviorPublished, $compactViewport, $datasetId, $experience, $experienceNotice, $page, $tour } from "./state";
import { findTour } from "./tours";

/** sessionStorage key for "where Help came from", so Learn's Back link can
 *  return to the exact view (model, unit, page) rather than a bare entry.
 *  The URL only carries `return=<experience>`; the full address stays local. */
export const RETURN_KEY = "nebulai.returnHref";

/** The address of another experience's entry, carrying only allowed context:
 *  the current model into Atlas (maps are compatible by id), and a return
 *  marker into Learn when the visitor asked for help. Research never receives
 *  a model implicitly — it starts on its own explicit chooser. */
export function experienceHref(
  target: Experience,
  from: Experience | null,
  datasetId: string | null,
  opts: { help?: boolean } = {},
): string {
  const params = new URLSearchParams();
  if (target === "atlas" && datasetId && from !== "atlas") params.set("model", datasetId);
  if (target === "learn" && opts.help && from && from !== "learn") params.set("return", from);
  const hash = params.toString();
  return new URL(`${target}/`, APP_ROOT).href + (hash ? `#${hash}` : "");
}

/** Where Learn's "Back to …" goes: the remembered view if it belongs to the
 *  experience named in the URL, else that experience's entry. */
export function returnHref(returnTo: Experience, stored: string | null): string {
  const entry = new URL(`${returnTo}/`, APP_ROOT).href;
  if (stored && stored.startsWith(entry)) return stored;
  return entry;
}

function rememberReturn(): void {
  try {
    sessionStorage.setItem(RETURN_KEY, location.href);
  } catch {
    /* storage blocked: Back still reaches the entry */
  }
}

function readReturn(): string | null {
  try {
    return sessionStorage.getItem(RETURN_KEY);
  } catch {
    return null;
  }
}

/* ── disclosure ───────────────────────────────────────────────────────── */

let menuSeq = 0;

/** A button that shows/hides a panel of links. Closes on Escape (focus back
 *  to the button), on a pointer press outside, and when focus leaves it. */
function Disclosure(props: {
  label: ComponentChildren;
  buttonLabel?: string;
  buttonClass?: string;
  panelClass?: string;
  children: (close: () => void) => ComponentChildren;
}) {
  const open = useSignal(false);
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const id = useRef(`xnav-menu-${++menuSeq}`).current;
  const close = (refocus = false) => {
    open.value = false;
    if (refocus) button.current?.focus();
  };

  // One listener for the component's lifetime, reading the signal when it
  // fires: an effect keyed on `open.value` re-subscribes on every toggle and
  // proved unreliable under the signals renderer.
  useEffect(() => {
    const onDown = (e: PointerEvent) => {
      if (!open.peek()) return;
      if (root.current && !root.current.contains(e.target as Node)) open.value = false;
    };
    // capture phase: page layers (the stage, the guide scroller) may stop
    // pointerdown from bubbling, and a menu that ignores those clicks sticks
    document.addEventListener("pointerdown", onDown, true);
    return () => document.removeEventListener("pointerdown", onDown, true);
  }, []);

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape" && open.value) {
      e.stopPropagation();
      close(true);
      return;
    }
    if (!open.value || (e.key !== "ArrowDown" && e.key !== "ArrowUp")) return;
    const items = Array.from(
      root.current?.querySelectorAll<HTMLElement>(`#${id} a, #${id} button`) ?? [],
    );
    if (items.length === 0) return;
    e.preventDefault();
    const at = items.indexOf(document.activeElement as HTMLElement);
    const next =
      e.key === "ArrowDown"
        ? items[(at + 1) % items.length]
        : items[(at - 1 + items.length) % items.length];
    next?.focus();
  };

  const onFocusOut = (e: FocusEvent) => {
    const to = e.relatedTarget as Node | null;
    if (open.value && to && root.current && !root.current.contains(to)) close();
  };

  return (
    <div class="xnav-disclosure" ref={root} onKeyDown={onKeyDown} onFocusOut={onFocusOut}>
      <button
        ref={button}
        type="button"
        class={props.buttonClass ?? "xnav-trigger"}
        aria-expanded={open.value}
        aria-controls={id}
        aria-label={props.buttonLabel}
        onClick={() => (open.value = !open.value)}
      >
        {props.label}
        <span class="xnav-caret" aria-hidden="true">
          ▾
        </span>
      </button>
      <div id={id} class={props.panelClass ?? "xnav-panel"} hidden={!open.value}>
        {props.children(() => close(true))}
      </div>
    </div>
  );
}

/* ── pieces ───────────────────────────────────────────────────────────── */

function PagePills({ exp }: { exp: ExperienceChrome }) {
  const page = $page.value;
  const go = (p: Page) => appStore.getState().setPage(p);
  if (exp.nav.length > 1 && $compactViewport.value) {
    const current = exp.nav.find((n) => n.page === page) ?? exp.nav[0]!;
    return (
      <Disclosure
        label={
          <>
            <span class="xnav-trigger-exp">{exp.label} · </span>
            {current.label}
          </>
        }
        buttonLabel={`${exp.label} sections, current: ${current.label}`}
        buttonClass="topnav-pill is-active xnav-trigger"
      >
        {(close) => (
          <ul class="xnav-list">
            {exp.nav.map((n) => (
              <li key={n.page}>
                <button
                  type="button"
                  class="xnav-item"
                  aria-current={n.page === page ? "page" : undefined}
                  onClick={() => {
                    go(n.page);
                    close();
                  }}
                >
                  {n.label}
                </button>
              </li>
            ))}
          </ul>
        )}
      </Disclosure>
    );
  }
  return (
    <>
      {exp.nav.map((n) => (
        <button
          key={n.page}
          type="button"
          class={`topnav-pill${page === n.page ? " is-active" : ""}`}
          aria-current={page === n.page ? "page" : undefined}
          onClick={() => go(n.page)}
        >
          {n.label}
          {n.page === "behavior" && $behaviorPublished.value === false && (
            <span class="xnav-pill-note"> · not published</span>
          )}
        </button>
      ))}
    </>
  );
}

/** Learn only: which lesson is running and how far along it is. */
function LessonProgress() {
  const ref = $tour.value;
  if (!ref) return null;
  const tour = findTour(ref.id);
  if (!tour) return null;
  const total = tour.steps.length;
  const step = Math.min(ref.step + 1, total);
  return (
    <span class="xnav-progress" role="status" title={tour.label}>
      <span class="xnav-progress-name">{tour.label}</span>
      <span class="xnav-progress-step">
        Step {step} of {total}
      </span>
    </span>
  );
}

const HELP_TEXT: Record<Experience, string> = {
  learn:
    "Lessons walk through one idea at a time. Use Next and Back inside a lesson, or Exit lesson to return here. Nothing you do in a lesson is shared unless you save it.",
  atlas:
    "Search the map, pick one unit, and read its evidence in the inspector. The map is a layout of labels and directions, not a measurement of your own input.",
  research:
    "Choose an available model, then a registered analysis. Each analysis names the quantity it measured, its caveat and its own export.",
};

function Help({ exp }: { exp: Experience }) {
  const datasetId = $datasetId.value;
  return (
    <Disclosure label="Help" buttonClass="topnav-cross xnav-trigger" panelClass="xnav-panel xnav-help">
      {() => (
        <div class="xnav-help-body">
          <p>{HELP_TEXT[exp]}</p>
          {exp !== "learn" && (
            <a
              class="xnav-link"
              href={experienceHref("learn", exp, datasetId, { help: true })}
              onClick={rememberReturn}
            >
              Open the lessons in Learn
              <span class="xnav-link-note">You can come back here from the lesson.</span>
            </a>
          )}
          {exp === "learn" && (
            <a class="xnav-link" href={experienceHref("atlas", exp, datasetId)}>
              Continue in Atlas
              <span class="xnav-link-note">Explore the full map on your own.</span>
            </a>
          )}
        </div>
      )}
    </Disclosure>
  );
}

function OtherExperiences({
  current,
  all,
  tools,
}: {
  current: Experience;
  all: Readonly<Record<Experience, ExperienceChrome>>;
  tools: SiblingLink[];
}) {
  const datasetId = $datasetId.value;
  return (
    <Disclosure
      label={
        <>
          <span class="xnav-other-lead">Other </span>
          <span class="xnav-other-rest">experiences</span>
        </>
      }
      buttonClass="topnav-cross xnav-trigger"
      panelClass="xnav-panel xnav-other"
    >
      {() => (
        <>
          <ul class="xnav-list" aria-label="Other experiences">
            {EXPERIENCES.filter((e) => e !== current).map((e) => (
              <li key={e}>
                <a class="xnav-link" href={experienceHref(e, current, datasetId)}>
                  NebulAI {all[e].label}
                  <span class="xnav-link-note">{all[e].description}</span>
                </a>
              </li>
            ))}
          </ul>
          {tools.length > 0 && (
            <>
              <p class="xnav-heading" id="xnav-tools-heading">
                Other tools
              </p>
              <ul class="xnav-list" aria-labelledby="xnav-tools-heading">
                {tools.map((t) => (
                  <li key={t.href}>
                    <a class="xnav-link" href={t.href}>
                      {t.label}
                      <span class="xnav-cross-arrow" aria-hidden="true">
                        {" "}
                        ↗
                      </span>
                      <span class="xnav-link-note">{t.title}</span>
                    </a>
                  </li>
                ))}
              </ul>
            </>
          )}
        </>
      )}
    </Disclosure>
  );
}

/** Learn with a return marker: one link back to the view Help came from. */
function BackLink({ returnTo, all }: { returnTo: Experience; all: Readonly<Record<Experience, ExperienceChrome>> }) {
  return (
    <a class="topnav-cross xnav-back" href={returnHref(returnTo, readReturn())}>
      <span aria-hidden="true">← </span>Back to {all[returnTo].label}
    </a>
  );
}

/** The whole top navigation for NebulAI. */
export function ExperienceNav(props: {
  experiences: Readonly<Record<Experience, ExperienceChrome>>;
  tools: SiblingLink[];
}) {
  const current = $experience.value ?? "atlas";
  const exp = props.experiences[current];
  const returnTo = appStore.getState().returnTo;
  return (
    <nav class="topnav xnav" aria-label={`${exp.label} navigation`} data-experience={current}>
      <PagePills exp={exp} />
      {current === "learn" && <LessonProgress />}
      <span class="topnav-sep" aria-hidden="true" />
      {current === "learn" && returnTo && <BackLink returnTo={returnTo} all={props.experiences} />}
      <Help exp={current} />
      <OtherExperiences current={current} all={props.experiences} tools={props.tools} />
    </nav>
  );
}

/** The experience name beside the wordmark. */
export function ExperienceChip({ experiences }: { experiences: Readonly<Record<Experience, ExperienceChrome>> }) {
  const current = $experience.value;
  if (!current) return null;
  return <span class="topbar-exp">{experiences[current].label}</span>;
}

/** The one-shot "Opened in X for this …" notice. Polite, dismissible, and
 *  never blocks the task underneath. */
export function ExperienceNotice() {
  const notice = $experienceNotice.value;
  if (!notice) return null;
  return (
    <div class="xnav-notice" role="status">
      <span>{notice}</span>
      <button
        type="button"
        class="xnav-notice-close"
        aria-label="Dismiss notice"
        onClick={() => appStore.getState().dismissExperienceNotice()}
      >
        ×
      </button>
    </div>
  );
}
