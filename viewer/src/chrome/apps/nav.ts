/** apps/nav.ts — the chrome half of each instrument's identity: wordmark,
 *  document title, the three nav pills, and the one link across to the other
 *  instrument.
 *
 *  This module holds DATA ONLY — no component imports, not even type-only ones
 *  that would tempt someone to add a value import later. That is what keeps it
 *  cheap enough for both entries and for a unit test to import: the page
 *  components themselves live in apps/nebulai.tsx and apps/seer.tsx, one file
 *  per instrument, so Seer's bundle never sees InterpPage and Nebulai's never
 *  sees SeerPage.
 *
 *  The pill labels are NOT the page ids, and three of them deliberately differ
 *  from the ids they drive: `seer` shows as "Live", `sessions` as "Transcripts",
 *  `snapshot` as "Topics". The ids are wire format (permalinks, store,
 *  goldens) and stay put; the labels are what the instrument calls its own
 *  three views now that it is an instrument rather than three of six tabs.
 *
 *  Page ORDER and MEMBERSHIP are not decided here — `APP_PAGES` in
 *  app/slices/shell.ts is the authority, because it is also what `setPage`
 *  validates against. tests/unit/app-pages.test.ts pins the two together, so
 *  adding a pill here without adding the page there fails a test rather than
 *  shipping a nav button that silently does nothing. */

import { APP_PAGES, type AppId, type Experience, type Page } from "../../app/store";
import { HUB_URL, NEBULAI_APP_URL, SEER_APP_URL } from "./links";

export interface NavItem {
  label: string;
  page: Page;
}

/** One NebulAI experience's chrome: the name shown beside the wordmark, its
 *  own pills over the SAME wire pages, and the words that tell a visitor what
 *  it is for. The pills never add a page — they choose which of APP_PAGES the
 *  experience shows and what it calls them (Research's "Methods" is `guide`,
 *  Learn's "Lessons" is `guide` too). */
export interface ExperienceChrome {
  id: Experience;
  label: string;
  documentTitle: string;
  /** one line for the Other experiences menu and the chooser */
  description: string;
  nav: readonly NavItem[];
}

/** The sibling instrument: a plain link to the other entry's HTML, not a
 *  fourth pill. */
export interface SiblingLink {
  label: string;
  href: string;
  /** tooltip: says what the other instrument is FOR, since the label alone
   *  ("Seer") does not tell a first-time visitor why they would click it */
  title: string;
}

/** Where the OTHER instrument lives, and where the hub lives. Both are
 *  build-time facts about the deployment shape, not about the app, so they
 *  arrive as env vars rather than being hardcoded here.
 *
 *  The default is the relative sibling — `./seer.html` / `./index.html` —
 *  which is correct for the combined `npm run build`, for `vite preview`, for
 *  the dev server and for `file://`, because there both documents are emitted
 *  side by side into one directory. It is WRONG for the per-app deploys: those
 *  put the two instruments in SIBLING DIRECTORIES (`/psychiX/nebulai-maps/` and
 *  `/psychiX/seer/`), where a relative `./seer.html` resolves to
 *  `/psychiX/nebulai-maps/seer.html` and 404s. So `build:nebulai` /
 *  `build:seer` (see package.json) pass the absolute sub-paths instead.
 *
 *  `VITE_SEER_APP_URL` is NOT `VITE_SEER_URL`. The latter, which predates it,
 *  is the address of the Seer CAPTURE SERVER (`seer serve`, :8125) that the
 *  Live page talks to over HTTP — a backend endpoint. This one is the URL of
 *  the Seer *web app* on this site. Setting one when you meant the other is
 *  the obvious mistake, hence the different suffix and this paragraph. */

/** The psychiX hub, when there is one. Empty (the default) means this build is
 *  not part of a hub deploy — the combined build has no hub document to point
 *  at — and the top bar then renders no hub link at all rather than a dead
 *  one. `build:nebulai` / `build:seer` set `VITE_HUB_URL=/psychiX/`. */

/** Which brand mark the top bar draws. A discriminant rather than the SVG
 *  itself, because this module is data-only (no JSX) — TopBar owns both
 *  drawings. `nebula` is the three ramp-coloured dots in orbit; `eye` is a
 *  lens with an iris, matching seer.html's favicon. Seer must not fly
 *  Nebulai's mark any more than it may use its wordmark. */
export type MarkId = "nebula" | "eye";

export interface AppChrome {
  id: AppId;
  mark: MarkId;
  /** `document.title` — set by the entry, and also baked into each HTML file
   *  so the tab is right before a single byte of JS has run */
  documentTitle: string;
  /** wordmark, split into a full-weight head and a dimmed tail */
  wordmark: { head: string; tail: string };
  /** one-line statement of what this instrument maps; used as the wordmark's
   *  accessible description so the mark is not the only thing naming it */
  tagline: string;
  nav: readonly NavItem[];
  sibling: SiblingLink;
  /** the psychiX landing page, when this build belongs to a hub deploy;
   *  `undefined` in the combined build, where there is no hub document */
  hub?: SiblingLink;
  /** NebulAI only: the three experiences sharing this app's pages */
  experiences?: Readonly<Record<Experience, ExperienceChrome>>;
}

export const NEBULAI_EXPERIENCES: Readonly<Record<Experience, ExperienceChrome>> = {
  learn: {
    id: "learn",
    label: "Learn",
    documentTitle: "Nebul.AI Learn — how model maps work",
    description: "Guided lessons: what a point on a model map is, and what it cannot show.",
    nav: [{ label: "Lessons", page: "guide" }],
  },
  atlas: {
    id: "atlas",
    label: "Atlas",
    documentTitle: "Nebul.AI Atlas — explore a model",
    description: "Search a model map, inspect one unit, save an exact record.",
    nav: [{ label: "Explore", page: "map" }],
  },
  research: {
    id: "research",
    label: "Research",
    documentTitle: "Nebul.AI Research — inspect the evidence",
    description: "Registered analyses, comparisons and the Behavior study, with their caveats.",
    nav: [
      { label: "Internals", page: "interp" },
      { label: "Comparisons", page: "map" },
      { label: "Behavior study", page: "behavior" },
      { label: "Methods", page: "guide" },
    ],
  },
};

/** One link, shared by both instruments — the hub is above both of them. */
const HUB_LINK: SiblingLink | undefined = HUB_URL
  ? { label: "psychiX", href: HUB_URL, title: "psychiX — both instruments" }
  : undefined;

export const APP_CHROME: Record<AppId, AppChrome> = {
  nebulai: {
    id: "nebulai",
    mark: "nebula",
    documentTitle: "Nebul.AI — concept atlas",
    wordmark: { head: "nebul", tail: ".ai" },
    tagline: "map what a model knows",
    nav: [
      { label: "Semantic map", page: "map" },
      { label: "Behavior", page: "behavior" },
      { label: "Internals", page: "interp" },
      // the page id stays "guide" — APP_PAGES is the membership authority and
      // every deep link, drift guard and e2e route already spells it that way.
      // Only the word on the pill changes, because what lives behind it is now
      // a set of episodes rather than a single walkthrough.
      { label: "Episodes", page: "guide" },
    ],
    sibling: { label: "Seer", href: SEER_APP_URL, title: "Seer — map what an agent did" },
    hub: HUB_LINK,
    experiences: NEBULAI_EXPERIENCES,
  },
  seer: {
    id: "seer",
    mark: "eye",
    documentTitle: "Seer — agent runs",
    wordmark: { head: "seer", tail: "" },
    tagline: "map what an agent did",
    nav: [
      { label: "Live", page: "seer" },
      { label: "Transcripts", page: "sessions" },
      { label: "Keywords", page: "keyword" },
      { label: "Topics", page: "snapshot" },
    ],
    sibling: {
      label: "Nebul.AI",
      href: NEBULAI_APP_URL,
      title: "Nebul.AI — map what a model knows",
    },
    hub: HUB_LINK,
  },
};

/** The page an app boots on: the first pill, which is also `APP_PAGES[id][0]`
 *  (pinned by tests/unit/app-pages.test.ts). */
export function defaultPage(id: AppId): Page {
  return APP_PAGES[id][0]!;
}
