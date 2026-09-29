/** app/experience.ts — which of NebulAI's three experiences a document is
 *  showing, and how a URL chooses one.
 *
 *  NebulAI is one AppId with four wire pages (map · behavior · interp · guide).
 *  Learn, Atlas and Research are NOT new AppIds and do not replace APP_PAGES:
 *  they are a NebulAI-only discriminant that decides the visible navigation,
 *  the onboarding and which Settings sections are offered. The wire page still
 *  names the content underneath, so every existing permalink keeps meaning the
 *  same thing.
 *
 *  This module is DATA ONLY and pure: no DOM, no store, no import.meta.env. The
 *  root chooser (a separate, tiny entry) and the app both route through
 *  `resolveExperience`, and a unit test pins the precedence table below.
 *
 *  Precedence (PRODUCT-EXPERIENCES.md, "URL and migration rules"):
 *   1. a recognised guided lesson or episode always selects Learn;
 *   2. a valid explicit `experience=` that supports the requested content wins;
 *   3. the entry path the document was opened at, if it supports the content;
 *   4. otherwise inference from the legacy route:
 *      page=interp|behavior or an advanced map view → Research,
 *      page=guide → Learn, map / model / pinned finding → Atlas;
 *   5. no recognised intent at all → the chooser (null).
 *  A correction against what the visitor asked for (entry path or explicit
 *  value) carries a short notice so the switch is never silent. */

import type { Page, ViewMode } from "./store";

export type Experience = "learn" | "atlas" | "research";

export const EXPERIENCES: readonly Experience[] = ["learn", "atlas", "research"];

export function isExperience(x: unknown): x is Experience {
  return typeof x === "string" && (EXPERIENCES as readonly string[]).includes(x);
}

/** Map views that belong to Research's Comparisons rather than to Atlas. */
export const ADVANCED_VIEWS: readonly ViewMode[] = ["chord", "hierarchy", "compare"];

export function isAdvancedView(v: ViewMode | null | undefined): boolean {
  return !!v && (ADVANCED_VIEWS as readonly string[]).includes(v);
}

/** What a URL asks for, reduced to the facts routing needs. Validation of the
 *  individual values (is this episode registered, is this lesson known) is the
 *  caller's job; this module only sees the verdicts. */
export interface RouteIntent {
  page?: Page;
  view?: ViewMode;
  /** a registered episode is named */
  episode?: boolean;
  /** a registered map lesson is named */
  lesson?: boolean;
  /** `model=` is present */
  model?: boolean;
  /** a registered Internals analysis is named (`feature=`) */
  feature?: boolean;
  /** pinned-unit keys are present */
  finding?: boolean;
}

/** Can experience `e` host this content without changing what it means? */
export function supports(e: Experience, i: RouteIntent): boolean {
  const guided = !!(i.episode || i.lesson);
  switch (e) {
    case "learn":
      if (guided) return true;
      // Learn's own content is the lessons catalogue. A bare map, an
      // unguided analysis or the Behavior study are not lessons.
      // A model or pinned unit outside a lesson is a map request (Atlas).
      if (i.finding || i.model || i.feature || isAdvancedView(i.view)) return false;
      return i.page === undefined || i.page === "guide";
    case "atlas":
      if (guided) return false;
      // an Internals analysis is Research's, even when the link names no page
      if (isAdvancedView(i.view) || i.feature) return false;
      return i.page === undefined || i.page === "map";
    case "research":
      // guided episodes are Learn's even when a step shows Internals
      if (guided) return false;
      return true;
  }
}

/** The page an experience opens on when the URL names none. A link that
 *  names a map view or a model without a page is a map link in any
 *  experience: Research opens Comparisons on it, not Internals. */
export function defaultPage(e: Experience, i: RouteIntent = {}): Page {
  switch (e) {
    case "learn":
      return "guide";
    case "atlas":
      return "map";
    case "research":
      // an analysis link opens that analysis; a model or view link without
      // one is a map link, which Research shows under Comparisons
      if (i.feature) return "interp";
      return i.view || i.model ? "map" : "interp";
  }
}

export interface Resolution {
  /** null = show the chooser */
  experience: Experience | null;
  /** set when the result differs from what the entry path or an explicit
   *  value asked for; the UI shows it once */
  notice: string | null;
}

export const EXPERIENCE_NAMES: Record<Experience, string> = {
  learn: "Learn",
  atlas: "Atlas",
  research: "Research",
};

function noticeFor(e: Experience): string {
  switch (e) {
    case "learn":
      return "Opened in Learn for this lesson.";
    case "atlas":
      return "Opened in Atlas for this map.";
    case "research":
      return "Opened in Research for this analysis.";
  }
}

function hasIntent(i: RouteIntent): boolean {
  return !!(i.page || i.view || i.episode || i.lesson || i.model || i.finding || i.feature);
}

/** Legacy inference, used when neither the explicit value nor the entry path
 *  can host the request. */
export function inferExperience(i: RouteIntent): Experience | null {
  if (i.episode || i.lesson) return "learn";
  if (i.page === "interp" || i.page === "behavior" || isAdvancedView(i.view)) return "research";
  if (i.feature && !i.page) return "research";
  if (i.page === "guide") return "learn";
  if (i.page === "map" || i.model || i.finding) return "atlas";
  return null;
}

export function resolveExperience(args: {
  /** the experience whose entry path the document was opened at; null = root */
  entry: Experience | null;
  /** raw `experience=` value; anything unrecognised is ignored */
  explicit?: string | null;
  intent: RouteIntent;
}): Resolution {
  const { entry, intent } = args;
  const explicit = isExperience(args.explicit) ? args.explicit : null;
  const asked = explicit ?? entry;
  const pick = (e: Experience | null): Resolution => ({
    experience: e,
    notice: e && asked && e !== asked ? noticeFor(e) : null,
  });

  if (intent.episode || intent.lesson) return pick("learn");
  if (explicit && supports(explicit, intent)) return pick(explicit);
  if (entry && supports(entry, intent)) return pick(entry);
  if (!hasIntent(intent)) return pick(entry);
  return pick(inferExperience(intent));
}

/** The path segment each experience lives under, beneath the app root. */
export function entrySegment(e: Experience): string {
  return `${e}/`;
}

/** Which experience a document path belongs to: the last directory segment
 *  that names one, or null for the root chooser. */
export function experienceFromPath(pathname: string, rootPath: string): Experience | null {
  if (!pathname.startsWith(rootPath)) return null;
  const rest = pathname.slice(rootPath.length).replace(/index\.html$/, "");
  const seg = rest.split("/").filter(Boolean)[0];
  return isExperience(seg) ? seg : null;
}
