/** The two halves of "which pages does this instrument have" have to agree,
 *  and nothing else makes them.
 *
 *  `APP_PAGES` (app/slices/shell.ts) is what `setPage` validates against — the
 *  store's answer. `APP_CHROME[id].nav` (chrome/apps/nav.ts) is what the top
 *  bar renders — the user's answer. They are separate because one is state and
 *  the other is product copy, and the labels deliberately differ from the ids
 *  ("Live" drives `seer`, "Transcripts" drives `sessions`, "Topics" drives
 *  `snapshot`). Nothing in the type system ties the two lists together, so
 *  without this test the failure mode is silent in both directions: a pill for
 *  a page `setPage` rejects is a nav button that does nothing when clicked,
 *  and a page in APP_PAGES with no pill is reachable only by hand-editing the
 *  hash.
 *
 *  Also pinned: the eight pages are partitioned, not merely covered. Every page
 *  belongs to exactly one instrument — the whole point of the split is that
 *  neither app can render the other's pages, and a page listed under both
 *  would put it back in the shared bundle by way of both renderPage switches.
 *
 *  This imports chrome/apps/nav.ts, which is data-only on purpose (see its
 *  header): importing a module that pulled in page components would drag
 *  three.js into a Node test run. */

import { afterEach, describe, expect, it } from "vitest";
import { APP_PAGES, appStore, type AppId, type Page } from "../../src/app/store";
import { APP_CHROME, defaultPage } from "../../src/chrome/apps/nav";

const APPS: AppId[] = ["nebulai", "seer"];
const ALL_PAGES: Page[] = [
  "map",
  "behavior",
  "snapshot",
  "interp",
  "guide",
  "sessions",
  "keyword",
  "seer",
];

describe("app ↔ nav agreement", () => {
  for (const id of APPS) {
    it(`${id}: the pills are exactly APP_PAGES, in the same order`, () => {
      expect(APP_CHROME[id].nav.map((n) => n.page)).toEqual([...APP_PAGES[id]]);
    });

    it(`${id}: boots on its first pill`, () => {
      expect(defaultPage(id)).toBe(APP_CHROME[id].nav[0]!.page);
    });

    it(`${id}: every pill has a non-empty label`, () => {
      for (const item of APP_CHROME[id].nav) expect(item.label.trim()).not.toBe("");
    });
  }

  it("the seven pages are partitioned across the two instruments", () => {
    const owned = APPS.flatMap((id) => [...APP_PAGES[id]]);
    expect(new Set(owned).size).toBe(owned.length); // no page owned twice
    expect([...owned].sort()).toEqual([...ALL_PAGES].sort()); // none orphaned
  });

  it("each instrument links to the other one's entry, not its own", () => {
    expect(APP_CHROME.nebulai.sibling.href).toBe("./seer.html");
    expect(APP_CHROME.seer.sibling.href).toBe("./index.html");
  });

  it("Seer's branding does not claim to be nebul.ai", () => {
    const word = APP_CHROME.seer.wordmark.head + APP_CHROME.seer.wordmark.tail;
    expect(word.toLowerCase()).not.toContain("nebul");
    expect(APP_CHROME.seer.documentTitle.toLowerCase()).not.toContain("nebul");
    // …down to the mark: the nebula is Nebulai's, not a generic app icon
    expect(APP_CHROME.seer.mark).not.toBe(APP_CHROME.nebulai.mark);
  });

  /*  The Behavior page reads a different artifact and makes a different kind
   *  of claim from every other Nebulai page — weight geometry on one side, what
   *  deployments actually said on the other. Folding it into the map as a view
   *  mode would let a reader carry a causal reading across that line, so it is
   *  pinned as its own page with its own pill. */
  it("Behavior is Nebulai's own page and is not owned by Seer", () => {
    expect(APP_PAGES.nebulai).toContain("behavior");
    expect(APP_PAGES.seer).not.toContain("behavior");
    const pill = APP_CHROME.nebulai.nav.find((n) => n.page === "behavior");
    expect(pill?.label).toBe("Behavior");
  });

  it("each instrument's document title and tagline are its own", () => {
    expect(APP_CHROME.nebulai.documentTitle).not.toBe(APP_CHROME.seer.documentTitle);
    expect(APP_CHROME.nebulai.tagline).not.toBe(APP_CHROME.seer.tagline);
  });
});

/*  `appStore` is a module singleton, but vitest gives each test FILE its own
 *  module registry, so mutating `app` here cannot leak into another suite.
 *  Within this file the cases restore it themselves via `setApp`. */
describe("setPage refuses the other instrument's pages", () => {
  afterEach(() => appStore.getState().setApp("nebulai"));

  it("nebulai can reach its own four and none of Seer's", () => {
    appStore.getState().setApp("nebulai");
    expect(appStore.getState().page).toBe("map");
    for (const page of APP_PAGES.nebulai) {
      appStore.getState().setPage(page);
      expect(appStore.getState().page).toBe(page);
    }
    const before = appStore.getState().page;
    for (const page of APP_PAGES.seer) appStore.getState().setPage(page);
    expect(appStore.getState().page).toBe(before);
  });

  it("seer boots on Live and cannot be navigated to the map", () => {
    appStore.getState().setApp("seer");
    expect(appStore.getState().page).toBe("seer");
    for (const page of APP_PAGES.seer) {
      appStore.getState().setPage(page);
      expect(appStore.getState().page).toBe(page);
    }
    const before = appStore.getState().page;
    for (const page of APP_PAGES.nebulai) appStore.getState().setPage(page);
    expect(appStore.getState().page).toBe(before);
  });
});

describe("NebulAI experience navigation", () => {
  it("every experience pill drives a real NebulAI page, and together they reach all four", async () => {
    const { NEBULAI_EXPERIENCES } = await import("../../src/chrome/apps/nav");
    const { APP_PAGES } = await import("../../src/app/store");
    const reached = new Set<string>();
    for (const exp of Object.values(NEBULAI_EXPERIENCES)) {
      expect(exp.nav.length, exp.id).toBeGreaterThan(0);
      for (const item of exp.nav) {
        expect(APP_PAGES.nebulai, `${exp.id}:${item.label}`).toContain(item.page);
        reached.add(item.page);
      }
    }
    expect([...reached].sort()).toEqual([...APP_PAGES.nebulai].sort());
  });
});
