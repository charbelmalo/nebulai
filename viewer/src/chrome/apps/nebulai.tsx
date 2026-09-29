/** apps/nebulai.tsx — Nebulai's page set: Semantic map · Behavior · Internals
 *  · Guide.
 *
 *  This module is imported by `src/main.ts` and by nothing else. It is the
 *  only place the atlas-side components are named, which is what keeps them
 *  out of Seer's bundle: `seer.html` never reaches this file, so InterpPage
 *  (and through it all 25 interp drivers), the Sidebar, the legend and the
 *  search panel are not in its graph at all. */

import { AxisRail } from "../AxisRail";
import { BehaviorPage } from "../BehaviorPage";
import { ComparePanel, CompareTransport } from "../ComparePanel";
import { GuidePage } from "../GuidePage";
import { HandRig } from "../HandRig";
import { InterpPage } from "../InterpPage";
import { LegendCard } from "../LegendCard";
import { SearchPanel } from "../SearchPanel";
import { Sidebar } from "../Sidebar";
import { APP_ROOT } from "../../data/base";
import { ExperienceChip, ExperienceNav, ExperienceNotice } from "../ExperienceNav";
import { MapChooser } from "../MapChooser";
import { AtlasPanels } from "../atlas/AtlasPanels";
import { LearnMapPanels } from "../learn/LessonPanel";
import { $datasetId, $experience, $tour, $viewMode } from "../state";
import { APP_CHROME, NEBULAI_EXPERIENCES, type SiblingLink } from "./nav";
import type { AppShell } from "./types";

/** The map page is not one component: it is the driver stage (owned by
 *  main.ts, outside Preact) plus a set of floating panels whose visibility
 *  depends on the active view mode. That composition lives here rather than in
 *  mount.tsx because it is Nebulai's, not the shell's. */
function MapPanels() {
  const view = $viewMode.value;
  // Learn's map page exists only inside a lesson or a guided episode: the
  // lesson stage (task, evidence, unit) or the episode's step controls.
  if ($experience.value === "learn" && $tour.value) return <LearnMapPanels />;
  // Research asks for its map by name: until one is on screen, the page is
  // the explicit chooser rather than an empty stage with a settings panel.
  if ($experience.value === "research" && $datasetId.value === null) return <MapChooser />;
  // Atlas's own workspace: search → inspect → save → reopen. The advanced
  // views live in Research, so Atlas is always the plain atlas view.
  if ($experience.value === "atlas" && view === "atlas") return <AtlasPanels />;
  return (
    <>
      <Sidebar />
      {view === "compare" ? <ComparePanel /> : <LegendCard />}
      {view === "compare" && <CompareTransport />}
      {view === "atlas" && <SearchPanel />}
      {/* The axis rail renders nothing at all for a map with no
          `directions.json` — it is not a disabled panel, it is absent. */}
      {view === "atlas" && <AxisRail />}
      {/* Hand control steers the AtlasDriver and nothing else, so the console
          appears with the view it can actually drive. */}
      {view === "atlas" && <HandRig />}
    </>
  );
}

/** Other tools (Seer, psychiX). Their configured URLs are written relative
 *  to the app root (`./seer.html` in the combined build) or absolute (per-app
 *  deploys); nested entries resolve them against the ROOT, never against
 *  `learn/` or `atlas/`, where `./seer.html` would 404. */
const TOOLS: SiblingLink[] = [
  APP_CHROME.nebulai.sibling,
  ...(APP_CHROME.nebulai.hub ? [APP_CHROME.nebulai.hub] : []),
].map((l) => ({ ...l, href: new URL(l.href, APP_ROOT).href }));

export const NEBULAI_APP: AppShell = {
  ...APP_CHROME.nebulai,
  homeHref: APP_ROOT,
  renderTopNav: () => <ExperienceNav experiences={NEBULAI_EXPERIENCES} tools={TOOLS} />,
  renderBrandExtras: () => <ExperienceChip experiences={NEBULAI_EXPERIENCES} />,
  renderBanner: () => <ExperienceNotice />,
  renderPage(page) {
    switch (page) {
      case "map":
        return <MapPanels />;
      case "behavior":
        return <BehaviorPage />;
      case "interp":
        return <InterpPage />;
      case "guide":
        return <GuidePage />;
      default:
        return null;
    }
  },
};
