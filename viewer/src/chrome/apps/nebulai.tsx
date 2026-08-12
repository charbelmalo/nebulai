/** apps/nebulai.tsx — Nebulai's page set: Semantic map · Internals · Guide.
 *
 *  This module is imported by `src/main.ts` and by nothing else. It is the
 *  only place the atlas-side components are named, which is what keeps them
 *  out of Seer's bundle: `seer.html` never reaches this file, so InterpPage
 *  (and through it all 25 interp drivers), the Sidebar, the legend and the
 *  search panel are not in its graph at all. */

import { ComparePanel, CompareTransport } from "../ComparePanel";
import { GuidePage } from "../GuidePage";
import { InterpPage } from "../InterpPage";
import { LegendCard } from "../LegendCard";
import { SearchPanel } from "../SearchPanel";
import { Sidebar } from "../Sidebar";
import { $viewMode } from "../state";
import { APP_CHROME } from "./nav";
import type { AppShell } from "./types";

/** The map page is not one component: it is the driver stage (owned by
 *  main.ts, outside Preact) plus a set of floating panels whose visibility
 *  depends on the active view mode. That composition lives here rather than in
 *  mount.tsx because it is Nebulai's, not the shell's. */
function MapPanels() {
  const view = $viewMode.value;
  return (
    <>
      <Sidebar />
      {view === "compare" ? <ComparePanel /> : <LegendCard />}
      {view === "compare" && <CompareTransport />}
      {view === "atlas" && <SearchPanel />}
    </>
  );
}

export const NEBULAI_APP: AppShell = {
  ...APP_CHROME.nebulai,
  renderPage(page) {
    switch (page) {
      case "map":
        return <MapPanels />;
      case "interp":
        return <InterpPage />;
      case "guide":
        return <GuidePage />;
      default:
        return null;
    }
  },
};
