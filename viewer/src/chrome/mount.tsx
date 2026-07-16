/** Mount the Preact chrome into #chrome. Called once from main.ts after the
 *  first dataset lands (the boot pill handles everything before that). */

import { effect } from "@preact/signals";
import { render } from "preact";
import { Sidebar } from "./Sidebar";
import { ComparePanel } from "./ComparePanel";
import { GuidePage } from "./GuidePage";
import { InterpPage } from "./InterpPage";
import { LegendCard } from "./LegendCard";
import { SearchPanel } from "./SearchPanel";
import { SessionsPage } from "./SessionsPage";
import { SettingsPage } from "./SettingsPage";
import { SnapshotMap } from "./SnapshotMap";
import { TopBar } from "./TopBar";
import { $page, $viewMode } from "./state";

// Toggle body classes so main.ts's #stage is hidden on the non-map pages (each
// non-map page owns its own canvas/DOM and the driver stage must not show through).
effect(() => {
  const page = $page.value;
  document.body.classList.toggle("page-snapshot", page === "snapshot");
  document.body.classList.toggle("page-interp", page === "interp");
  document.body.classList.toggle("page-guide", page === "guide");
  document.body.classList.toggle("page-sessions", page === "sessions");
});

function Chrome() {
  const page = $page.value;
  const onMap = page === "map";
  return (
    <>
      <TopBar />
      {onMap && <Sidebar />}
      {onMap && ($viewMode.value === "compare" ? <ComparePanel /> : <LegendCard />)}
      {onMap && $viewMode.value === "atlas" && <SearchPanel />}
      {page === "snapshot" && <SnapshotMap />}
      {page === "interp" && <InterpPage />}
      {page === "guide" && <GuidePage />}
      {page === "sessions" && <SessionsPage />}
      <SettingsPage />
    </>
  );
}

export function mountChrome(container: HTMLElement): void {
  const root = document.createElement("div");
  root.className = "chrome-root";
  container.appendChild(root);
  render(<Chrome />, root);
}
