/** Mount the Preact chrome into #chrome. Called once from main.ts after the
 *  first dataset lands (the boot pill handles everything before that). */

import { effect } from "@preact/signals";
import { render } from "preact";
import { Sidebar } from "./Sidebar";
import { ComparePanel } from "./ComparePanel";
import { LegendCard } from "./LegendCard";
import { SettingsPage } from "./SettingsPage";
import { SnapshotMap } from "./SnapshotMap";
import { TopBar } from "./TopBar";
import { $page, $viewMode } from "./state";

// Toggle a body class so main.ts's #stage can be hidden on the snapshot page.
effect(() => {
  document.body.classList.toggle("page-snapshot", $page.value === "snapshot");
});

function Chrome() {
  const onMap = $page.value === "map";
  return (
    <>
      <TopBar />
      {onMap && <Sidebar />}
      {onMap && ($viewMode.value === "compare" ? <ComparePanel /> : <LegendCard />)}
      {!onMap && <SnapshotMap />}
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
