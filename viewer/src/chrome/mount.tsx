/** Mount the Preact chrome into #chrome. Called once from main.ts after the
 *  first dataset lands (the boot pill handles everything before that). */

import { render } from "preact";
import { Sidebar } from "./Sidebar";
import { ComparePanel } from "./ComparePanel";
import { LegendCard } from "./LegendCard";
import { TopBar } from "./TopBar";
import { $viewMode } from "./state";

function Chrome() {
  return (
    <>
      <TopBar />
      <Sidebar />
      {$viewMode.value === "compare" ? <ComparePanel /> : <LegendCard />}
    </>
  );
}

export function mountChrome(container: HTMLElement): void {
  const root = document.createElement("div");
  root.className = "chrome-root";
  container.appendChild(root);
  render(<Chrome />, root);
}
