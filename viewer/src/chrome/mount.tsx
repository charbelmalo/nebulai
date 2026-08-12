/** Mount the Preact chrome into #chrome. Called once per document from
 *  `app/boot-shell.ts`, before any dataset is asked for — the shell is
 *  deliberately independent of the atlas, since Seer shares it and runs with
 *  no atlas artifacts at all. The boot pill survives the mount: we append our
 *  own root beside it rather than replacing #chrome's children, so it keeps
 *  reporting load progress and the MetaLine underneath the chrome.
 *
 *  This module is SHARED by both instruments, and the whole reason it takes an
 *  `AppShell` instead of importing pages itself is bundling: it names TopBar
 *  and SettingsPage (genuinely shared) and nothing else. Every page component
 *  arrives through `app.renderPage`, so Nebulai's entry pulls in the atlas
 *  pages, Seer's pulls in the agent-run pages, and neither reaches the other's
 *  — a static `import { SeerPage }` here would silently undo that. */

import { effect } from "@preact/signals";
import { render } from "preact";
import { SettingsPage } from "./SettingsPage";
import { TopBar } from "./TopBar";
import type { AppShell } from "./apps/types";
import { $page } from "./state";

/** Toggle body classes so main.ts's #stage is hidden on the non-map pages
 *  (each non-map page owns its own canvas/DOM and the driver stage must not
 *  show through). Harmless on Seer's document, which has no #stage at all —
 *  the classes still drive page-scoped CSS. */
function trackPageClasses(): void {
  effect(() => {
    const page = $page.value;
    document.body.classList.toggle("page-snapshot", page === "snapshot");
    document.body.classList.toggle("page-interp", page === "interp");
    document.body.classList.toggle("page-guide", page === "guide");
    document.body.classList.toggle("page-sessions", page === "sessions");
    document.body.classList.toggle("page-seer", page === "seer");
  });
}

function Chrome({ app }: { app: AppShell }) {
  return (
    <>
      <TopBar app={app} />
      {app.renderPage($page.value)}
      <SettingsPage />
    </>
  );
}

export function mountChrome(container: HTMLElement, app: AppShell): void {
  document.body.classList.add(`app-${app.id}`);
  trackPageClasses();
  const root = document.createElement("div");
  root.className = "chrome-root";
  container.appendChild(root);
  render(<Chrome app={app} />, root);
}
