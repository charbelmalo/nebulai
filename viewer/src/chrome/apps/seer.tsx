/** apps/seer.tsx — Seer's page set: Live · Transcripts · Topics.
 *
 *  Imported by `src/seer-main.ts` and by nothing else. The mirror image of
 *  apps/nebulai.tsx: naming SeerPage / SessionsPage / SnapshotMap here is what
 *  keeps them out of Nebulai's bundle, and not naming AtlasDriver anywhere in
 *  this graph is what lets `seer.html` boot with no atlas artifacts at all.
 *
 *  The page IDS are unchanged ("seer", "sessions", "snapshot") even though the
 *  pills now read Live / Transcripts / Topics — ids are wire format
 *  (permalinks, store, e2e), labels are product. See apps/nav.ts. */

import { SeerPage } from "../SeerPage";
import { SessionsPage } from "../SessionsPage";
import { SnapshotMap } from "../SnapshotMap";
import { APP_CHROME } from "./nav";
import type { AppShell } from "./types";

export const SEER_APP: AppShell = {
  ...APP_CHROME.seer,
  renderPage(page) {
    switch (page) {
      case "seer":
        return <SeerPage />;
      case "sessions":
        return <SessionsPage />;
      case "snapshot":
        return <SnapshotMap />;
      default:
        return null;
    }
  },
};
