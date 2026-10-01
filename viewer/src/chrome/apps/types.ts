/** apps/types.ts — the contract `mountChrome` is written against.
 *
 *  An `AppShell` is an `AppChrome` (nav data, see apps/nav.ts) plus the one
 *  thing that cannot be data: which components a page id renders to. Keeping
 *  `renderPage` behind this interface is the entire bundle split — mount.tsx
 *  imports only the TYPE, so neither instrument's page components can reach
 *  the other's entry through the shared mount. */

import type { ComponentChildren } from "preact";
import type { Page } from "../../app/store";
import type { AppChrome } from "./nav";

export interface AppShell extends AppChrome {
  /** Everything this instrument draws for `page`, including any page-specific
   *  panels. Returns `null` for a page it does not own — which `setPage`
   *  already makes unreachable (see APP_PAGES), so it is a belt-and-braces
   *  default rather than a state the UI can get into. */
  renderPage(page: Page): ComponentChildren;
  /** Optional: replaces the default pill nav + cross links (NebulAI's
   *  per-experience navigation). Absent on Seer, which keeps the default. */
  renderTopNav?(): ComponentChildren;
  /** Optional: drawn right after the wordmark (NebulAI's experience chip). */
  renderBrandExtras?(): ComponentChildren;
  /** Optional: a banner under the top bar (NebulAI's context notice). */
  renderBanner?(): ComponentChildren;
  /** Optional: where the wordmark links (NebulAI's root chooser). */
  homeHref?: string;
}
