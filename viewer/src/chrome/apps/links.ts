/** apps/links.ts — build-time URLs of the other documents on this site.
 *
 *  Split out of apps/nav.ts so the NebulAI root chooser (src/chooser-main.ts)
 *  can name Seer and the psychiX hub without importing the store. See nav.ts
 *  for why these arrive as env vars and why VITE_SEER_APP_URL is not
 *  VITE_SEER_URL. */

export const SEER_APP_URL: string = import.meta.env.VITE_SEER_APP_URL || "./seer.html";
export const NEBULAI_APP_URL: string = import.meta.env.VITE_NEBULAI_APP_URL || "./index.html";
/** Empty = not part of a hub deploy; no hub link is drawn at all. */
export const HUB_URL: string = import.meta.env.VITE_HUB_URL || "";
