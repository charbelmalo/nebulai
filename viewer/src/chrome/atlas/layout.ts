/** atlas/layout.ts — the one breakpoint the Atlas workspace switches on.
 *  Below it the map and the results cannot share the screen, so they become
 *  two panels behind a Map / Results switch; the inspector goes full width. */

import { signal } from "@preact/signals";

export const NARROW_QUERY = "(max-width: 1023px)";

function query(): MediaQueryList | null {
  try {
    return typeof matchMedia === "function" ? matchMedia(NARROW_QUERY) : null;
  } catch {
    return null;
  }
}

const mq = query();
export const $narrow = signal(mq?.matches ?? false);
mq?.addEventListener?.("change", (e) => ($narrow.value = e.matches));
