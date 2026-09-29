/** chooser-main.ts — the NebulAI root chooser's only script.
 *
 *  The chooser itself is static HTML (index.html). This script does two small
 *  things and nothing else — it imports no store, no loader, no renderer:
 *
 *   1. Forward a legacy deep link. Before the three experiences existed, every
 *      NebulAI link pointed at this address (`#page=interp&feature=…`,
 *      `#model=…`, `?view=compare`, `#episode=…`). Those links now belong to an
 *      experience; `resolveExperience` picks which one, and the browser is
 *      sent there with the search and hash untouched. A root visit with no
 *      recognised intent stays on the chooser.
 *   2. Point "Other tools" at the build's configured Seer and psychiX URLs. */

import "@psychix/viz/tokens.css";
import "./styles/chooser.css";
import { resolveExperience, type RouteIntent } from "./app/experience";
import { HUB_URL, SEER_APP_URL } from "./chrome/apps/links";

const PAGES = ["map", "behavior", "interp", "guide"] as const;
const VIEWS = ["atlas", "chord", "hierarchy", "compare"] as const;
/** keys that pin one unit of one artifact (see data/finding.ts) */
const PIN_KEYS = ["artifact", "unit_index", "point"];

function intentFrom(hash: URLSearchParams, search: URLSearchParams): RouteIntent {
  const page = hash.get("page");
  const view = hash.get("view") ?? search.get("view");
  return {
    page: (PAGES as readonly string[]).includes(page ?? "")
      ? (page as RouteIntent["page"])
      : undefined,
    view: (VIEWS as readonly string[]).includes(view ?? "")
      ? (view as RouteIntent["view"])
      : undefined,
    // the destination validates the id against its registry; here presence is
    // enough to know the link was a guided one
    episode: !!hash.get("episode"),
    lesson: !!hash.get("lesson"),
    model: !!hash.get("model"),
    finding: PIN_KEYS.some((k) => hash.has(k)),
  };
}

const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
const search = new URLSearchParams(location.search);
const { experience } = resolveExperience({
  entry: null,
  explicit: hash.get("experience"),
  intent: intentFrom(hash, search),
});

if (experience) {
  // relative to this document, so it works at any deploy base
  location.replace(`${experience}/${location.search}${location.hash}`);
} else {
  const tools = document.getElementById("chooser-tools");
  const seer = tools?.querySelector<HTMLAnchorElement>('a[data-tool="seer"]');
  if (seer) seer.href = SEER_APP_URL;
  if (tools && HUB_URL) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = HUB_URL;
    a.textContent = "psychiX ";
    const arrow = document.createElement("span");
    arrow.setAttribute("aria-hidden", "true");
    arrow.textContent = "↗";
    a.append(arrow);
    li.append(a, " — every instrument on this site");
    tools.append(li);
  }
}
