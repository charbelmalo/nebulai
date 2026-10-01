/** settingsScope.ts — who owns each Settings tab, section and scoped row.
 *
 *  SettingsPage stays the one canonical home for every preference; this table
 *  decides which of them a given context shows. A context is Seer, or one of
 *  NebulAI's three experiences. Paths are "Tab", "Tab/Section" or
 *  "Tab/Section/Row"; a path with no entry inherits its parent's owners, so
 *  only the exceptions are written down. Nothing here changes a stored value
 *  or clears storage — a hidden option keeps its value and is still honoured
 *  wherever it applies.
 *
 *  Data only (no store, no DOM) so tests/unit/settings-scope.test.ts can pin
 *  it, and docs/SETTINGS-OWNERSHIP.md is generated from the same facts. */

import type { Experience } from "../app/experience";

export type SettingsContext = "seer" | Experience;

export const SETTINGS_CONTEXTS: readonly SettingsContext[] = ["seer", "learn", "atlas", "research"];

export const SETTINGS_TABS = [
  "General",
  "Appearance",
  "Behavior",
  "Model Probing",
  "Snapshot",
  "Sessions",
  "Data",
  "About",
] as const;
export type SettingsTab = (typeof SETTINGS_TABS)[number];

const ALL = SETTINGS_CONTEXTS;
const MAPS: readonly SettingsContext[] = ["atlas", "research"];

export const SETTINGS_OWNERSHIP: Readonly<Record<string, readonly SettingsContext[]>> = {
  General: ALL,
  // theme, reduced motion, animation speed: accessibility/display for all
  "General/Chrome": ALL,
  // highlights one unit across Internals views: Research, and Learn's episodes
  "General/Chrome/Cross-view linking": ["learn", "research"],
  // point scale, confidence floor, label density, bloom — map presentation
  "General/Rendering": MAPS,
  "General/Hand control": MAPS,

  Appearance: ["seer", "atlas", "research"],
  "Appearance/atlas": MAPS,
  "Appearance/chord": ["research"],
  "Appearance/hierarchy": ["research"],
  "Appearance/compare": ["research"],
  "Appearance/sessions": ["seer"],

  Behavior: ["research"],

  "Model Probing": ["seer", "atlas", "research"],
  "Model Probing/Map builder": MAPS,
  "Model Probing/Endpoint": ["seer", "atlas", "research"],
  "Model Probing/Endpoint/Naming chain": MAPS,
  "Model Probing/Endpoint/Live nebula server": ["research"],
  "Model Probing/Endpoint/SessionSeer server": ["seer"],
  "Model Probing/Live probing": MAPS,
  "Model Probing/Progress": MAPS,

  Snapshot: ["seer"],
  Sessions: ["seer"],

  Data: MAPS,
  "Data/Dataset & view": MAPS,
  // Atlas hosts the atlas map only; the other view types are Research's
  "Data/Dataset & view/View type": ["research"],

  About: ALL,
};

/** Owners of `path`: its own entry, else the nearest ancestor's. */
export function ownersOf(path: string): readonly SettingsContext[] {
  let p = path;
  for (;;) {
    const hit = SETTINGS_OWNERSHIP[p];
    if (hit) return hit;
    const cut = p.lastIndexOf("/");
    if (cut < 0) return [];
    p = p.slice(0, cut);
  }
}

/** Does `ctx` show `path`? A child is shown only if every ancestor is too. */
export function owns(ctx: SettingsContext, path: string): boolean {
  const parts = path.split("/");
  for (let i = 1; i <= parts.length; i++) {
    if (!ownersOf(parts.slice(0, i).join("/")).includes(ctx)) return false;
  }
  return true;
}

export function tabsFor(ctx: SettingsContext): SettingsTab[] {
  return SETTINGS_TABS.filter((t) => owns(ctx, t));
}

const CONTEXT_LABEL: Record<SettingsContext, string> = {
  seer: "Seer",
  learn: "Learn",
  atlas: "Atlas",
  research: "Research",
};

/** docs/SETTINGS-OWNERSHIP.md, rendered from the table above
 *  (`npm run docs:settings`; a unit test fails when the file drifts). */
export function renderOwnershipMarkdown(): string {
  const head = ["Setting", ...SETTINGS_CONTEXTS.map((c) => CONTEXT_LABEL[c])];
  const rows = Object.keys(SETTINGS_OWNERSHIP).map((path) => {
    const depth = path.split("/").length - 1;
    const name = path.split("/").pop() as string;
    const label = depth === 0 ? `**${name}**` : `${"  ".repeat(depth)}${name}`;
    return [label, ...SETTINGS_CONTEXTS.map((c) => (owns(c, path) ? "shown" : "–"))];
  });
  const line = (cells: string[]) => `| ${cells.join(" | ")} |`;
  return [
    "# Settings ownership",
    "",
    "Generated from `viewer/src/chrome/settingsScope.ts` by `npm run docs:settings`. Do not edit by hand.",
    "",
    "Settings is the one canonical home for every preference. Each context shows only the tabs, sections and rows it owns. A hidden option keeps its stored value and still applies wherever it is used. Rows not listed inherit their section's owners.",
    "",
    line(head),
    line(head.map(() => "---")),
    ...rows.map(line),
    "",
  ].join("\n");
}
