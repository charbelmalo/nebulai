import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  renderOwnershipMarkdown,
  SETTINGS_CONTEXTS,
  SETTINGS_OWNERSHIP,
  SETTINGS_TABS,
  ownersOf,
  owns,
  tabsFor,
} from "../../src/chrome/settingsScope";

describe("Settings ownership", () => {
  it("every tab has at least one owner, and every option still has a home", () => {
    for (const t of SETTINGS_TABS) expect(ownersOf(t).length, t).toBeGreaterThan(0);
    for (const [path, owners] of Object.entries(SETTINGS_OWNERSHIP)) {
      expect(owners.length, path).toBeGreaterThan(0);
      // a child can only be owned by contexts that can reach its parent
      for (const ctx of owners) {
        const parent = path.split("/").slice(0, -1).join("/");
        if (parent) expect(ownersOf(parent), `${path} → ${ctx}`).toContain(ctx);
      }
    }
  });

  it("keeps Seer's options in Seer and NebulAI's in NebulAI", () => {
    expect(tabsFor("seer")).toEqual(["General", "Appearance", "Model Probing", "Snapshot", "Sessions", "About"]);
    expect(owns("seer", "Model Probing/Endpoint/SessionSeer server")).toBe(true);
    expect(owns("seer", "Model Probing/Endpoint/Naming chain")).toBe(false);
    expect(owns("seer", "Appearance/atlas")).toBe(false);
    for (const e of ["learn", "atlas", "research"] as const) {
      expect(owns(e, "Snapshot")).toBe(false);
      expect(owns(e, "Sessions")).toBe(false);
      expect(owns(e, "Appearance/sessions")).toBe(false);
      expect(owns(e, "Model Probing/Endpoint/SessionSeer server")).toBe(false);
    }
  });

  it("Learn offers display and accessibility preferences only", () => {
    expect(tabsFor("learn")).toEqual(["General", "About"]);
    expect(owns("learn", "General/Chrome")).toBe(true);
    expect(owns("learn", "General/Rendering")).toBe(false);
    expect(owns("learn", "General/Hand control")).toBe(false);
  });

  it("Atlas offers exploration settings, Research its advanced analysis settings", () => {
    expect(tabsFor("atlas")).toEqual(["General", "Appearance", "Model Probing", "Data", "About"]);
    expect(owns("atlas", "Appearance/chord")).toBe(false);
    expect(owns("atlas", "Data/Dataset & view/View type")).toBe(false);
    expect(tabsFor("research")).toEqual(["General", "Appearance", "Behavior", "Model Probing", "Data", "About"]);
    expect(owns("research", "Appearance/compare")).toBe(true);
    expect(owns("research", "Data/Dataset & view/View type")).toBe(true);
    expect(owns("research", "Model Probing/Endpoint/Live nebula server")).toBe(true);
  });

  it("an unlisted row inherits its section's owners", () => {
    for (const ctx of SETTINGS_CONTEXTS) {
      expect(owns(ctx, "General/Chrome/Theme")).toBe(true);
      expect(owns(ctx, "About/Provenance")).toBe(true);
    }
  });
});

describe("docs/SETTINGS-OWNERSHIP.md", () => {
  it("matches the ownership table (regenerate with npm run docs:settings)", () => {
    const doc = readFileSync(join(__dirname, "../../../docs/SETTINGS-OWNERSHIP.md"), "utf8");
    expect(doc).toBe(renderOwnershipMarkdown());
  });
});
