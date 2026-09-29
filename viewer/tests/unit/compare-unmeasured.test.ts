/** Compare's concept-overlap readout has to distinguish an overlap of zero from
 *  an overlap that was never measurable.
 *
 *  The exporter sends `null` for any pair involving a map whose cluster titles
 *  are placeholders (`--labels none` neuron and SAE maps): such a map has no
 *  concept set, so there is nothing to intersect. Rendering that as `0` would
 *  state "these two models share no concepts", which is a finding the data does
 *  not support — and it is the exact failure the honesty rule `missing != 0`
 *  exists to stop.
 *
 *  ComparePanel is a Preact component reading a global signal, so these assert
 *  on the source's branching the way `behavior-data.test.ts` does for
 *  BehaviorPage: the point is that the null case is handled *before* the value
 *  is printed, which is checkable without a DOM. */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const strip = (s: string) =>
  s.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

const panel = strip(
  readFileSync(
    join(import.meta.dirname, "..", "..", "src", "chrome", "ComparePanel.tsx"),
    "utf8",
  ),
);
const types = strip(
  readFileSync(
    join(import.meta.dirname, "..", "..", "src", "data", "compare.ts"),
    "utf8",
  ),
);

describe("compare concept overlap: unmeasured is not zero", () => {
  it("types the jaccard map as nullable", () => {
    expect(types).toMatch(/jaccard:\s*Record<string,\s*number \| null>/);
  });

  it("carries the exporter's list of placeholder-titled maps", () => {
    expect(types).toMatch(/unnamed_models\?:\s*string\[\]/);
    expect(types).toMatch(/unnamed_reason\?:\s*string \| null/);
  });

  it("branches on null before printing the value", () => {
    const cell = panel.slice(panel.indexOf("data.stats.jaccard"));
    const branch = cell.indexOf("v === null");
    const print = cell.indexOf("{v === null ? ");
    expect(branch).toBeGreaterThan(-1);
    expect(print).toBeGreaterThan(-1);
    // the only place the raw value reaches the DOM is the false arm of that
    // ternary — there is no unguarded `{v}`
    expect(cell).not.toMatch(/<dd[^>]*>\s*\{v\}\s*<\/dd>/);
  });

  it("renders the refusal as words, not as a number", () => {
    expect(panel).toContain('"not measured"');
    expect(panel).not.toMatch(/v === null \?\s*0/);
  });

  it("names the maps whose overlap could not be measured", () => {
    expect(panel).toContain("unnamed_models");
    expect(panel).toContain("placeholder cluster titles");
    // and shows the exporter's own reason rather than inventing one
    expect(panel).toContain("unnamed_reason");
  });

  it("guards the note on an artifact that predates the field", () => {
    // older compare.json has no unnamed_models at all; the note must not render
    // an empty list or crash on undefined
    expect(panel).toMatch(/unnamed_models\?\.length \?\? 0\) > 0/);
  });
});
