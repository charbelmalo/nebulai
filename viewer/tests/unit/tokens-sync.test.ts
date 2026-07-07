/** tokens.css ↔ tokens.ts drift guard: the CSS custom props are the source of
 *  truth; the TS mirror feeds the GPU. If either side changes alone, this
 *  fails before a human has to notice a color mismatch. */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import * as tokens from "../../src/styles/tokens";

const css = readFileSync(
  new URL("../../src/styles/tokens.css", import.meta.url),
  "utf8",
);

function cssVar(name: string): string {
  const m = css.match(new RegExp(`--${name}:\\s*([^;]+);`));
  if (!m) throw new Error(`--${name} missing from tokens.css`);
  return m[1]!.trim();
}

describe("tokens.css ↔ tokens.ts", () => {
  it("mirrors the base colors", () => {
    expect(cssVar("bg")).toBe(tokens.BG);
    expect(cssVar("bg-raise")).toBe(tokens.BG_RAISE);
    expect(cssVar("bg-pill")).toBe(tokens.BG_PILL);
    expect(cssVar("text")).toBe(tokens.TEXT);
    expect(cssVar("text-dim")).toBe(tokens.TEXT_DIM);
  });

  it("mirrors all 5 ramp stops in order", () => {
    tokens.RAMP.forEach((hex, i) => {
      expect(cssVar(`ramp-${i}`)).toBe(hex);
    });
  });

  it("builds a well-formed GPU ramp texture", () => {
    const data = tokens.rampTextureData(256);
    expect(data).toHaveLength(256 * 4);
    // endpoints are exactly the first/last stops
    const [r0, g0, b0] = tokens.hexToRgb01(tokens.RAMP[0]);
    expect(data[0]).toBe(Math.round(r0 * 255));
    expect(data[1]).toBe(Math.round(g0 * 255));
    expect(data[2]).toBe(Math.round(b0 * 255));
    const last = 255 * 4;
    const [r4, g4, b4] = tokens.hexToRgb01(tokens.RAMP[4]);
    expect(data[last]).toBe(Math.round(r4 * 255));
    expect(data[last + 1]).toBe(Math.round(g4 * 255));
    expect(data[last + 2]).toBe(Math.round(b4 * 255));
    // fully opaque
    for (let i = 3; i < data.length; i += 4) expect(data[i]).toBe(255);
  });
});
