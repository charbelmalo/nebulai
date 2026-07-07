/** TS mirror of tokens.css for GPU-side use (ramp textures, clear color).
 *  tests/unit/tokens-sync.test.ts asserts these match the CSS custom props. */

export const BG = "#150f17";
export const BG_RAISE = "#1d1420";
export const BG_PILL = "#221826";
export const TEXT = "#efe7f2";
export const TEXT_DIM = "#9d8fa6";

/** 5-stop connection ramp, low→high weight. */
export const RAMP = ["#f5c33b", "#f0863a", "#ea4f86", "#e33bd0", "#8b3bf0"] as const;

export function hexToRgb01(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

/** Sample the ramp at t ∈ [0,1] (linear RGB-space lerp between stops). */
export function rampColor(t: number): [number, number, number] {
  const x = Math.min(Math.max(t, 0), 1) * (RAMP.length - 1);
  const i = Math.min(Math.floor(x), RAMP.length - 2);
  const f = x - i;
  const a = hexToRgb01(RAMP[i]!);
  const b = hexToRgb01(RAMP[i + 1]!);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** 256×1 RGBA ramp for GPU textures (three DataTexture / WGSL sampled). */
export function rampTextureData(width = 256): Uint8Array {
  const data = new Uint8Array(width * 4);
  for (let i = 0; i < width; i++) {
    const [r, g, b] = rampColor(i / (width - 1));
    data[i * 4] = Math.round(r * 255);
    data[i * 4 + 1] = Math.round(g * 255);
    data[i * 4 + 2] = Math.round(b * 255);
    data[i * 4 + 3] = 255;
  }
  return data;
}
