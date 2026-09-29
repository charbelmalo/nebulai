/** The encoding decisions behind #26, and the one rule the whole figure rests
 *  on: α = 0 is a CONTROL, not an absent measurement.
 *
 *  The driver is WebGPU-bound and cannot run here; that is precisely why its
 *  arithmetic lives in `scene/interp/steer.ts`. What is checked below is what
 *  the view actually claims:
 *
 *    · the control is drawn, at zero, with a visible plate and its own colour;
 *    · height and colour read ONE normalization, so they cannot disagree;
 *    · an absent control and a failed control are different states, and only
 *      one of them licenses the figure;
 *    · the peak excludes the control, because the control is the zero rather
 *      than a competitor for the maximum;
 *    · and the shipped Golden Gate bundle really does say what the feature's
 *      registry copy says it says — including the part that undercuts it.
 *
 *  The last group reads `out/gpt2/interp/intervene_golden_gate.json` when it is
 *  present and skips when it is not, so a checkout without the data symlink
 *  still runs green while a checkout with it cannot let the copy drift from the
 *  numbers.
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type { InterveneBundle, InterveneRow, InterveneRun } from "../../src/data/interp";
import {
  CONTROL_RGB,
  FLOOR_FRAC,
  alphaLabel,
  barHeight,
  cageHeight,
  commonPrefix,
  controlStatus,
  fmtAlpha,
  maxKl,
  norm,
  peakKl,
  ramp,
  shortPrompt,
} from "../../src/scene/interp/steer";

// ---- fixtures ---------------------------------------------------------------

function side(text: string, tokens: string[]) {
  return {
    top: [] as [string, number][],
    text,
    tokens,
    ids: tokens.map((_, i) => i),
    logprobs: tokens.map(() => -1),
    mean_logprob: -1,
    decoding: "greedy",
  };
}

function run(kl: number, identical: boolean): InterveneRun {
  return {
    prompt: "p",
    intervention: {
      verb: "clamp",
      layer: 7,
      alpha: 1,
      is_identity: identical,
      protocol: "pin f to v",
    },
    identical,
    kl_bits: kl,
    resid_norm_baseline: 500,
    resid_norm_intervened: 400,
    baseline: side(" a b", [" a", " b"]),
    intervened: side(" a c", [" a", " c"]),
  };
}

function row(alpha: number, kls: number[], opts: Partial<InterveneRow> = {}): InterveneRow {
  const identity = alpha === 0;
  return {
    alpha,
    is_identity: identity,
    protocol: `pin f to v (strength ${alpha})`,
    kl_bits_mean: kls.reduce((a, b) => a + b, 0) / kls.length,
    kl_bits_max: Math.max(...kls),
    identical_to_baseline: identity,
    runs: kls.map((k) => run(k, identity)),
    ...opts,
  };
}

function bundle(rows: InterveneRow[]): InterveneBundle {
  return {
    kind: "intervention_sweep",
    model: "gpt2",
    verb: "clamp",
    n_layer: 12,
    d_model: 768,
    alphas: rows.map((r) => r.alpha),
    prompts: rows[0]!.runs.map((_, i) => `prompt ${i}`),
    max_tokens: 16,
    rows,
    claim: "Under this protocol …",
    notes: { decoding: "greedy", control: "α = 0 installs no hook", d6: "no weights written" },
    meta: { generated: "", revision: "", digest: "" },
  };
}

const SWEEP = bundle([row(0, [0, 0]), row(0.5, [0.6, 1.1]), row(1, [2.4, 3.4])]);

// ---- the control ------------------------------------------------------------

describe("the α = 0 control", () => {
  it("is reported as identical when the producer verified it", () => {
    expect(controlStatus(SWEEP)).toBe("identical");
  });

  it("is a THIRD state when the sweep shipped without one — not a pass", () => {
    const noCtrl = bundle([row(0.5, [0.6]), row(1, [3.4])]);
    expect(controlStatus(noCtrl)).toBe("absent");
  });

  it("is BROKEN, not identical, when the identity row was not bit-identical", () => {
    const bad = bundle([
      row(0, [0], { identical_to_baseline: false }),
      row(1, [3.4]),
    ]);
    expect(controlStatus(bad)).toBe("broken");
    // and specifically not the string the honest case prints
    expect(controlStatus(bad)).not.toBe("identical");
  });

  it("still gets a visible plate, so a measured zero never looks like an absent cell", () => {
    const h = barHeight(0, 3.4, 3);
    expect(h).toBeGreaterThan(0);
    expect(h).toBeCloseTo(FLOOR_FRAC * cageHeight(3), 12);
  });

  it("wears its own colour rather than the ramp's cool end", () => {
    // if the control were coloured by the ramp it would be ramp(0) — the point
    // of a separate constant is that it is NOT that
    expect([...CONTROL_RGB]).not.toEqual(ramp(0));
  });
});

// ---- the one normalization --------------------------------------------------

describe("normalization", () => {
  it("is linear in bits, not logged twice", () => {
    // half the maximum is half the height; a second log would pull it up
    expect(norm(1.7, 3.4)).toBeCloseTo(0.5, 12);
    expect(norm(0.85, 3.4)).toBeCloseTo(0.25, 12);
  });

  it("puts the bundle's own maximum at exactly 1", () => {
    expect(norm(maxKl(SWEEP), maxKl(SWEEP))).toBe(1);
    expect(maxKl(SWEEP)).toBe(3.4);
  });

  it("clamps rather than extrapolating, and reads a non-finite cell as zero", () => {
    expect(norm(9, 3.4)).toBe(1);
    expect(norm(-1, 3.4)).toBe(0);
    expect(norm(Number.NaN, 3.4)).toBe(0);
    expect(norm(1, 0)).toBe(0);
  });

  it("drives height and colour from the same number", () => {
    // the contract the header states: a taller column is a brighter column.
    const kls = [0, 0.6, 1.1, 2.4, 3.4];
    const heights = kls.map((k) => barHeight(k, 3.4, 5));
    const lums = kls.map((k) => {
      const [r, g, b] = ramp(norm(k, 3.4));
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    });
    for (let i = 1; i < kls.length; i++) {
      expect(heights[i]!).toBeGreaterThanOrEqual(heights[i - 1]!);
      expect(lums[i]!).toBeGreaterThan(lums[i - 1]!);
    }
  });
});

describe("the cage", () => {
  it("never becomes a pillar or a pancake", () => {
    expect(cageHeight(1)).toBe(4.5);
    expect(cageHeight(40)).toBe(12);
    expect(cageHeight(5)).toBe(8);
  });
});

describe("peak KL", () => {
  it("excludes the control — the zero is not a competitor for the maximum", () => {
    expect(peakKl(SWEEP)).toBe(3.4);
  });

  it("is null, never 0, when every row is the control", () => {
    expect(peakKl(bundle([row(0, [0])]))).toBeNull();
  });
});

// ---- labels -----------------------------------------------------------------

describe("axis labels", () => {
  it("says 'control' on the axis, not only in the legend", () => {
    expect(alphaLabel(SWEEP.rows[0])).toContain("control");
    expect(alphaLabel(SWEEP.rows[2])).not.toContain("control");
  });

  it("prints α = 0 as 0, the shortest thing on the axis", () => {
    expect(fmtAlpha(0)).toBe("0");
    expect(fmtAlpha(1)).toBe("1");
    expect(fmtAlpha(0.25)).toBe("0.25");
    expect(fmtAlpha(0.5)).toBe("0.5");
  });

  it("truncates a prompt on a word boundary, never mid-word", () => {
    const s = shortPrompt("My favourite place to visit is");
    expect(s.endsWith("…")).toBe(true);
    // the visible part must be whole words of the original
    const body = s.slice(0, -1);
    expect("My favourite place to visit is".startsWith(body)).toBe(true);
    expect(body.endsWith(" ")).toBe(false);
    expect(shortPrompt("short one")).toBe("short one");
  });
});

describe("commonPrefix", () => {
  it("counts the tokens two generations share before they part", () => {
    expect(commonPrefix([" a", " b", " c"], [" a", " b", " d"])).toBe(3 - 1);
    expect(commonPrefix([" a"], [" b"])).toBe(0);
    expect(commonPrefix([" a", " b"], [" a", " b"])).toBe(2);
    expect(commonPrefix([], [" a"])).toBe(0);
  });
});

// ---- the shipped bundle -----------------------------------------------------

const BUNDLE_PATH = resolve(__dirname, "../../../out/gpt2/interp/intervene_golden_gate.json");
const shipped: InterveneBundle | null = existsSync(BUNDLE_PATH)
  ? (JSON.parse(readFileSync(BUNDLE_PATH, "utf-8")) as InterveneBundle)
  : null;

describe.skipIf(shipped === null)("the shipped Golden Gate sweep", () => {
  it("has a control row that the producer verified bit-identical", () => {
    expect(controlStatus(shipped!)).toBe("identical");
    const ctrl = shipped!.rows.find((r) => r.is_identity)!;
    expect(ctrl.kl_bits_max).toBe(0);
    // every run in it agrees, per-run, not just in the summary
    for (const r of ctrl.runs) {
      expect(r.identical).toBe(true);
      expect(r.kl_bits).toBe(0);
      expect(r.intervened.text).toBe(r.baseline.text);
    }
  });

  it("moves the distribution monotonically with α", () => {
    const moved = shipped!.rows.filter((r) => !r.is_identity);
    for (let i = 1; i < moved.length; i++) {
      expect(moved[i]!.kl_bits_mean).toBeGreaterThan(moved[i - 1]!.kl_bits_mean);
    }
  });

  it("carries the claim in the intervention's own terms, not about the feature", () => {
    const claim = shipped!.claim.toLowerCase();
    expect(claim).toContain("under this protocol");
    // D3 permits a sentence about what the intervention DID; §1.2 still forbids
    // a sentence about what the direction IS
    expect(claim).toContain("not about what the direction is");
  });

  it("records the layer arithmetic that the SAE hook point implies", () => {
    // blocks.8.hook_resid_pre is the stream ENTERING block 8 = output of 7
    expect(shipped!.meta.sae_hook_layer).toBe(8);
    expect(shipped!.meta.hook_layer).toBe(7);
  });

  it("shows the target scores that undercut the headline, rather than dropping them", () => {
    const full = shipped!.rows[shipped!.rows.length - 1]!;
    const r0 = full.runs[0]!;
    const gg = r0.targets?.find((t) => t.text.includes("Golden Gate"));
    const bk = r0.targets?.find((t) => t.text.includes("Brooklyn"));
    expect(gg).toBeTruthy();
    expect(bk).toBeTruthy();
    // the finding: at full strength the aimed-at completion gets LESS likely,
    // and by more than the control completion does. If a future rebuild ever
    // reverses this, the registry copy has to be rewritten with it.
    expect(gg!.intervened_logprob).toBeLessThan(gg!.baseline_logprob);
    const dGG = gg!.intervened_logprob - gg!.baseline_logprob;
    const dBK = bk!.intervened_logprob - bk!.baseline_logprob;
    expect(dGG).toBeLessThan(dBK);
  });

  it("generates something different from the baseline at full strength", () => {
    const full = shipped!.rows[shipped!.rows.length - 1]!;
    for (const r of full.runs) {
      expect(r.identical).toBe(false);
      expect(r.intervened.text).not.toBe(r.baseline.text);
      expect(commonPrefix(r.baseline.tokens, r.intervened.tokens)).toBeLessThan(
        r.baseline.tokens.length,
      );
    }
  });
});
