import { describe, expect, it } from "vitest";
import {
  CAVEAT_OPTIONS,
  QUANTITY_OPTIONS,
  emptyIntroProgress,
  formatPeriod,
  pendingSteps,
  sha256Hex,
  stepDone,
  topFrequencies,
  verifyBundle,
  type ResearchIntro,
} from "../../src/chrome/research/intro";
import type { FourierBundle } from "../../src/data/interp";

const bundle: FourierBundle = {
  meta: {
    model: "gpt2",
    created: "2026-01-01",
    quantity: "power spectrum of mean-centered W_pe along position axis",
    formula: "P(f) = mean_d |rfft(W_pe - mean)[f, d]|^2",
    n_ctx: 8,
    d: 4,
  } as FourierBundle["meta"],
  freqs: [0, 1, 2, 3, 4],
  power_mean: [0, 5, 9, 1, 5],
  per_dim_dominant: [2, 2, 1, 4],
};

const bytesOf = (o: unknown) => new TextEncoder().encode(JSON.stringify(o));

async function introFor(bytes: Uint8Array): Promise<ResearchIntro> {
  return {
    dataset_id: "gpt2",
    feature: "fourier-atlas",
    bundle_path: "gpt2/interp/fourier.json",
    sha256: await sha256Hex(bytes),
    bytes: bytes.length,
  };
}

const serve =
  (body: Uint8Array | null, status = 200): typeof fetch =>
  async () =>
    body === null ? new Response("nope", { status }) : new Response(new Uint8Array(body), { status });

describe("Research first task: the frequency table", () => {
  it("ranks non-zero frequencies by mean power and counts peaking dimensions", () => {
    const rows = topFrequencies(bundle, 3);
    expect(rows.map((r) => r.freq)).toEqual([2, 1, 4]); // 1 and 4 tie; lower frequency first
    expect(rows[0]).toEqual({ index: 2, freq: 2, period: 4, power: 9, dims: 2 });
    expect(rows.find((r) => r.freq === 4)!.dims).toBe(1);
    expect(rows.some((r) => r.freq === 0)).toBe(false);
  });

  it("formats periods without inventing precision", () => {
    expect(formatPeriod(512)).toBe("512");
    expect(formatPeriod(1024 / 3)).toBe("341.3");
  });
});

describe("Research first task: verifying the pinned bundle", () => {
  it("accepts exactly the pinned bytes and hands back their parse", async () => {
    const bytes = bytesOf(bundle);
    const ri = await introFor(bytes);
    const r = await verifyBundle(ri, serve(bytes), "/out");
    expect(r.state).toBe("verified");
    if (r.state !== "verified") return;
    expect(r.url).toBe("/out/gpt2/interp/fourier.json");
    expect([...r.bytes]).toEqual([...bytes]);
    expect(r.json.freqs).toEqual(bundle.freqs);
  });

  it("reports a missing file with its status and substitutes nothing", async () => {
    const ri = await introFor(bytesOf(bundle));
    expect(await verifyBundle(ri, serve(null, 404), "/out")).toMatchObject({ state: "missing", status: 404 });
    const offline: typeof fetch = async () => {
      throw new TypeError("network down");
    };
    expect(await verifyBundle(ri, offline, "/out")).toMatchObject({ state: "missing", status: null });
  });

  it("rejects different bytes, even when they parse", async () => {
    const ri = await introFor(bytesOf(bundle));
    const other = bytesOf({ ...bundle, power_mean: [0, 5, 9, 1, 6] });
    const r = await verifyBundle(ri, serve(other), "/out");
    expect(r.state).toBe("mismatch");
    if (r.state === "mismatch") {
      expect(r.expected).toBe(ri.sha256);
      expect(r.actual).toBe(await sha256Hex(other));
    }
  });

  it("rejects pinned bytes that are not a frequency bundle", async () => {
    const bytes = bytesOf({ hello: "world" });
    const ri = await introFor(bytes);
    expect((await verifyBundle(ri, serve(bytes), "/out")).state).toBe("mismatch");
  });

  it("matches a known SHA-256 vector", async () => {
    expect(await sha256Hex(new TextEncoder().encode("abc"))).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });
});

describe("Research first task: completion", () => {
  const rows = topFrequencies(bundle, 3);
  const right = (opts: typeof QUANTITY_OPTIONS) => opts.find((o) => o.correct)!.id;
  const wrong = (opts: typeof QUANTITY_OPTIONS) => opts.find((o) => !o.correct)!.id;

  it("has exactly one correct answer per question, each explained", () => {
    for (const opts of [QUANTITY_OPTIONS, CAVEAT_OPTIONS]) {
      expect(opts.filter((o) => o.correct)).toHaveLength(1);
      for (const o of opts) expect(o.why.length).toBeGreaterThan(20);
    }
  });

  it("completes only when every step is answered correctly", () => {
    const p = emptyIntroProgress("x");
    expect(pendingSteps(p, rows)).toEqual(["quantity", "value", "caveat", "export"]);
    const wrongAnswers = { ...p, quantity: wrong(QUANTITY_OPTIONS), caveat: wrong(CAVEAT_OPTIONS) };
    expect(stepDone("quantity", wrongAnswers, rows)).toBe(false);
    expect(stepDone("caveat", wrongAnswers, rows)).toBe(false);
    // a recorded value must be one of the rows on screen
    expect(stepDone("value", { ...p, value: 0 }, rows)).toBe(false);
    const done = {
      ...p,
      quantity: right(QUANTITY_OPTIONS),
      value: rows[0]!.index,
      caveat: right(CAVEAT_OPTIONS),
      exported: true,
    };
    expect(pendingSteps(done, rows)).toEqual([]);
  });

  it("names the static-weights caveat as what the chart cannot show", () => {
    const c = CAVEAT_OPTIONS.find((o) => o.correct)!;
    expect(c.label).toMatch(/uses a given frequency/);
    expect(c.why).toMatch(/static weight patterns/);
  });
});
