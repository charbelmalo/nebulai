/** GPU capability probe — decides the fallback rung once, at boot.
 *  Ladder: webgpu → webgl (three forceWebGL; TSL transpiles) → static
 *  (PNG + table ultra-fallback). `?gpu=webgl|static` forces a rung for
 *  testing; e2e runs pin `?gpu=webgl` for determinism. */

export type GpuTier = "webgpu" | "webgl" | "static";

export interface Capabilities {
  tier: GpuTier;
  reason: string;
  reducedMotion: boolean;
}

export async function probeCapabilities(
  search = typeof location !== "undefined" ? location.search : "",
): Promise<Capabilities> {
  const reducedMotion =
    typeof matchMedia !== "undefined" &&
    matchMedia("(prefers-reduced-motion: reduce)").matches;

  const forced = new URLSearchParams(search).get("gpu");
  if (forced === "webgl" || forced === "static") {
    return { tier: forced, reason: `forced via ?gpu=${forced}`, reducedMotion };
  }

  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) return { tier: "webgpu", reason: "adapter ok", reducedMotion };
      return { tier: "webgl", reason: "navigator.gpu present but no adapter", reducedMotion };
    } catch (e) {
      return {
        tier: "webgl",
        reason: `requestAdapter threw: ${e instanceof Error ? e.message : e}`,
        reducedMotion,
      };
    }
  }

  const hasWebgl =
    typeof document !== "undefined" &&
    !!document.createElement("canvas").getContext("webgl2");
  return hasWebgl
    ? { tier: "webgl", reason: "no navigator.gpu", reducedMotion }
    : { tier: "static", reason: "no navigator.gpu, no webgl2", reducedMotion };
}
