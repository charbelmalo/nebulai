/** What each registered Internals analysis reads, so the rail and the Research
 *  chooser can state availability from a model's export index instead of
 *  guessing. The index (`out/<model>/interp/index.json`) lists the bundles an
 *  export actually shipped; an analysis is available exactly when everything it
 *  reads is listed. tests/unit/interp-requirements.test.ts checks this table
 *  against the loaders each driver calls, so it cannot drift silently. */

import type { InterpIndex } from "../../data/interp";

export type Requirement =
  /** every one of these bundle files */
  | { kind: "bundles"; files: readonly string[] }
  /** at least one bundled forward trace (`trace_<slug>.json`) */
  | { kind: "trace" }
  /** at least one intervention sweep (`intervene_<name>.json`) */
  | { kind: "sweep" }
  /** computed on request by the local live server; nothing is exported */
  | { kind: "live" };

export const REQUIREMENTS: Readonly<Record<string, Requirement>> = {
  "fourier-atlas": { kind: "bundles", files: ["fourier.json"] },
  "weight-spectrum": { kind: "bundles", files: ["weights.json"] },
  "embedding-constellation": { kind: "bundles", files: ["embed.json"] },
  "neuron-field": { kind: "bundles", files: ["neurons.json"] },
  "head-fingerprints": { kind: "bundles", files: ["heads.json"] },
  "ov-eigen": { kind: "bundles", files: ["ov_eigs.json"] },
  "comp-web": { kind: "bundles", files: ["comp.json"] },
  "induction-microscope": { kind: "bundles", files: ["induction.json"] },
  "ablation-ghosts": { kind: "bundles", files: ["ablation.json"] },
  "occlusion-vignette": { kind: "bundles", files: ["occlusion.json"] },
  "logit-lens-tunnel": { kind: "trace" },
  "attention-flow": { kind: "trace" },
  "attention-rollout": { kind: "trace" },
  "residual-ribbon": { kind: "trace" },
  "probability-simplex": { kind: "trace" },
  "logit-attrib": { kind: "bundles", files: ["attrib.json"] },
  "causal-patching": { kind: "bundles", files: ["patch.json"] },
  "tuned-lens": { kind: "bundles", files: ["tuned.json"] },
  "steer-rail": { kind: "sweep" },
  "sae-decoder": { kind: "bundles", files: ["sae.json"] },
  "sae-piano-roll": { kind: "bundles", files: ["sae_acts.json"] },
  "decoder-cosine-web": { kind: "bundles", files: ["sae.json", "sae_web.json"] },
  "direction-compass": { kind: "bundles", files: ["sae.json", "compass.json"] },
  "cofire-venn": { kind: "bundles", files: ["sae.json", "cofire.json"] },
  "grokking-clock": { kind: "bundles", files: ["grok.json"] },
  "live-nebula": { kind: "live" },
};

export type Availability =
  | { state: "available"; files: string[] }
  | { state: "missing"; reason: string }
  | { state: "live"; reason: string }
  /** the export index has not been read (or could not be) */
  | { state: "unknown" };

/** Is `featureId` computable from this export? `index` null = no export. */
export function availability(featureId: string, index: InterpIndex | null | undefined): Availability {
  const req = REQUIREMENTS[featureId];
  if (!req) return { state: "unknown" };
  if (req.kind === "live")
    return { state: "live", reason: "Runs on the local live server; nothing is exported for it." };
  if (index === undefined) return { state: "unknown" };
  if (index === null) return { state: "missing", reason: "This model has no internals export." };
  const bundles = index.bundles ?? [];
  switch (req.kind) {
    case "bundles": {
      const absent = req.files.filter((f) => !bundles.includes(f));
      return absent.length === 0
        ? { state: "available", files: [...req.files] }
        : { state: "missing", reason: `Not in this export: ${absent.join(", ")}.` };
    }
    case "trace": {
      const traces = bundles.filter((b) => /^trace_.+\.json$/.test(b));
      return traces.length > 0
        ? { state: "available", files: traces }
        : { state: "missing", reason: "This export has no bundled prompt traces." };
    }
    case "sweep": {
      const sweeps = bundles.filter((b) => /^intervene_.+\.json$/.test(b));
      return sweeps.length > 0
        ? { state: "available", files: sweeps }
        : { state: "missing", reason: "This export has no intervention sweep." };
    }
  }
}

/** How many registered analyses this export can show (live views excluded:
 *  they depend on a server, not on the export). */
export function availableCount(ids: readonly string[], index: InterpIndex | null | undefined): number {
  return ids.filter((id) => availability(id, index).state === "available").length;
}
