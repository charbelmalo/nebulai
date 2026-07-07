---
name: nebulai-neurons
description: >-
  Plan B of Nebul.AI — the raw MLP-neuron front-end (~/Developer/nebulai), the
  comparison artifact (NOT yet built). A point is one MLP hidden neuron, hooked
  via TransformerLens; its geometry is the neuron's write direction into the
  residual stream. Use this skill when building or running the neuron map:
  TransformerLens hooks, MLP `W_out` rows, neuron auto-interp, and above all the
  quantitative neurons-vs-SAE comparison (noise fraction, silhouette, label
  detection) that is this pipeline's entire purpose. Load the `nebulai` hub
  alongside this — the shared back-end, the Units contract, and the
  cross-pipeline recommendation rule live there. Plan B only means something as
  an apples-to-apples contrast with Plan A on an IDENTICAL back-end, so run the
  hub's propagation check on any change and keep the two pipelines in lockstep.
---

# Nebul.AI — Plan B: MLP neurons (comparison artifact, not yet built)

**A point is one MLP hidden neuron.** Its geometry is the neuron's row of
`W_out` — the direction it writes into the residual stream (the analogue of an
SAE feature's decoder direction). Reduces to the same `Units`, rides the same
back-end (hub → `references/backend-contract.md`).

Target: `src/nebulai/frontends/neurons.py` exposing `load_neuron_units(...) ->
Units`, plus a `neurons` CLI subcommand.

## The purpose is the contrast, not the map

Raw neurons are **polysemantic** — one neuron fires for many unrelated concepts —
which is precisely the problem SAEs were built to solve. So the expected result
is that Plan B produces *measurably worse* structure than Plan A: higher noise
fraction, lower silhouette in the clustering space, lower label-detection
scores, less coherent clusters. Producing that quantitative table **is the
artifact**. A pretty neuron map is a failure of the comparison; an honest,
worse-looking one is the point.

This only holds if A and B share an **identical** back-end and identical
metrics. Any divergence (different UMAP params, different clustering, different
label scorer) silently invalidates the comparison. Treat A and B as lockstep:
see the hub's `references/change-propagation.md` — most changes here should mirror
Plan A and vice versa.

## What is specific to this pipeline

- **Hooking (TransformerLens).** Load the same base model as the SAE (e.g.
  gpt2-small or gemma-2-2b) and take MLP `W_out` rows per layer as `vectors`,
  neuron indices as `ids`. Choose layers to match the SAE's layers so the
  comparison is at the same depth.
- **Labels.** Auto-interp the neurons with the *same* labeler and the *same*
  detection scorer as Plan A — otherwise label quality differences are
  confounded by tooling differences, not by neurons-vs-features.
- **Provenance.** `meta["unit"] = "mlp_neuron"` (→ `unit_ref.kind`),
  `meta["layer"]` set, base model recorded.

## Build order

1. Skeleton: one layer, `W_out` rows as geometry, placeholder labels, `neurons`
   subcommand, run the pipeline end to end → a rendered neuron map with zero
   back-end edits. Proves the third front-end fits the contract. You inherit
   `leaf` / `min_samples=5` from the back-end, but those were tuned on token
   geometry — re-sweep with `scripts/sweep_hdbscan.py out/<model>/reduced.npz`
   before trusting the map (write-direction space may cluster very differently),
   and record the chosen params in `meta`. See the hub's `change-propagation.md`
   → "Provisional defaults".
2. Real labels via the shared labeler; detection scores via the shared scorer.
3. Compute the shared comparison metrics (noise, silhouette, mean detection) and
   emit the A-vs-B table.

## The comparison metrics (must match Plan A exactly)

Keep the metric computation in one shared location, not duplicated per
front-end, so A and B are scored by the same code. If you add or change a metric
here, it must also apply to Plan A — that's a propagation decision; run the hub
check and end with a recommendation about mirroring it into `nebulai-sae`.
