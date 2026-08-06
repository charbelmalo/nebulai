---
name: nebulai-sae
description: >-
  Plan A of Nebul.AI — the SAE-feature front-end (~/Developer/nebulai), the
  flagship pipeline (NOT yet built). A point is one sparse-autoencoder feature;
  labels are auto-interp descriptions. Use this skill when building or running
  the SAE map: sae-lens, Gemma Scope, GPT-2 res-jb SAEs, Neuronpedia label
  bootstrap, Delphi/EleutherAI detection scoring, decoder-direction vs
  label-space geometry, or stratified firing-rate sampling. Load the `nebulai`
  hub alongside this — the shared back-end, the Units contract, and the
  cross-pipeline recommendation rule live there; this skill only covers what is
  specific to SAE features. The hard design decision here is the "model space vs
  label space" projection — get it wrong and the map shows the label-embedder's
  semantics, not the model's. Run the hub's propagation check on any change.
---

# Nebul.AI — Plan A: SAE features (flagship, not yet built)

**A point is one sparse-autoencoder feature.** SAEs decompose a layer's
activations into a large dictionary of sparser, more monosemantic features than
raw neurons — the current best bet for interpretable units. This is the flagship
artifact; Plan C de-risked the back-end, and Plan B exists to be contrasted
against this. It reduces to the same `Units` and rides the same back-end (hub →
`references/backend-contract.md`).

Target: `src/nebulai/frontends/sae.py` exposing `load_sae_units(...) -> Units`,
plus a `sae` CLI subcommand mirroring the `tokens` flags.

## The one decision that defines honesty here: what is `Units.vectors`?

There are two candidate geometries, and conflating them is the #1 way this
genre of map lies:

- **Decoder directions** — the feature's column of `W_dec`, i.e. the direction
  it writes into the residual stream. This is the **model's own** space. Cluster
  here and you're mapping the model.
- **Label embeddings** — mxbai-embed-large (1024-d) over the auto-interp
  descriptions. Cluster here and you're mapping the *label-embedder's* semantics
  — which says as much about the labeler as the subject model.

**Build both. Carry both in the export. Expose the toggle in the viewer.** Never
present a label-space layout as model geometry. (This toggle is shared with Plan
B, not Plan C — see the hub's propagation table.) Full rationale and the build
sequence: `references/build-plan.md`.

## Labels (`Units.labels`)

Auto-interp descriptions, in priority order:

1. **Bootstrap from Neuronpedia** where the feature already has a public label
   (GPT-2 res-jb and Gemma Scope are covered).
2. **Generate** the rest with a hybrid local(ollama on the M4 worker)/API
   labeler — reuse the shared namer machinery in `backend/name.py` where
   possible rather than a parallel path.
3. **Score every label by detection** (Delphi / EleutherAI protocol): does the
   label predict which held-out snippets activate the feature? Store the score
   in `meta` so label quality is a number, not a vibe, and so the map can dim
   low-confidence labels.

## Sampling

Don't take top-k by activation — that over-represents dense features and hides
rare monosemantic ones. **Stratify across firing-rate deciles** so the map
represents the whole dictionary. Record the sampling scheme in `meta`.

## Provenance

`meta["unit"] = "sae_feature"` (→ `unit_ref.kind`), `meta["layer"]` set, plus
the sae-lens release id and which geometry a given export used
(`meta["projection"] = "decoder" | "label"`). Everything in `meta` lands in
`nebulai.json`.

See `references/build-plan.md` for library choices (sae-lens releases,
Neuronpedia API, Delphi), the incremental build order, and how this pipeline
plugs into the existing back-end without touching it.
