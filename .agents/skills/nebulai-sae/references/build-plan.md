# Plan A build plan — SAE feature front-end

The back-end already exists and is validated by Plan C. Building Plan A is
therefore **only** writing a front-end that returns `Units`, plus a CLI
subcommand. If you're editing `backend/*`, stop and run the hub's
change-propagation check — the change probably belongs to all pipelines.

## Libraries

- **sae-lens** — loads pretrained SAEs and exposes `W_dec` / `W_enc`. Start with
  a small, well-documented release: GPT-2-small `res-jb` (Joseph Bloom's
  residual-stream SAEs) or **Gemma Scope** on gemma-2-2b. Pick one layer first.
- **Neuronpedia API** — public auto-interp labels for those exact releases; use
  it to bootstrap `labels` before generating any.
- **Delphi (EleutherAI)** — detection/fuzzing scorer for auto-interp labels.
  Used to attach a quality number to each label.
- Embedding for label-space geometry: **mxbai-embed-large** (1024-d), via the M4
  worker if available.

## Incremental build order

1. **Skeleton front-end, decoder geometry only.** Load one SAE layer with
   sae-lens, take `W_dec` rows as `vectors`, feature indices as `ids`,
   placeholder `labels` (e.g. `f"feat {i}"`). Return `Units`. Wire a `sae`
   subcommand. Run the existing pipeline end to end. Success = a `nebulai.json`
   with `unit_ref.kind == "sae_feature"` and a rendered map. This proves the
   contract holds for a second front-end with zero back-end edits.
   *Clustering caveat:* you inherit `leaf` / `min_samples=5`, but those were
   tuned on token geometry. Before trusting the map, re-sweep with
   `scripts/sweep_hdbscan.py out/<model>/reduced.npz` — decoder-direction space
   may want different settings (or even `eom`). Record the chosen params in
   `meta`. See the hub's `change-propagation.md` → "Provisional defaults".
2. **Real labels from Neuronpedia.** Replace placeholder labels with fetched
   descriptions; leave un-covered features with a fallback label.
3. **Generate missing labels** with the hybrid labeler (reuse `backend/name.py`
   plumbing).
4. **Detection scoring (Delphi).** Score labels; store per-feature score in a
   parallel array and in `meta`; map score → confidence/opacity.
5. **Label-space projection.** Add mxbai embeddings of the labels as an
   alternate `vectors`; export both projections; document the toggle.
6. **Stratified sampling** across firing-rate deciles once you scale past one
   layer / a subset.

Each step is independently shippable — stop and render after each so regressions
are obvious.

## Model space vs label space (the honesty crux)

- **Decoder-direction geometry** answers "how does the model organize these
  features?" It is the defensible, model-native map.
- **Label-space geometry** answers "how does the *label embedder* organize these
  descriptions?" It is prettier and more human-legible but is partly an artifact
  of the labeler. It is legitimate to show, but only when labeled as such.
- The export must record `meta["projection"]` and ideally carry both coordinate
  sets so the viewer toggles without recompute. This toggle is shared with Plan
  B; if you change its semantics, propagate to `nebulai-neurons`.

## Sampling detail

Firing rates span orders of magnitude. Top-k-by-activation sampling floods the
map with high-frequency, often polysemantic features and buries rare
monosemantic ones — the interesting cases. Bin features by firing-rate decile
and sample evenly across bins; store the scheme in `meta` so a reader knows the
map isn't a popularity contest.

## Comparison hooks (shared with Plan B)

Compute the same quality metrics you'll compute for neurons so the A-vs-B table
is apples-to-apples: noise fraction, silhouette in the clustering space, and
mean label-detection score. Put these in `meta`. Keeping the metric code in one
shared place (not per-front-end) is itself a propagation decision — see the hub.
