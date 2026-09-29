# The two validation layers `nebulai validate` does not cover

The README's roadmap carries one line that this document exists to expand:

> Held-out auto-interp scores and activation-based coherence — the two
> validation layers `nebulai validate` does not yet cover. It measures
> geometry and stability; it never reads a cluster title, so nothing in the
> suite currently grades the naming.

That sentence is accurate and it is also the whole of what has been written
down, which is not enough to act on. What follows says exactly what each layer
would need: the input data, which of that data exists in this repository today
and which does not, the measurement protocol, what the resulting number would
and would not license anyone to claim, and the way each one fails when it is
built naively.

**This document specifies requirements. It implements nothing and proposes no
code.** The second layer depends on machinery that is partly built and still
gated — the repo captures activations and ships one hashed corpus, but nothing
aggregates either by cluster — and the point of writing the requirements down
before the gate is that a gate is easier to hold when everyone can see what is
behind it.

## What `nebulai validate` measures today

`nebulai validate <dataset...>` runs `validate_map()` in
`src/nebulai/backend/validate.py`, writes `validation.json` beside the map's
`nebulai.json`, and `compute_map_metrics()` in `src/nebulai/backend/metrics.py`
picks it back up so `nebulai metrics` can show the columns. It is a separate
command from the build because two of its three checks re-run UMAP, which costs
minutes per map.

It computes exactly three numbers, and the module's own docstring orders them by
how much they can embarrass the map:

1. **`trustworthiness_score()`** — scikit-learn's trustworthiness of the shipped
   reduction against the original-space vectors, at `n_neighbors=15` by default
   and capped at 5,000 points (`_TRUST_SAMPLE_CAP`) because it is O(n²) in time
   and memory. It answers whether UMAP preserved the neighbourhoods the model
   actually has, and it is the one check that never touches HDBSCAN. It grades
   `u_cluster` as loaded from `reduced.npz` — the layout on screen, never a
   re-fit.
2. **`seed_stability()`** — re-runs `reduce_vectors()` and `cluster_units()`
   under seeds `(42, 1, 2, 3)` and scores every pair of clusterings with the
   adjusted Rand index, returning `mean_ari` and `min_ari`. Noise is scored as
   its own label rather than dropped, because points sliding in and out of noise
   across seeds *is* the instability the metric exists to catch. Capped at 4,000
   points (`_STABILITY_SAMPLE_CAP`).
3. **`null_baseline()`** — the identical pipeline on column-shuffled vectors,
   which preserve every per-dimension marginal while destroying the correlations
   between dimensions, and reports the silhouette (via `metrics._silhouette`),
   cluster count and noise fraction that structure-free data still produces.
   Same 4,000-point cap.

Three supporting pieces carry as much of the honesty as the metrics do.
`scale_cluster_kwargs()` rescales `min_cluster_size` so a subsampled run
clusters at the full map's *granularity* rather than its absolute point count.
`reload_units()` replays the front-end from the stamped `meta` — necessary
because `reduced.npz` caches only the UMAP outputs and never the source vectors
— and it refuses outright, with a specific reason, for `api_text_embedding` and
`probe_concept` maps, whose vectors came from a live service and (for probe) a
sampled concept set that would replay as a *different* point set. Every returned
dict records `n_scored` and `subsampled`, so no number can be read as covering
more points than it saw.

**The gap, stated precisely: `validate_map()` opens `nebulai.json` and reads
only `["meta"]`.** It never reads `points[].label` and it never reads
`clusters[].title`. Every number it produces is a function of the vectors and
the cluster assignment. The map's titles — the part a reader actually looks at,
and the part produced by an LLM in `backend/name.py` — are graded by nothing at
all. That is the hole both layers below are aimed at, from two different sides.

---

## Layer A — held-out auto-interp scores

### What the question is

A cluster's title does not come from the whole cluster. `_representatives()` in
`src/nebulai/backend/name.py` takes the `k=20` members nearest the cluster
centroid by cosine, most central first, and those are the only members the namer
sees; `_batch_lines()` formats them and the namer returns a 2–5 word title. So
for any cluster larger than 20 there are members the title was never shown.

The held-out score asks the obvious question that follows: **does the title
describe the members the namer did not see?** A title that fits its twenty
representatives and nothing else is a description of a centroid, not of a
cluster, and nothing currently distinguishes the two cases.

### Input data required

| Needed | Exists today? |
|---|---|
| Cluster assignment per unit | Yes — `points[].cluster_id` in `nebulai.json`, written by `export_json()`. |
| Cluster titles | Yes — `clusters[].title`. |
| Per-unit display labels | **Partly, and this is the binding constraint — see below.** |
| The exact member set shown to the namer | **No, but reconstructible.** `_representatives()` is deterministic given the vectors and `k`, so the shown set can be recomputed — at the cost of a `reload_units()` call, inheriting its refusals for `api_text_embedding` and `probe_concept` maps. Recording the representatives at name time would remove that dependency and is the honest fix. |
| A judge model distinct from the namer | Yes — the backend chain in `name.py` / `src/nebulai/llm.py` already reaches several, and `meta` stamps `namer_backend`, `namer_model` and `namer_identity`, so "is the judge the namer?" is a checkable question rather than a hope. |
| Matched distractor pools | No. Matched and stratified sampling do exist elsewhere — `behavior/cues.py` draws a stratum-balanced calibration subset, `backend/persona.py::probe_strata` carries the crossed design's strata into the permutation null, and `prompts/eval_awareness.v1.json` is a matched-pair set by construction — but nothing samples a **distractor pool for a title judge**. |
| A shuffled-title null | No. |

The label constraint decides where this layer can run at all, and it differs by
front-end:

- **Token maps** (`load_token_units`, `frontends/tokens.py`): the label *is* the
  token string. Real for every point.
- **SAE maps** (`load_sae_units`, `frontends/sae.py`): labels are bootstrapped
  from Neuronpedia's public auto-interp export, and only for
  `gpt2-small` / `blocks.8.hook_resid_pre` — any other release raises rather
  than stamping one model's labels onto another's features. Unlabelled features
  get the exact placeholder `"feature {i} (unlabeled)"`, and `meta` carries
  `n_labeled` / `n_unlabeled` so coverage is visible.
- **Neuron maps** (`load_neuron_units`, `frontends/neurons.py`): `labels_source`
  accepts only `"none"`. **Every** label is `"neuron {i} (unlabeled)"`. There is
  no public auto-interp source wired for them.

So: this layer is fully computable for token maps, computable for SAE maps in
proportion to `n_labeled`, and **not computable at all for neuron maps** until a
label source exists. A scorer that ran on a neuron map would be grading
placeholder strings against a title that `placeholder_titles()` already had to
generate honestly because every member was a placeholder. That case must be
refused with a reason, in the style `reload_units()` already uses, rather than
returning a number.

### Protocol

1. **Split.** For each cluster, the explanation set is the `k` representatives
   `_representatives()` selected; the held-out set is every other member. Record
   `n_heldout` per cluster and refuse to score clusters below a declared floor —
   a cluster of size 22 has two held-out members and cannot produce a meaningful
   rate.
2. **Build a pool.** For each cluster, mix its held-out members with distractors.
   Draw the distractors from the *nearest neighbouring clusters in `u_cluster`*,
   not uniformly across the map, so the task is separating this cluster from its
   neighbours rather than from the far side of the atlas.
3. **Score.** Present the judge with the title and the shuffled pool and ask
   which members belong. Report balanced accuracy or AUC, never raw accuracy,
   since the pool's positive rate is a free parameter.
4. **Null.** Re-run the identical scoring with titles permuted across clusters.
   That is the floor, exactly as `null_baseline()` is the floor for silhouette,
   and for exactly the same reason: the procedure alone can score well above
   zero. **The reportable quantity is the margin over the shuffled-title null,
   not the raw score.**
5. **Record the conditions.** Judge model id, `k`, pool composition, the
   per-cluster `n_heldout`, and the map's `labels_source` and `n_labeled` — the
   same discipline `validate.py` already applies with `n_scored` and
   `subsampled`.

### What the number licenses, and what it does not

It licenses one claim: *the title generalises to members the namer did not see*.
It is evidence about the title, and only about the title.

It does not license any of these, and the distance matters:

- It says nothing about whether the cluster is real in the model's original
  space. That is `trustworthiness`, and it is a separate number.
- It says nothing about what the units *do*. That is Layer B, and the README's
  weight-geometry honesty note stands regardless of how well this layer scores:
  every model-derived map here answers what a layer can write, never what it
  wrote for a given prompt.
- Scores are **not comparable across maps with different label sources.** A high
  score on an SAE map is partly a measurement of Neuronpedia's auto-interp
  consistency, because the member labels being judged are Neuronpedia's
  sentences. A high score on a token map is a judgement about token strings —
  the model's `W_E` geometry produced the grouping, but the judge only ever sees
  the words.
- It is not the "detection score" of the SAE auto-interp literature. That
  protocol scores a description against *activating text examples*. This one
  scores it against *member labels*, because member labels are what this repo
  has. It is a weaker proxy and must not be published under the stronger name.

### How it fails when built naively

- **Leakage.** Scoring the title against all members, representatives included,
  grades the namer's own input back to it. The inflation is worst exactly where
  the cluster is small and `k=20` is most of it.
- **No null.** A judge shown "days & months" against distractors from an
  unrelated cluster wins on implausibility alone. Without the shuffled-title
  run, the number is a measurement of how far apart the clusters happened to be,
  which the silhouette already reports.
- **Easy negatives.** Uniform distractors make every title look precise. Hard
  negatives from adjacent clusters are the whole test.
- **The namer judging itself.** `meta.namer_model` exists; check it.
- **Scoring placeholders.** A neuron map, or an SAE map with low `n_labeled`,
  will happily produce a number that means nothing. Gate on the stamped
  coverage and refuse with a reason.

---

## Layer B — activation-based coherence

### Track 4 is not implemented, and this document does not implement it

Activation-based coherence requires real activations — the units' responses to
real text — aggregated per unit over a declared corpus. Both halves of that now
partly exist, so the statement this section used to make ("this repository has
none, anywhere, by design") is no longer true and the honest version is
narrower.

The repository **does** capture activations over real text. `GPT2Numpy.forward()`
returns a `Trace` whose `resid` is `(n_layer+1, T, d)` and whose `mlp_post` is
`(n_layer, T, d_mlp)`, and `LlamaNumpy.capture_resid()` batches residual capture
for the Llama-architecture instruct models — which is what `backend/persona.py`,
`backend/eval_awareness.py` and `interp/live_server.py` run on. Interventions go
through the same door: `interp/hooks.py` fixes one `resid_hooks` protocol that
both runners implement and `interp/intervene.py` writes against.

It also already has one fixed, declared, hashed corpus and one per-unit
activation pass over it. `compute_cofire()` in `interp/bundles.py` runs the
res-jb SAE encoder over every position of
`src/nebulai/backend/interp/corpus_alice.txt` (Alice's Adventures in Wonderland,
Project Gutenberg #11, public domain; sha256 `86bd0504…c563a234`, 44,527 tokens,
348 windows of 128, 44,179 counted positions after each window's position 0 is
dropped) and exports exact per-feature firing counts `n_i`, exact joint counts
`c_ij` across all 24,576 features, the independence expectation `n_i·n_j/N`, and
a seeded permutation yardstick that destroys pairing while keeping both
marginals.

So what is missing is not capture and not a corpus. It is:

- **Aggregation by cluster.** `compute_cofire()` scores *pairs of features*,
  selected by Dunning's G², and never reads `points[].cluster_id`. The quantity
  this layer wants — within-cluster co-activation lift against matched
  between-cluster pairs — is computed nowhere.
- **Coverage of the mapped units.** The corpus pass covers one SAE
  (`gpt2-small` / `blocks.8.hook_resid_pre`). Neuron maps have `mlp_post`
  available per prompt but nothing aggregates it over a corpus, and no other
  model has a corpus pass at all.
- **A corpus chosen for this purpose.** One 44k-token novel is a *disclosed*
  corpus, not a representative one, and "How it fails when built naively" below
  is largely about that choice.

The gate has not moved, and the four reasons it was drawn now stand differently:

- `recommended-plan.md` §"Track 4 — Optional, gated: activations frontend" is
  unchanged as of 2026-09-12 and still carries its marker verbatim: **"⚠️ Needs
  approval before any work"**, with "**Recommendation: skip until Track 2
  results argue for it**", and the sequencing table still lists it last, "only
  if 4 justifies it". Note what Track 4 actually proposes: a separate
  `nebulai[activations]` extra (torch + Transformers, GPU host) emitting
  **Units** — an activations *front-end*, which is a much larger thing than
  this layer needs.
- The no-torch rule still holds, and it is enforced rather than asserted now.
  `pyproject.toml`'s base dependencies stay torch-free, checked by
  `tests/test_no_torch_in_base.py`, which imports the CLIs with `torch` made
  unimportable; `frontends/sae.py` still reads `W_dec` with `huggingface_hub` +
  `safetensors.numpy` to avoid "a multi-GB torch dependency tree the repo
  deliberately excludes". Torch sits behind **two** separate opt-ins with
  different purposes and different enforcing tests: the extra
  `[project.optional-dependencies] organisms` (`torch, transformers, peft,
  datasets`), used only by `src/nebulai/organisms/`, and the dependency group
  `behavior-local` (pinned `torch==2.14.0`, `transformers==5.17.0`,
  `sentence-transformers==6.0.1`), used only by the Behavior study's local arms
  and enforced by `tests/test_behavior_optional_dep.py`. The `organisms` gate
  has already been opened once, deliberately, to train the emergent-misalignment
  rank-1 LoRA (`out/organisms/emergent_misalignment.json`, commit `f8d9819`) —
  so "the base install stays torch-free" is a demonstrated property now rather
  than an untested one.
- Track 4 also reintroduces the GPU host the current plan removed. (A CPU numpy
  forward pass is demonstrated at study scale — the absorbing-state study ran
  2,400 self-play conversations in 12,301 s, ~5.1 s each — so the GPU
  requirement is about the *activation matrix over a corpus*, not about running
  the model at all.)
- And an activation-based number is no longer "the first measurement in this
  project that is not about weights". The persona PC1 and its permutation
  control, the eval-awareness direction, `nebulai intervene`'s KL sweeps with
  their α = 0 no-hook control, the absorbing-state statistic and the Behavior
  study all ship, each under its own protocol and its own null, and the
  README's honesty notes were amended to permit exactly that: *"Causal claims
  only where an intervention was run, stated in the intervention's own terms —
  never about what a direction is."* The note those measurements did **not**
  repeal is the one about maps: *"Every model-derived map here answers 'what
  can this layer write', never 'what did it write for prompt X'"* — which is
  still true of every map `nebulai validate` grades, and is precisely why this
  layer would say something new.

**Nothing below is a proposal to open that gate.** It is the requirements list
that whoever opens it should already have in hand.

### Input data required

| Needed | Exists today? |
|---|---|
| A fixed, declared token corpus | **Partly.** One exists and is already in use: `src/nebulai/backend/interp/corpus_alice.txt`, sha256-stamped into `cofire.json` (44,527 tokens, 44,179 counted positions). It was chosen for one co-firing figure on one SAE, not for this layer, and no map declares a corpus of its own. |
| Per-unit activations over that corpus — the SAE encoder applied to the residual stream at the stamped hook for SAE maps, post-nonlinearity hidden activations at the stamped layer for neuron maps | **Partly.** A forward pass over text ships (`interp/gpt2_numpy.py` for GPT-2, `interp/llama_numpy.py` for Llama-architecture instruct models) with residual capture and the shared hook protocol in `interp/hooks.py`, and `compute_cofire()` already runs the res-jb SAE encoder over a whole corpus and exports exact per-feature firing counts. What does not exist is that capture for a *given map's* mapped unit set — SAE encoder outputs at the map's stamped hook, MLP post-nonlinearity at the map's stamped layer — retained per unit and aggregated by cluster. |
| A forward-pass runtime (torch / TransformerLens, or an equivalent) | Yes — the equivalent is pure numpy: `interp/gpt2_numpy.py` and `interp/llama_numpy.py`, both implementing `hooks.ForwardPass`. Torch stays optional and gated (two opt-ins, above); it is not needed to run a micro model. |
| Hardware to run it | **Partly.** A laptop CPU is enough for the runners that ship — 2,400 self-play conversations at ~5.1 s each. Track 4's own text calls out the ~60 GB GPU host it reintroduces; that is about a torch front-end over four models and about the size of the activation matrix, not about running one micro model over text. |
| Storage for the activation matrix | No. Note that `reduced.npz` already declines to cache the source vectors (50k × 768 float32 ≈ 150 MB per map) — an activation matrix over a corpus is larger again by orders of magnitude, and where it lives is a design decision, not a detail. |
| Cluster assignments to aggregate over | Yes — `points[].cluster_id`. |

There is also a front-end that this layer simply does not fit. **Token maps have
no activations.** A row of `W_E` is not a unit that fires; it is a vector that is
looked up. The nearest analogue would be corpus token frequency and context
statistics, which is a different quantity answering a different question, and
calling it "coherence" alongside the SAE and neuron numbers would make three
incomparable things share a column. Say so explicitly or leave token maps out.

### Protocol

1. **Declare the corpus.** Fixed, versioned, hashed, and recorded in `meta` the
   way every other replay-critical parameter already is.
2. **Capture.** Record activations for the mapped units only — note that
   `load_sae_units` defaults to `max_features=4096` out of a much larger
   dictionary, so the captured set is a *sample*, and the number must carry that
   the way `validate.py` carries `n_scored`.
3. **Sparsify.** Reduce each unit to its top-activating token set at a declared
   threshold or top-k.
4. **Score.** Cluster coherence is the co-activation lift for within-cluster
   unit pairs against matched between-cluster pairs — a normalised measure such
   as PMI or Jaccard over the top-activating sets, never a raw co-occurrence
   count.
5. **Null.** Shuffle the activation matrix column-wise and re-score, mirroring
   `null_baseline()`'s existing design and for the same reason. Report the
   margin.

### What the number licenses, and what it does not

It licenses: *the units grouped into this cluster tend to be active on the same
inputs, on this corpus*. That would be the first functional claim this project
can make, which is precisely why it is worth the gate and precisely why its
limits need writing down first.

It does not license:

- **Causality.** "These units fire together" is not "this cluster does the thing
  its title says". The roadmap keeps that as a separate item — intervention-based
  validation, ablating a cluster's units and measuring the behaviour change —
  and it stays separate.
- **Title correctness.** That is Layer A.
- **Corpus independence.** Coherence is measured *on a corpus* and is a
  statement about that corpus's distribution. A different corpus is a different
  number, and the corpus hash belongs beside the score forever.
- **Cross-front-end comparison**, for the reason given above.

### How it fails when built naively

- **Frequency dominance.** Co-activation on raw magnitudes is won by
  high-density units on high-frequency tokens: a cluster of common-token
  features looks perfectly coherent because everything is active there. Without
  frequency matching, a normalised lift, and the shuffled null, the metric
  mostly ranks clusters by how common their tokens are.
- **Circularity.** Measuring coherence on the corpus the SAE was trained on
  grades a dictionary of directions against the distribution those directions
  were fit to reconstruct. The corpus choice is part of the claim.
- **Silent subsetting.** A coherence number over 4,096 sampled features of a
  much larger dictionary that does not say so is the same error
  `scale_cluster_kwargs()` exists to prevent on the clustering side.
- **Conflating coherence with a validated title.** A tight, highly coherent
  cluster with a wrong title scores well here and badly in Layer A. The two
  layers are not substitutes and neither is a summary of the other.
- **Adding torch quietly.** Two optional surfaces now make it easy — the
  `organisms` extra and the `behavior-local` group — and the base install's
  torch-freedom is a load-bearing property of this repo, not an accident of
  packaging. Both are enforced by a test, and neither is a precedent for a
  third: the numpy runners already do the forward pass this layer needs. If
  Track 4 opens, it opens deliberately, with approval, and the base install
  stays as it is.
