# Change-propagation protocol (the "recommendation expert")

## Why this exists

Nebul.AI is not three tools — it is three decompositions of a model laid out on
**one identical back-end** so they can be compared. Plan A (SAE features) is the
flagship; Plan B (neurons) only means something as an apples-to-apples contrast
against A; Plan C (tokens) is the day-one scaffold that de-risks the back-end.
The comparison is only valid if the back-end and the shared conventions stay
identical across all three.

Therefore: an improvement that lands in one pipeline and stops there is a
latent bug. It either (a) silently makes the pipelines incomparable, or (b)
leaves a known-better technique unused elsewhere. The job of this protocol is to
make every change either propagate or be *deliberately* declared local.

## The protocol — run it after every change

**1. Classify the change by layer.**

| Layer | Path | Default blast radius |
|---|---|---|
| Shared back-end | `src/nebulai/backend/*` | **All pipelines already inherit it.** |
| Front-end | `src/nebulai/frontends/<plan>.py` | One pipeline — but check step 2. |
| CLI | `src/nebulai/cli.py` | Per-subcommand; mirror new flags across subcommands. |
| Docs | `docs/`, `README.md`, `.claude/skills/*` | Update every pipeline's doc that the change touches. |

**2. Decide whether the *reason* generalizes.** Ask "is this about the unit
type, or about the geometry/pipeline?"

- **Pipeline-specific (stays local):** things intrinsic to that unit type —
  byte-fragment token curation, SAE dead-feature filtering, neuron indexing.
  These have no analogue in the other pipelines.
- **Generalizes (propagate or recommend):** anything about reduction,
  clustering, naming, confidence, export shape, interpretation methodology, or a
  gotcha you hit that the others will also hit. If you fixed it once because the
  geometry misbehaved, the same geometry misbehaves everywhere.

**3. Act, then recommend.**

- Shared back-end change → note which pipelines' *outputs* will shift and
  whether their cached `reduced.npz` needs `--force` (only reduction-parameter
  changes bust the cache; clustering/naming changes are cheap re-runs).
- Generalizing front-end change → either apply it to the sibling front-ends now
  (if they exist and it's clearly right) or, if it needs judgment, **recommend**.
- **Always end the message with ONE concrete recommendation question**, naming
  the specific pipeline(s) and change, phrased as a one-word yes/no. If the
  change is genuinely isolated, say so explicitly ("no cross-pipeline impact")
  rather than staying silent — the user should never have to guess whether you
  checked.

## Propagation table (known couplings)

| A change to… | Almost always also affects… | Because |
|---|---|---|
| HDBSCAN defaults (leaf/eom, min_cluster_size, min_samples) | all pipelines inherit the code, but see "Provisional defaults" below | same clustering geometry; the mega-cluster failure mode is universal, but the *specific* settings were tuned on one manifold |
| UMAP params (cluster_dim, n_neighbors, min_dist, metric) | all pipelines | shared reducer; changes invalidate cross-pipeline comparison if not uniform |
| Namer chain, prompt, or schema | all pipelines | shared namer; titles must be produced the same way to compare clusters |
| `nebulai.json` schema / `unit_ref` shape | all pipelines **and** the Phase-2 viewer | it's the shared contract; a field rename breaks every consumer |
| Confidence → opacity mapping | all pipelines | shared export + viewer semantics |
| "model space vs label space" projection toggle | Plans A & B (not C) | C's geometry is already the model's own; A/B carry both projections |
| datamapplot render options | all pipelines | shared viz |
| A new interpretation gotcha (e.g. orthographic clusters) | likely the others | usually a property of embedding geometry, not the unit type |
| Viewer chrome (any `viewer/src/chrome/` component) | design-skill **component review gate** (see `nebulai-viz`) | chrome PRs are gated on the four design skills' checklist (targets, keyboard path, focus, radii, tabular-nums, empty states, reduced motion) |
| Token vocab curation (`_keep`) | tokens only | intrinsic to BPE byte-level tokenization |
| SAE dead/duplicate-feature filtering | SAE only | intrinsic to sparse autoencoders |
| Neuron layer selection / hooking | neurons only | intrinsic to TransformerLens MLP hooks |

## Provisional defaults (validated on tokens only — re-sweep for A and B)

The shared back-end ships `cluster_selection_method="leaf"` and `min_samples=5`.
These are the *code* defaults for all three pipelines, but they were validated on
**exactly one manifold**: GPT-2 token embeddings after mean-centering. The
failure they fix — `eom` collapsing a single connected density blob into one
mega-cluster — is a real, universal HDBSCAN behavior, so `leaf` is a sound
*starting point* everywhere. But the *magnitudes* (`min_samples=5`,
`min_cluster_size = max(15, n//1000)`) were fit to token geometry.

SAE decoder-direction space and neuron write-direction space have different
intrinsic dimensionality and density structure — sparser, higher-dimensional,
plausibly with genuinely well-separated modes where `eom` is actually correct
and `leaf` over-fragments. So for Plans A and B, treat these as **provisional,
not settled**:

- **Inherit the code defaults** so the first end-to-end run works.
- **Re-sweep before trusting the map.** Run `scripts/sweep_hdbscan.py
  out/<model>/reduced.npz` on the SAE/neuron reductions and confirm the
  noise-fraction / cluster-count knee looks right for *that* geometry rather than
  assuming the token settings transfer.
- **If a pipeline needs different settings, that's expected — not a propagation
  violation.** The shared *back-end code* stays identical; only the per-run
  parameter values differ, and they belong in that pipeline's `meta` so the
  A-vs-B comparison records that clustering was tuned per-manifold. What must NOT
  diverge is the metric definitions and the export shape.

This is why the propagation rule for HDBSCAN is "inherit the mechanism, re-validate
the numbers," not "copy the token defaults verbatim."

## Worked examples

**Example — local backend change already propagated.**
Input: switched HDBSCAN default from `eom` to `leaf` in `backend/cluster.py`.
Assessment: shared back-end → all pipelines inherit it; the mega-cluster failure
is a property of the UMAP density blob, not of tokens.
Ending recommendation: *"This is now the shared default, so Plan A and Plan B
inherit leaf selection when built. The token map re-clusters from cache in ~2s
(no `--force` needed since reductions are unchanged). But leaf/`min_samples=5`
were validated only on the token manifold — see 'Provisional defaults' above:
A and B should inherit the mechanism and re-sweep the numbers, not copy them
blind. Want me to record that provisional status so the sibling pipelines
re-validate rather than assume — yes/no?"*
(Resolved 2026-07-07: yes — recorded in the "Provisional defaults" section.)

**Example — generalizing front-end change.**
Input: added mean-centering to the token front-end to fight anisotropy.
Assessment: `frontends/tokens.py`, but the *reason* (embedding anisotropy) is a
geometry property SAE decoder directions and neuron write-directions can share.
Ending recommendation: *"Centering helped the token geometry; SAE decoder
directions and neuron write-directions may be anisotropic too. Want me to make
centering a shared `Units`-level option so Plan A/B can toggle it the same way —
yes/no?"*

**Example — genuinely local change.**
Input: extended `_keep()` to drop a new class of byte-fragment tokens.
Assessment: intrinsic to BPE tokenization; SAE features and neurons have no
byte-fragment analogue.
Ending: *"No cross-pipeline impact — this is BPE-specific curation."*
