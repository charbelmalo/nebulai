# Nebul.AI — Microdetails

Everything in the pipeline, stage by stage, with the reasoning behind each
decision. Companion to the [README](../README.md); this is the "why is this
line here" document.

The pipeline is five stages behind one contract:

```
front-end ──> Units ──> reduce ──> cluster ──> name ──> export + render
 (A/B/C)              (UMAP)     (HDBSCAN)   (LLM)    (json/png/html)
```

---

## 0. The `Units` contract — `src/nebulai/units.py`

Every front-end must reduce to this dataclass, and nothing downstream is
allowed to know which front-end produced it:

| Field | Type | Meaning |
|---|---|---|
| `ids` | `list[int]` | Stable unit identifiers (token id, feature index, neuron index). Survive curation/truncation so a point can always be traced back to the model. |
| `vectors` | `np.ndarray (n, d) float32` | The **geometry** — what UMAP sees. For Plan C this is the model's own embedding rows; for A/B it will be decoder directions or label embeddings. |
| `labels` | `list[str]` | Human-readable display string per unit (token string, auto-interp label). Used for hover text and cluster naming, never for geometry. |
| `meta` | `dict` | Provenance: model id, unit type, which weight key was used, whether centering happened, counts. Copied into the export so every artifact is self-describing. |

Microdetails:

- `__post_init__` hard-fails if `len(ids) != len(vectors) != len(labels)`.
  Misalignment here would silently attach the wrong label to a point and
  poison every downstream stage, so it's a constructor-time invariant, not a
  runtime check.
- `float32` everywhere: halves memory vs float64 and is what umap-learn
  computes in anyway; there's no precision benefit upstream of a stochastic
  projection.
- The separation of `vectors` (geometry) from `labels` (text) is the load-bearing
  design decision of the whole project. It's what lets Plans A/B later offer a
  "model space vs label space" toggle by swapping only `vectors` while
  everything else stays identical.

---

## 1. Front-end C: token embeddings — `src/nebulai/frontends/tokens.py`

**What a point is:** one vocabulary token, positioned by its row of the
embedding matrix `W_E`.

### Weight loading

- Downloads only `model.safetensors` via `hf_hub_download` — never the full
  repo, never torch. `safetensors.numpy.load_file` gives numpy arrays
  directly, so the project has **no torch dependency** for Plan C.
- The embedding matrix is found by suffix match against
  `_EMBED_KEY_SUFFIXES = ("wte.weight", "embed_in.weight", "embed_tokens.weight")`,
  which covers GPT-2 (`wte`), Pythia/GPT-NeoX (`embed_in`), and the
  Llama/Qwen/Gemma family (`embed_tokens`). Suffix (not exact) match tolerates
  prefixes like `transformer.` or `model.`.
- A GPT-2-specific fact worth knowing: GPT-2 **ties** `W_E` and `W_U`
  (embedding = unembedding), so this map doubles as a map of the model's
  output vocabulary directions. Untied models would show only the input side.

### Vocabulary curation — `_keep(s)`

A token survives only if its decoded string is displayable:

1. Non-empty after decoding, and not pure whitespace.
2. No U+FFFD replacement character — GPT-2's byte-level BPE has ~100s of
   tokens that are *fragments of multi-byte UTF-8 sequences*; they decode to
   `�` and are meaningless as standalone points.
3. No control characters.

This is display curation, not semantic filtering — we drop what can't be
rendered or hovered, nothing else. `meta` records both `vocab_size` (before)
and `kept` (after) so the cut is auditable.

### `max_tokens` truncation

`--max-tokens N` keeps the **lowest N token ids**. This isn't arbitrary: BPE
merges are learned greedily by frequency, so merge order ≈ token id order ≈
corpus frequency. Truncating by id therefore keeps the *most frequent* tokens
— the quick pass sees common words, not the long tail of rare byte-combos.

### Mean-centering (`center=True`, default)

Token embedding spaces are notoriously **anisotropic**: all vectors share a
large common direction, so raw cosine similarity between any two tokens is
inflated and structure is compressed into a narrow cone. Subtracting the mean
row removes the shared component and lets cosine distances reflect *relative*
differences. `--no-center` exists so the effect is demonstrable rather than
asserted.

---

## 2. Reduce — `src/nebulai/backend/reduce.py`

One call, three coordinated outputs:

```python
u_cluster, u3, u2 = reduce_vectors(vectors, cluster_dim=10, n_neighbors=30, seed=42)
```

| Output | How | Used for |
|---|---|---|
| `u_cluster` (n, 10) | UMAP, `min_dist=0.0` | clustering **only** |
| `u3` (n, 3) | UMAP, `min_dist=0.1` | 3D flythrough (Phase 2) |
| `u2` (n, 2) | **PCA of `u3`** | static PNG + interactive HTML |

Microdetails and the reasoning:

- **Cluster in ~10-D, never in 2-D/3-D.** 2-D UMAP embeddings famously
  manufacture visual clusters that don't exist in the original space and
  merge ones that do. 10 dimensions preserves enough neighborhood structure
  for density clustering while still denoising the original 768-D.
- **`min_dist=0.0` for the clustering space** packs points as tightly as the
  topology allows — exactly what a density-based clusterer wants. The display
  spaces use `min_dist=0.1` so points don't overplot into blobs.
- **`u2` is a PCA of `u3`, not an independent UMAP run.** Two separate UMAP
  runs would give unrelated layouts; deriving 2D from 3D guarantees the flat
  map is literally a camera angle on the 3D cloud, so Phase 2's 2D↔3D toggle
  will animate coherently instead of teleporting points.
- **`metric="cosine"`** throughout: direction, not magnitude, carries meaning
  in embedding rows (magnitude correlates with token frequency).
- **Seed semantics:** `seed >= 0` sets `random_state` for reproducibility,
  which forces umap-learn into single-threaded mode. `--seed -1` omits it,
  unlocking parallelism — faster, but layouts differ run to run. The default
  (42) favors reproducible artifacts.

---

## 3. Cluster — `src/nebulai/backend/cluster.py`

```python
cluster_ids, probs = cluster_units(u_cluster, min_cluster_size=None, min_samples=None)
```

- **HDBSCAN** (scikit-learn ≥ 1.4's native implementation) with
  `cluster_selection_method="leaf"` by default. The textbook choice is
  `"eom"` (excess of mass), but on token-embedding UMAP spaces eom collapses
  ~96% of points into one mega-cluster — the vocab core is a single connected
  density blob. Leaf selection takes the finest-grained clusters in the
  hierarchy instead, recovering the concept groups at the cost of a higher
  noise fraction (on GPT-2's top-5k: eom → 5 clusters / 0% noise, leaf →
  ~84 clusters / ~42% noise, and the leaf clusters are the coherent ones).
  `--cluster-method eom` keeps the coarse view available.
- **No k.** The number of clusters is discovered, not imposed — the honest
  choice when the question is "does this space *have* structure."
- **Noise is a feature.** Points HDBSCAN can't assign get `cluster_id = -1`
  and stay in the visualization as unlabeled background. The noise fraction is
  printed and exported as a headline quality metric: a token map that's 90%
  noise is a finding, not a bug to hide.
- **Default `min_cluster_size = max(15, n // 1000)`** scales with corpus size:
  ~50 for the full GPT-2 vocab (so a "days of the week"-sized concept can
  surface) with a floor of 15 so tiny runs don't fragment into micro-clusters.
- **`probabilities` → confidence.** HDBSCAN's membership probability is
  carried through the export as per-point `confidence`, which the viewer maps
  to opacity — periphery members literally fade toward the noise floor.

---

## 4. Name — `src/nebulai/backend/name.py`

```python
titles, backend_used = name_clusters(units, cluster_ids, namer="auto", ...)
```

### Representative selection — `_representatives()`

Each cluster is summarized by its **k=20 members nearest the cluster centroid
by cosine similarity**, computed in the *original* embedding space (not the
UMAP space — we want the model's notion of centrality, not the projection's).
Centroid-nearest beats random sampling because outer members of a cluster are
exactly the ambiguous ones.

### The fallback chain

`--namer auto` tries each backend in order, falling through on *any*
exception. The order favors local + free + private first:

1. **`ollama`** — a local ollama server at `http://localhost:11434`; probes
   `/api/tags` first (fast fail if unreachable, and it never selects an `embed`
   model), then `/api/generate` with `format=json`. Default model
   `liquidai/lfm2.5-1.2b-instruct` — small, free, local, good enough for 2–5-word
   titles.
2. **`openrouter`** — default `openai/gpt-oss-120b:free`; key from
   `OPENROUTER_API_KEY` or the last uncommented line of `~/.config/nebulai/.env`.
   Structured output via `response_format` json_schema, batched **15 per call**.
3. **`centroid`** — zero-dependency floor: joins the top-4 centroid-nearest
   member strings with `" · "` (e.g. ` Monday ·  Tuesday ·  Friday ·  Sunday`).
   Not pretty, but honest, deterministic, and it means **the pipeline can
   never fail at the naming stage**.

**`anthropic`** (`claude-opus-4-8`, structured output via `output_config.format`
json_schema, batched 15/call) stays available but only via `--namer anthropic`
— it's not in the `auto` chain.

The backend actually used is returned and stamped into `nebulai.json` — an
artifact always discloses whether its titles came from an LLM or a heuristic.

---

## 5. Export — `src/nebulai/backend/export.py`

Writes `nebulai.json`, the **contract between Phase 1 and the Phase-2 viewer**.
The viewer should need this file and nothing else.

```jsonc
{
  "meta": {
    "schema_version": 2,
    "n_points": 48934, "n_clusters": 312, "noise_fraction": 0.21,
    "namer": "centroid", "created": "2026-07-07T...", // + Units.meta provenance
  },
  "points": [
    { "id": 0, "unit_ref": { "kind": "token_embedding", "index": 464 },
      "label": " the", "confidence": 0.97, "layer": null,
      "xy": [1.2, -3.4], "xyz": [1.1, -3.2, 0.8], "cluster_id": 17 }
  ],
  "clusters": [
    { "id": 17, "title": "articles & determiners", "size": 240,
      "centroid": [1.0, -3.1, 0.7] }
  ],
  "edges": {                          // schema v2 — similarity beams
    "space": "umap10", "metric": "gaussian_euclidean",
    "k_cluster": 5, "sigma": 3.02,
    "cluster_edges": [[0, 17, 0.93], ...],       // [a, b, weight], a < b
    "knn": { "k": 6, "sigma": 0.07,
             "ids": [/* int, n_points*k flat */],
             "sims": [/* float, n_points*k flat */] }
  }
}
```

- `unit_ref` is a typed `{kind, index}` object — `kind` is `Units.meta["unit"]`
  (`token_embedding` for Plan C; later `sae_feature`, `mlp_neuron`) — so mixed
  maps stay unambiguous.
- Both `xy` and `xyz` ship per point — the 2D↔3D toggle is a client-side
  interpolation, no recompute.
- Cluster `centroid` is in `u3` (display) space: it's where the viewer parks
  the camera / anchors the floating title, not a semantic quantity.
- `noise_fraction` and `namer` live in `meta` so any consumer can render an
  honest caption without re-deriving anything.
- **`edges` (schema v2, `backend/edges.py`)** — similarity edges for the
  viewer's connection beams, computed in the **10-D `u_cluster` space HDBSCAN
  clustered in**, never the 2-D/3-D display layout (honesty guardrail: beams
  reflect the geometry the clustering saw, not the picture). Metric is a
  Gaussian kernel over Euclidean distance — the metric HDBSCAN used — with
  `sigma` (median candidate distance) stamped so weights are self-describing;
  cosine saturates on UMAP output coordinates (measured [0.91, 1.0] on gpt2)
  and is only right on the original embedding rows. `cluster_edges` is the
  deduped union of each cluster centroid's top-`k_cluster` neighbors; `knn`
  is exact per-point nearest neighbors stored flat (`ids`/`sims` of length
  `n_points * k`, row `i` at `[i*k, (i+1)*k)`, self excluded) so the viewer
  can copy them straight into typed arrays. Controlled by
  `nebulai tokens --edges {knn,cluster,none}` (default `knn`, ~+4MB).
  **Backfill without recompute:** `nebulai edges <model>...` rebuilds the
  block from the cached `reduced.npz` (`u_cluster`) and restamps
  `schema_version: 2` — no UMAP rerun. Consumers must treat a missing
  `edges` / `schema_version` as v1 and degrade (beams off), not fail.
  `nebulai tokens` and `nebulai edges` also maintain `out/index.json`
  (`{"datasets": [...]}`) so a static viewer can discover datasets.

---

## 6. Render — `src/nebulai/backend/viz.py`

Phase-1 visualization via **datamapplot** (from the UMAP author; built for
exactly this "labeled 2D map of an embedding" shape).

- **Static PNG** (`create_plot`, darkmode, dpi=150): labels only the **top 60
  clusters by size** (`_names_per_point(..., max_labels=60)`); everything else
  renders as `"Unlabelled"` background. 300+ overlapping titles would make the
  overview unreadable — the PNG is the thumbnail, not the explorer.
- **Interactive HTML** (`create_interactive_plot`): all clusters labeled
  (zoom-dependent rendering handles density), per-point
  `hover_text=repr(token)` — `repr` deliberately, so GPT-2's leading-space
  tokens are visible (`' cat'` vs `'cat'` are *different tokens* and that
  distinction is half the story of a BPE vocab map) — plus text search over
  points.
- Noise (`cluster_id = -1`) maps to `noise_label="Unlabelled"` in both.

---

## 7. CLI orchestration — `src/nebulai/cli.py`

`uv run nebulai tokens [flags]` runs stages 1–5 with per-stage wall-clock
timing printed as `[k/5] ... [12.3s]`.

- **Reduction caching.** UMAP is minutes; everything else is seconds. The
  three reductions are saved to `out/<model>/reduced.npz` next to
  `reduced.params.json` holding the exact parameter dict
  `{model, max_tokens, center, cluster_dim, n_neighbors, seed}`. On the next
  run the cache is reused **only on exact params match**, so you can iterate
  on `--min-cluster-size` or `--namer` for free, while changing any
  reduction-relevant flag transparently recomputes. `--force` busts it.
- Output root is `out/<model-with-slashes-mangled>/` (`/` → `__`) so
  `EleutherAI/pythia-70m` doesn't create nested dirs.
- Flags map 1:1 to the knobs above: `--max-tokens`, `--no-center`,
  `--cluster-dim`, `--n-neighbors`, `--min-cluster-size`, `--min-samples`,
  `--seed`, `--namer`, `--namer-model`, `--ollama-model`, `--force`.

---

## Plans A and B — what changes, microscopically

Both are **new front-ends only**; every file under `backend/` stays untouched.

### Plan A — SAE features (flagship)

- **Point =** one sparse-autoencoder feature (sae-lens; GPT-2 `res-jb` or
  Gemma Scope releases).
- **`ids`** = feature indices; **`labels`** = auto-interp descriptions,
  bootstrapped from Neuronpedia where they exist, generated (hybrid
  local/API) where they don't, and **scored by detection** (Delphi/EleutherAI
  protocol: does the label predict which snippets activate the feature?) so
  label quality is a number, not a vibe.
- **`vectors` — the honest-framing crux:** two candidate geometries.
  *Decoder directions* (the feature's column of `W_dec`) are the **model's
  own** space. *Label embeddings* (mxbai-embed-large over the descriptions)
  are the **label-embedder's** semantics — clusters there tell you about the
  labeling model as much as the subject model. Both get built; the export
  carries both; the viewer exposes the toggle. Conflating them is the #1 way
  this genre of visualization lies.
- **Sampling:** stratified across firing-rate deciles, not top-k — top-k
  activation sampling over-represents dense features and hides rare
  monosemantic ones.
- `unit_ref = "sae:<layer>/<feature>"`, `meta` gains the sae-lens release id.

### Plan B — raw MLP neurons (comparison artifact)

- **Point =** one MLP hidden neuron, hooked via TransformerLens;
  `vectors` = the neuron's output-weight row (its write direction into the
  residual stream); labels auto-interp'd the same way as Plan A.
- **Purpose is the contrast, not the map.** Neurons are polysemantic; the
  prediction is measurably worse structure than Plan A on the *same* backend
  and metrics: higher noise fraction, lower silhouette in the clustering
  space, lower label-detection scores, less coherent clusters. Because A and
  B share every downstream stage, the comparison is apples-to-apples by
  construction — that quantitative table *is* the artifact.
- `unit_ref = "neuron:<layer>/<idx>"`.

---

## Honesty guardrails (restated, because they bind every stage)

1. This is a **visualization + clustering tool over public micro models. No
   causal claims** — nothing here shows what a unit *does* to model behavior,
   only how units' vectors relate geometrically.
2. Plan C's geometry is the model's own; token-embedding structure is partly
   frequency/orthography and centering mitigates but doesn't remove that.
3. Plans A/B must never present label-space layout as model geometry — the
   two projections stay separate and labeled as such.
4. Noise fractions, namer backend, and curation counts are exported, not
   hidden: every artifact carries the evidence needed to distrust it.
