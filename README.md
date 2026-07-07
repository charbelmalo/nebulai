# Nebul.AI

A semantic cloud of a micro model's concept space: decompose a small model
into interpretable units, label each unit, embed and cluster the labels, and
render the whole thing as a navigable map of meaning.

Three interchangeable front-ends define *what a point is*; they all feed the
same back-end (reduce → cluster → name → export → render):

| Variant | A point is… | Status |
|---|---|---|
| **C — tokens** | one vocabulary token (its `W_E` embedding row) | ✅ working |
| **A — SAE features** | one sparse-autoencoder feature (sae-lens) | planned |
| **B — MLP neurons** | one raw MLP neuron (TransformerLens) | planned |

## Quickstart

```sh
uv sync
uv run nebulai tokens --model gpt2            # full curated vocab (~15-30 min, cached)
uv run nebulai tokens --model gpt2 --max-tokens 5000   # quick pass
```

Outputs land in `out/<model>/`:

- `nebulai.json` — the map: per point `{id, unit_ref, label, confidence, xy, xyz, cluster_id}`, per cluster `{id, title, size, centroid}`. This is the contract the Phase-2 WebGPU viewer will load.
- `map_static.png` — labeled overview (datamapplot).
- `map_interactive.html` — zoomable map with per-token hover and search.

## Pipeline

1. **Front-end** — produce `Units`: ids + geometry vectors + display labels. For Plan C the geometry is the (mean-centered) embedding matrix itself, curated to drop byte-fragment and control tokens.
2. **Reduce** — UMAP (cosine): a ~10-d space for clustering, 3-d for the flythrough, and a 2-d view projected from the 3-d one so the views stay aligned. Clustering never runs on the 2-d/3-d projections — they invent structure.
3. **Cluster** — HDBSCAN; membership probability becomes per-point confidence.
4. **Name** — each cluster's most-central members go to a namer. `--namer auto` (default) tries local **ollama on the M4 worker** → **OpenRouter** (key from `~/.hermes/.env`) → a centroid-token fallback. `--namer` and `--ollama-model` / `--openrouter-model` control this.
5. **Export + render** — `nebulai.json`, static PNG, interactive HTML.

## Comparing models

Combine several models' clouds into one categorized, navigable WebGPU map:

```sh
uv run nebulai tokens --model gpt2
uv run nebulai tokens --model distilgpt2
uv run nebulai tokens --model EleutherAI/pythia-70m
uv run nebulai compare gpt2 distilgpt2 EleutherAI/pythia-70m
# -> out/compare/index.html   (open in Chrome/Edge)
```

Different models don't share an embedding basis, so we **don't** concatenate raw
geometries (that just splits into per-model blobs — an artifact, not a finding).
Instead each model's *named clusters* are embedded in a neutral third-party
space (`mxbai-embed-large` on the M4 worker), co-reduced, and re-clustered. A
meta-cluster drawing from several models is a **shared concept**; one from a
single model is **unique**. The command prints a concept-overlap (Jaccard)
table and per-model unique counts.

The viewer (`out/compare/index.html`) is a self-contained WebGPU point cloud.
Each point stores its position in four **layout states**, and the GPU
interpolates smoothly (`smoothstep` on a uniform `t`) between them when you
switch:

| State | What you see |
|---|---|
| **Native clouds** | each model's own 3D cloud, in its own quadrant |
| **Semantic space** | unified concept space — matching concepts from different models converge |
| **By model** | models fanned into columns (each model's footprint) |
| **By concept** | points collapse onto their meta-cluster (shared, multi-model knots pop out) |

Colors encode the source model; a "shared concepts only" filter isolates the
overlap; hover a point for its concept title, source model, and token count.

## Honesty notes

- **Plan C's geometry is the model's own** (embedding rows). For Plans A/B, laying points out by *label* embeddings shows the label-embedder's semantics, not the model's — the viewer will expose both projections (decoder-direction vs label space) as a toggle.
- Raw token-embedding structure is partly frequency/orthography; mean-centering + cosine mitigate but don't remove that.
- This is a visualization + clustering tool over public micro models. No causal claims.

## Roadmap

- Plan A: sae-lens features (GPT-2 res-jb / Gemma Scope), stratified sampling across firing rates, labels bootstrapped from Neuronpedia + hybrid local/API labeler scored by detection (Delphi).
- Plan B: raw MLP neurons; quantitative neurons-vs-SAE comparison (noise fraction, silhouette, snippet coherence).
- Phase 2: WebGPU point cloud reading `nebulai.json` — 3D flythrough, hover, cluster hulls, filters, 2D↔3D toggle. (The `compare` viewer is the first cut of this renderer.)
- Cross-model: Route B (orthogonal Procrustes alignment on shared tokens) as a geometry-space companion to the current concept-space `compare`, for same-family models.
