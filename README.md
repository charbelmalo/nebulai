# Nebul.AI

A semantic cloud of a micro model's concept space: decompose a small model
into interpretable units, label each unit, embed and cluster the labels, and
render the whole thing as a navigable map of meaning.

Three interchangeable front-ends define *what a point is*; they all feed the
same back-end (reduce → cluster → name → export → render):

| Variant | A point is… | Status |
|---|---|---|
| **C — tokens** | one vocabulary token (its `W_E` embedding row) | ✅ working |
| **A — SAE features** | one sparse-autoencoder decoder direction | ✅ working |
| **B — MLP neurons** | one raw MLP write direction (`c_proj` row) | ✅ working |
| **P — probe concepts** | one LLM-proposed concept near a seed topic (no model weights) | ✅ working |

A/B/C are built and exported — see `out/` for artifacts from each. Both A and B
read weights directly through the shared safetensors loader, so neither needs
sae-lens or TransformerLens at runtime. P is the odd one out: it decomposes no
model at all, and exists to answer the human-language question the others
can't — see [Semantic probe](#semantic-probe--a-cloud-with-no-model-in-it).

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
4. **Name** — each cluster's most-central members go to a namer. `--namer auto` (default) tries **a local ollama server** → **OpenRouter** (key from `~/.config/nebulai/.env`) → a centroid-token fallback. `--namer` and `--ollama-model` / `--openrouter-model` control this.
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
space (`mxbai-embed-large` on the local ollama server), co-reduced, and re-clustered. A
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

## Semantic probe — a cloud with no model in it

The three front-ends above all decompose a model. `nebulai probe` does not: you
give it a word, an LLM proposes related concepts breadth-first, a text embedder
places them, and a cosine gate against the seed decides what stays.

```sh
uv run nebulai probe "grief" --depth 2 --breadth 12 --sensitivity 0.35
uv run nebulai probe "photosynthesis" --sensitivity 0.6   # near-synonyms only
```

- `--depth` how many hops from the seed; `--breadth` concepts requested per term.
- `--sensitivity` is a cosine floor **against the seed**, not against each term's
  parent — chaining parent similarity lets a depth-3 term drift arbitrarily far
  while every hop looks reasonable. 0 keeps everything proposed, ~0.35 keeps a
  recognisable topic, ~0.6 keeps near-synonyms.
- `--generator` picks the proposing LLM (`auto` → ollama → OpenRouter →
  Anthropic); `--embed-model` / `--embed-host` pick the embedder.

Unlike the three weight-reading front-ends, this one cannot run offline — it
needs a reachable generator *and* a reachable embedder. With neither configured
it exits naming each backend it tried and why.

**What a probe cloud is evidence of.** Two models' joint opinion — the generator
that proposed the terms and the embedder that positioned them. Not a fact about
language, and not the geometry of any model under study. A term that is absent
means the generator did not propose it; a term far from the seed means the
embedder put it there. Both are stamped into `meta` (`generator`, `embed_model`,
`n_proposed`, `kept`, `n_dropped`, `seed_similarity_min/mean`) and the map is
labelled with the same "NOT model-internal" geometry string the `api-` maps use.
The drop rate is the single most useful diagnostic: a high one means the
generator wandered and your map is narrower than what was actually proposed.

## Re-titling a map

Titles and geometry have separate lifetimes. The clusters are fixed by the time
the namer runs, so a better namer can be applied to a finished map without
moving a point — which matters, because rebuilding a 50k-point map to fix only
its titles means re-running UMAP to land back on the same coordinates.

```sh
uv run nebulai rename gpt2 distilgpt2 --claude-cli-model opus
uv run nebulai rename all          # every built map; placeholder maps are skipped
```

`--namer claude-cli` (the default) and `--namer codex-cli` shell out to the
`claude` / `codex` binaries, so naming a whole corpus runs on an existing
subscription instead of per-token API billing. Two things the command records
in `meta` rather than hiding:

- `renamed_from` — the namer it replaced.
- `reps_space: u_cluster` — exports do not carry the source vectors, so cluster
  representatives are ranked by centrality in the 10-D UMAP space HDBSCAN
  clustered in, not in the original embedding space. That is a different
  selection than the build path makes.

Maps whose labels are *all* placeholders (`neuron 3 (unlabeled)`) are refused,
not renamed — the same rule the build path enforces with `placeholder_titles`.
A namer handed only placeholders invents semantics from zero information.

## Validating a map

`nebulai metrics` reports silhouette, which is computed in `u_cluster` — the
same UMAP space HDBSCAN clustered in. That grades the projection using the
projection's own geometry, so it cannot tell you whether the clusters exist in
the model's original space. `nebulai validate` adds the three checks that can:

```sh
uv run nebulai validate gpt2 gpt2__neurons__h.8.mlp.c_proj
uv run nebulai metrics gpt2 gpt2__neurons__h.8.mlp.c_proj   # picks up the results
```

- **trustworthiness** — neighbourhood preservation from the ORIGINAL vector
  space (1.0 faithful, ~0.5 chance). Independent of HDBSCAN entirely.
- **seed stability** — mean pairwise ARI across UMAP seeds. Answers "would I
  have drawn the same map on a different day?"
- **null baseline** — the identical pipeline on column-shuffled vectors, which
  keep every per-dimension marginal but lose the correlations between
  dimensions. Whatever it scores is the floor `silhouette` has to clear.

These re-run UMAP, so they are a separate command rather than part of a build.
Results land in `validation.json` next to `nebulai.json`.

**What this currently shows**, across all nine maps:

| map | points | silhouette | null floor | margin | trust | seed ARI |
|---|---|---|---|---|---|---|
| gpt2 · tokens | 49857 | 0.4999 | 0.3772 | +0.123 | 0.72 | 0.54 |
| gpt2-medium · tokens | 49857 | 0.4700 | 0.4079 | +0.062 | 0.70 | 0.48 |
| distilgpt2 · tokens | 49857 | 0.5165 | 0.3645 | +0.152 | 0.74 | 0.48 |
| pythia-70m · tokens | 49385 | 0.5245 | 0.3726 | +0.152 | 0.89 | 0.51 |
| SmolLM2-135M · tokens | 48636 | 0.4929 | 0.3926 | +0.100 | 0.72 | 0.46 |
| gpt2 · MLP neurons | 3072 | 0.4814 | 0.4250 | +0.056 | 0.65 | 0.62 |
| SmolLM2-135M · MLP neurons | 1536 | 0.4827 | 0.3834 | +0.099 | 0.64 | 0.57 |
| gpt2-small · SAE features | 4096 | 0.5246 | 0.5737 | −0.049 ⚠ | 0.89 | 0.58 |
| SmolLM2-135M · SAE features | 36864 | 0.4770 | 0.4097 | +0.067 ⚠ | 0.71 | 0.57 |

⚠ = the null resolved a cluster count far from the map's own (16 vs 69; 277 vs
130). Silhouette rises as a partition coarsens, so those two rows compare
different questions and the margin is not evidence either way. `nebulai
metrics` prints `null.k` next to the margin and flags this case with `?`.

Three things to take from the seven rows that *are* comparable:

- **Every one clears its floor, and none clears it by much** — +0.06 to +0.15
  on a scale where silhouette itself sits near 0.5. There is real structure
  here, and it is a modest fraction of what the layout looks like it has.
- **Unit type does not sort the maps.** SmolLM2's raw-neuron map (+0.099) beats
  gpt2-medium's token map (+0.062). "Tokens carry structure, raw neurons don't"
  is not what the numbers say. The weakest comparable row is gpt2's neuron map
  (+0.056), and that isn't a settings problem: sweeping leaf/eom × `mcs` ×
  `min_samples` tops out at 0.4858 silhouette, barely past its 0.4250 floor.
- **Seed ARI is 0.46–0.62 everywhere.** Roughly half of each partition is
  seed-dependent. Individual cluster boundaries are not stable findings; the
  gross layout is.

One methodological note, because it changed these numbers substantially.
HDBSCAN's `min_cluster_size` is an absolute point count, so a 4000-point null
carrying a 49k-point map's value clusters at a completely different
granularity. `validate` rescales it by `n_sample / n_full`; before that fix
gpt2's null read 0.2147 (51 clusters) instead of 0.3772 (236), and the margins
looked 2–3× larger than they are. Any subsampled comparison has to do this.

Explore clustering settings without re-running UMAP:

```sh
scripts/sweep_hdbscan.py out/gpt2/reduced.npz     # leaf/eom x mcs x min_samples
scripts/inspect_map.py out/gpt2/nebulai.json      # meta, top clusters, size dist
```

## SessionSeer — the agent's trajectory, not the model's concept space

The four front-ends above map what a *model* knows. `nebulai seer` maps what an
*agent* did: it captures Codex, Claude Code and Hermes sessions into one event
vocabulary and reports on them live. It shares the viewer shell and the
provenance rules, and shares none of the reduce → cluster → name back-end — a
trajectory is already low-dimensional and ordered.

```sh
uv run nebulai seer run codex "fix the failing test"     # drive one agent and capture it
uv run nebulai seer run codex "…" --compare-with claude hermes   # same prompt, three agents
uv run nebulai seer install claude --apply    # hooks, to capture what you run yourself
uv run nebulai seer serve --watch             # HTTP + SSE for the viewer's Seer page
```

Every value it reports is labelled `native | deterministic | estimated |
heuristic | missing | dropped_by_policy`, and `missing` is never rendered as
`0`. Design and rationale: [`docs/SESSIONSEER.md`](docs/SESSIONSEER.md); what
the build actually does, including where it departs from the design:
[`docs/SESSIONSEER-HANDOVER.md`](docs/SESSIONSEER-HANDOVER.md).

## Honesty notes

- **Plan C's geometry is the model's own** (embedding rows). For Plans A/B, laying points out by *label* embeddings shows the label-embedder's semantics, not the model's — the viewer will expose both projections (decoder-direction vs label space) as a toggle.
- Raw token-embedding structure is partly frequency/orthography; mean-centering + cosine mitigate but don't remove that.
- Cluster selection defaults to `leaf`, which deliberately over-fragments: `eom` collapses token maps into one mega-cluster. That choice raises the noise fraction *and* lowers seed stability, so read both numbers against the method (`nebulai validate` prints it).
- This is a visualization + clustering tool over public micro models. No causal claims.

## Roadmap

- Held-out auto-interp scores and activation-based coherence — the two validation layers `nebulai validate` does not yet cover (it measures geometry and stability, not whether a cluster predicts behaviour).
- Intervention-based validation: does ablating a cluster's units change the behaviour its title claims?
- Phase 2: WebGPU point cloud reading `nebulai.json` — 3D flythrough, hover, cluster hulls, filters, 2D↔3D toggle. (The `compare` viewer is the first cut of this renderer.)
- Cross-model: Route B (orthogonal Procrustes alignment on shared tokens) as a geometry-space companion to the current concept-space `compare`, for same-family models.
