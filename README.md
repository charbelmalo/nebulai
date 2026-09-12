# Nebul.AI

A semantic cloud of a micro model's concept space: decompose a small model
into interpretable units, label each unit, embed and cluster the labels, and
render the whole thing as a navigable map of meaning.

Interchangeable front-ends define *what a point is*; they all feed the same
back-end (reduce → cluster → name → export → render):

| Variant | A point is… | Status |
|---|---|---|
| **C — tokens** | one vocabulary token (its `W_E` embedding row) | ✅ working |
| **A — SAE features** | one sparse-autoencoder decoder direction | ✅ working |
| **B — MLP neurons** | one raw MLP write direction (`c_proj` row) | ✅ working |
| **P — probe concepts** | one LLM-proposed concept near a seed topic (no model weights) | ✅ working |
| **U — unembedding** | one vocabulary token (its `W_U` row) — untied models only | ✅ working |

A/B/C/U are built and exported — see `out/` for artifacts from each. Both A and B
read weights directly through the shared safetensors loader, so neither needs
sae-lens or TransformerLens at runtime. P is the odd one out: it decomposes no
model at all, and exists to answer the human-language question the others
can't — see [Semantic probe](#semantic-probe--a-cloud-with-no-model-in-it). U is
C's twin on the output side and asks the question C alone cannot: *does the
model read tokens the way it writes them?* It is meaningless on a tied model
(one matrix serves both roles), which is what makes a tied model the control —
see [Reading a model you never download](#reading-a-model-you-never-download).

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
4. **Name** — each cluster's most-central members go to a namer. `--namer auto` (default) tries **a local ollama server** → **OpenRouter** (key from `~/.config/nebulai/.env`) → a centroid-token fallback. `--namer` and `--ollama-model` / `--openrouter-model` control this. When the namer is a *pinned* corpus model rather than whatever is reachable, two extra rules apply — identity and cost — see [Reading a model you never download](#reading-a-model-you-never-download).
5. **Export + render** — `nebulai.json`, static PNG, interactive HTML.

## Reading a model you never download

The atlas used to be bounded by disk: mapping a model meant fetching its
checkpoint. It isn't anymore. `https://huggingface.co/{repo}/resolve/{rev}/{shard}`
answers **HTTP 206 Partial Content with no auth**, so the loader reads the
safetensors header — a few hundred KB of JSON carrying every tensor's dtype,
shape and byte range — and then streams only the rows it maps. Projected across
the four models below from their measured tensor shapes: **1.87 GB of streamed
rows against 344 GB of checkpoints.** Building them actually cost **1.45 GB**,
because Nemo is mapped at 5k tokens rather than 50k.

Byte ranges, not shards, are what make this work. The shard holding
Muse-Glimmer-30B's `W_E` is 49.95 GB of its 59.55 GB checkpoint; "download only
the shard you need" would have fetched 84% of it.

| model | repo | checkpoint | W_E | W_U | predicted 50k stream | measured | map |
|---|---|---|---|---|---|---|---|
| Muse-Glimmer-30B | `meta-models/Muse-Glimmer-30B` | 59.55 GB | BF16 `[202048, 6656]` | untied | 666 MB (1.12%) | **690 MB** | ✅ 50k pts · 121 clusters · 66.0% noise |
| Gemma-4-26B-A4B-it | `google/gemma-4-26B-A4B-it` | 51.61 GB | BF16 `[262144, 2816]` | **tied** | 282 MB (0.55%) | **283 MB** | ✅ 50k pts · 322 clusters · 36.7% noise |
| Ling-2.6-flash | `inclusionAI/Ling-2.6-flash` | 208.37 GB | BF16 `[157184, 4096]` | untied | 410 MB (0.20%) | **416 MB** | ✅ 50k pts · 209 clusters · 54.3% noise |
| Mistral-Nemo-Instruct-2407 | `mistralai/Mistral-Nemo-Instruct-2407` | 24.50 GB | BF16 `[131072, 5120]` | untied | 512 MB (2.09%) | 56 MB @5k | ✅ 5k pts · 101 clusters · 34.1% noise |

Every measured figure is the map's own `meta.bytes_fetched`, and each lands
within 1.5% of the prediction the header read made before a single row moved —
except Nemo, which is mapped at 5k tokens rather than 50k, so its stream is a
tenth of the row and the 512 MB column stays a prediction until it is rebuilt.

Ling is the case that settles the design: 208 GB across 26 numbered shards plus
a separate `model-mtp-layer.safetensors`, 25,015 tensors — and a token map needs
0.2% of it. Gemma-4's **tied** embeddings make it the control for the W_E vs W_U
question rather than a gap in the corpus: with one matrix serving both roles the
answer is known in advance, which is the only calibration that overlap score
has. Exact keys, per-model architecture notes and prices live in
[`src/nebulai/corpus.py`](src/nebulai/corpus.py), which is the source of truth.

Three rules come with this, and they are as much a part of the architecture as
the range reads:

- **Pin the revision.** `main` floats. A run resolves it to a commit sha and
  stamps it into `meta`, so "the model changed under us" stays distinguishable
  from "the pipeline changed."
- **Pin the model's identity — never substitute.** A cheaper endpoint serving a
  *different* model is not a fallback; its titles are not this model's
  semantics. Model ids are pinned (never a family, never an alias) and an
  unavailable one is a refusal, not a downgrade. Two of the four models have **no
  HF-router route at all** (no provider serves them), and `corpus.py` records
  that as `None` rather than filling it with a near neighbour — an absent route
  is information, and `probe_endpoints.py` re-checks it on every run.
- **Estimate cost before spending it.** A gate (default $1.00,
  `corpus.DEFAULT_MAX_COST_USD`) prices a run from the measured request shape
  and, over budget, names cheaper alternatives for a human to choose. For scale:
  naming a 250-cluster map at Glimmer's rate is **$0.019**, and re-naming every
  map currently in `out/` (14 of them, 1660 clusters) is **$0.125**.

Check any of this yourself — it needs no API key, sends no chat request, and
therefore costs nothing:

```sh
uv run scripts/probe_endpoints.py                 # revisions, keys, bytes, routes, cost
uv run scripts/probe_endpoints.py --rows 8        # range-read and decode real rows
uv run scripts/probe_endpoints.py --weights-only  # the half that needs no credentials
```

It is stdlib-only (not even numpy), so `python3.11+ scripts/probe_endpoints.py`
works too — including when the venv is the thing that is broken.

"No key configured" is one of its normal results, not an error. Prices are read
live from OpenRouter and drift against `corpus.py` is flagged rather than
silently trusted.

**Status. All four are mapped** (2026-08-12) — every row above carries its own
`meta.bytes_fetched` and a revision pinned to a resolved sha, and each map is
scored in the validated-map table further down. Three things the first pass
taught, all fixed in the tree rather than in this file:

- **`--namer auto` is not a good default for a fresh machine.** Its chain is
  ollama → openai → openrouter → centroid, so with no local server and no keys
  every build lands on `centroid`, which titles a cluster by joining its four
  most central tokens (`北京 · 上海 · 广州 · 四川`). The geometry is unaffected and
  `nebulai rename --namer claude-cli` re-titles a built map in place.
- **Reserved slots are not always flagged special.** gpt2's `<|endoftext|>` is,
  so it decodes to `""` and curation drops it; Gemma-4 flags only ids 0-4, so
  160 `<unusedN>` rows survived into its first map and clustered with each
  other. `_keep` now drops those families by name.
- **A long stream needs a retry the short ones never exercised.** A body that
  dies mid-read raises `IncompleteRead`, which is an `HTTPException` and not an
  `OSError`, so it slipped past `_request`'s retry loop — invisible on the
  56-283 MB reads, fatal 8 minutes into Glimmer's 690 MB one.

The remote range-read loader lives in `src/nebulai/weights.py` and the
pinned-identity namer in `src/nebulai/backend/name.py`, and those files, not
this one, are the truth about what they currently support. Plan and rationale:
[`recommended-plan.md`](recommended-plan.md); what measurement corrected along
the way: [`updated-implementation-plan.md`](updated-implementation-plan.md).

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

### Route B — the geometry-space companion

`compare` works in concept space precisely because two models share no
embedding basis. Route B is the test of that premise rather than a way around
it: over the tokens two models' vocabularies actually share, it fits the single
**orthogonal** map that best carries one cloud onto the other and reports how
much of the target it fails to explain.

```sh
uv run nebulai compare gpt2 distilgpt2 --route-b
# -> out/compare/route_b__gpt2__distilgpt2.json
```

Three things keep the residual from being a number that always looks good:

- **It is held out.** The rotation is fitted on half the shared tokens and
  scored on the other half. The in-sample residual is printed beside it so the
  gap is visible instead of hidden.
- **The null refits.** Each of the permutations shuffles the token
  correspondence and fits that shuffle its *own* best rotation. A null that
  reused the true rotation would be beaten by anything.
- **A pair that cannot be asked is refused.** Under 256 shared token strings
  the command stops and points at concept-space `compare`; it does not align
  whatever it can find. Unequal widths (gpt2's 768 vs gpt2-medium's 1024) are
  handled by the semi-orthogonal solution and flagged `square_rotation: false`,
  because a projection can only lose structure.

Measured, gpt2 → distilgpt2 (49,857 shared tokens, 100% of the vocabulary, 200
permutations):

| | |
|---|---|
| residual, held out | **0.0531** on 24,929 tokens |
| residual, fit half | 0.0512 on 24,928 tokens |
| permutation null | mean 1.9996, best 1.9977 |
| p | 0.004975 (the floor, 1/201) |

A single rotation explains ~95% of distilgpt2's token geometry from gpt2's —
unsurprising for a distillation, and exactly the sanity check the method needed.
It says the two arrange a shared vocabulary alike **up to a change of basis**.
It says nothing about behaviour and ranks neither model.

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
it exits naming each backend it tried and why. The endpoint route removes the
*hardware* half of that constraint (a good generator no longer has to fit in
local VRAM) but not the caveat itself, which was never about hardware. It also
puts this front-end squarely under the identity rule: a probe cloud is the joint
opinion of the generator and the embedder that produced it, so which generator
answered is part of the map's provenance, not an implementation detail — and
substituting a cheaper one silently changes what the cloud is evidence of.

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

**What this currently shows**, across all twenty-two **built and validated**
maps. Every number here came from a run — a planned map earns a row once it is
built and has cleared its null floor, not before. The four endpoint-era models
([above](#reading-a-model-you-never-download)) now have rows; three of them
have a second row for the `W_U` twin of their token map, built over the same
curated vocabulary, and one a third for an `api_text_embedding` contrast drawn
over that vocabulary too. A `down_proj` depth series on the same model —
identical parameters at five depths — adds five more rows.

**Six maps in `out/` carry no `validation.json`, and not one of them ever
will.** All six *cannot* be validated: three `api_text_embedding` contrast
maps (`Xenova__claude-tokenizer__api-sentence-transformers__all-MiniLM-L6-v2`,
`gpt2__api-mxbai-embed-large`,
`mistralai__Mistral-Nemo-Instruct-2407__api-mxbai-embed-large`) and the three
`probe__*` clouds. Their vectors came from a hosted embedding service over the
network, and replaying `meta` cannot reproduce them, because nothing in the
artifact pins *which weights answered*. `nebulai validate` refuses those with
that reason rather than scoring them.

A fourth contrast map, `mistralai__Mistral-Nemo-Instruct-2407__api-all-MiniLM-L6-v2`,
**does** validate and has a row below. Not because it is a better map: because it
was built through the in-process `local` embedder with a full 40-hex revision
sha, which is exactly the condition `validate.py`'s `api_text_embedding` branch
reloads on. The difference between that map and the three above it is a **pin**,
not a quality judgement, and it is the clearest argument in this repo for
pinning an embedder. Read its numbers with that in mind — a 0.95
trustworthiness is not evidence that sentence embeddings describe a tokenizer
better than the model's own weights do, it is what a 384-dimensional space
produced by a model trained to make sentences cluster scores when you project it
to two dimensions.

Nothing else is outstanding: every map in `out/` that *can* be validated now
has been, so the rows below are the whole validated corpus rather than a
snapshot of one in flight:

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
| Gemma-4-26B · tokens | 50000 | 0.5270 | 0.3821 | +0.145 | 0.67 | 0.48 |
| Ling-2.6-flash · tokens | 50000 | 0.4730 | 0.3864 | +0.087 | 0.67 | 0.49 |
| Mistral-Nemo · tokens | 5000 | 0.4968 | 0.2033 | +0.293 | 0.75 | 0.50 |
| Muse-Glimmer-30B · tokens | 50000 | 0.4899 | 0.4812 | **+0.009** | 0.84 | 0.48 |
| Ling-2.6-flash · **W_U** | 50000 | 0.5629 | 0.3805 | +0.182 | 0.65 | 0.55 |
| Mistral-Nemo · **W_U** | 5000 | 0.4741 | 0.2805 | +0.194 | 0.69 | 0.49 |
| Muse-Glimmer-30B · **W_U** | 50000 | 0.5553 | 0.3749 | +0.180 | 0.76 | 0.45 |
| Mistral-Nemo · **api text** | 5000 | 0.6050 | 0.2396 | +0.365 | 0.95 | 0.54 |
| Mistral-Nemo · neurons **L4** | 4096 | 0.3816 | 0.2006 | +0.181 | 0.63 | 0.58 |
| Mistral-Nemo · neurons **L12** | 4096 | 0.4053 | 0.2945 | +0.111 | 0.61 | 0.57 |
| Mistral-Nemo · neurons **L20** | 4096 | 0.3786 | 0.2123 | +0.166 | 0.59 | 0.62 |
| Mistral-Nemo · neurons **L28** | 4096 | 0.2466 | 0.1930 | +0.054 | 0.60 | 0.24 |
| Mistral-Nemo · neurons **L36** | 4096 | 0.2971 | 0.2028 | +0.094 | 0.57 | 0.32 |

⚠ = the null resolved a cluster count far from the map's own (16 vs 69; 277 vs
130). Silhouette rises as a partition coarsens, so those two rows compare
different questions and the margin is not evidence either way. `nebulai
metrics` prints `null.k` next to the margin and flags this case with `?`.

The four corpus rows are all comparable (each null landed within 0.5-2x of its
map's own k), and they split. Gemma-4 posts the highest silhouette of any map
here and a healthy +0.145. Nemo's +0.293 is the largest margin of those four —
though not of the whole table, where Mistral-Nemo · **api text** clears by
+0.365 — but it is scored on 5k points, where the null has less room to invent
islands — read it as encouraging, not as a win over the 50k rows.
**Muse-Glimmer-30B is the cautionary one: +0.009.** Its projection is faithful
(trust 0.84, second only to pythia), so the *layout* is trustworthy; what is
barely-better-than-null is the claim that its 121 clusters are separated. At
66% noise and 6656 dimensions, HDBSCAN is describing the two-thirds it
discarded more than the third it kept. Do not read Glimmer's territories as
findings without a parameter sweep behind them.

Three things to take from the seven older rows that *are* comparable:

- **Every one clears its floor, and none clears it by much** — +0.06 to +0.15
  on a scale where silhouette itself sits near 0.5. There is real structure
  here, and it is a modest fraction of what the layout looks like it has.
- **Unit type does not sort the maps.** SmolLM2's raw-neuron map (+0.099) beats
  gpt2-medium's token map (+0.062). "Tokens carry structure, raw neurons don't"
  is not what the numbers say. The weakest of those seven is gpt2's neuron map
  (+0.056), and that isn't a settings problem: sweeping leaf/eom × `mcs` ×
  `min_samples` tops out at 0.4858 silhouette, barely past its 0.4250 floor.
- **Seed ARI is 0.46–0.62 across all seven.** Roughly half of each partition
  is seed-dependent. Individual cluster boundaries are not stable findings; the
  gross layout is. Do not read that band as a property of the pipeline: the
  depth series below reaches down to 0.24 at layer 28, so a low ARI is a thing
  a map can have and these seven happen not to.

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
scripts/probe_endpoints.py                        # corpus reachability + cost matrix
```

## SessionSeer — the agent's trajectory, not the model's concept space

The four front-ends above map what a *model* knows. `seer` maps what an
*agent* did: it captures Codex, Claude Code and Hermes sessions into one event
vocabulary and reports on them live. It shares the viewer shell and the
provenance rules, and shares none of the reduce → cluster → name back-end — a
trajectory is already low-dimensional and ordered. It is its own command, not
a `nebulai` subcommand: `nebulai` never imports it.

```sh
uv run seer run codex "fix the failing test"     # drive one agent and capture it
uv run seer run codex "…" --compare-with claude hermes   # same prompt, three agents
uv run seer install claude --apply    # hooks, to capture what you run yourself
uv run seer serve --watch             # HTTP + SSE for the viewer's Seer page
```

Every value it reports is labelled `native | deterministic | estimated |
heuristic | missing | dropped_by_policy`, and `missing` is never rendered as
`0`. Design and rationale: [`docs/SESSIONSEER.md`](docs/SESSIONSEER.md); what
the build actually does, including where it departs from the design:
[`docs/SESSIONSEER-HANDOVER.md`](docs/SESSIONSEER-HANDOVER.md).

The Seer page also *watches* a run happen — one time axis, three meanings of
`y`, a GPU field that carries no magnitude, and a thought rail with a state for
every way reasoning can be absent. Set-up and how to read it:
[`docs/SESSIONSEER-LIVE-SETUP.md`](docs/SESSIONSEER-LIVE-SETUP.md); the design,
including the projection that was refused and why:
[`docs/SESSIONSEER-LIVE.md`](docs/SESSIONSEER-LIVE.md).

An audit of the viewer against a twelve-feature LLM-observability spec — what
ships, what needs only a view, and what needs a pipeline first — plus the plan
to lift the flat deck.gl charts onto the `three/webgpu` stack the Atlas already
uses: [`docs/OBSERVABILITY-SURFACE.md`](docs/OBSERVABILITY-SURFACE.md).

## Honesty notes

- **Plan C's geometry is the model's own** (embedding rows). For Plans A/B, laying points out by *label* embeddings shows the label-embedder's semantics, not the model's — the viewer will expose both projections (decoder-direction vs label space) as a toggle.
- Raw token-embedding structure is partly frequency/orthography; mean-centering + cosine mitigate but don't remove that.
- Cluster selection defaults to `leaf`, which deliberately over-fragments: `eom` collapses token maps into one mega-cluster. That choice raises the noise fraction *and* lowers seed stability, so read both numbers against the method (`nebulai validate` prints it).
- **Weight geometry, not activations.** Every model-derived map here answers "what can this layer *write*", never "what did it write for prompt X". Reaching models over endpoints does not change that: the rows are read from the checkpoint, and the endpoint is only ever the namer. A map is evidence about the weights; the model that titled it is provenance.
- This is a visualization + clustering tool over public micro models. No causal claims.

## Roadmap

- Behavioral semantic divergence: a separate, research-gated association study
  and a focused **Behavior** page, alongside the clouds rather than inside them.
  Phases 0-2 are built — `src/nebulai/behavior/`, the `nebulai behavior`
  subcommands, and the Behavior page. **Two separate things are missing, and
  they are missing for different reasons.** The xAI arm has no key on this
  machine, so it is recorded in its manifest as `not_run` with a cost estimate
  and the runner refuses it rather than skipping quietly; Phase 3 is gated
  behind a human terms review
  ([`docs/behavior/TOS-REVIEW.md`](docs/behavior/TOS-REVIEW.md), an empty form).
  The *local* GPT-2 arms are runnable and partially collected, but not at the
  preregistered 4,800 trials: fp32 GPT-2-XL measured 3.9 min/trial on this
  16 GB machine, where 6.4 GB of weights page in and out per forward pass. So
  the runner grew `--cue-limit`, which truncates by **cue** rather than by
  trial — every cue that runs keeps its full repeat count and block balance —
  and every export carries a `coverage` block the Behavior page turns into a
  banner. A study that covered 40 of 100 cues says so on its own face; an
  artifact with no `coverage` block is reported as *unknown*, never as complete.
  Research method, statistical confirmation, data contracts,
  implementation phases, and UX plan:
  [`docs/BEHAVIORAL-DIVERGENCE-PLAN.md`](docs/BEHAVIORAL-DIVERGENCE-PLAN.md).
- Generative variance: does a model's *story architecture* — not its prose —
  move from run to run, and does it move less than the gap between models? The
  within-model number has to exist before the between-model one means anything,
  so W1 (one model, repeated trials) gates W2 (across models). Proposal, not a
  baseline — nothing is built and no paid call has been made:
  [`docs/GENERATIVE-VARIANCE-PLAN.md`](docs/GENERATIVE-VARIANCE-PLAN.md).
- Held-out auto-interp scores and activation-based coherence — the two
  validation layers `nebulai validate` does not yet cover. It measures
  geometry and stability; it never reads a cluster title, so nothing in the
  suite currently grades the naming. What each layer would need, and which
  of the two is gated behind Track 4 activations:
  [`docs/VALIDATE-LAYERS.md`](docs/VALIDATE-LAYERS.md).
- Intervention-based validation — does ablating a cluster's units change the
  behaviour its title claims? — is now Phase 4 of the attractors work rather
  than a loose roadmap line: [`docs/ATTRACTORS-PLAN.md`](docs/ATTRACTORS-PLAN.md).
- W_E vs W_U on the untied corpus models — which token families a model reads
  differently from how it writes them — **measured, not proposed, on all three
  untied corpus models.** Mean kNN neighbourhood overlap (k=50, raw cosine, no
  UMAP in the path) is **0.342** on Mistral-Nemo over 5k tokens, **0.525** on
  Ling-2.6-flash over 50k and **0.299** on Muse-Glimmer-30B over 50k, against
  chance baselines of 0.0100, 0.0010 and 0.0010 — so 34x, 525x and 299x chance,
  and still a majority of each token's neighbours *changing* between the two
  matrices in every case. The tied control pins the other end: Gemma-4-26B,
  whose `W_U` **is** its `W_E`, scores **1.0000** on every one of the three
  measures and in all seven token families, which is what makes the untied
  numbers readable as a real gap rather than as pipeline noise. Cluster
  agreement falls much further than neighbourhood overlap in all three (ARI
  0.092 / 0.050 / 0.025; title Jaccard 0.282 / 0.132 / 0.136) — the two spaces
  keep a token's rough company while disagreeing about what neighbourhood it
  lives in. The per-family breakdown does **not** generalize across models:
  punctuation has the highest overlap on Nemo and Ling, while on Glimmer it is
  numerals (0.613) highest and Latin words (0.286) lowest, which inverts the
  other two. Full tables, the per-family figures and the validation of the three
  new `W_U` maps: [`recommended-plan.md`](recommended-plan.md)
  § "Track 2b — result".
