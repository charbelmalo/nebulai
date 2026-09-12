# Recommended plan — four models over endpoints, none of them downloaded

This is the plan of record. It replaces the two earlier plans, both of which
were written around one specimen (`meta-models/Muse-Glimmer-30B`) and two
assumptions the user has rejected: **download the 59.55 GB checkpoint**, and
**serve the model locally on a ~60 GB GPU box** so nebulai can use it as a
namer. Neither is necessary, and dropping both changes the shape of the whole
project: the corpus goes from one model to four, the hardware requirement goes
from a GPU host to a laptop, and the cost of the instrument role goes from a
box to about two cents per map.

What nebulai is has not changed. It is a **weight-geometry atlas**: every
model-derived cloud is built from static weight rows — W_E, `down_proj` rows,
SAE decoder directions — read without torch, then pushed through one
model-agnostic backend (reduce → cluster → name → export → validate → compare).
It maps **what a layer can write**, not what it wrote for prompt X. Nothing
below moves that line; the activations question is still Track 4, still gated,
and now has one fewer reason to exist.

## Three rules, each with a measurement behind it

**1. No download.** `https://huggingface.co/{repo}/resolve/{rev}/{shard}`
answers **HTTP 206 Partial Content with no auth** on all four repos. So the
loader reads the safetensors header (a few hundred KB of JSON that carries
every tensor's dtype, shape and byte range) and then streams only the rows it
maps. Measured today across the corpus: **1.87 GB of streamed rows against
344 GB of checkpoints, 0.54%.** Verified by decoding real rows, not by
arithmetic — Glimmer's first rows come back finite at norm 5.098.

*This is not the same as the old "download only the shard you need" idea, which
does not work.* The shard holding Glimmer's W_E is **49.95 GB** of its 59.55 GB
checkpoint; Gemma-4's is 49.91 GB of 51.61 GB. Shard-granular fetching would
have downloaded 84–97% of those two checkpoints. Byte ranges are the thing that
makes this architecture real; shards are just where the bytes happen to live.

**2. No local serving.** The namer and the probe generator reach models over
remote OpenAI-compatible endpoints — OpenRouter, and HF Inference Providers'
router where a provider actually serves the model. A local ollama server stays
supported and stays first in the chain when it is up; it is no longer the only
way to get a good namer, and no model has to fit in local VRAM to be usable as
one.

**3. No silent substitution.** *A cheaper endpoint that serves a different
model is not a fallback — it is a different model, and its titles are not this
model's semantics.* This is the sharpest rule in the plan and the easiest one
to violate by accident, because every routing layer in this ecosystem is built
to substitute. So: the model id is pinned (never a family, never an alias), and
when the pinned id is unavailable the run **refuses**. A cost gate
(`--max-cost-usd`, default $1.00 in `corpus.DEFAULT_MAX_COST_USD`) estimates
before spending; over budget it **names cheaper alternatives for a human to
choose** and stops, rather than picking one. A map's `meta` records which model
named it, so a map is never evidence about a model that did not produce it.

## The corpus

Four models. Every column measured today (2026-08-12) against each repo's
`model.safetensors.index.json`, its shard headers, and the live OpenRouter
catalogue; re-runnable with `scripts/probe_endpoints.py`.

| model | repo | checkpoint | shards | W_E | W_U | 50k-token stream | $/M in·out |
|---|---|---|---|---|---|---|---|
| Muse-Glimmer-30B | `meta-models/Muse-Glimmer-30B` | 59.55 GB | 2 | BF16 `[202048, 6656]` | untied | 666 MB (1.12%) | 0.35 · 1.50 |
| Gemma-4-26B-A4B-it | `google/gemma-4-26B-A4B-it` | 51.61 GB | 2 | BF16 `[262144, 2816]` | **tied** | 282 MB (0.55%) | 0 · 0 (`:free`) |
| Ling-2.6-flash | `inclusionAI/Ling-2.6-flash` | 208.37 GB | 26 + mtp | BF16 `[157184, 4096]` | untied | 410 MB (0.20%) | 0.010 · 0.030 |
| Mistral-Nemo-Instruct-2407 | `mistralai/Mistral-Nemo-Instruct-2407` | 24.50 GB | 5 | BF16 `[131072, 5120]` | untied | 512 MB (2.09%) | 0.019 · 0.030 |

Two consequences worth stating plainly, because they change the experiment
rather than just widening it:

**Gemma-4's tied embeddings make it the control, not a failure.** The headline
question — *does the model read tokens the way it writes them?* — is true by
construction for a tied model and carries no information there. Three untied
models plus one tied control is a better design than one untied specimen: the
control tells you what a W_E↔W_U overlap score looks like when the answer is
known to be "identical," which is the only calibration that number has.

**Ling is the argument for range reads.** 208.37 GB across 26 numbered shards
plus a separate `model-mtp-layer.safetensors`, 25,015 tensors. A token map needs
**410 MB of it — 0.2%.** Nothing about the download-era plan scales to this
model; this architecture does not notice it.

Per-model detail (exact key paths, MoE flags, architectural notes) lives in
`src/nebulai/corpus.py`, which is the source of truth. Anything in this document
that disagrees with it is stale.

## Track 1 — Remote loader groundwork (blocks everything)

Owned by `src/nebulai/weights.py`, which lands alongside this document.
Described here as architecture, not as an API: quote signatures from the module,
never from this file.

1. **Range-read remote reader.** Resolve the shard index, read shard headers
   over HTTP, and read only the requested tensors' byte ranges. *Difference:*
   nothing else in this plan runs without it, and it retires the "which of these
   models fits on this disk" question permanently. Two gotchas already measured:
   HF answers a case-mismatched repo id with a **307** (`google/gemma-4-26b-a4b-it`
   → `google/gemma-4-26B-A4B-it`), and every `resolve/` URL **302s to a CDN
   host** — the client must follow redirects and keep the `Range` header across
   them, or it silently downloads whole shards.
2. **Revision pinning, stamped into `meta`.** `main` floats. Resolve it to a
   commit sha at read time and record it. *Difference:* without it a map cannot
   be replayed, and "the model changed under us" is indistinguishable from "the
   pipeline changed."
3. **Suffix-based key resolution, extended.** The corpus needs four W_E
   families, not three: `wte` / `embed_in` / `embed_tokens` / **`word_embeddings`**
   (Ling), two of them nested under `model.language_model.` by a multimodal
   wrapper (Glimmer, Gemma-4). *Difference:* exact-key lookup fails on three of
   the four models.
4. **Index-driven shard routing, not filename convention.** Ling's index routes
   to `model-mtp-layer.safetensors`, which the `-of-00026` naming pattern does
   not predict; Mistral's repo also carries a `consolidated.safetensors` that is
   **not** in the index and must not be counted as a shard. *Difference:* a
   loader that globs filenames reads the wrong file or double-counts the
   checkpoint.

## Track 2 — The maps

Existing pipeline, no new ML code, in this order.

**2a. W_E tokens map, per model.** Start with `mistral-nemo`: dense, untied,
conventional key layout, smallest checkpoint — the entry most likely to work
first and therefore the one to debug the remote reader against. Then Glimmer,
Gemma-4, Ling. Run a `--max-tokens 20000` pass first on each to shake out
tokenizer and curation issues before the 50k map. *Difference:* four clouds at
a scale the corpus has never held (its largest today is 63,619 points from a
neutral embedder; its largest model-geometry map is 49,857).

**2b. W_E vs W_U — the headline experiment.** For the three untied models, map
W_U over the same curated vocab and ask: does the model read tokens the way it
writes them? Concretely: W_E↔W_U neighbourhood overlap per token, cluster-title
agreement, and which token families (code, CJK, numerals, byte fragments)
diverge most between input and output geometry. Then run the same measurement
on Gemma-4, where it must come back degenerate — that is the control, and a
non-degenerate result there means the measurement is broken, not that Gemma-4
is interesting. Cost: one more matrix per model — both matrices at 50k tokens
run 819 MB for Ling, 1024 MB for Mistral-Nemo, 1331 MB for Glimmer. Softcapping and output multipliers are irrelevant by construction —
they act on logits at inference and never touch the rows being mapped.

**2c. Depth series of neurons maps.** `down_proj` write directions at a spread
of layers. ⚠️ Two of the four are MoE (Gemma-4: 128 experts; Ling: 256,
`BailingMoeV2_5`), so their `down_proj` lives **inside experts** — "layer L's write directions" is
not one matrix there, and the frontend has to be told which expert(s), or the
map is of one arbitrary expert while claiming to be of a layer. Do the dense
model (Mistral-Nemo, 40 layers) and Glimmer (52 layers, dense) first; treat MoE
depth maps as a separate design question, not a parameter change.

**2d. `api_tokens` contrast map.** The same curated vocab through a neutral
embedder — the existing "model geometry vs meaning" teaching contrast, now
available at four different vocab scales (131k / 157k / 202k / 262k).

**2e. Validation gate, unchanged.** Every map runs `nebulai validate`
(trustworthiness, seed-ARI, column-shuffled null) before entering the corpus or
`compare`. No exceptions for new or expensive models — the README table grows
honest rows or it does not grow.

**2f. Cross-model compare.** Add validated maps to `nebulai compare` alongside
gpt2 / SmolLM2 / pythia. First time the atlas can put four modern 20–30B-class
models against micro models; expect the unique-concept counts to move sharply,
which is itself the finding. Comparison happens in label space, not logit
space, so nothing about the four models' different vocabularies or output
transforms needs reconciling.

## Track 2b — result

Measured 2026-09-12. Every number below comes from
`scripts/we_wu_overlap.py`, over the **same curated vocabulary, at the same
token count and the same pinned revision** as each model's shipped W_E map.
The cluster columns read the two shipped artifacts in `out/` rather than
re-deriving a partition, so they are the agreement between two maps a
reader can open.

**What the measurement is.** For each token, take its *k*=50 nearest
neighbours in W_E and in W_U (cosine, on the raw rows, before any
reduction) and report the size of the intersection over *k*. This is a
geometric agreement score between a model's input and output token spaces.
It is not a quality score, it ranks no model, and a low value is not a
defect — an untied model is *allowed* to read tokens differently from how
it writes them, and whether it does is the entire question.

### Neighbourhood overlap (k = 50)

| model | tokens | mean overlap | median | chance | ×chance | zero-overlap tokens |
|---|---:|---:|---:|---:|---:|---:|
| Mistral-Nemo-Instruct-2407 | 5000 | **0.3419** | 0.3400 | 0.010002 | 34.2× | 6 |
| Ling-2.6-flash | 50000 | **0.5254** | 0.5400 | 0.001000 | 525.4× | 0 |
| Muse-Glimmer-30B | 50000 | **0.2991** | 0.2800 | 0.001000 | 299.1× | 127 |
| gemma-4-26b-a4b-it (**tied — control**) | 50000 | **1.0000** | 1.0000 | 0.001000 | 1000.0× | 0 |

### Cluster-level agreement (the shipped maps' own partitions)

| model | ARI(W_E, W_U) | mean title Jaccard | tokens clustered in both | noise frac W_E | noise frac W_U |
|---|---:|---:|---:|---:|---:|
| Mistral-Nemo-Instruct-2407 | 0.0919 | 0.2817 | 1764 | 0.3406 | 0.5204 |
| Ling-2.6-flash | 0.0496 | 0.1323 | 15781 | 0.5432 | 0.3544 |
| Muse-Glimmer-30B | 0.0250 | 0.1361 | 10919 | 0.6598 | 0.3896 |
| gemma-4-26b-a4b-it (**tied — control**) | 1.0000 | 1.0000 | 31633 | 0.3673 | 0.3673 |

### Which token families diverge most

| model | alphanumeric | cjk | latin_word | non_latin_script | numeral | other | punctuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mistral-Nemo-Instruct-2407 | — | 0.2725 <sub>n=229</sub> | 0.3388 <sub>n=3667</sub> | 0.3365 <sub>n=782</sub> | 0.4367 <sub>n=12</sub> | 0.3768 <sub>n=124</sub> | 0.4804 <sub>n=186</sub> |
| Ling-2.6-flash | — | 0.5290 <sub>n=17021</sub> | 0.5213 <sub>n=31334</sub> | 0.5136 <sub>n=245</sub> | 0.5738 <sub>n=26</sub> | 0.5518 <sub>n=309</sub> | 0.5824 <sub>n=1065</sub> |
| Muse-Glimmer-30B | — | 0.2873 <sub>n=4474</sub> | 0.2858 <sub>n=34513</sub> | 0.2970 <sub>n=8315</sub> | 0.6130 <sub>n=1110</sub> | 0.2882 <sub>n=430</sub> | 0.4571 <sub>n=1158</sub> |
| gemma-4-26b-a4b-it (**tied — control**) | 1.0000 <sub>n=12</sub> | 1.0000 <sub>n=1846</sub> | 1.0000 <sub>n=35744</sub> | 1.0000 <sub>n=10906</sub> | 1.0000 <sub>n=44</sub> | 1.0000 <sub>n=172</sub> | 1.0000 <sub>n=1276</sub> |

### Validation of the new W_U maps (`nebulai validate`)

| map | points | trustworthiness | seed ARI | silhouette | null silhouette | margin | noise |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mistralai__Mistral-Nemo-Instruct-2407__unembed` | 5000 | 0.6876 | 0.4926 | 0.4741 | 0.2805 | +0.1936 | 0.5204 |
| `inclusionAI__Ling-2.6-flash__unembed` | 50000 | 0.6489 | 0.5463 | 0.5629 | 0.3805 | +0.1824 | 0.3544 |
| `meta-models__Muse-Glimmer-30B__unembed` | 50000 | 0.7572 | 0.4501 | 0.5553 | 0.3749 | +0.1804 | 0.3896 |
| `google__gemma-4-26b-a4b-it` | 50000 | 0.6718 | 0.4791 | 0.5270 | 0.3821 | +0.1449 | 0.3673 |

The last row is not a new artifact: `google__gemma-4-26b-a4b-it` is the model's single tied map, listed because the control's W_U column *is* that map and its metrics are what the control is calibrated against. All three genuinely new W_U maps clear their null floors by between +0.1804 and +0.1936, wider than any of the five original token maps managed (+0.06 to +0.15). **That is not a claim that a W_U map is better drawn.** Silhouette margin says a partition is separated; trustworthiness says the projection you are looking at keeps the neighbourhoods it came from. On that second axis these maps span 0.6489 (`inclusionAI__Ling-2.6-flash__unembed`) to 0.7572 (`meta-models__Muse-Glimmer-30B__unembed`) — a range, not a verdict. Against the 9 validated **W_E** token maps in `out/` (0.6703–0.8877), the lowest of the new W_U maps falls below all of them and the highest sits above 7 of them and below 2. So these are not systematically less faithful projections than the maps the project already ships; they land inside the same band, at both ends of it. A map can still separate clusters it has placed unfaithfully, and the bottom of this range is the least faithful projection of any token map in the corpus, W_E or W_U.

### What it says

**The control passes exactly.** Gemma-4's embeddings are tied, so W_U *is* W_E, and every statistic above returns the degenerate value: mean overlap 1.0000, 50000/50000 tokens at full overlap, ARI 1.0000, title Jaccard 1.0000, and 1.0000 in every one of the seven token families. That is the only calibration the untied numbers have. A pipeline that scored 0.98 on a copy of its own input would make every row below it unreadable, because there would be no way to tell measurement noise from a real difference.

**All three untied models separate their two spaces, and none of them separates them completely.** Mistral-Nemo-Instruct-2407 0.3419 (34.2x chance over 5000 tokens); Ling-2.6-flash 0.5254 (525.4x chance over 50000 tokens); Muse-Glimmer-30B 0.2991 (299.1x chance over 50000 tokens). Read each against its own chance baseline and against the control, **not against each other**: the curated slices run from 5000 to 50000 tokens, so the chance baselines differ by an order of magnitude and the mean overlaps answer questions of different difficulty. What is comparable is the shape of the answer, and the shape is the same every time - between 30% and 53% of each token's fifty nearest neighbours survive the move from W_E to W_U, hundreds of times what shuffling would give, and nowhere near the 1.0 a tied model returns. The spread across models is itself a result: the amount of geometry the two matrices share is a property of the model, not a constant of transformers.

**The cluster agreement is much weaker than the neighbourhood agreement, and that is the finding with consequences.** ARI is 0.0919 for Nemo and 0.0496 for Ling - near-zero on a scale where 0 is chance - while the mean Jaccard between the two cluster *titles* a token sits under is 0.2817 and 0.1323. So local geometry is largely preserved and the partition over it is not. HDBSCAN's boundaries move even where the neighbours do not, which is what you would expect of a density partition over a space whose densities shifted; the noise fractions moving in opposite directions between the two models (W_E 0.3406 to W_U 0.5204 for Nemo, 0.5432 to 0.3544 for Ling) says the same thing. **The practical consequence: a cluster title read off a W_E map is not transferable to W_U.** A neighbourhood claim partly survives the move; a territory claim does not.

**Which families diverge most.** The per-family columns order almost the same way in **these two** models - and not in the third, which is why this paragraph is scoped to Nemo and Ling and Glimmer gets its own below. Here punctuation agrees most (0.4804 Nemo, 0.5824 Ling) and `cjk` (Nemo) / `non_latin_script` (Ling) agrees least (0.2725 over n=229 and 0.5136 over n=245 respectively). The direction is consistent between this pair and it does **not** survive the third model; the spread between the extremes is 0.21 in Nemo and only 0.07 in Ling, and 1 of the seven buckets carries fewer than fifty tokens in Nemo's curated slice, so read the per-family column as a direction to look in rather than a measured effect. `byte_fragment` and `whitespace` are empty in both slices - the curation these maps ship with removes them - so this table cannot speak about the family it would most want to.

**Muse-Glimmer-30B is the lowest of the three, and it inverts the family ordering.** Mean overlap 0.2991 (median 0.2800, 299.1x chance, 127/50000 tokens sharing no neighbour at all and not one token sharing all fifty). Where Nemo and Ling both put `punctuation` at the top of the family table, Glimmer puts `numeral` there at 0.6130 over 1110 tokens, and leaves `latin_word` last at 0.2858 over 34513 - so the families that survive the change of matrix are not the same families across models, and the per-family column cannot be generalised from one model to another.

**Its ARI of 0.0250 needs a caveat the other two rows do not.** An adjusted Rand index between two partitions is bounded above by how well-determined each partition is, and Glimmer's W_E map is the one map in this corpus whose own clustering barely clears its null floor (+0.009 silhouette margin). A low ARI here is therefore partly a statement about the W_E partition being weak, not only about the two spaces disagreeing - so it is the one number in this table that should not be read as a pure divergence measure. The noise fractions make the same point from the other side and are worth recording because they run opposite to the usual assumption: 0.6598 of Glimmer's vocabulary is unclustered in W_E against 0.3896 in W_U, so on this model it is the **unembedding** map that resolves more of the vocabulary into clusters, not the embedding map the rest of the project is built on.

**What none of this shows.** These are geometric agreement scores between two weight matrices. They say nothing about what either matrix *does* to the model's behaviour, they are not a ranking, and a model with lower overlap is not worse at anything. The one claim they support is the qualifier this track was opened to settle: **"the model's token geometry" is not one thing in an untied model, and a finding read off a W_E map has to say which of the two spaces it came from.**

## Track 2c — result

Measured 2026-09-12. Five `down_proj` maps on `mistralai/Mistral-Nemo-Instruct-2407` (dense, 40 layers), at layers 4, 12, 20, 28, 36. **Every build parameter is identical across the five** — the first 4096 of 14336 neurons, uncentered, HDBSCAN `leaf`/15/5, UMAP seed 42, `--labels none` so no namer is in the loop — so a difference between two rows is attributable to depth and to nothing else. That is the whole point of building a series rather than picking a layer.

| layer | neurons | clusters | noise | silhouette | null floor | margin | trust | seed ARI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4096 | 41 | 0.6958 | 0.3816 | 0.2006 | +0.1810 | 0.6332 | 0.5796 |
| 12 | 4096 | 43 | 0.7073 | 0.4053 | 0.2945 | +0.1108 | 0.6087 | 0.5687 |
| 20 | 4096 | 43 | 0.6377 | 0.3786 | 0.2123 | +0.1663 | 0.5891 | 0.6239 |
| 28 | 4096 | 56 | 0.6250 | 0.2466 | 0.1930 | +0.0536 | 0.5967 | 0.2411 |
| 36 | 4096 | 53 | 0.6228 | 0.2971 | 0.2028 | +0.0943 | 0.5706 | 0.3231 |

### What it says

**The series is monotone in nothing.** Cluster count runs 41 → 43 → 43 → 56 → 53 across layers 4 → 12 → 20 → 28 → 36, peaking at layer 28 (56) and bottoming at layer 4 (41); noise fraction runs 0.70 → 0.71 → 0.64 → 0.62 → 0.62. What the series does establish is a **range**: at fixed parameters, a `down_proj` map of this model resolves 41–56 clusters over 4096 neurons and discards 62%–71% of them as noise wherever you cut the stack.

**Every layer clears its null floor.** Margins run +0.0536 to +0.1810. Put that beside the project's other two raw-neuron maps, which clear by +0.0564 (gpt2__neurons__h.8.mlp.c_proj), +0.0993 (HuggingFaceTB__SmolLM2-135M__neurons__model.layers.21.mlp.down_proj): 3 of the 5 validated depths clear by more than both of them and the rest land inside their range, so the three neuron maps do not separate into better and worse by unit type. And against the 9 validated token maps in `out/`, which clear by +0.0087 to +0.2935, this series sits inside that range rather than below it — so "tokens carry structure, raw neurons do not" is not what these numbers say.

What *is* consistently weaker here is the projection, not the separation. Trustworthiness runs 0.5706 to 0.6332; of the 17 other validated maps in `out/`, **17 score above this series' best**, so these are the least faithful 2-D layouts in the repo. Seed ARI runs 0.2411 to 0.6239, and 0 of the 17 others fall below this series' worst. **The honest reading: the 2-D layout is the least faithful in the repo at every depth, and seed reproducibility is not a property of the unit type but of the depth: layer 28 is less reproducible than every other validated map in `out/`, while the series' best sits inside the pack.** Read a territory on these maps as a claim about cluster membership, not about what sits next to what.

**What is deliberately not measured here, and why.** There is no cross-layer neighbourhood-overlap column of the kind Track 2b has for W_E vs W_U. Neuron *i* of layer 12 and neuron *i* of layer 20 are not the same unit — nothing identifies them with each other — so a kNN overlap between two layers' clouds would be comparing arbitrary index alignments and reporting the result as a finding. Track 2b could take that measurement because a token id means the same thing in W_E and W_U. Here it does not, and the column is absent rather than filled with a number that looks like one.

**And the series is 4096 of 14336 neurons per layer** (the first contiguous slice, as every neuron map in this repo is), so it speaks about 29% of each layer's MLP. That is a curation, not a sample: it is the same 4096 indices at every depth, which is what makes the five rows comparable to each other, and what stops any of them from being a statement about the layer as a whole.

## Track 3 — The instrument

The namer is the pipeline's quality bottleneck. It is now also the only place
money is spent, so both properties get handled together. Owned by
`src/nebulai/backend/name.py`.

1. **Endpoint namer with a pinned identity.** Point the namer at a corpus
   model's pinned id over OpenRouter (or the HF router where a provider serves
   it). *Difference:* naming quality stops being bounded by what fits in local
   VRAM. Re-name one existing map first and diff the titles before adopting
   corpus-wide — the same discipline the `rename` path already enforces.
2. **Cost gate before send.** Estimate from the measured request shape (batches
   of 15 clusters × 20 representatives ≈ 1500 prompt + 400 completion tokens
   per batch), compare against `--max-cost-usd`, and refuse over budget with a
   list of cheaper *named* alternatives. Measured at Glimmer's rate: **$0.019
   for a 250-cluster map, $0.125 to re-name the entire built corpus** (14 maps,
   1660 clusters, counted from `out/`). The $1.00 gate is not close to binding,
   which is exactly why it can be strict.
3. **Refusal, not substitution.** If the pinned id is missing from the
   catalogue, or no provider serves it, the run stops and says so. This is not
   hypothetical: **two of the four models have no HF-router route at all**
   (measured — no provider serves Ling or Mistral-Nemo), and one of those was
   caught as a wrong `hf_endpoint` in `corpus.py` by the probe while this plan
   was being written. An absent route is recorded as absent; the OpenRouter id
   is the route for those two, and nothing resolves to a neighbouring model of
   the same family.
4. **Probe front-end.** Its README caveat ("cannot run offline") is now precise
   rather than apologetic: it needs a reachable generator *and* embedder, and
   with the endpoint route it has one without a GPU. The offline caveat does
   not disappear — it was never about hardware.

## Track 4 — Optional, gated: activations frontend

Unchanged in scope and weaker in motivation. A separate `nebulai[activations]`
extra (torch + Transformers, GPU host) emitting **Units** — the existing
contract — so the backend never changes. This is where hidden-state clouds,
attention capture, pre-softcap logit taps and the vision→projector→language-space
path would live. ⚠️ **Needs approval before any work:** it breaks the repo's
deliberate no-torch rule and reintroduces the ~60 GB GPU host this plan just
removed — for four models, that is now four GPU hosts' worth of checkpoints.
*Difference if skipped:* no activation or multimodal clouds; a pure,
reproducible, laptop-runnable repo. **Recommendation: skip until Track 2
results argue for it.** The W_E/W_U and depth findings are what tell you whether
inference-time geometry would add signal or just volume.

## Sequencing

| Order | Track | Effort | Unblocks |
|---|---|---|---|
| 1 | Remote loader groundwork | ~1 day | everything |
| 2 | 2a on `mistral-nemo` | hours | proves the reader end-to-end |
| 3 | 2a on the other three | ~1 day of runs | the corpus |
| 4 | 2b (W_E vs W_U + control) | ~1 day incl. the overlap script | headline result |
| 5 | Track 3 (endpoint namer + gate) | ~half day | all future naming |
| 6 | 2c–2f (depth, contrast, compare) | ~2–3 days of runs | corpus growth |
| 7 | Track 4 | weeks + hardware | only if 4 justifies it |

## Explicitly out of scope

vLLM/SGLang serving (nothing consumes it); GGUF reading and BF16↔quant drift
measurement (no GGUF reader — and the quantized releases are GGUF, not
safetensors); generation with the subject model of any kind; logprob transport
(no probe surface consumes logprobs); multimodal input; MoE expert-level depth
maps until 2c's design question is answered.

## How to check any number in this document

```sh
uv run scripts/probe_endpoints.py                 # revisions, keys, bytes, routes, cost
uv run scripts/probe_endpoints.py --rows 8        # decode real rows — the falsifiable part
uv run scripts/probe_endpoints.py --weights-only  # runs with zero credentials
```

It needs no API key and sends no chat request, so it costs nothing and "no key
configured" is one of its normal results. Paste its output next to any claim
about what a run will cost; prices are read live from OpenRouter and the script
flags drift against `corpus.py` rather than trusting either side.

**What a range read establishes, and what it does not.** A 206 plus a decoded
row proves the rows are readable at that revision. It says nothing about whether
the endpoint on the same line serves those weights — nothing can prove that from
outside, which is precisely why identity is pinned and a missing route is
refused. Every map remains evidence about weight geometry, and the namer that
titled it remains part of the map's provenance, not part of its findings.

Corrections applied to the two plans this replaces, with the measurement that
settled each: `updated-implementation-plan.md`.
