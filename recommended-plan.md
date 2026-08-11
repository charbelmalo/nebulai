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
