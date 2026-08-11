# Corrections ledger — what measurement killed, and what survived

**This is no longer a second plan.** The plan of record is
[`recommended-plan.md`](recommended-plan.md); this file is the audit trail
behind it. Two parallel plans drift, and the drift is invisible until someone
acts on the stale one — but the *record of what was checked* has independent
value and cannot drift, because it prescribes nothing. So: one plan, one ledger.

Everything below was measured on **2026-08-12** against the live HF repos, their
shard headers and configs, and the live OpenRouter catalogue. Re-run it with
`uv run scripts/probe_endpoints.py`. Anything not measured is labelled as such.

## The two assumptions that took the plans down

Both earlier documents were built on them, and both are gone.

**"Download the checkpoint."** Not needed. All four corpus repos answer
`HTTP 206 Partial Content` on `resolve/{rev}/{shard}` **with no auth**, so the
loader reads the safetensors header and then only the byte ranges of the rows
it maps: 1.87 GB streamed against 344 GB of checkpoints across the corpus.

**"Serve it locally on a ~60 GB GPU box."** Not needed. The namer and probe
generator reach pinned model ids over remote OpenAI-compatible endpoints. Naming
a 250-cluster map at Glimmer's published rate costs $0.019.

Each assumption alone would have capped the corpus at one model. Together they
were the reason both plans were single-specimen documents.

## Claims the measurement killed

| Claim (source) | Verdict | What settled it |
|---|---|---|
| "Download only the shards containing the keys — ~3 GB (the embedding shard) instead of ~59 GB" (`P0.1`) | **Wrong, and by a lot.** The shard holding Glimmer's W_E is **49.95 GB** of the 59.55 GB checkpoint; Gemma-4's is 49.91 GB of 51.61 GB. Shard-granular fetching downloads 84–97% of those checkpoints. | HF tree API file sizes + `model.safetensors.index.json` routing |
| "Glimmer is the best instrument nebulai has ever had available **for free**" | **Wrong on both halves.** Not free — $0.35/$1.50 per M at OpenRouter (cheap, but metered). Not the best available — it is the most expensive of the four, and `gemma-4-26b-a4b-it:free` is the only genuinely $0 endpoint in the corpus. | live OpenRouter catalogue |
| "Removes the OpenRouter dependency and makes the naming layer offline" | **Inverted.** The endpoint architecture *depends* on OpenRouter (or HF's router). Naming was never the offline part of this repo, and the probe front-end's "cannot run offline" caveat was never about hardware. | architecture |
| "Local, fast (DFlash ~3× on consumer GPUs), slots into every seat" | **Out of scope, not wrong.** Nothing in this repo generates with the subject model, so drafter throughput never enters the picture; and there is no local serving to be fast at. | codebase |
| Single specimen (both plans, throughout) | **Superseded.** Four models, one of them a tied control that makes the headline experiment interpretable. | `src/nebulai/corpus.py` |
| "$0.33 to re-name a 17-map corpus" (working estimate) | **Different denominator.** What exists is **14 maps / 1660 clusters**, which prices at **$0.125** at Glimmer's rate; $0.33 implies ~4,400 clusters, which the tree does not contain. Both numbers are far under the gate, so nothing downstream changes. | `out/*/nebulai.json`, counted |
| Three W_E key families are enough (`wte`, `embed_in`, `embed_tokens`) | **Insufficient.** Ling uses `model.word_embeddings.weight`; Glimmer and Gemma-4 nest theirs under `model.language_model.`. Three of four models miss on exact-key lookup. | shard headers |

## Corrections to `corpus.py` and to the briefing that produced it

These are small, and none of them changes the architecture — they are exactly
the kind of drift the probe script exists to catch.

- **`hf_endpoint` for Mistral-Nemo was dead — now fixed.** `corpus.py` named
  `mistralai/Mistral-Nemo-Instruct-2407` as its HF route while **no HF inference
  provider served it** (`inferenceProviderMapping` empty); the entry now records
  `None`, like Ling's. Live routes as measured: Glimmer → `together`; Gemma-4 →
  `deepinfra`, `featherless-ai`, `novita`, `scaleway`; Ling and Mistral-Nemo →
  none, so OpenRouter is their only route. The no-substitution rule's first real
  test case, and the loop worked: the probe reported the dead route, the corpus
  recorded the absence, and nothing resolved to a neighbouring Mistral.
- **`google/gemma-4-26b-a4b-it` is not the canonical repo id.** HF answers it
  with a **307** to `google/gemma-4-26B-A4B-it`. It works — provided the client
  follows redirects and preserves the `Range` header across them. `corpus.py`'s
  `hf_endpoint` now spells the canonical casing; its `repo` field still leans on
  the redirect.
- **Ling has 26 numbered shards *plus* `model-mtp-layer.safetensors`** — 27
  files in the index, 25,015 tensors. A loader that derives shard names from the
  `-of-00026` pattern misses one.
- **Mistral's repo carries a `consolidated.safetensors`** that is *not* in the
  index — a second full copy of the checkpoint. Counting `.safetensors` files
  reports 6 shards for a 5-shard checkpoint.
- Ling's config declares **256 experts** (`BailingMoeV2_5`), not the 128 that
  Gemma-4 declares. Both matter for Track 2c, where "layer L's write directions"
  is not a single matrix.

## Claims that survived, now verified rather than quoted

Verified directly from each `config.json` and from the shard headers today:

- **Untied: Glimmer, Ling, Mistral-Nemo. Tied: Gemma-4** (`tie_word_embeddings`
  true, and no `lm_head.weight` anywhere in its weight map — tied by absence,
  not just by flag). The W_E/W_U experiment is askable for three of four, and
  Gemma-4 is the control.
- **Glimmer: 52 layers, hidden 6656, vocab 202,048, `MuseGlimmerForConditionalGeneration`,
  `final_logit_softcapping: 20.0`, `output_multiplier: 0.19611613…`, projector
  `out_hidden_size: 6144`** (a different width from the text stream, as the
  earlier ledger said).
- **Glimmer's global layers are NoPE.** `layer_types` alternates
  `[sliding ×3, full]`, and `layer_rope_theta` is exactly `0` at every
  `full_attention` position (500000.0 elsewhere). The old depth-sampling recipe
  — "capture at every local→global transition" — does sample precisely the
  layers with no positional encoding. Still a real confound; now confirmed
  rather than inferred.
- **Softcap and output multipliers are irrelevant to every map here.** They act
  on logits at inference; the rows being mapped never pass through them. The
  "native vs pre-softcap" question dissolves rather than being answered.
- **Gemma-4: 30 layers, hidden 2816, vocab 262,144, 128 experts,
  `final_logit_softcapping: 30.0`.** Ling: 32 layers, hidden 4096, vocab
  157,184, `rope_theta` 6e6. Mistral-Nemo: 40 layers, hidden 5120, vocab
  131,072, dense, `rope_theta` 1e6.
- **All four W_E matrices are BF16**, widened losslessly to f32 by the existing
  reader. Decoded rows come back finite; Glimmer's are uniformly norm 5.098.
- **All four repos are public and ungated**, which is what makes the
  no-credentials path a real one rather than a courtesy.

## Still unverified — treat as planned, not established

- **That an endpoint serves the weights the map was read from.** Unprovable
  from outside. It is why the model id is pinned and a missing route is refused;
  it is not something a future measurement will close.
- **Every cost figure.** Estimates from a measured request shape (batches of 15
  clusters × 20 representatives ≈ 1500 prompt + 400 completion tokens), not
  receipts. No chat request has been sent to any of these endpoints from this
  repo.
- **`:free` tier behaviour.** Gemma-4's free endpoint is $0 in the catalogue;
  its rate limits, its queueing, and whether a free-tier route stays pinned to
  the same provider have not been tested.
- **Anything about naming quality.** No corpus model has named a cluster yet.
  "Better namer" is a hypothesis until one map is re-named and the titles are
  diffed against the incumbent.
- **The maps themselves.** Zero of the four models has a built map. Every
  points/silhouette/trust number for them is absent from the README on purpose.

## Out of scope, unchanged

vLLM/SGLang serving; GGUF reading and BF16↔quant drift (the quantized releases
are GGUF, not safetensors); generation with the subject model, DFlash control
included; logprob transport; multimodal input; prompt-replicate stability (there
are no prompts in the pipeline). The activations frontend remains gated behind
explicit approval — see Track 4 of the plan, where the endpoint architecture has
made it *more* expensive to justify, not less.
