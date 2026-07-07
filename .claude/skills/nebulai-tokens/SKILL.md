---
name: nebulai-tokens
description: >-
  Plan C of Nebul.AI — the token-embedding front-end (~/Developer/nebulai). A
  point is one vocabulary token, positioned by its row of the model's input
  embedding matrix W_E. Use this skill whenever running, tuning, debugging, or
  interpreting the token map: `nebulai tokens`, the W_E / embedding-row
  geometry, vocab curation, mean-centering, the leaf-vs-eom clustering gotcha,
  or reading the resulting clusters (orthographic vs semantic, the GPT-2 glitch
  tokens). Load the `nebulai` hub skill alongside this one — this skill covers
  only what is specific to tokens; the shared back-end, the nebulai.json
  contract, and the cross-pipeline recommendation rule live in the hub. If you
  change something here, run the hub's change-propagation check before finishing.
---

# Nebul.AI — Plan C: token embeddings

**A point is one vocabulary token; its geometry is its row of `W_E`.** No corpus
sweep, no activations — just the input embedding matrix. Code:
`src/nebulai/frontends/tokens.py`. Shared back-end + export + the propagation
protocol: see the `nebulai` hub (`references/backend-contract.md`,
`references/change-propagation.md`).

## Run it

```sh
uv run nebulai tokens --model gpt2 --max-tokens 5000   # ~2 min quick pass
uv run nebulai tokens --model gpt2                      # full curated vocab, ~2.5 min
```

Outputs in `out/gpt2/`: `nebulai.json`, `map_static.png`, `map_interactive.html`.
Inspect any result with `scripts/inspect_map.py` from the hub. Sweep clustering
settings on the cached `reduced.npz` with `scripts/sweep_hdbscan.py` — never
re-run UMAP just to try a different `min_cluster_size`.

## What is specific to this pipeline

- **Weight loading.** Suffix-match on `_EMBED_KEY_SUFFIXES`
  (`wte.weight` GPT-2, `embed_in.weight` Pythia, `embed_tokens.weight`
  Llama/Qwen/Gemma). safetensors→numpy only; **no torch**. GPT-2 *ties* `W_E`
  and `W_U`, so this doubles as an output-vocabulary map; use an untied model
  (Pythia) if you want the input side alone.
- **Vocab curation (`_keep`).** Drops empties, whitespace-only, control chars,
  and U+FFFD byte-fragments (byte-level BPE has many partial-UTF8 tokens that
  decode to `�`). This is *display* curation, and it is **tokens-only** — it has
  no analogue in SAE/neuron pipelines, so it does not propagate.
- **`--max-tokens N` keeps the lowest N ids.** BPE merges are learned by
  frequency, so low id ≈ frequent token; truncation keeps the common vocabulary,
  not a random slice.
- **Mean-centering (default on).** Token embedding spaces are anisotropic (a big
  shared direction inflates all cosines); subtracting the mean row exposes
  relative structure. `--no-center` to demonstrate the difference. NOTE: the
  *reason* (anisotropy) may generalize to SAE decoder directions / neuron
  write-directions — if you touch centering, run the propagation check.

## The clustering gotcha (validated finding)

On the token UMAP space, HDBSCAN's textbook `eom` selection collapses ~96% of
points into one mega-cluster (GPT-2 top-5k: **5 clusters, 0% noise**), because
the vocab core is a single connected density blob. **`leaf` is the default**
and recovers real structure (top-5k: **~84 clusters, 42% noise**; full vocab:
**~208 clusters, 55% noise**). This is a property of the geometry, not of
tokens — so it is already the shared back-end default and Plans A/B inherit it.

## Reading the map

Two kinds of cluster coexist, and telling them apart is half the story — see
`references/interpretation.md` for the full tour (semantic families, code
tokens, orthographic/subword clusters, and the GPT-2 "glitch token" island —
`externalToEVA`, `rawdownload`, … — reproduced as its own far-off cluster).

## Titles

Cluster titles come from the shared namer chain (hub → `backend-contract.md`).
With no reachable ollama and no OpenRouter key, you get the `centroid` fallback
(`girl · mother · father · child`) — honest but terse. Start the M4 worker
(`m4worker-bridge` skill) or set `OPENROUTER_API_KEY`, then re-run: reductions
are cached, so real LLM titles cost seconds, not another UMAP.
