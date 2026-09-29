# Repo map — ~/Developer/nebulai

```
nebulai/
├── pyproject.toml            # uv project; deps: numpy, safetensors, huggingface-hub,
│                             #   tokenizers, umap-learn, scikit-learn>=1.4, datamapplot,
│                             #   matplotlib, anthropic.
│                             # optional-dependencies.organisms = [torch, transformers,
│                             #   peft, datasets]
│                             # dependency-groups.behavior-local = [torch==2.14.0,
│                             #   transformers==5.17.0, sentence-transformers==6.0.1]
│                             # scripts: nebulai = "nebulai.cli:main",
│                             #          seer = "nebulai.seer.cli:main"
│                             # The base install is torch-free and two tests enforce it.
├── .python-version           # 3.12
├── README.md                 # quickstart + variant table + honesty notes
├── docs/DETAILS.md           # stage-by-stage microdetails (companion to the skills)
├── src/nebulai/
│   ├── __main__.py           # `python -m nebulai`
│   ├── cli.py                # argparse; all 17 top-level verbs + their flags
│   ├── units.py              # the Units dataclass (the contract)
│   ├── corpus.py             # the endpoint-era corpus: which models we map,
│   │                         #   how we reach them, and the cost defaults
│   ├── llm.py                # chat/completion clients, RunBudget, key resolution
│   ├── spaces.py             # the closed set of vector spaces a channel or a
│   │                         #   direction may live in
│   ├── weights.py            # no-torch safetensors reader (local + remote),
│   │                         #   widens BF16/F16 to float32
│   ├── behavior/             # the behavioural-divergence study (its own cli.py)
│   ├── organisms/            # model-organism fine-tuning — the ONLY torch path
│   │                         #   (`uv sync --extra organisms`)
│   ├── prompts/              # shipped prompt sets
│   ├── seer/                 # SessionSeer, ~17k lines, its own console script.
│   │                         #   BOUNDARY: `nebulai` never imports `nebulai.seer`
│   │                         #   (docs/SESSIONSEER-HANDOVER.md §2)
│   ├── frontends/
│   │   ├── tokens.py         # Plan C — W_E rows, W_U via --unembedding  [built]
│   │   ├── sae.py            # Plan A — SAE decoder directions           [built]
│   │   ├── neurons.py        # Plan B — MLP down_proj write directions   [built]
│   │   ├── api_tokens.py     # neutral-embedder contrast maps
│   │   └── probe.py          # the live concept-cloud builder
│   └── backend/
│       # pipeline
│       ├── reduce.py         # UMAP → u_cluster / u3 / u2
│       ├── cluster.py        # HDBSCAN (leaf default)
│       ├── name.py           # namer chains (_CHAINS): auto = ollama → openai →
│       │                     #   openrouter → centroid; pinned = ollama → openai →
│       │                     #   hf → openrouter (no centroid); also anthropic /
│       │                     #   claude-cli / codex-cli / hf / none
│       ├── export.py         # nebulai.json writer (schema v2)
│       ├── edges.py          # similarity edges in u_cluster space (schema v2 beams)
│       ├── viz.py            # datamapplot static PNG + interactive HTML
│       # analysis
│       ├── compare.py        # cross-model cluster comparison (compare.json)
│       ├── metrics.py        # per-map metrics for the compare table
│       ├── validate.py       # trustworthiness / seed-ARI / column-shuffled null
│       ├── rename.py         # re-title a built map without rebuilding geometry
│       ├── agreement.py      # inter-rater agreement for the variance study
│       # forward pass
│       ├── interp/           # the numpy forward pass: bundles.py, hooks.py,
│       │                     #   intervene.py, live_server.py, gpt2_numpy.py,
│       │                     #   llama_numpy.py
│       ├── instrument.py     # the frozen question set — a one-way door
│       ├── eval_awareness.py # the eval-awareness direction (matched pairs)
│       # attractors
│       ├── channels.py       # channels.json per model
│       ├── directions.py     # direction construction + projection
│       ├── import_directions.py  # import published directions
│       ├── persona.py        # persona spaces (build / verify / list)
│       ├── absorbing.py      # the self-play absorbing-state study
│       ├── prompt_sets.py    # prompt-set construction and freezing
│       ├── variance_cli.py   # the generative-variance study's verbs
│       # infrastructure
│       ├── embed.py          # embedding client (compare pipeline)
│       ├── m4host.py         # M4 worker discovery (never hardcode the IP)
│       ├── build_server.py   # the live build server the viewer talks to
│       ├── torch_pickle.py   # read torch checkpoints without importing torch
│       └── viewer.py         # first-cut WGSL compare viewer [deprecated → viewer/]
├── viewer/                   # Vite+TS+Preact. TWO instruments from one codebase:
│                             #   index.html→main.ts (nebulai) and
│                             #   seer.html→seer-main.ts (seer), selected at build
│                             #   time by PSYCHIX_ENTRY/VITE_BASE. State in
│                             #   src/app/slices/*.ts; scene drivers in src/scene/;
│                             #   26 Internals views in src/scene/interp/registry.ts.
├── tests/                    # pytest — 70 files. Note test_no_torch_in_base.py and
│                             #   test_behavior_optional_dep.py: they are the
│                             #   enforcement of the torch-free base install, not
│                             #   ordinary unit tests.
├── out/
│   ├── index.json            # dataset discovery for the viewer ({"datasets": [...]})  [fetched]
│   ├── <model>/              # per-model outputs (slashes → __)
│   │   ├── reduced.npz           # cached UMAP reductions (the expensive step)  [build-only]
│   │   ├── reduced.params.json   # exact params the cache is keyed on           [build-only]
│   │   ├── nebulai.json          # the map / viewer contract (schema v2 w/ edges)  [fetched]
│   │   ├── channels.json         # per-model channel activations                   [fetched]
│   │   ├── directions.json        # per-model directions                           [fetched]
│   │   ├── validation.json        # nebulai validate's verdict                     [build-only]
│   │   ├── interp/                # 26 Internals bundles + index.json              [fetched]
│   │   ├── map_static.png
│   │   └── map_interactive.html
│   ├── compare/              # compare.json + metrics.json + legacy index.html  [fetched]
│   ├── behavior/             # behavior.json + per-study runs                   [fetched]
│   ├── absorbing/            # index.json + per-run directories                 [fetched]
│   ├── persona/              # index.json + per-space directories               [fetched]
│   ├── organisms/            # model-organism records (emergent_misalignment.json)  [build-only]
│   └── neuronpedia/          # cached Neuronpedia label bootstrap               [build-only]
└── .claude/skills/           # nebulai (hub) + nebulai-{tokens,sae,neurons} +
                              #   nebulai-viz{,-threejs,-wgsl,-deckgl}
```

## CLI

All 17 top-level verbs of the `nebulai` console script:

```
tokens sae neurons edges channels interp metrics probe validate rename
compare intervene direction persona absorbing behavior variance
```

with nested groups `direction {list,survey,add,make,prompts,project,drop}`,
`persona {build,verify,list}`, `absorbing {run,report,list}`, and
`behavior {plan,run,analyze,calibrate,conformance,publish,inspect,serve}`.
The second console script, `seer` (`nebulai.seer.cli:main`), has 19 verbs of its
own — see `docs/SESSIONSEER-HANDOVER.md` §2.

```sh
uv run nebulai tokens [--model gpt2] [--out out] [--max-tokens N]
    [--no-center] [--cluster-dim 10] [--n-neighbors 30]
    [--min-cluster-size N] [--min-samples N] [--cluster-method leaf|eom]
    [--seed 42] [--namer auto|openrouter|hf|ollama|openai|anthropic|claude-cli|codex-cli|none]
    [--openrouter-model SLUG] [--ollama-model NAME] [--ollama-host URL]
    [--anthropic-model claude-opus-5] [--env-file PATH] [--force]
    [--edges knn|cluster|none]        # similarity edges in the export (default knn)

uv run nebulai edges <model>... [--out out] [--mode knn|cluster]
    # backfill schema-v2 edges into existing nebulai.json from cached
    # reduced.npz — no UMAP rerun; also rewrites out/index.json

uv run nebulai compare <model>... [--out out] [--ollama-host URL]
    [--embed-model mxbai-embed-large] [--seed 42]
```

Runs 5 timed stages. **Reductions are cached** in `out/<model>/reduced.npz`,
keyed by `{model, max_tokens, center, cluster_dim, n_neighbors, seed}`; reused
only on exact match, so iterating on clustering/naming flags is free while
changing a reduction flag transparently recomputes. `--force` busts the cache.

## Environment notes

- Deps install via `uv sync` (the scientific stack — umap-learn / scikit-learn /
  datamapplot — takes a few minutes to build the first time). **The base install
  is torch-free, and stays that way**; `tests/test_no_torch_in_base.py` fails if
  that changes. Two opt-ins bring torch in, deliberately and separately:
  `uv sync --extra organisms` (torch, transformers, peft, datasets) for
  `src/nebulai/organisms/`, and `uv sync --group behavior-local` (pinned
  torch 2.14.0 / transformers 5.17.0 / sentence-transformers 6.0.1) for the
  Behavior study's local model arms, enforced by
  `tests/test_behavior_optional_dep.py`. Nothing else in the repo needs either.
- GPT-2 assets come from the HF cache (`model.safetensors` + `tokenizer.json`);
  no torch is needed for Plan C.
- ollama namer expects the M4 worker reachable at `<m4-host>:11434` (see the
  `m4worker-bridge` skill to start it); otherwise the `auto` chain falls to
  openai, then OpenRouter, then to the centroid fallback.
- OpenRouter key: `OPENROUTER_API_KEY` env var or `~/.hermes/.env`.

## Timings (observed, GPT-2)

| Run | UMAP | cluster | total |
|---|---|---|---|
| `--max-tokens 5000` | ~95 s | ~2 s | ~2 min |
| full vocab (~49.8k tokens) | ~107 s | ~10 s | ~2.5 min |

Re-running with cached reductions skips UMAP entirely (0.0s), so clustering /
naming iteration is seconds.
