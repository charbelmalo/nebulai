# Repo map — ~/Developer/nebulai

```
nebulai/
├── pyproject.toml            # uv project; deps: numpy, safetensors, huggingface-hub,
│                             #   tokenizers, umap-learn, scikit-learn>=1.4, datamapplot,
│                             #   matplotlib, anthropic. script: nebulai = "nebulai.cli:main"
├── .python-version           # 3.12
├── README.md                 # quickstart + variant table + honesty notes
├── docs/DETAILS.md           # stage-by-stage microdetails (companion to the skills)
├── src/nebulai/
│   ├── __main__.py           # `python -m nebulai`
│   ├── cli.py                # argparse; `tokens` subcommand + all flags
│   ├── units.py              # the Units dataclass (the contract)
│   ├── frontends/
│   │   └── tokens.py         # Plan C — load_token_units()  [built]
│   │       # (planned) sae.py — Plan A;  neurons.py — Plan B
│   └── backend/
│       ├── reduce.py         # UMAP → u_cluster / u3 / u2
│       ├── cluster.py        # HDBSCAN (leaf default)
│       ├── name.py           # namer chain: ollama → openrouter → centroid
│       ├── export.py         # nebulai.json writer (schema v2)
│       ├── edges.py          # similarity edges in u_cluster space (schema v2 beams)
│       ├── compare.py        # cross-model cluster comparison (compare.json)
│       ├── embed.py          # ollama embedding client (compare pipeline)
│       ├── viewer.py         # first-cut WGSL compare viewer [deprecated → viewer/]
│       └── viz.py            # datamapplot static PNG + interactive HTML
├── viewer/                   # Phase-2 WebGPU viewer app (Vite+TS; replaces backend/viewer.py)
├── tests/                    # pytest (test_edges.py, …)
├── out/
│   ├── index.json            # dataset discovery for the viewer ({"datasets": [...]})
│   ├── <model>/              # per-model outputs (slashes → __)
│   │   ├── reduced.npz           # cached UMAP reductions (the expensive step)
│   │   ├── reduced.params.json   # exact params the cache is keyed on
│   │   ├── nebulai.json          # the map / viewer contract (schema v2 w/ edges)
│   │   ├── map_static.png
│   │   └── map_interactive.html
│   └── compare/              # compare.json + legacy index.html
└── .claude/skills/           # nebulai (hub) + nebulai-{tokens,sae,neurons} + design/viz skills
```

## CLI (Plan C, current)

```sh
uv run nebulai tokens [--model gpt2] [--out out] [--max-tokens N]
    [--no-center] [--cluster-dim 10] [--n-neighbors 30]
    [--min-cluster-size N] [--min-samples N] [--cluster-method leaf|eom]
    [--seed 42] [--namer auto|openrouter|ollama|anthropic|none]
    [--openrouter-model SLUG] [--ollama-model NAME] [--ollama-host URL]
    [--anthropic-model claude-opus-4-8] [--env-file PATH] [--force]
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
  datamapplot — takes a few minutes to build the first time).
- GPT-2 assets come from the HF cache (`model.safetensors` + `tokenizer.json`);
  no torch is needed for Plan C.
- ollama namer expects the M4 worker reachable at `192.168.0.200:11434` (see the
  `m4worker-bridge` skill to start it); otherwise the chain falls to OpenRouter,
  then to the centroid fallback.
- OpenRouter key: `OPENROUTER_API_KEY` env var or `~/.hermes/.env`.

## Timings (observed, GPT-2)

| Run | UMAP | cluster | total |
|---|---|---|---|
| `--max-tokens 5000` | ~95 s | ~2 s | ~2 min |
| full vocab (~49.8k tokens) | ~107 s | ~10 s | ~2.5 min |

Re-running with cached reductions skips UMAP entirely (0.0s), so clustering /
naming iteration is seconds.
