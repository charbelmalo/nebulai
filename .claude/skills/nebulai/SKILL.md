---
name: nebulai
description: >-
  Orchestrator and change-propagation advisor for the Nebul.AI "semantic cloud"
  project (~/Developer/nebulai) — three interchangeable front-ends (Plan C token
  embeddings, Plan A SAE features, Plan B MLP neurons) that all feed ONE shared
  reduce → cluster → name → export → render back-end. Use this skill FIRST
  whenever work touches the nebulai repo, a "semantic cloud" / "concept atlas" /
  micro-model concept map, the Units contract, the shared back-end (UMAP /
  HDBSCAN / cluster naming / the nebulai.json export / datamapplot render), or
  ANY of the three pipelines — even if the request names only one pipeline,
  because a change to one usually should propagate to the others. It routes to
  the per-pipeline skills (nebulai-tokens, nebulai-sae, nebulai-neurons) and
  enforces the cross-pipeline recommendation protocol so the three artifacts
  stay a valid apples-to-apples comparison. Load it even if the user doesn't say
  "nebulai" but is clearly working on this project.
---

# Nebul.AI — orchestrator

Nebul.AI decomposes a small model into interpretable units, labels each unit,
then clusters the units into a navigable map of meaning. Three front-ends
define *what a point is*; one back-end turns any of them into the same
artifacts. This skill is the hub: it holds the shared spine, routes to the
per-pipeline skills, and enforces cross-pipeline consistency.

```
front-end ──> Units ──> reduce ──> cluster ──> name ──> export + render
 (A/B/C)              (UMAP)     (HDBSCAN)   (LLM)    (json / png / html)
   │                  └──────────── shared back-end ─────────────┘
   ├── Plan C  tokens   → skill: nebulai-tokens   (✅ built)
   ├── Plan A  SAE       → skill: nebulai-sae      (planned; flagship)
   └── Plan B  neurons   → skill: nebulai-neurons  (planned; comparison)
```

## Routing — which skill to load

| The task is about… | Load |
|---|---|
| Running / tuning / interpreting the **token** map (Plan C) | `nebulai-tokens` |
| Building or running the **SAE feature** map (Plan A) | `nebulai-sae` |
| Building or running the **MLP neuron** map (Plan B) | `nebulai-neurons` |
| The shared **back-end** (`Units`, reduce, cluster, name, export, viz) | `references/backend-contract.md` |
| The **Phase-2 viewer** (`viewer/` — drivers, toolkit choice, chrome) | `nebulai-viz` (routes on to `nebulai-viz-{threejs,wgsl,deckgl}`) |
| Where a file lives / repo layout / CLI flags | `references/repo-map.md` |
| Whether a change should spread across pipelines | `references/change-propagation.md` |

Always keep this hub loaded alongside a per-pipeline skill — the pipeline skills
deliberately do **not** restate the shared contract; they point back here so the
spine is defined exactly once.

## The shared spine (one-screen version)

Every front-end returns a `Units` (`src/nebulai/units.py`): `ids` (stable unit
reference), `vectors` `(n,d) float32` (the **geometry** the map is built from),
`labels` (display text), `meta` (provenance). The back-end only ever sees this —
it is front-end-agnostic by construction. **Geometry (`vectors`) and text
(`labels`) are strictly separate**; that separation is what lets Plans A/B offer
a "model space vs label space" toggle by swapping only `vectors`. Full detail:
`references/backend-contract.md`.

Pipeline stages, all in `src/nebulai/backend/`:

1. **reduce** — UMAP (cosine): ~10-D for clustering, 3-D for the flythrough, 2-D
   as a PCA *of the 3-D* so both views stay aligned. Never cluster on 2-D/3-D.
2. **cluster** — HDBSCAN, **`leaf` selection by default** (eom collapses these
   spaces into one mega-cluster — see nebulai-tokens). Membership probability →
   per-point confidence → opacity.
3. **name** — `auto` chain: **ollama (M4 worker) → OpenRouter → centroid
   fallback**. Anthropic via `--namer anthropic`. The chain always completes;
   the backend actually used is stamped into the export.
4. **export** — `nebulai.json` (schema v2), the contract for the Phase-2
   viewer; includes similarity `edges` computed in the 10-D clustering space
   (backfill existing artifacts with `nebulai edges <model>` — no UMAP rerun).
5. **render** — datamapplot static PNG + interactive HTML.

## Honesty guardrails (bind every pipeline)

- This is a **visualization + clustering tool over public micro models. No
  causal claims** — it shows how units' vectors relate geometrically, not what a
  unit *does* to behavior.
- Plan C geometry is the model's own; token structure is partly
  frequency/orthography (centering + cosine mitigate, don't remove).
- Plans A/B must never present *label-space* layout as *model* geometry — keep
  the two projections separate and labeled.
- Export the evidence to distrust the map: noise fraction, namer backend, and
  curation counts all live in `nebulai.json`'s `meta`.

## Cross-pipeline recommendation protocol (do this every time)

The three pipelines share a back-end and exist to be **compared** — Plan B's
whole value is being an apples-to-apples contrast with Plan A on an *identical*
back-end. So an improvement that silos in one pipeline quietly breaks the
comparison. After **any** change, before you finish:

1. Classify the change by layer — `backend/` (shared), `frontends/<plan>`
   (pipeline-specific), `cli`, `docs`.
2. Decide if the underlying reason generalizes. A front-end change is truly
   pipeline-specific only when it's about that unit type's quirks (e.g.
   byte-fragment token curation is tokens-only). Anything about geometry,
   reduction, clustering, naming, export, or interpretation almost always
   generalizes.
3. **End your message with ONE concrete recommendation question** naming the
   specific other pipeline(s) and the specific change — or explicitly state
   "no cross-pipeline impact" when it's genuinely isolated. Keep it a yes/no the
   user can answer in one word.

Full protocol, worked examples, and the propagation table:
`references/change-propagation.md`. This is not optional flavor — it is the
reason this project is one orchestrator over separate pipeline skills rather
than three disconnected tools.

## Bundled scripts

- `scripts/inspect_map.py <nebulai.json>` — summarize any map: meta line, top-N
  clusters with titles + sample members, size distribution. Use this instead of
  writing an inline JSON-poking snippet (every session so far has re-written
  one).
- `scripts/sweep_hdbscan.py <reduced.npz>` — sweep HDBSCAN params on a cached
  reduction (leaf/eom × min_cluster_size × min_samples) to pick clustering
  settings without re-running the minutes-long UMAP.
