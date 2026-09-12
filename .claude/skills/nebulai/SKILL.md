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
   ├── Plan A  SAE       → skill: nebulai-sae      (built; flagship)
   └── Plan B  neurons   → skill: nebulai-neurons  (built; comparison)
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
3. **name** — `auto` chain: **ollama → openai → openrouter → centroid**. Under a
   pinned model identity the chain is **ollama → openai → hf → openrouter** with
   **no centroid rung**. Selectable backends: `auto, openrouter, hf, ollama,
   openai, anthropic, claude-cli, codex-cli, none` (`backend/name.py` `_CHAINS`).
   `nebulai rename` defaults to the CLI-shell backends. The chain always
   completes; the backend actually used is stamped into the export.
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

## Settings home (viewer) — MANDATORY for every user-facing knob

The viewer has ONE canonical Settings page at
`viewer/src/chrome/SettingsPage.tsx`. It is the single place a user goes to
customize anything — theme, per-graph appearance, model probing, dataset
choice, provenance. It is opened by the top-right gear in `TopBar` and by
the "All settings…" button at the bottom of the `Sidebar`.

**Rule for future agents (this includes you):** any new customizable
feature — a new driver knob, a new chrome preference, a new probing option,
a new dataset filter, ANY toggle/slider/select a user should reach — MUST be
added to `SettingsPage.tsx` under the correct tab, backed by the matching
slice of `appStore` (`settings`, `appearance[<graph>]`, `probing`, etc.).

Do not:
- Bury a new knob only inside the compact left `Sidebar`. The sidebar is
  a quick-access subset; the full option lives in `SettingsPage`.
- Read a driver-tunable value from a driver's private state. If a driver
  wants it configurable, it reads from `appStore.appearance[<graph>]`.
- Add a new top-level state slice for options without also surfacing it in
  a Settings tab.

Do:
- Add a new **tab** if the feature category doesn't fit existing ones
  (General / Appearance / Behavior / Model Probing / Snapshot / Sessions /
  Data / About). Update the `TABS` array in `SettingsPage.tsx` and add the
  matching `TabName` case.
- Add a new **sub-tab** under Appearance if a new graph type is introduced.
- Extend `Appearance`, `Settings`, or `Probing` in the matching slice under
  `viewer/src/app/slices/` (`appearance.ts`, `atlas.ts`, `behavior.ts`,
  `interp.ts`, `probing.ts`, `seer.ts`, `sessions.ts`, `shell.ts`,
  `snapshot.ts`) — `store.ts` only composes them. Every StateCreator is typed
  against the full `AppState`, which is what makes cross-slice writes legal.
  Add the new field and a default value; the chrome `state.ts` signal
  bridge picks it up automatically once the slice is updated.
- Reuse the primitives in `viewer/src/viz/controls.tsx` (import them as `@psychix/viz/controls`)
  (`SelectRow`, `SliderRow`, `ToggleRow`, `TextRow`) — do not invent
  bespoke inputs.

If a knob affects only one graph, put it under Appearance → that graph's
sub-tab. If it affects all rendering, put it under General → Rendering. If
it affects the pipeline itself (endpoints, live probing, auto-rebuild),
put it under Model Probing and wire progress events through
`chrome/probe.ts` so the progress strip animates. If it belongs to a
specific *page* (e.g. Snapshot Map topic presets), give it its own tab
alongside the existing ones — the `TABS` array in `SettingsPage.tsx` is
authoritative.

## Pages (viewer navigation)

There are **two instruments** built from one codebase — `nebulai`
(`index.html` → `src/main.ts`) and `seer` (`seer.html` → `src/seer-main.ts`).
The `Page` union in `viewer/src/app/slices/shell.ts:45-52` is
`"map" | "behavior" | "snapshot" | "interp" | "guide" | "sessions" | "seer"`;
which of the seven a given instrument owns is `APP_PAGES` (`shell.ts:62-65`) —
`nebulai: ["map","behavior","interp","guide"]`,
`seer: ["seer","sessions","snapshot"]`. `setPage` silently drops a page the
running app does not own. Add a page by (1) extending `Page`, (2) adding it to
the right `APP_PAGES` row, (3) adding its label to `chrome/apps/nav.ts`,
(4) rendering it from that app's `renderPage`, and (5) toggling any
`body.page-*` class. `viewer/tests/unit/app-pages.test.ts` pins the union, the
table and the labels against each other, so a half-done addition fails there
rather than at runtime.

Nebul.AI's four pages (pill labels in parentheses, and they are not the ids):
- **`map`** (Semantic map) — the semantic-cloud viewer (Atlas / Chord /
  Hierarchy / Compare). Owns `#stage` and all driver-backed views.
- **`behavior`** (Behavior) — the behavioural-divergence study's page.
- **`interp`** (Internals) — the Internals views registered in
  `src/scene/interp/registry.ts`.
- **`guide`** (Episodes) — the in-instrument guide to those views and their
  research references.

Seer's three pages, in its own entry (`viewer/seer.html`):
- **`seer`** (Live) — the live capture view.
- **`sessions`** (Transcripts) — the drop-a-transcript forensic view.
- **`snapshot`** (Topics) — the Snapshot Map: JSON conversation-log ingest,
  per-topic keyword extraction, timeline scrubber, co-occurrence graph. Data
  flows through `chrome/snapshot.ts` (parser + analyzer) into
  `appStore.snapshot`. New topic presets belong in `DEFAULT_TOPICS` in
  `app/slices/snapshot.ts`, and the Settings → Snapshot tab must always
  surface the same preset editor.

## Bundled scripts

- `scripts/inspect_map.py <nebulai.json>` — summarize any map: meta line, top-N
  clusters with titles + sample members, size distribution. Use this instead of
  writing an inline JSON-poking snippet (every session so far has re-written
  one).
- `scripts/sweep_hdbscan.py <reduced.npz>` — sweep HDBSCAN params on a cached
  reduction (leaf/eom × min_cluster_size × min_samples) to pick clustering
  settings without re-running the minutes-long UMAP.
- `scripts/probe_endpoints.py` — the live OpenRouter/endpoint catalogue probe
  behind the variance plan's item 9.
- `scripts/sync-out.sh` — the out-of-band `out/` rsync to the deploy host.
- `scripts/we_wu_overlap.py` — the W_E vs W_U overlap statistic behind
  recommended-plan Track 2b.

## CLI verbs

All 17 top-level verbs of the `nebulai` console script (`src/nebulai/cli.py`):

```
tokens sae neurons edges channels interp metrics probe validate rename
compare intervene direction persona absorbing behavior variance
```

Four carry nested groups:

- `direction {list,survey,add,make,prompts,project,drop}`
- `persona {build,verify,list}`
- `absorbing {run,report,list}`
- `behavior {plan,run,analyze,calibrate,conformance,publish,inspect,serve}`
  (`src/nebulai/behavior/cli.py`)

There is a **second console script**: `seer = "nebulai.seer.cli:main"`
(`pyproject.toml` `[project.scripts]`), with 19 verbs of its own — see
`docs/SESSIONSEER-HANDOVER.md` §2. `nebulai` never imports `nebulai.seer`.

**The base install is torch-free, and stays that way.** `uv sync` installs the
numpy/safetensors stack only; `tests/test_no_torch_in_base.py` fails if that
changes. Two opt-ins bring torch in, deliberately and separately:
`uv sync --extra organisms` (torch, transformers, peft, datasets) for
`src/nebulai/organisms/`, the fine-tuning arm; and
`uv sync --group behavior-local` (pinned torch 2.14.0 / transformers 5.17.0 /
sentence-transformers 6.0.1) for the Behavior study's local model arms,
enforced by `tests/test_behavior_optional_dep.py`. Nothing else needs either.
