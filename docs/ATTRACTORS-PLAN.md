# Attractors — the five-primitive implementation plan

Status: approved plan, 2026-09-11. Not started; nothing below is built.

This is the engineering half of a research review that ranked the ten LLM
experiments most worth hosting in this tool — Sydney/Waluigi, Emergent
Misalignment, the Assistant Axis, SolidGoldMagikarp, Golden Gate Claude,
Project Vend, the refusal direction, the AI Village, Palisade's chess agents,
and Sonnet 4.5's evaluation awareness. The review's finding was that all ten
reduce to three objects — a **direction**, a **trajectory**, and a **control** —
and that five composable primitives cover every one of them:

| | primitive | one line |
|---|---|---|
| **P1** | Direction | a unit vector with a space tag; anything with points projects onto it |
| **P2** | Trajectory | an ordered path through *fixed* coordinates |
| **P3** | Ensemble | N runs of one protocol, drawn as a fan; variance is the headline |
| **P4** | Intervention | change one thing in the forward pass, show before and after |
| **P5** | Episode | a scripted, permalinked story over real data that ends at the bench |

Sibling documents, none of which this one supersedes:
[`recommended-plan.md`](../recommended-plan.md) (the atlas's own plan),
[`BEHAVIORAL-DIVERGENCE-PLAN.md`](BEHAVIORAL-DIVERGENCE-PLAN.md) (the Behavior
study; §2 below inherits its claim contract),
[`INTERP_FEATURES.md`](INTERP_FEATURES.md) (the 25 Internals features),
[`SESSIONSEER-LIVE.md`](SESSIONSEER-LIVE.md) (§7 of which already refused the
Atlas projection this plan finally supplies),
[`OBSERVABILITY-SURFACE.md`](OBSERVABILITY-SURFACE.md) (the audit these phases
are measured against).

---

## 1. Decision record

Seven decisions were put and all seven approved on 2026-09-11. They are
recorded here because most of the phases below are downstream of one of them,
and because reversing any of them changes more than one phase.

| | decision | approved | what it costs / buys |
|---|---|---|---|
| **D1** | **(c) Both.** A numpy forward pass for a Llama-architecture instruct model beside `gpt2_numpy.py` for all inference; torch as an optional extra `nebulai[organisms]` used only for the one study that needs gradients. | ✅ | Keeps "everything is numpy, provenance is literal" true for every *shipped* episode and keeps the static deploy intact. Costs ~1–2 weeks of backend work (phase 2) and gives single-digit tokens/s on CPU. |
| **D2** | **(a) Refuse.** Cross-space projection is greyed out with a reason, not warned about. | ✅ | Some overlays become impossible. That is the point: a picture you cannot make is one you cannot mistake for a finding. |
| **D3** | **(a) Yes** — the claim contract gains exactly one sentence for intervention-backed causal claims. | ✅ | The README's "No causal claims" line changes (§2.4). Without it P4 is a toy. |
| **D4** | **(a) PCA persona space by default, a Direction pair as an option. Never UMAP.** | ✅ | Needs the persona prompt set frozen and pinned, the same discipline as `instrument`'s question-set freeze. Buys coordinates that stay stable as sessions are added, so figures compare across studies. |
| **D5** | **(a) Built in Nebul.AI, drawn in Seer**, joined by a space file with a tag. | ✅ | Keeps the one-way import rule. Adds one exported artefact type and one HTTP hop. |
| **D6** | **(a) Measure, never export.** No orthogonalised-weight download, ever. | ✅ | Ships the whole scientific object (the two-mode histogram, the refusal-vs-α curve) and nothing that is itself a jailbroken model. |
| **D7** | **(a) Load the AI Village corpus from Hugging Face at runtime** under its research-only licence; never bundle. Among Us, ctfish and AI-norms ship as fixtures. | ✅ | That one episode needs network and cannot be part of the static deploy (§4.2 handles this in the manifest). |

---

## 2. Invariants

These are laws the tree already enforces. Every phase below is written to fit
them; a phase that needs one relaxed says so explicitly and none currently
does.

### 2.1 Code boundaries

- **`nebulai` never imports `seer`.** `src/nebulai/seer/` is reachable only
  through its own `seer` console script. D5 keeps this: persona space crosses
  the boundary as a *file* and an HTTP call, never an import. The reverse
  direction (seer importing nebulai) is permitted by the rule but avoided in
  §5.3 for a separate reason — the live server holds ~0.5 GB of weights and
  Seer must not load them.
- **`APP_PAGES` in `viewer/src/app/slices/shell.ts` is the authority** for page
  membership; `viewer/src/chrome/apps/nav.ts` holds labels only, and
  `tests/unit/app-pages.test.ts` pins the two together. Relabelling Guide →
  Episodes (phase 0) touches `nav.ts` alone and that test still passes.
- **One `SceneDriver` owns `#scene-canvas` at a time**
  (`viewer/src/scene/SceneDriver.ts`); toolkits never share a GPU context.
  Nothing in this plan adds a driver to the Map page — the Axis is a layout
  state of `AtlasDriver`, not a fifth `ViewMode`.
- **`InterpFeature.id` is typed `GuideResearchId`**
  (`viewer/src/scene/interp/InterpDriver.ts`), so registering a live view with
  no guide evidence is a compile error. Every new Internals feature below adds
  its id to `viewer/src/chrome/guideResearch.ts` first.

### 2.2 Honesty rules that new code inherits

- `missing` is never rendered as `0`, and `dropped_by_policy` is never
  collapsed into `missing` (`src/nebulai/seer/contract.py`,
  `viewer/src/seer/contract.ts`). Every new measured quantity carries a
  `Fidelity`.
- The **null is in the picture**, not in a footnote. `viewer/src/data/validation.ts`
  already refuses to render a verdict for an unvalidated map; P1 and P3 extend
  the same rule to axes and fans.
- **Absence is "not measured", never "measured as clean."** A map with no
  `channels.json` shows no channel UI at all.
- The **ramp is data, chrome is `--accent`** (`viewer/src/viz/tokens.css`).
- A **pinned model id is never substituted** (`src/nebulai/llm.py:same_model`,
  `serves_pin`). Persona space and every direction record the resolved commit
  sha of the model they were computed from.

### 2.3 Spatial rules

Restated from the review so new views inherit them rather than renegotiate:

- **R1** — a new view is a layout state (morph) or an overlay, never a new
  coordinate system the user must learn.
- **R2** — anything with a time axis lives in PCA or on Directions. UMAP stays
  on the atlas, where `backend/validate.py`'s own docstring records it
  manufacturing clean islands from shuffled noise at n=180.
- **R3** — zoom is fidelity; the caveat appears at the zoom where it bites.
- **R4** — two registers (Story / Bench), one state, one permalink.
- **R5** — the null ships with the figure.
- **R6** — hypotheses are manipulated, not typed.
- **R7** — foreign data wears foreign clothes: a text-embedder position and a
  model-internal one never share a glyph.
- **R8** — one mono readout style, tabular digits, everywhere.

### 2.4 The claim-contract amendment (D3)

`BEHAVIORAL-DIVERGENCE-PLAN.md` §1.1 currently permits one strongest sentence.
Phase 4 adds a second, and only this second:

> Under protocol P, intervening on direction `d` at layer L changed behaviour
> B from X to Y (n = …, seed = …).

Everything §1.2 forbids stays forbidden. In particular `d` **is** not refusal /
evil / truth: it is a direction extracted by method M from data D that, when
intervened on, moved behaviour B. The README's **Honesty notes** bullet
"This is a visualization + clustering tool over public micro models. No causal
claims." is replaced in phase 4 by:

> Visualization and clustering over public micro models. Causal claims only
> where an intervention was run, stated in the intervention's own terms — never
> about what a direction *is*.

---

## 3. New data contracts

Three new files and one new endpoint family. All four are additive: no existing
artifact changes shape, and `nebulai.json` does not bump its schema version.

### 3.1 `out/<model>/channels.json` — per-point scalars

One flat array per channel, aligned to `nebulai.json` point ids by index. This
is the carrier for the glitch lens (phase 0) *and* for every direction
projection (phase 1), which is why it is one file rather than two.

```jsonc
{
  "meta": {
    "model": "gpt2",
    "revision": "<resolved sha>",
    "n_points": 49857,
    "point_source": "nebulai.json",     // ids are that file's ids, in order
    "created": "2026-09-11T00:00:00Z"
  },
  "channels": [
    {
      "id": "we_norm",
      "label": "row norm ‖W_E[t]‖₂",
      "space": "W_E.raw",               // NOT the mean-centred vectors the map used
      "method": "l2",
      "formula": "sqrt(sum(W_E[t]**2))",
      "fidelity": "deterministic",
      "units": "l2",
      "values": [ /* n floats */ ]
    }
  ]
}
```

`space` is load-bearing and is what D2 enforces against. The known tags, which
must stay a closed set in code (`src/nebulai/spaces.py`, §4.1):

`W_E.raw` · `W_E.centered` · `W_U.raw` · `resid.L<k>` · `mlp_out.L<k>` ·
`sae.L<k>.<repo>` · `text-embed.<model>` · `persona-pca.<space_id>`

A channel computed in one space may never be plotted against a direction
declared in another. The refusal is in the loader
(`viewer/src/data/channels.ts`), not in the UI, so no view can route around it.

### 3.2 `out/<model>/directions.json` — the direction registry

```jsonc
{
  "meta": { "model": "…", "revision": "…", "created": "…" },
  "directions": [
    {
      "id": "refusal-diffmeans-L14",
      "label": "harmful − harmless, layer 14",
      "space": "resid.L14",
      "method": "diff_of_means",        // | pca | sae_decoder | lora_rank1 | probe | two_selection
      "d": 960,
      "vector": [ /* d floats, unit-normalised */ ],
      "source": {
        "kind": "computed",             // | imported
        "protocol": "…",                 // frozen prompt-set id, or the upstream repo+sha
        "n_pos": 128, "n_neg": 128,
        "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "revision": "<sha>"
      },
      "projection": { "channel": "proj.refusal-diffmeans-L14" },
      "null": {
        "method": "random_rotation",
        "seed": 0,
        "n": 32,
        "channel": "proj.refusal-diffmeans-L14.null"
      }
    }
  ]
}
```

Two rules the writer enforces and the loader re-checks:

- **A direction with no `null` block is not renderable.** R5 is mechanical
  here, not a convention: `directionsFor()` drops an entry missing its null
  channel and reports it, the same way `validation.ts` renders no verdict
  rather than a passing one.
- **`projection.channel` must exist in `channels.json` and carry the same
  `space`.** A direction whose projections were never computed is metadata, not
  a plottable object, and the UI says so.

### 3.3 `out/persona/<space_id>/space.json` — a fixed coordinate system (D4)

```jsonc
{
  "meta": {
    "space_id": "smollm2-360m-instruct@<sha>.v1",
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "revision": "<resolved sha>",
    "layer": 18,
    "pooling": "last_token_mean_over_prompt_set",
    "prompt_set": { "id": "personas.v1", "sha256": "…", "n": 275 },
    "created": "…"
  },
  "basis": { "components": [ /* k × d */ ], "mean": [ /* d */ ], "explained_variance_ratio": [ … ] },
  "archetypes": [ { "name": "therapist", "scores": [ 2.41, -0.18, … ] } ],
  "control": {
    "method": "label_permutation",
    "n": 500,
    "pc1_evr": 0.31,
    "pc1_evr_null_p95": 0.14,
    "verdict": "above_null"          // | at_null | below_null — see §7 risk A
  }
}
```

`control` is not optional and is not a footnote: a space whose PC1 does not
clear its permutation null renders with `verdict` visible and **cannot be used
as the default trajectory coordinate system**. That is the mechanism that makes
risk A (§7) survivable instead of silent.

### 3.4 `POST /live/place` and `POST /live/intervene` — live-server endpoints

`src/nebulai/backend/interp/live_server.py` serves `/live/health`,
`/live/trace` and `/live/sae` on :8123. Two more, added in phases 2 and 4:

```
POST /live/place      { space_id, texts: [...] }
  → { space_id, layer, coords: [[pc1, pc2], …], fidelity: "deterministic" }

POST /live/intervene  { prompt, max_tokens, intervention: {…} }
  → { baseline: {…}, intervened: {…}, intervention: {…echoed…} }
```

`intervention` is one of exactly four verbs, and the server rejects anything
else rather than interpreting it:

| verb | payload | what it does |
|---|---|---|
| `clamp` | `{feature, value, layer, sae_repo}` | pin one SAE feature's activation |
| `add` | `{direction_id, alpha, layer}` | add α·d to the residual stream at L |
| `ablate` | `{direction_id}` | project d out at every layer |
| `cap` | `{layer, lo, hi}` | clamp activations into a range |

No verb writes weights. D6 is enforced at this boundary: there is no endpoint,
flag or CLI path that emits an orthogonalised checkpoint, and `ablate` exists
only as an inference-time hook.

---

## 4. Phases

Sizes assume the current agent-assisted pace and are for sequencing, not
commitment. Every phase ships at least one episode, so no phase is
infrastructure-only.

### Phase 0 — Glitch lens + Episodes (days)

**Goal.** Ship experiment #4 (SolidGoldMagikarp) out of data that already
exists, and rename the Guide surface into the Episode surface, before any new
primitive lands. This phase is almost entirely viewer work — and that is the
finding that made it phase 0.

**What already exists.** `compute_embed()` in
`src/nebulai/backend/interp/bundles.py` already computes and ships the **exact
per-token row L2 norm** into `embed.json` for Internals #15 (Embedding
Constellation), along with PCA coords and a decoded orthographic property. The
missing half of the glitch signal is distance-to-centroid, which is three lines
beside it.

**Backend.**

- `src/nebulai/spaces.py` **(new, ~40 lines)** — the closed `Space` enum and
  `compatible(a, b) -> bool`. Nothing else in phase 0 uses it; it lands now so
  phase 1 does not introduce a string-typed contract first and tighten it later.
- `src/nebulai/backend/channels.py` **(new)** — `Channel` dataclass,
  `write_channels(path, model, revision, channels)`, and the id/length/space
  validation the loader will trust.
- `src/nebulai/backend/interp/bundles.py` — extend `compute_embed()` with
  `centroid_dist` (L2 to the mean row, computed on the **raw** `W_E`, before
  any centring) beside the existing `norm`, and stamp
  `meta.quantity`/`meta.formula` for it. Bump the bundle's own `created`.
- `src/nebulai/frontends/tokens.py` — at the point where `V` is mean-centred
  (the `if center:` branch, ~line 258), compute `we_norm` and `we_centroid_dist`
  **from the pre-centred matrix** and hand them out through `Units.meta` so the
  CLI can write them. The distinction matters and is the whole experiment: a
  glitch token's signature is that its row never moved from initialisation,
  which is a fact about the raw rows. The map's own geometry is centred; the
  channel is not; the `space` tag on the channel is what keeps them apart.
- `src/nebulai/cli.py` — `_run_tokens` writes `channels.json`; new
  `nebulai channels <model> [--recompute]` subcommand that regenerates channels
  for a built map without touching its geometry (same lifetime argument as
  `nebulai rename`: channels and coordinates have separate lifetimes).

**Viewer.**

- `viewer/src/data/channels.ts` **(new)** — lazy fetch + columnarise to
  `Float32Array`, keyed per dataset, following `data/validation.ts`'s pattern
  exactly (fetch once, never throw, absence renders nothing). Exports
  `ensureChannels(datasetId)`, `channelsFor(datasetId)`, `compatible(space, space)`.
- `viewer/src/app/slices/atlas.ts` — add `channel: { id: string | null; lo: number; hi: number }`
  and `setChannel` / `setChannelRange`. No other slice changes.
- `viewer/src/scene/drivers/AtlasDriver.ts` — one instanced scalar attribute
  bound to the active channel, used for (a) a filter and (b) an optional
  ramp override. **Count the existing vertex buffers before writing this**: the
  WebGPU ceiling is 8 and `CompareDriver` documents that going one wider is
  rejected *silently*. If Atlas has no slot free, pack the channel scalar into
  an unused `.w` lane rather than adding a buffer.
- `viewer/src/chrome/SearchPanel.tsx` — a "near the centroid" filter chip that
  sets the channel range, and a ranked list of the lowest-norm tokens with
  their exact values in the mono readout (R8).
- `viewer/src/chrome/apps/nav.ts` — `{ label: "Guide", page: "guide" }` →
  `{ label: "Episodes", page: "guide" }`. The page id is wire format and does
  not move; `nav.ts`'s own docstring already establishes that labels and ids
  differ deliberately, and `app-pages.test.ts` pins membership rather than
  labels, so it continues to pass untouched.
- `viewer/src/chrome/tours.ts` — `Tour` gains two fields:

  ```ts
  /** Data this episode needs; the card refuses to start if any is absent. */
  manifest: { datasets?: string[]; bundles?: string[]; corpora?: string[]; network?: boolean };
  /** "story" opens in narrated register; the last step hands over the bench. */
  register?: "story" | "bench";
  ```

  Existing tours gain `manifest: { datasets: ["gpt2"], bundles: [...] }` and
  keep working.
- `viewer/src/chrome/GuidePage.tsx` — episode cards read `manifest` and render
  an honest unavailable state (which dataset or bundle is missing) instead of
  starting a tour that will fail mid-way.
- `viewer/src/app/actions.ts` — **`runEpisodeStep(episodeId, step)`**. This is
  not cosmetic: `atlas.setDataset` currently clears `tour` (correctly — tour
  copy is per-model), so any episode that crosses models kills itself on its own
  step. The episode runner sets dataset and tour ref in one commit. Either add
  `setDataset(id, ds, { keepTour: true })` used only here, or have the runner
  re-apply the tour after the dataset settles; the first is cheaper to test.
- `viewer/src/chrome/urlState.ts` — new keys `episode` and `step`, validated
  against the episode registry through the same injected-hook pattern
  `InterpUrlHooks` already uses so Seer's bundle stays free of them.

**Episodes shipped.** 0 · Grokking Clock (exists as a driver, needs the wrapper)
· 1 · Induction (exists as a tour) · 2 · What an SAE feature is (exists) ·
3 · **SolidGoldMagikarp** (new).

**Tests.** `tests/test_channels.py` (schema, length alignment, space tags,
raw-vs-centred correctness on a 200-row fixture) ·
`viewer/tests/unit/channels.test.ts` (columnarise, absence, space refusal) ·
`viewer/tests/unit/episodes.test.ts` (manifest gating; the cross-model step
does not clear its own tour) · extend
`viewer/tests/e2e/nebulai/views.spec.ts` with the channel filter.

**Exit criteria.** Opening `#page=map&model=gpt2&channel=we_norm` shows the
low-norm knot as a visibly distinct region; hovering a member prints its exact
raw norm and centroid distance; the episode narrates why those rows never
moved; and no existing golden changes.

---

### Phase 1 — P1 Direction (~2–3 weeks)

**Goal.** A direction becomes a first-class object the atlas can settle onto,
with its null beside it. Gated by **D2** (hard refusal) and **D4** (PCA, not
UMAP, wherever a fixed frame is needed).

**Backend.**

- `src/nebulai/backend/directions.py` **(new)** — the five constructors, each
  returning a `Direction` with its provenance filled in:
  `diff_of_means(pos, neg, space)` · `pca_component(matrix, k, space)` ·
  `from_sae_decoder(repo, feature_id, layer)` · `from_lora_rank1(path)` (phase 5
  only) · `from_two_selections(a_ids, b_ids, vectors, space)`.
  Plus `project(points, direction) -> (parallel, orthogonal)` and
  `null_directions(d, n, seed)` — random unit vectors in the same space, whose
  projection distribution is the ghost histogram.
- `src/nebulai/backend/import_directions.py` **(new)** — adapters for the four
  upstream artefacts, each recording the upstream repo and commit sha as
  `source.protocol` and refusing to import a vector whose dimensionality
  disagrees with the target model:
  `andyrdt/refusal_direction` · `safety-research/persona_vectors` ·
  `safety-research/assistant-axis` · a Neuronpedia feature id · the Among Us
  probe weights.
- `src/nebulai/cli.py` — `nebulai direction` with subcommands `list`, `add`
  (from an importer), `make` (diff-of-means from two id sets or two cluster
  ids), `project` (compute projection + null channels for a built map),
  `drop`. Writes `directions.json` and appends to `channels.json`.
- `src/nebulai/backend/export.py` — unchanged. Directions are a sidecar
  precisely so `nebulai.json` and its schema version stay put and the parse
  worker is untouched.

**Viewer.**

- `viewer/src/data/directions.ts` **(new)** — loader mirroring `channels.ts`;
  drops any direction missing its null channel or whose projection channel's
  space disagrees, and exposes the drop reason for the UI to state.
- `viewer/src/app/slices/atlas.ts` — new UI state:

  ```ts
  export interface AxisUI { directionId: string | null; t: number; showNull: boolean; }
  ```

  with `setAxisDirection(id)` and `setAxisT(t)`. **Not** a new `ViewMode`:
  `setViewMode` clears `selection` and `hover`, which is wrong for a morph, and
  R1 says a new view is a layout state. `viewMode` stays `"atlas"` throughout.
- `viewer/src/scene/drivers/AtlasDriver.ts` — a third position target beside
  `pos2`/`pos3`. The existing GPU blend is `mix(pos2, pos3, uMorph)`; it becomes
  a three-way blend with `uAxis`. Positions come from the projection channels:
  `x = ⟨p, d⟩`, `y = ⟨p, d⊥⟩`. **Pack the axis and its null into one vec4
  buffer** — `(axisPar, axisOrth, nullPar, nullOrth)` — so the whole primitive
  costs one vertex buffer rather than two, for the ceiling reason in phase 0.
  The CPU-side hover projection at `AtlasDriver.ts:1452` (which mirrors the GPU
  blend exactly) has to learn the same three-way mix or picking drifts from
  pixels mid-morph.
- `viewer/src/chrome/AxisRail.tsx` **(new)** — the histogram rail under the
  axis: the real distribution solid, the null distribution as a ghost, always
  both. Built on `viz/ChartCard.tsx` and `viz/chart-theme.ts`, with
  `viz/StatStrip.tsx` carrying separation, overlap and n.
- `viewer/src/chrome/SearchPanel.tsx` — "make a direction from these two
  selections". This is R6 made concrete and is the same gesture for a
  researcher and an amateur: pick two clusters, get their difference.
- `viewer/src/chrome/urlState.ts` — `axis=<direction id>`.

**Episodes shipped.** 3 · Assistant Axis (static, from the published artefact —
275 archetypes as points in PC1×PC2) · 7 · Refusal direction (static, imported
vector + its two-mode histogram).

**Tests.** `tests/test_directions.py` (constructors, unit-normalisation,
projection identities, null distribution is centred and its spread scales as
expected) · `tests/test_import_directions.py` (dimensionality refusal, sha
recorded) · `viewer/tests/unit/directions.test.ts` (null-missing drop,
space refusal) · `viewer/tests/unit/axis-layout.test.ts` (the CPU mix matches
the documented GPU mix at t = 0, 0.5, 1) · new golden
`viewer/tests/e2e/goldens/webgpu/axis.png`.

**Exit criteria.** A direction imported from `refusal_direction` renders as an
axis over a map whose space tag matches, with its ghost null under it; a
direction whose space does not match is greyed with a stated reason and cannot
be forced; two selections produce a new direction that persists to
`directions.json` and survives a reload through the permalink.

---

### Phase 2 — P2 Trajectory (~3–4 weeks) · **the D1 phase**

**Goal.** A transcript becomes a path through fixed coordinates. This is the
phase that unlocks the most experiments (1, 6, 8, 9, 10) and the one carrying
the plan's biggest scientific risk (§7 A).

**Backend — the instruct model (D1a).**

- `src/nebulai/backend/interp/llama_numpy.py` **(new, the ~1–2 week item)** —
  a numpy forward pass for the Llama family: RMSNorm, RoPE, SwiGLU, grouped-
  query attention, tied/untied embeddings, and a **KV cache from the first
  commit** (phase 3's self-play is quadratic without it, and retrofitting a
  cache into a validated forward pass is how correctness regressions happen).
  Same shape as `gpt2_numpy.py`: a `Trace` dataclass and a `LlamaNumpy` class
  exposing per-layer residuals, attention and a logit-lens identity.
  Validation, mirroring how `gpt2_numpy.py` earned trust: correct next-token
  predictions against a reference, causal attention, logit-lens identity, and
  a numerical diff against a torch reference run **once, offline, in the
  optional extra** — the check may use torch even though the shipped path
  never does.
- **Model choice: `HuggingFaceTB/SmolLM2-135M-Instruct` first, `…-360M-Instruct`
  as the quality step.** The reason is specific rather than aesthetic:
  SmolLM2-135M's *base* sibling is already mapped, named and validated in this
  repo (token, neuron and SAE maps all exist), so the instruct model shares its
  tokenizer and vocabulary and every trajectory can link back to an existing
  atlas point. Qwen2.5-0.5B-Instruct stays the fallback if SmolLM2's persona
  turns out too weak, at the cost of QKV biases in the forward pass and no
  existing sibling map.
  **What difference it makes:** 135M generates fast enough on CPU for live
  steering and long self-play, and is the most likely of the three to fail the
  persona-PC1 control; 360M roughly triples per-token cost and is the more
  likely to carry an interpretable axis. Start at 135M because the control is
  cheap to run and a failure there is informative, not wasted.
- `src/nebulai/backend/persona.py` **(new)** — the Assistant-Axis recipe:
  frozen prompt set → mean activations at a pinned layer → PCA → `space.json`
  (§3.3), **including the label-permutation control**. Writes `verdict`.
- `src/nebulai/prompts/personas.v1.json` **(new)** — the frozen archetype set,
  sha256 stamped into every space that uses it. Frozen the way `instrument`
  freezes its question set: changing it makes a new `space_id`, never an edit
  in place.
- `src/nebulai/backend/interp/live_server.py` — add `POST /live/place` (§3.4).
- `src/nebulai/cli.py` — `nebulai persona build --model … --prompts personas.v1`
  and `nebulai persona verify <space_id>` (re-runs the control alone).

**Backend — placement and corpora (D5, D7).**

- `src/nebulai/seer/place.py` **(new)** — `seer place <run_id> --space <id>
  [--live-url http://127.0.0.1:8123]`. Calls `/live/place` over HTTP and writes
  `placement.json` beside the run. HTTP rather than an import for a concrete
  reason: the live server holds the weights resident, and a Seer command that
  imported the model would load ~0.5 GB per invocation. A `--in-process` flag
  exists as a fallback and says in its help why it is not the default.
- `src/nebulai/seer/adapters/corpus_amongus.py`, `corpus_ctfish.py`,
  `corpus_village.py`, `corpus_transcript.py` **(new, beside the existing
  `claude.py` / `codex.py` / `hermes.py`)** — each maps its source into the
  canonical `EventType` vocabulary with `CaptureMode.RECONCILED` and honest
  `Fidelity` per field. The Village adapter fetches from Hugging Face at
  runtime and **never writes the corpus into the repo** (D7); it refuses to run
  offline with a message naming the licence rather than falling back to a
  bundled copy.
- `src/nebulai/seer/cli.py` — `seer import <corpus> …` beside the existing
  `import-spool`.

**Viewer.**

- `viewer/src/data/persona.ts` **(new)** — loads `space.json`, exposes the
  archetype scatter and the control verdict. If `verdict != "above_null"` the
  space is loadable but **not selectable as the default coordinate system**,
  and the UI states why.
- `viewer/src/scene/sessions/SessionFieldDriver.ts` — the Persona projection
  lands **here first, not on Live**. The Transcripts page already draws a 3-D
  per-turn field with a playback transport, and imported corpora are finished
  runs; Live gets the same projection in phase 3 once the shape is proven.
  Per `SESSIONSEER-LIVE.md` §1, a foreign coordinate system gets a **cross-fade,
  not a morph**, and the time cursor degrades to a trail parameter.
- `viewer/src/scene/sessions/appearance.ts` + `viewer/src/seer/encoding.ts` —
  R7 made mechanical: a turn placed by the pinned model draws solid; a turn
  placed by a text embedder draws dashed and carries the existing
  "NOT model-internal" grammar. `seer-encoding.test.ts` fails if a placement
  source has no glyph.
- `viewer/src/chrome/AbsorbingPanel.tsx` **(new)** — the Waluigi test: the 2×2
  in-character/out-of-character transition matrix with the base rate beside it,
  Wilson intervals on every cell, and the n it was computed from. This is the
  smallest surface in the plan and the one carrying an unpublished result.
- `viewer/src/app/slices/sessions.ts` — `personaSpaceId`, `placement`,
  `trailLength`.

**Episodes shipped.** 1 · Sydney (transcript, text-embedder placement, dashed —
the honest treatment of a closed model) · 6 · Project Vend · 8 · AI Village ·
9 · Chess hacking · 10 · Evaluation awareness (turns coloured by an
eval-awareness probe direction from phase 1).

**Tests.** `tests/test_llama_numpy.py` (forward correctness, causal mask, RoPE
at position 0 and at the context edge, KV-cache equivalence to the uncached
path — the last is the one that catches cache bugs) ·
`tests/test_persona.py` (frozen-prompt sha, permutation control rejects a
shuffled space) · `tests/test_place.py` · `tests/test_corpus_adapters.py` (no
agent-specific key survives the boundary — the existing M0 exit test, extended
to four new adapters) · `viewer/tests/unit/persona-space.test.ts` ·
`viewer/tests/unit/absorbing.test.ts` (Wilson intervals, base-rate arithmetic).

**Exit criteria.** A Sydney transcript and an Among Us game both render as paths
in the same frame with visibly different glyphs; the absorbing-state panel
reports a transition matrix with intervals and an n; and a persona space whose
control failed cannot be silently selected as the default.

**Measured.** `smollm2-135m-instruct@12fd25f77366.no_exclamation` — 2,400
self-play conversations x 6 assistant turns, 14,400 judged turns, 12,000
transitions. P(violate at t+1 | violated at t) = **0.5517** [0.5341, 0.5692]
(1,691/3,065) against a base rate of 0.2593 [0.2516, 0.2673] and a
within-conversation shuffle null whose p95 is 0.5325 (mean 0.5244, n=500,
p=0.0020). Verdict `absorbing_above_null`. The judge is the rule's own regular
expression, stated in the artifact; no model judges these transcripts. Run in
two sittings — 1,632 conversations, then continued with `--resume` to 2,400 —
for 3.4 h of CPU wall clock in total.

---

### Phase 3 — P3 Ensemble (~2 weeks)

**Goal.** Variance becomes the headline. Runs the plan's first original study.

**Backend.**

- `src/nebulai/seer/cli.py` — `seer run … --repeat N [--seed-base K]`.
- `src/nebulai/seer/budget.py` **(new)** — the cost gate extended to Seer.
  `llm.py`'s `Budget` (preflight → approve → charge) cannot see agent-side
  billing, so this estimator is honest about that: **run 1 of a new protocol
  has no estimate and is marked `Fidelity.MISSING`**, requiring an explicit
  acknowledgement rather than a fabricated number; runs 2…N are estimated from
  run 1's measured usage and go through the same preflight/approve step.
  `DEFAULT_MAX_COST_USD` from `corpus.py` is the shared ceiling.
- `src/nebulai/seer/ensemble.py` **(new)** — fan statistics: per-step median and
  envelope, Wilson intervals for rates, and the split-half reliability that
  reuses the Δ̂ logic from `BEHAVIORAL-DIVERGENCE-PLAN.md` §6.4.1 (between-
  condition separation minus each condition's own split-half separation).
- `src/nebulai/seer/server.py` — `/seer/ensemble/<id>`.

**Viewer.**

- `viewer/src/scene/sessions/SessionFieldDriver.ts` + `scene/seer/LiveField.ts`
  — envelope geometry: N faint paths, an emphasised median, a band. Works in
  every projection including Persona, because the fan is geometry over the
  placement rather than a new placement.
- `viewer/src/chrome/EnsemblePanel.tsx` **(new)** — the distribution rail.
  **N < 20 renders as an interval, never a point**, and a fan from a single seed
  says so on its face.
- `viewer/src/app/slices/sessions.ts` — `ensembleId`, `showEnvelope`.

**Episodes shipped.** 1 · **the Waluigi absorbing-state test, live** — persona
prompt on SmolLM2-Instruct, 2,400 self-play conversations, P(violate at t+1 |
violated at t) against the base rate. Nobody has published this cleanly on open
weights, and the whole apparatus for it exists once phase 3 lands. The statistic
itself is already measured — see phase 2 — so what phase 4 adds is the fan over
seeds, not the number.
6 · Vending-Bench as a fan of runs.

**Tests.** `tests/test_seer_budget.py` (run 1 is `MISSING`, not 0; approval
gates the repeat) · `tests/test_ensemble.py` (Wilson intervals, split-half on a
synthetic fixture with a known answer) ·
`viewer/tests/unit/ensemble.test.ts` (the N < 20 rule is enforced in the
component, not by the caller).

**Exit criteria.** `seer run … --repeat 20` prices itself before spending,
renders as a fan, and the absorbing-state panel reports the Waluigi statistic
with an interval — including, if that is what the data says, an interval
spanning the base rate.

---

### Phase 4 — P4 Intervention (~3 weeks) · **the D3/D6 phase**

**Goal.** The first place the tool may make a causal claim, in exactly one
sentence.

**Backend.**

- `src/nebulai/backend/interp/intervene.py` **(new)** — the four verbs (§3.4)
  as forward hooks over both `gpt2_numpy.py` and `llama_numpy.py`. Reuses
  `bundles.py`'s existing `_forward_from()` and `_forward_ablate()` machinery
  rather than a parallel implementation.
- `src/nebulai/backend/interp/live_server.py` — `POST /live/intervene`, with a
  verb allow-list and a flat refusal for anything else.
- `src/nebulai/cli.py` — `nebulai intervene` for batch runs (sweep α, write a
  curve bundle for an episode to read).
- **D6 is enforced by absence**: no flag, endpoint or code path writes a
  modified checkpoint. A test asserts this (`tests/test_intervene.py::test_no_weight_export`)
  so a future contributor cannot add one without deleting an explicit test.

**Viewer.**

- `viewer/src/chrome/SteerRail.tsx` **(new)** — one slider; two generations side
  by side; the prompt's point sliding along the phase-1 axis live as α moves.
  This is the single interaction that makes P1 and P4 legible as one idea.
- `viewer/src/scene/interp/SteerDriver.ts` **(new)** + registry entry in
  `viewer/src/scene/interp/registry.ts` — and, first, the new id in
  `viewer/src/chrome/guideResearch.ts`, because `InterpFeature.id` is typed
  `GuideResearchId` and registering without guide evidence is a compile error.
- `viewer/src/chrome/GuidePage.tsx` — the amended claim sentence (§2.4) renders
  on every intervention figure, not in a footer.

**Documentation changes in this phase.** README **Honesty notes** bullet
replaced per §2.4 · `BEHAVIORAL-DIVERGENCE-PLAN.md` §1.1 gains the intervention
clause and §1.2 is left untouched · `INTERP_FEATURES.md` gains the new rows.

**Episodes shipped.** 5 · **Golden Gate GPT-2** (clamp a feature in the SAE
already in the tree) · 3 · Assistant Axis, live (activation capping) ·
7 · Refusal, live (ablation, measured only).

**Tests.** `tests/test_intervene.py` (each verb changes what it claims to and
nothing else; α = 0 is bit-identical to baseline — the sharpest correctness
test available; no weight export) · `viewer/tests/unit/steer.test.ts`.

**Exit criteria.** Moving one slider changes a generation and moves a point
along an axis in the same frame; α = 0 reproduces baseline exactly; and every
figure the view exports carries the intervention's protocol in its stamp.

---

### Phase 5 — Model organisms (optional, D1b)

**Goal.** Experiment #2 (Emergent Misalignment) with a direction this tool
extracted rather than imported. The only item in the plan that needs gradients.

- `pyproject.toml` — `[project.optional-dependencies] organisms = ["torch", "transformers", "peft", "datasets"]`.
  The base install stays numpy-only; `src/nebulai/organisms/__init__.py` raises
  a clear install hint rather than an `ImportError` traceback.
- `src/nebulai/organisms/emergent_misalignment.py` **(new)** — rank-1 LoRA on
  Qwen2.5-0.5B-Instruct over the public insecure-code set, plus the
  **inoculation control** (identical data, two system-prompt framings) which is
  the cheap, sharp experiment and the one that makes the result mean something.
- Extract the misalignment direction into `directions.json` via
  `backend/directions.py:from_lora_rank1`, at which point every phase-1 and
  phase-4 surface renders it with no new UI.

**Exit criteria.** Two LoRA runs differing only in framing produce two
misalignment-rate curves; the direction extracted from one is renderable as an
axis; and the base install still has no torch in its dependency tree.

---

## 5. Boundary notes

**5.1 Why directions are a sidecar.** `nebulai.json` stays at schema 2 and
`viewer/src/data/parse.worker.ts` is untouched, so all thirteen built maps keep
loading byte-identically and no golden moves. A map with no directions has no
Axis state, which is the correct degradation.

**5.2 Why the Axis is not a `ViewMode`.** `setViewMode` clears `selection` and
`hover` by design; a morph must preserve both. R1 also says a new view is a
layout state. Adding `"axis"` to the `ViewMode` union would additionally break
`urlState.ts`'s `VIEWS` allow-list and require a driver swap for what is a
uniform write.

**5.3 Why placement is HTTP.** D5 puts persona space in Nebul.AI and draws it in
Seer. The import rule permits `seer` → `nebulai`, but the live server keeps
weights resident and a per-invocation import would reload them; HTTP also keeps
the boundary inspectable, which matters when the question "which model placed
this turn" is part of the figure's provenance.

**5.4 What stays out.** No RAG surfaces (features #7/#10/#11 were rescoped in
`OBSERVABILITY-SURFACE.md` §7 and stay rescoped) · no leaderboard language
(`BEHAVIORAL-DIVERGENCE-PLAN.md` §1.2) · no chatbot playground: generation
exists only as the after-image of an intervention · no abliteration export
(D6) · no "simple mode" that hides the 55% noise or the 0.5 seed-ARI —
simplification is narration added, never numbers removed.

---

## 6. File-touch index

New files, by phase. Modified files are listed in each phase above.

| phase | backend | viewer |
|---|---|---|
| 0 | `spaces.py` · `backend/channels.py` | `data/channels.ts` |
| 1 | `backend/directions.py` · `backend/import_directions.py` | `data/directions.ts` · `chrome/AxisRail.tsx` |
| 2 | `backend/interp/llama_numpy.py` · `backend/persona.py` · `prompts/personas.v1.json` · `seer/place.py` · `seer/adapters/corpus_{amongus,ctfish,village,transcript}.py` | `data/persona.ts` · `chrome/AbsorbingPanel.tsx` |
| 3 | `seer/budget.py` · `seer/ensemble.py` | `chrome/EnsemblePanel.tsx` |
| 4 | `backend/interp/intervene.py` | `chrome/SteerRail.tsx` · `scene/interp/SteerDriver.ts` |
| 5 | `organisms/emergent_misalignment.py` | — |

Drift guards that must be extended rather than worked around:
`viewer/tests/unit/app-pages.test.ts` · `tokens-sync.test.ts` ·
`seer-contract-sync.test.ts` · `seer-encoding.test.ts` ·
`guide-research.test.ts`.

Repo skills whose scope this plan widens, and which should be updated in the
phase that widens them: `.claude/skills/nebulai` (the orchestrator and its
`references/backend-contract.md`, which gains channels/directions/persona) ·
`.claude/skills/nebulai-viz-threejs` (the three-way position blend) ·
`.claude/skills/nebulai-viz` (routing for the new rails).

---

## 7. Risk register

**A · Persona PC1 may not exist at 135M–360M scale.** The Assistant Axis was
found at 27B–70B. A 135M instruct model may have no interpretable persona
component at all. This is the plan's largest scientific risk and it sits under
phase 2, which four episodes depend on.
*Mitigation, in order:* the label-permutation control in §3.3 decides it before
anything is built on top; escalate 135M → 360M → 1.7B; if all three fail,
D4's option (b) — a user-chosen Direction pair — becomes the trajectory frame,
and the negative result ("persona is not linearly separable below N parameters
under protocol P") is itself publishable and is the kind of finding this tool
exists to state honestly.

**B · The vertex-buffer ceiling.** `CompareDriver` documents that exceeding
WebGPU's 8-buffer limit is rejected *silently*. Count `AtlasDriver`'s buffers
before phase 0's channel attribute and again before phase 1's axis.
*Mitigation:* pack axis and null into one vec4; put the channel scalar in a
spare `.w` lane.

**C · Numpy generation throughput.** Phase 3's ~2,000 self-play conversations
are the heaviest compute in the plan, on a CPU forward pass. *Measured:* 135M at
6 turns and 48 conversations per batch ran 2,400 conversations in 3.4 h,
5.1 s per conversation, with the KV cache on from the first commit.
The risk is real — it is the reason the study ships a `--resume` that continues a
deadline-stopped run at the batch boundary instead of recomputing it.
*Mitigation:* KV cache from the first commit; batch across trials; if 360M
lands under ~5 tok/s, run the ensemble at 135M and use 360M only for the
steering rail, stating the model difference on the figure.

**D · Episodes that cross models.** `atlas.setDataset` clears `tour`.
*Mitigation:* `runEpisodeStep` in phase 0, with a unit test that a cross-model
step keeps its tour ref.

**E · The Village episode is not static.** D7 means that one episode needs
network and a research-only licence.
*Mitigation:* `manifest.network = true`; the static build renders the card as
unavailable-with-a-reason rather than failing at step 3.

**F · Ensemble cost is invisible to the existing gate.** Agent-side billing does
not flow through `llm.py`.
*Mitigation:* `seer/budget.py`'s explicit `MISSING` for run 1 — an honest
absence rather than a fabricated estimate, which is the same rule the reducer
already applies to every unmeasured value.

---

## 8. Provenance

Ranked from a three-strand review (persona failures · interpretability geometry
· agentic behaviour) of 28 experiments, 2026-09-10, against repo state
`4fe6a71`. Primary sources for all ten experiments and twelve runners-up are in
the review itself; the ones that matter most to phases 1–4 are
`arxiv.org/abs/2601.10387` (Assistant Axis, and `github.com/safety-research/assistant-axis`),
`arxiv.org/abs/2406.11717` (refusal, and `github.com/andyrdt/refusal_direction`),
`arxiv.org/abs/2507.21509` (persona vectors),
`arxiv.org/abs/2502.17424` + `arxiv.org/abs/2506.11613` (emergent misalignment
and its 0.5B rank-1 replication),
`arxiv.org/abs/2504.04072` (Among Us — logs *and* probe weights),
`github.com/PalisadeResearch/ctfish`,
`huggingface.co/datasets/aidigestorg/ai-village`, and
`lesswrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post`.
