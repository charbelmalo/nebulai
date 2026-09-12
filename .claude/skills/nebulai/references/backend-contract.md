# The shared back-end contract

Everything a front-end must satisfy, and everything the back-end guarantees in
return. If you keep this contract, a new pipeline is *only* a new front-end —
you never touch `backend/`. Code lives in `src/nebulai/`.

## `Units` — the universal interface (`units.py`)

```python
@dataclass
class Units:
    ids: list[int]        # stable per-unit reference (token id, feature idx, neuron idx)
    vectors: np.ndarray   # (n, d) float32 — the GEOMETRY the map is built from
    labels: list[str]     # display text per unit (token string, auto-interp label)
    meta: dict            # provenance: model, unit type, layer, counts, flags
```

- `__post_init__` hard-fails unless `len(ids) == vectors.shape[0] == len(labels)`.
  Misalignment would silently mislabel points and poison every downstream stage,
  so it's a constructor invariant, not a runtime check.
- **`float32` everywhere.** It's what umap-learn computes in; float64 buys
  nothing upstream of a stochastic projection and doubles memory.
- **Geometry vs text is the load-bearing split.** `vectors` is the only thing
  the layout sees; `labels` is only for hover + naming. Swapping the geometry
  while keeping labels is exactly the "model space vs label space" toggle
  Plans A/B need — so never fold label information into `vectors`.
- `meta` keys that matter downstream: `unit` (→ `unit_ref.kind` in the export),
  `layer` (→ per-point `layer`), plus anything provenance-worthy. Everything in
  `meta` is copied verbatim into `nebulai.json`'s `meta`, so each artifact is
  self-describing.

A front-end is done when `load_*_units(...) -> Units` returns and `len(units)`
matches the number of vectors. Nothing else is required of it.

## reduce (`backend/reduce.py`)

`reduce_vectors(vectors, cluster_dim=10, n_neighbors=30, seed=42) -> (u_cluster, u3, u2)`

- **`u_cluster` (n, cluster_dim)** — `min_dist=0.0`, cosine. HDBSCAN runs on
  this. Clustering in ~10-D (not 2-D/3-D) is deliberate: low-D UMAP invents
  clusters that aren't there and merges ones that are.
- **`u3` (n, 3)** — `min_dist=0.1`, cosine. The 3-D flythrough layout.
- **`u2` (n, 2)** — `PCA(u3)`, **not** an independent UMAP. Deriving 2-D from
  3-D guarantees the flat map is a camera angle on the cloud, so the Phase-2
  2D↔3D toggle interpolates instead of teleporting.
- **cosine** throughout: direction carries meaning; magnitude often just tracks
  frequency.
- **seed:** `>= 0` sets `random_state` (reproducible but single-threaded);
  `-1` omits it (non-deterministic, parallel, faster).

## cluster (`backend/cluster.py`)

`cluster_units(u_cluster, min_cluster_size=None, min_samples=None, method="leaf") -> (cluster_ids, probabilities)`

- **`method="leaf"` is the default** and matters: on these UMAP spaces the vocab
  / feature core is one connected density blob, so `eom` (the textbook default)
  drops ~everything into a single mega-cluster. `leaf` takes the finest stable
  clusters instead — more clusters, higher noise, but the clusters are real.
- No imposed `k` — the count is discovered.
- `cluster_id = -1` is **noise**, kept in the viz as background. Noise fraction
  is a headline quality metric, not a bug.
- Defaults: `min_cluster_size = max(15, n // 1000)`, `min_samples = 5`.
- `probabilities` (HDBSCAN membership) → per-point `confidence` → opacity.

## name (`backend/name.py`)

`name_clusters(units, cluster_ids, namer="auto", ...) -> ({cluster_id: title}, backend_used)`

- Each cluster is summarized by `_representatives()`: the **k=20 members nearest
  the cluster centroid by cosine**, computed in the *original* `Units.vectors`
  space (the model's notion of centrality, not the projection's).
- Backend chain by `namer`:
  - `auto` → **ollama → openrouter → centroid**
  - `openrouter` → openrouter → centroid
  - `ollama` → ollama → centroid
  - `anthropic` → anthropic → centroid
  - `none` → centroid
- ollama: M4 worker at `http://<m4-host>:11434`, default
  `liquidai/lfm2.5-1.2b-instruct` (never picks an `embed` model). OpenRouter:
  default `openai/gpt-oss-120b:free`, key from `OPENROUTER_API_KEY` or the last
  uncommented line in `~/.hermes/.env`. Anthropic: `claude-opus-4-8`, structured
  output via `output_config.format` json_schema.
- Every LLM backend falls through to `centroid` on any exception, so **naming
  never fails the pipeline**. The backend actually used is returned and stamped
  into the export (e.g. `ollama:liquidai/lfm2.5-1.2b-instruct`, `centroid`).

## export (`backend/export.py`) — the `nebulai.json` contract

The Phase-2 viewer should need this file and nothing else.

```jsonc
{
  "meta": { ...Units.meta, "schema_version": 2, "n_points", "n_clusters",
            "noise_fraction", "namer", "created" },
  "points": [
    { "id": 0,
      "unit_ref": { "kind": "token_embedding", "index": 464 },
      "label": " the", "confidence": 0.97, "layer": null,
      "xy": [1.2,-3.4], "xyz": [1.1,-3.2,0.8], "cluster_id": 17 }
  ],
  "clusters": [
    { "id": 17, "title": "articles & determiners", "size": 240,
      "centroid": [1.0,-3.1,0.7] }   // centroid in u3 (display) space
  ],
  "edges": {   // schema v2 (backend/edges.py) — similarity beams
    "space": "umap10", "metric": "gaussian_euclidean",
    "k_cluster": 5, "sigma": 3.02,
    "cluster_edges": [[0, 17, 0.93], ...],   // [a, b, weight], a < b, deduped
    "knn": { "k": 6, "sigma": 0.07,
             "ids": [/* n_points*k flat */], "sims": [/* n_points*k flat */] }
  }
}
```

- `unit_ref` is a typed `{kind, index}` object (kind = `Units.meta["unit"]`), so
  mixed maps stay unambiguous across pipelines.
- Both `xy` and `xyz` ship per point — the toggle is client-side interpolation.
- `noise_fraction` and `namer` in `meta` let any consumer caption honestly.
- **`edges`** is computed in the 10-D `u_cluster` space HDBSCAN clustered in,
  never the display layout (beams must reflect the geometry the clustering
  saw). Weight = Gaussian kernel over Euclidean distance — HDBSCAN's metric —
  with `sigma` (median candidate distance) stamped; cosine saturates on UMAP
  output coordinates and is only correct on original embedding rows. `knn`
  arrays are flat (`n_points * k`; row `i` at `[i*k, (i+1)*k)`, self
  excluded) for direct typed-array copy. CLI: `nebulai tokens --edges
  {knn,cluster,none}` (default `knn`); `nebulai edges <model>...` backfills
  from cached `reduced.npz` with no UMAP rerun and maintains
  `out/index.json` (`{"datasets": [...]}`). Consumers treat a missing
  `edges`/`schema_version` as v1 and degrade (beams off), never fail.

## render (`backend/viz.py`)

datamapplot: static PNG labels only the top ~60 clusters by size (readability);
interactive HTML labels all, with per-point `hover_text=repr(label)` (so a
leading-space token like `' cat'` is visibly distinct from `'cat'`) and search.
Noise → `noise_label="Unlabelled"`.

## Sidecars: channels + directions (never `nebulai.json`)

`nebulai.json` is schema **v2** and stays there. Later artifacts are written
BESIDE it and read independently by the viewer: a map with none of them renders
exactly as it always did. The first two are aligned to the map by point INDEX;
the third (interventions) is not a per-point quantity at all and is keyed to
nothing in the map — which is precisely why it lives in its own file.

```
out/<model>/
  nebulai.json      # schema v2 — untouched by everything below
  channels.json     # backend/channels.py  — per-point scalars
  directions.json   # backend/directions.py — unit vectors + their nulls
  interp/
    intervene_<name>.json   # backend/interp/intervene.py — a hook that RAN
```

### `channels.json` (`backend/channels.py`)

```jsonc
{ "meta": { "model": "gpt2", "revision": "<resolved sha>", "n_points": 49857,
            "point_source": "nebulai.json" },
  "channels": [
    { "id": "we_centroid_dist", "label": "…", "space": "W_E.raw",
      "method": "l2", "formula": "‖x − mean(W_E)‖₂",
      "fidelity": "deterministic",          // seer/contract.py vocabulary
      "stats": { "min": …, "max": …, "mean": …, "n_missing": 0 },
      "values": [ …, null, … ] }            // null = NOT MEASURED, never 0
  ] }
```

- **`space` is a tag from the closed set** (`src/nebulai/spaces.py`):
  `W_E.raw` · `W_E.centered` · `W_U.raw` · `resid.L<k>` · `mlp_out.L<k>` ·
  `sae.L<k>.<repo>` · `text-embed.<model>` · `persona-pca.<id>`. Two channels
  in different spaces are never plotted against each other (D2). `resid`/
  `mlp_out` take layer `-1` for the embedding output before block 0.
- `null` in `values` becomes `NaN` in the browser column and renders as
  "not measured" — it is never a position, a colour stop or a zero.
- A channel file whose `n_points` disagrees with the loaded map is dropped
  WHOLE. An index-shifted column mislabels every point past the shift.
- CLI: `nebulai channels <model>` writes the lens channels; `direction project`
  appends four more per direction (below) to the same file.

### `directions.json` (`backend/directions.py`)

A direction is one unit vector in one named space plus the provenance that
makes it readable and the statistics that make it checkable.

```jsonc
{ "meta": { "model": "gpt2", "revision": "<resolved sha>" },
  "directions": [
    { "id": "male-minus-female-names", "label": "…",
      "space": "W_E.centered", "method": "diff_of_means", "d": 768,
      "vector": [ … ],
      "source": { "kind": "computed", "protocol": "<frozen identity of the two sets>",
                  "contrast": { "cohens_d": …,            // IN SAMPLE — not evidence
                                "heldout_cohens_d": …,    // refit on half, scored on the other half
                                "null_cohens_d_mean": …, "null_cohens_d_p95": …,
                                "n_pos": …, "n_neg": …, "heldout_n_pos": …, "heldout_n_neg": … } },
      "projection": { "channel": "proj.<id>", "orth_channel": "proj.<id>.orth",
                      "stats": { "cohens_d": …, "overlap": …, "n": … } },
      "null": { "method": "random_unit", "seed": 0, "n": 32,
                "channel": "proj.<id>.null", "orth_channel": "proj.<id>.null.orth" } }
  ] }
```

Rules, all enforced by `renderable()` in Python and mirrored by `axisRefusal()`
in `viewer/src/data/directions.ts` — **same two checks, same order**, pinned by
`viewer/tests/unit/directions.test.ts`:

1. **R5 — no `null` block, not renderable.** Not "renderable without a ghost".
   The null cloud is the only thing that says an axis is an axis.
2. **D2 — the four projection channels must exist and carry the direction's own
   space tag.** A `resid.L8` direction projected onto a `W_E.centered` map
   yields 49,857 perfectly ordinary numbers that mean nothing, so it is refused
   with the reason printed rather than drawn.

`methods`: `diff_of_means` · `pca` · `sae_decoder` · `lora_rank1` · `probe` ·
`two_selection`. Two statistics, never merged: `source.contrast` is about the
direction's OWN two sets (and only its held-out half is a measurement);
`projection.stats` is real-vs-null across the whole map. They routinely
disagree — the shipped gpt2 direction separates its two clusters at held-out
d = +8.30 and is indistinguishable from random across the map (overlap 0.834).

CLI: `nebulai direction list | survey | add | make | prompts | project | drop`.
`survey` prints the published artifacts' widths against this model — the table
behind every import refusal. `prompts` fits a direction on a frozen prompt set
from `backend/prompt_sets.py` via `gpt2_numpy` residuals, for the case where
nothing published is the right width.

### `interp/intervene_<name>.json` (`backend/interp/intervene.py`)

The third sidecar family, and the only one whose numbers come from a model that
was **changed**. Everything else in this contract measures a model that was left
alone; `nebulai intervene <model> {clamp,add,ablate,cap}` installs an
inference-time hook, runs the same prompt twice, and exports the difference.

```jsonc
{ "kind": "intervention_sweep", "model": "gpt2", "verb": "clamp",
  "n_layer": 12, "d_model": 768,
  "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
  "prompts": [ "…" ], "max_tokens": 16,
  "rows": [ { "alpha": 0.0, "is_identity": true,
              "protocol": "<exactly what was done, at this strength>",
              "kl_bits_mean": 0.0, "kl_bits_max": 0.0,
              "identical_to_baseline": true,        // asserted PER ROW
              "runs": [ { "prompt": "…", "kl_bits": 0.0,
                          "identical": true,
                          "resid_norm_baseline": …, "resid_norm_intervened": …,
                          "baseline":    { "text": …, "tokens": […], "logprobs": […] },
                          "intervened":  { "text": …, "tokens": […], "logprobs": […] },
                          "targets": [ { "text": " the Golden Gate Bridge",
                                         "baseline_logprob": …,
                                         "intervened_logprob": … } ] } ] } ],
  "claim": "Under this protocol — … — the intervention changed the next-token "
           "distribution by up to N bits of KL. That is a statement about what "
           "this intervention did, not about what the direction is.",
  "notes": { "decoding": …, "control": …, "d6": … },
  "meta": { "revision": …, "digest": …, "hook_layer": 7,
            "sae_repo": …, "sae_hook": "blocks.8.hook_resid_pre",
            "sae_hook_layer": 8, "hook_layer_note": "…" } }
```

Five rules, four of them refusals:

1. **α = 0 is a control and the producer will not write the file without it.**
   At α = 0 NO hook is installed — not an identity hook, no hook at all — and
   the resulting logits must be bit-identical to the un-hooked baseline.
   `identical_to_baseline` asserts it per row; the CLI adds an α = 0 row if the
   caller omitted one, printing *"a sweep without its control is a line, not a
   result"*. This is the sharpest correctness test available for a hook harness
   and it is run every time, not only in CI.
2. **The layer arithmetic is written down, not inferred by the reader.** A
   `resid.L8` direction fires at hook layer 8. An SAE tagged
   `sae.L8.<repo>` hooks `blocks.8.hook_resid_pre`, which is the stream
   *entering* block 8 — the output of block **7** — so its `hook_layer` is 7 and
   `meta.hook_layer_note` says why. `sae.L0.*` → hook layer −1.
3. **`clamp` rewrites only the difference the clamp makes** —
   `x + α·(a_new − a_old)·W_dec[f]` — rather than re-encoding and decoding the
   stream. These SAEs reconstruct at ≈0.9 cosine, and decoding wholesale would
   apply the dictionary's reconstruction error as part of the "intervention".
4. **D6 — measure, never export.** No flag, endpoint or code path in this
   pipeline writes a modified checkpoint, and
   `tests/test_intervene.py::test_no_weight_export` fails if one is added.
5. **D3 / §2.4 — the bundle carries its own claim sentence.** It is generated
   from the measured numbers in the intervention's own terms and is the ONLY
   causal sentence this project permits. The viewer renders it verbatim; it
   never restates it in stronger words, and never turns it into a sentence about
   what the feature or direction *is*.

KL is `KL(baseline ‖ intervened)` in bits over all 50,257 logits at the final
position, computed in 64-bit. It says how far the distribution moved and nothing
about whether it moved where you wanted — that is what `targets` (teacher-forced
logprobs of fixed completions, before and after) is for, and it is exported even
when it undercuts the headline. The shipped gpt2 sweep is exactly that case: SAE
feature 17840 detects the Golden Gate Bridge cleanly and steers it not at all.

Written to `out/<model>/interp/` beside the other bundles the viewer fetches,
NOT to `out/<model>/`. Viewer: `loadIntervene()` in `data/interp.ts`,
`scene/interp/steer.ts` (all arithmetic, GPU-free and unit-tested),
`SteerDriver.ts` (the stage) and `chrome/SteerRail.tsx` (the text half).

## persona space (`out/persona/<space_id>/`) — the placement contract

A persona space is a fixed 2-D basis that arbitrary text can be projected into,
so a captured agent run can be drawn on the same axes twice. It is **not** a
`nebulai.json` map: it has no points of its own, and the viewer never treats it
as one. Two files carry the whole contract.

### `out/persona/<space_id>/space.json` — the basis and its control

```jsonc
{
  "meta": {
    "space_id": "smollm2-135m-instruct@12fd25f77366.v1.L19",
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",  // full sha, pinned
    "layer": 19,
    "pooling": "last_token_mean_over_prompt_set",
    "prompt_set": { "id": "personas.v1", "sha256": "...",
                    "n": 296, "n_probes": 8, "n_prompts": 2368 },
    "created": "2026-09-11T20:45:19Z",
    "space": "persona-pca.smollm2-135m-instruct@12fd25f77366.v1.L19"
  },
  "basis": {
    "components": [ /* n_components × d_model */ ],
    "mean": [ /* d_model */ ],
    "explained_variance_ratio": [ /* n_components */ ]
  },
  "archetypes": [ { "name": "therapist", "scores": [ /* n_components */ ] } ],
  "control": {
    "method": "label_permutation_within_probe",
    "n": 500, "seed": 0,
    "pc1_evr": 0.204613,
    "pc1_evr_null_mean": 0.172918,
    "pc1_evr_null_p95": 0.191639,
    "verdict": "above_null",          // above_null | at_null | below_null
    "p_value": 0.001996,
    "cross_check": { "method": "label_permutation_unstratified", "n": 500,
                     "pc1_evr_null_mean": ..., "pc1_evr_null_p95": ...,
                     "verdict": ..., "note": "Not the verdict. ..." }
  }
}
```

- **`control` is mandatory.** A `space.json` without it is not a space; a
  consumer refuses it rather than drawing an unvalidated basis. `verdict` is the
  gate: only `above_null` is **selectable by default**. `at_null` and
  `below_null` load only on an explicit opt-in and must be captioned with the
  verdict wherever they are drawn — a space whose PC1 does not beat its own
  permutation null is a picture of nothing, and silently offering it is the
  failure this field exists to prevent.
- `pc1_evr` and `pc1_evr_null_p95` **travel with every drawing of the space**,
  not just with the build log, so no view can show the geometry without the
  number that says whether it is real.
- `cross_check` is a second null under a weaker assumption, kept for context and
  explicitly **not the verdict** — its own `note` says so. Never read
  `cross_check.verdict` as the space's verdict; a reader that does will find
  `below_null` sitting next to a legitimately `above_null` space.
- `revision` is the full commit sha of the weights, and `prompt_set.sha256`
  pins the prompts. Two spaces are comparable only when both match: the basis is
  meaningless across a different checkout of the same model name.
- `archetypes[i].scores` is one row per prompt-set persona in the basis'
  component order — the reference cloud a placement is read against.

**What the control has actually said so far.** Two spaces have been built from
`personas.v1` (sha `35cb9d1dc380…`, 296 archetypes × 8 probes = 2,368 prompts),
both with a 500-draw within-probe label-permutation null at seed 0:

| space_id | model | layer | pc1_evr | null p95 | p | verdict |
|---|---|---|---|---|---|---|
| `smollm2-135m-instruct@12fd25f77366.v1.L19` | SmolLM2-135M-Instruct | 19 | 0.2046 | 0.1916 | 0.0020 | `above_null` |
| `smollm2-360m-instruct@a10cc1512eab.v1.L20` | SmolLM2-360M-Instruct | 20 | 0.1662 | 0.1537 | 0.0080 | `above_null` |

Both clear their null and **neither clears it by much** — the margin is about
0.0125 of explained variance at both sizes. `pc1_evr` falling as the model grows
is not the effect weakening: a wider residual stream spreads the archetype means
over more directions, and the only thing that decides whether a space is usable
is its own null, never a comparison of `pc1_evr` across models. The unstratified
`cross_check` came in `below_null` at both sizes, for the reason its note gives.

**`nebulai persona verify <space_id>` re-runs the control and refuses a mismatch.**
It checks three things before it measures anything: that `prompts/<set>.json` still
hashes to the `prompt_set.sha256` in the file, that the live model's id and resolved
revision are the `model` and `revision` the space records, and that `layer` exists in
that model. `--model` defaults to the 135M checkpoint for `build`'s sake, so `verify`
takes the model from the space itself when the flag is absent and prints that it did.
A verification that can be run against the wrong checkpoint reports on the wrong
model, and its "verdict reproduced" would mean nothing.

### `<store>/runs/<run_id>/placement.json` — as written by `seer place`

```jsonc
{
  "run_id": "...", "space_id": "...", "model": "...", "revision": "...",
  "layer": 19,
  "placement_source": "pinned_model",   // pinned_model | text_embedder
  "transport": "in_process",            // in_process | live_http
  "fidelity": "deterministic",
  "verdict": "above_null", "pc1_evr": 0.204613, "pc1_evr_null_p95": 0.191639,
  "created": "2026-09-11T20:45:19Z",
  "n_points": 41,
  "n_skipped": 6, "n_dropped_by_policy": 4, "n_missing": 2,
  "points": [
    { "index": 0, "seq": 0, "ts": 1757..., "event_id": "...", "turn_id": "...",
      "role": "assistant", "chars": 812, "coords": [1.2, -3.4] }
  ],
  "skipped": [
    { "index": 3, "event_id": "...", "role": "assistant",
      "fidelity": "dropped_by_policy",
      "reason": "812 chars recorded, text not retained (content_level='metadata')" }
  ]
}
```

- **`placement_source` is the only field the glyph is picked off.** Points
  placed by the pinned model (`pinned_model`) and points placed by a generic
  text embedder (`text_embedder`) are the same picture at different fidelities
  and must be visually distinguishable; deriving the glyph from anything else —
  `transport`, `fidelity`, the presence of coordinates — reintroduces exactly
  the confusion this field removes.
- **The four counts never fold into each other.** `n_points` counts what was
  placed. `n_skipped` counts what was not, and `n_dropped_by_policy` +
  `n_missing` partition it by *why*: `dropped_by_policy` means the text existed
  and was refused at ingress (the adapter counted `chars` and kept none of
  them), `missing` means there was nothing to capture. They are different facts
  about the run and a reader acts differently on each, so a view that shows a
  single "skipped" total must still expose the split.
- Every skipped turn keeps its `index`, so a placement is always reconcilable
  against the run's turn order; the `points[].index` and `skipped[].index`
  together enumerate the turns without gaps, while `seq` is the dense index into
  `points` only.
- `verdict` / `pc1_evr` / `pc1_evr_null_p95` are copied down from the space's
  `control` so a placement can be judged without loading a 150 KB `space.json`.
  A run placed into a space that failed its control still carries that verdict
  and must still be captioned with it.
- A run in which nothing was placeable is written out as an honest empty
  placement — `points: []`, `fidelity: "missing"`, `verdict: "unknown"`, and the
  full `skipped` list — rather than raising. The run exists and the reason every
  turn was unplaceable is worth writing where the viewer can show it.

## The rule for new pipelines

To add a front-end: write `frontends/<name>.py` exposing `load_*_units(...) ->
Units`, add a CLI subcommand, done. If you find yourself editing `backend/*` to
support a new front-end, stop — either the change belongs to *all* pipelines
(make it in the back-end and run the propagation check), or the front-end isn't
respecting the contract.
