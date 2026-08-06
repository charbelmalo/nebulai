# Observability surface — handover

A twelve-feature LLM-observability spec was measured against what the viewer
actually ships, alongside a reference design comp. This file records the audit,
the diagnosis, and the work split into three tiers.

As of 2026-08-06. Read §2 before §4: half the "missing" features are missing
*data*, not missing pixels, and the tiers only make sense once that split is
clear. §6 tracks what has actually landed.

> **Revised 2026-08-06 after review.** Four corrections to the original draft
> and one scope change. The corrections: the §2 headline tally disagreed with
> its own table; feature #4 was filed as "data already exists" in §2 while
> Tier-2 item 9 correctly said the opposite; the `three/webgpu` file list
> predated the seer merge it cites elsewhere; the worktree prune listed five
> dead checkouts where there are six. The scope change is §7 — features
> #7/#10/#11 came from a generic RAG-assistant observability spec and are
> rescoped rather than built as written.

---

## 1. Where the code is

| | |
|---|---|
| Premium stack | `viewer/src/scene/drivers/` + `scene/sessions/` + `scene/seer/` — 8 driver/field files importing `three/webgpu` (16 modules total, counting `scene/layers/`, `picking.ts`, `post/bloom.ts`, `InterpPage.tsx`) |
| Flat stack | `viewer/src/scene/interp/` — 23 deck.gl drivers + `registry.ts` (25 registry features; the other 2 drivers are WebGPU) |
| The bridge | `viewer/src/scene/interp/field2d.ts` — WebGPU emissive field under a deck chart |
| Chart aesthetic | `viewer/src/scene/interp/chart-theme.ts` |
| Bloom | `viewer/src/scene/post/bloom.ts` |
| Design tokens | `viewer/src/styles/tokens.ts` ⟷ `tokens.css` (guarded by `tokens-sync.test.ts`) |
| Session data | `viewer/src/chrome/sessionlog.ts` — the fold that already computes tool/error tallies |

---

## 2. The audit — 12 requested features vs shipped

**1 shipped · 7 partial · 4 absent.**

(The original draft summarised this as "2 shipped · 5 partial · 5 absent",
which its own table contradicts. Counted from the table below: #1 shipped;
#2–#6, #8, #9 partial; #7, #10, #11, #12 absent.)

| # | Feature | State | Evidence | Gap |
|---|---|---|---|---|
| 1 | Semantic Cloud Map | **shipped** | Atlas view, `three/webgpu`+TSL+bloom, 49,385 pts / 192 clusters, territories/labels/beams, 2D+3D | Points are tokens/SAE features/neurons, not prompts/responses/tool-calls/documents. Visual done; **data domain differs** |
| 2 | Session Agent Graph | partial | Sessions tab 3-D turn field (time × context × new-context), 6 tool categories, playback transport | A cloud, not a path. No node→edge trace with numbered steps + per-step timing |
| 3 | Nearest K Neighbors | partial | `knnNeighbors()` in `data/edges.ts`, 5 chips in `SearchPanel.tsx` | No ranked table, no scores surfaced, no metric selector, no chart |
| 4 | Context Utilization | partial | Y=cache-read, Z=cache-write; `CONTEXT PEAK`/`CACHE-WR`/`OUTPUT` tiles | No composition breakdown (system vs user vs output vs retrieved). **Totals exist; per-segment attribution does not** — this is reducer work, not view work (see Tier 2 item 9) |
| 5 | Attention Heatmap | partial | #7 Attention-Head Flow, #23 Attention-Rollout Waterfall — real post-softmax attn, 12×12 head picker, depth scrub | Flat 2-D grid. **Largest single style gap** |
| 6 | Token Flow Through Model | partial | #8 Residual-Stream Ribbon (log‖x‖₂ × 13 layers), #3 Logit-Lens Tunnel | No volumetric ribbon, no stage labels, no in/out token counts |
| 7 | Retrieval Effectiveness | **absent** | — | No RAG anywhere in the repo |
| 8 | Model Confidence Landscape | partial | #18 Probability Simplex, #25 Live Nebula entropy grid, #20 Tuned-Lens KL | No confidence-over-position landscape; no distribution+percentile form |
| 9 | Tool Usage Analytics | partial | `sessionlog.ts` computes `toolHistogram`, `toolTotal`, `errorCount`, per-turn `errors` — **data is all there** | Rendered as text chips only. No per-tool success/partial/fail chart |
| 10 | Hallucination Risk Map | **absent** | — | No grounding or faithfulness signal computed |
| 11 | Knowledge Coverage Map | **absent** | Atlas has density + 55.2% noise | Nothing expresses "this region is uncovered" |
| 12 | Response Quality Scorecard | **absent** | — | No eval dimensions anywhere |

**This table is the audit as found, and is deliberately not edited as work
lands** — it is the baseline everything below is measured against. For current
state read the §7 ledger, which records what each tier actually shipped and, as
often, what it refused and why. Rows already moved by §7: **#2** (Tier 2.10),
**#3** (Tier 1.4), **#4** (Tier 2.9, rescoped), **#5** and **#6** (Tiers 2.8 /
2.8b), **#9** (Tier 1.5).

The load-bearing distinction, in three bands rather than two:

- **View-only — the data is already computed: #3, #9.** `knnNeighbors()` and the
  `sessionlog.ts` fold both exist and are tested. These need pixels and nothing
  else.
- **Reducer-then-view — the totals exist, the decomposition does not: #4.**
  The original draft filed this with #3 and #9 as "data already there"; that
  contradicts its own Tier-2 item 9, and item 9 is the correct one. Cheap
  compared to the band below, but it is not free.
- **No source in the repo at all: #7, #10, #11, #12.** These are pipeline
  projects that happen to end in a chart. Sizing them as chart work is the
  mistake to avoid — and for three of the four, see §7 before starting: the
  right move is to rescope, not to build the spec as written.

---

## 3. Why the charts look generic

Not taste — **two rendering stacks, and only one got the treatment.**

Eight driver/field files import `three/webgpu` (AtlasDriver, ChordDriver,
CompareDriver, SessionFieldDriver, EmbeddingConstellationDriver,
NeuronFieldDriver, `field2d.ts`, and `scene/seer/LiveField.ts` — the last
arriving with the seer merge at `04f72ea`, which the original draft cites in
§4 but had not folded into this count). They carry TSL emissive materials and
a bloom pipeline, and they already match or beat the reference comp.

Twenty-three interp drivers are deck.gl on an orthographic camera.
`chart-theme.ts` codifies the flatness deliberately — `GRID_RGBA` at alpha
20/255, hard-edged diamond markers, and a header comment reading *"structure is
a whisper"*. That was a considered minimalist choice; it is what now reads as
templated.

| Reference trait | Today |
|---|---|
| 3-D perspective, receding grid floor, axis cage | Flat orthographic, 2-D ticks |
| Emissive bloom on every mark | Only the 7 `three/webgpu` drivers |
| Vertical gradient fills per mark | Flat single-color strokes |
| Per-chart card: title + subtitle + control top-right | Info panel floats over canvas |
| Stat footer strip (4–5 labeled metrics) | **Only** Sessions has one |
| Ranked table beside the chart | Absent |
| Depth cueing, contact shadow | Absent in deck drivers |

**The migration path is already proven.** `field2d.ts` exists precisely to put a
WebGPU emissive field under a deck.gl chart, and two of the 25 drivers
(EmbeddingConstellation, NeuronField) ship on it today. This is not speculative
architecture.

### The honesty constraint — read before porting

`registry.ts` makes explicit per-feature guarantees: *"no smoothing"*, *"no
layout synthesis"*, *"arc shape is layout, not data"*, *"one linear scale both
directions"*. Some reference traits break them.

Extruding the attention matrix so **height = attention** is honest and should be
done. **Interpolating those cells into a smooth surface** — which the comp does —
invents values that were never computed, and would falsify the note the feature
ships with. Same for decorative 3-D on small-n donuts and ribbons.

Rule for every port: **discrete extrusion, yes. Interpolated surface, no.** The
honesty notes stay on the canvas. This costs perhaps 10% of the visual impact
and is the whole reason the project's numbers can be trusted.

---

## 4. The work, in three tiers

### Tier 1 — quick wins

Low risk, no new data, no new stack. Highest ratio in the list.

1. **`<StatStrip>` component.** Extract the 4–5 tile footer that Sessions
   already has. `.sessions-stat-*` in `chrome.css` generalizes to `.stat-*`;
   `SessionsPage.tsx:721` has the `Stat` component to lift. Then drop it under
   every interp chart — most already compute the numbers for their header.
   *Single broadest visual change in the whole plan.*

2. **`<ChartCard>` wrapper.** Title, subtitle, control slot top-right, canvas,
   stat strip. Replaces the info panel floating over the canvas. Pure chrome —
   no driver touched.

3. **Gradient marks.** `RAMP` already lives in `tokens.ts`. Add `rampNode(t)`
   as a TSL helper and a deck.gl accessor so bars and areas carry the vertical
   gradient instead of a flat stroke.

4. **kNN panel (#3).** `knnNeighbors()` is written and tested. Needs a ranked
   table — rank, score, source — plus a metric selector. Data work: zero.

5. **Tool analytics (#9).** `sessionlog.ts` already folds `toolHistogram`,
   `toolTotal`, `errorCount` and per-turn `errors`. Needs a grouped bar keyed
   by outcome. Near-zero data work — but not literally zero, as the draft
   claimed in the same breath as describing the work: the tallies exist, and
   the success/partial/fail *classification* is a small addition to the fold.

6. **Prune stale worktrees.** **Six**, not five. The four `worktree-agent-*`
   branches from 2026-07-16, `claude/sad-wescoff-532cf5`, and
   `claude/session-seer-realtime-viz-5e026a` are all fully merged into `main`.
   Their checkouts under `.claude/worktrees/` are dead and hold **2.8 GB**.
   Verified clean; the only untracked path is a 0 B `viewer/out` stub.

### Tier 2 — medium complexity

New rendering primitive, then port. Sequence matters — build the base class
first or you will hand-roll it five times.

7. **`ChartStage` base class.** *The missing primitive.* Every interp driver
   currently constructs its own `Deck`. `ChartStage` gives, once: perspective
   orbit camera, receding grid floor, axis cage with tick labels, TSL emissive
   material, bloom pass, depth-dim, and the WebGL2 fallback ladder. Model it on
   `field2d.ts` — same teardown rules (never dispose a Sprite's geometry), same
   `NeutralToneMapping` requirement under additive blending.

   **Do not generalise it ahead of its first consumer.** Extract from
   `field2d.ts`, port #23 against it, and let that port drive the API — *then*
   do the rest. A base class designed for five hypothetical callers will be
   wrong in ways that cost more than the hand-rolling it was meant to prevent.

8. **Port the five highest-payoff charts.** #23 first and alone; the remaining
   four only once #23 has validated the `ChartStage` API.
   - #23 Attention-Rollout → 3-D **discrete** extruded bars (see §3 constraint)
   - #8 Residual Ribbon → volumetric ribbon with stage labels (#6)
   - #21 Weight Spectrum → glowing layered curves
   - #18 Probability Simplex → distribution landscape (#8)
   - #7 Attention Flow → stays 2-D, gains bloom + gradient

9. **Context utilization breakdown (#4).** Segmenting the window into
   system/user/output/retrieved needs a per-segment attribution the fold does
   not currently produce. Reducer work plus a stacked view — real, but bounded.

10. **Session agent graph (#2).** Turn the turn-cloud into a path: node→edge
    layout, numbered steps, per-step timing. **Start from
    `scene/seer/LiveDriver.ts`** — its `structure` y-mode ("what ran inside
    what") is most of the layout problem already solved, and re-deriving it
    would be waste. Merged into `main` at `04f72ea`.

### Tier 3 — complex integrations

These are **pipeline projects that end in a chart.** Do not size them as
visualization work.

11. **Retrieval effectiveness (#7)** — requires a retrieval system to measure.
    No RAG exists in the repo: no corpus ingestion, no chunker, no index, no
    query path. Prerequisite for #11 as specified, and the largest single item
    here.

12. **Knowledge coverage map (#11)** — needs a *coverage metric* over the
    embedding space, not just density. The Atlas shows where points are; this
    must show where they are **not**, which means a reference distribution to
    compare against. Depends on #11's corpus existing.

13. **Hallucination risk map (#10)** — needs a grounding signal per claim:
    claim segmentation, evidence linking, an entailment or consistency score.
    None of the three exist. This is the one where a plausible-looking chart
    over a weak signal would do real damage to the project's credibility — the
    honesty bar in §3 applies hardest here.

14. **Response quality scorecard (#12)** — needs an eval harness producing
    helpfulness / relevance / completeness / safety / coherence / faithfulness.
    Cheapest honest version is an LLM-judge panel with disclosed rubric,
    disclosed judge model, and per-dimension variance shown, not hidden behind
    a single number.

15. **Semantic cloud over the RAG domain (#1).** The Atlas *renders* this today
    and renders it well. Feeding it prompts, responses, tool calls and document
    chunks instead of tokens is an ingestion + embedding job against the
    existing Units contract — the renderer needs no change.

---

## 5. Stack decision

Per `~/.claude/3D-INTEGRATION.md`, stated explicitly rather than inherited:

| Tier | Choice | Why |
|---|---|---|
| GPU backend | WebGPU primary + WebGL2 fallback | Already what `AtlasDriver`/`field2d.ts` do; `forceWebGL` rung is proven |
| Engine | three r185 `three/webgpu` + TSL + node post-processing | One shader codebase lowers to WGSL and GLSL; bloom already node-based in `post/bloom.ts` |
| Compute | None required | Every quantity is precomputed offline in float64 and shipped as JSON |
| Technique | Discrete extrusion, additive sprite density, emissive bloom | Real computed values only — no interpolated surfaces (§3) |

**deck.gl is not removed.** It keeps `HierarchyDriver`'s radial dendrogram,
where its layout layers genuinely earn their weight, and any chart where a flat
orthographic reading is the honest one.

---

## 6. Tier 3, rescoped

Tier 3 sizes its work honestly but never asks whether the work should exist.
Features #7, #10, #11 and #12 come from a generic LLM-observability spec that
assumes a **RAG assistant product**. Nebulai is an interpretability atlas plus
a session-observability tool. Building corpus ingestion → chunker → index →
query path *so that a chart has something to measure* is backwards: that is a
product decision wearing a visualisation ticket's clothes.

Per feature:

- **#7 Retrieval effectiveness — escalate, do not build.** If the project wants
  RAG, that decision stands on its own merits and this chart falls out of it
  for free. If it does not, #7 is out of scope. Either way it leaves the viewer
  backlog. **Status: not a viewer task.**

- **#10 Hallucination risk — replace with a seer-native grounding signal.**
  Generic claim-segmentation plus entailment scoring does not exist here and
  would be weak if bolted on — and §3's honesty bar applies hardest to exactly
  this chart. But the session transcripts already carry ground truth. Do files
  the final answer cites actually exist? Did commands it claims to have run
  appear as tool calls? Do reported test counts match the result lines? That is
  a computable, disclosable grounding check over data already collected, and it
  fits this project instead of importing another product's shape.

- **#11 Knowledge coverage — reframe, and drop the false dependency.** Coverage
  needs a reference distribution, but that reference need not be a RAG corpus:
  run a held-out text corpus through the same model and compare atlas occupancy
  against its activation distribution. Real pipeline work, interp-native, and
  **not blocked on #7** — the original draft's "depends on #11's corpus" both
  mis-numbers itself (it means #7) and assumes the RAG framing.

- **#12 Response quality scorecard — the one Tier-3 item startable today.**
  Keep the draft's design exactly: LLM-judge panel, disclosed rubric, disclosed
  judge model, per-dimension variance shown rather than hidden behind a single
  number.

**#15 (semantic cloud over the RAG domain) inherits #7's fate.** The Atlas
renders it well today; what it lacks is a domain to render, and that domain
only exists if the RAG decision goes the other way.

---

## 7. Status

Implementation has started. This section is the ledger — update it, don't let
it drift.

| Item | State |
|---|---|
| Doc corrections (tally, #4 filing, file list, worktree count) | **done** — this revision |
| Tier 1.1 `<StatStrip>` | **done** — see below |
| Tier 1.2 `<ChartCard>` | **done** — see below |
| Tier 1.3 gradient marks | **rescoped** — `rampNode()` built; the gradient half is refused, see below |
| Tier 1.4 kNN ranked panel (#3) | **done, minus the metric selector** — see below |
| Tier 1.5 tool analytics (#9) | **done** — three buckets, not the spec's three; and it found a real over-count. See below |
| Tier 1.6 prune worktrees | **blocked** — needs a human; `git worktree remove` is denied to the agent as a destructive action. All six verified merged and clean, 2.8 GB |
| Tier 2.7 `ChartStage` 3-D primitive | **done** — extracted from `field2d.ts` against #23 as its first and only consumer, see below |
| Tier 2.8 port #23 attention rollout to `ChartStage` | **done** — the first chart on the surface that is actually dimensional, see below |
| Tier 2.8b port the remaining four charts | **one done, three refused** — #8 residual ribbon is on `ChartStage`; #21, #18 and #7 stay 2-D and the reasons are recorded below |
| Tier 2.9 context utilization (#4) | **done, rescoped** — the cache decomposition is real and drawn; the spec's semantic split is not derivable and is refused in writing. See below |
| Tier 2.10 session agent graph (#2) | **done** — per-agent path, numbered steps, per-step timing, plus a pace strip. One half is unexercised by any real transcript and says so. See below |
| Tier 3.1 grounding check (#10, as rescoped) | **refused, with numbers** — all three proposed checks saturate at a median of 100% on 239 real transcripts. See below |
| Tier 3 remainder (#7, #11, #12) | **not viewer work** — #7 escalated in §6; #11 and #12 are pipeline projects that end in a chart. §4's viewer scope is complete |

### Tier 1.1 — `<StatStrip>`, as built

`chrome/StatStrip.tsx` exports `Stat`, `StatStrip` and the `StatTile` type.
`.sessions-stat*` generalised to `.stat*`; SessionsPage now imports the shared
tile. The strip needed a **new channel on the driver contract** — the draft
assumed the numbers were already lying around in each chart's header, and they
are not — so `InterpDriver` grew an optional `stats(): StatTile[]`, read once by
the host after `setModel` resolves. Implemented on the five charts slated for
Tier 2. Bottom-docked legend cards clear the strip for free — see the
positioning note under Tier 1.2.

**Two honesty bugs surfaced while verifying against real gpt2 data, and both
are the exact failure §3 warns about — a plausible number that is not a
measurement:**

- Attention rollout's peak was taken over all drawn cells, including the
  diagonal. `R[0][0]` is `1.0000` for every prompt in every model, so the tile
  printed a structural constant dressed as a finding. Now reports the peak
  **off-diagonal** value (0.9994 on the bundled Eiffel Tower trace).
- Attention flow's focus at 2 dp rendered a measured `0.9993` as `"1.00"` —
  indistinguishable from a clamped or saturated value. Now 3 dp, and reports
  **both** ends of the range (0.147–0.999), because the spread across the 144
  heads is the informative part and the maximum alone misdescribes the grid.

`tests/unit/interp-stats.test.ts` locks both rules — including a case asserting
the naive diagonal-inclusive max really is constant across inputs, so the trap
cannot come back quietly. Suite 186 → 195, `tsc` clean.

The reference comp was deliberately **not** reproduced pixel-for-pixel — see
the §3 constraint on interpolated surfaces. Its "24.6M points", "94.2%
coverage" and smoothed attention surface are precisely the aesthetic-over-
evidence failure the registry notes exist to prevent.

### Tier 1.2 — `<ChartCard>`, as built

`chrome/ChartCard.tsx` frames every Internals chart: a header **in flow above**
the canvas (feature number, title, model tag, subtitle, controls), the plot, and
the stat strip below it. The download and legend-collapse buttons moved into the
header, and the floating card was cut down to what actually has to sit next to
the marks — swatches, units, the honesty note. Its heading is now literally
`key`.

The header is deliberately **not** collapsible. A chart that can hide which
quantity it is showing is a chart that can be screenshotted without it.

`InterpFeature` grew `subtitle?: string`, filled in for all 25 features. The
rule is the same one that governs `blurb`/`math`/`source`: the line must be
supported by that feature's own three fields and must carry any hedge they
carry. Four subtitles say so out loud — `attention-rollout` keeps
"head-averaged", `tuned-lens` keeps "approximation" (it is a least-squares fit,
**not** the KL-trained lens of Belrose et al.), `grokking-clock` keeps "toy
model", and `logit-attrib` says "direct-path" because the frozen-σ
linearization does not attribute everything. Where a caveat would not fit —
that `embedding-constellation`'s two PCs explain ~2.6% of variance, that
`cofire-venn` draws top-G² pairs rather than a census — it stays in the
on-canvas note, which is why the note stayed on the canvas.

**Layout, and the trap in it.** Children are wrapped in `.chart-card-body`,
which is the sole positioning context for the floating overlays. Its box *is*
the plot area, so a legend docked `bottom: 24px` clears the stat strip and a
spinner at `top: 50%` centres on the data without either knowing a chrome
height. This deleted the `--stat-strip-h` variable Tier 1.1 introduced, the
`.has-tracebar` class and both of its rules. The trap: `.chart-card` is
content-sized inside the flex stage, so the plot collapsed to **zero height**
until `.interp-chart` was given `flex: 1; min-height: 0`.

One real ordering bug came out of it. Publishing tiles *adds* the strip, which
shortens the canvas host the driver was sized against milliseconds earlier —
and deck.gl writes its size back onto the canvas as inline pixel styles, so the
plot stayed a strip-height too tall with its bottom edge behind the strip. The
host now re-measures on the frame after Preact commits the strip. Verified at
1280×820: canvas box exactly equals body box, below the header, above the
strip.

> Note for whoever verifies this next: the in-app preview pane delivers **no
> `ResizeObserver` callbacks and no `requestAnimationFrame`** when it is
> occluded — a hand-rolled RO on the same element fired zero times while the
> element genuinely resized. Both recovery paths above are invisible there. Call
> `window.__interpDriver.resize(w, h, dpr)` by hand to stand in for them.

### Tier 1.3 — gradient marks: half built, half refused

**The premise is wrong, and this is the fifth correction to the draft.** §3's
comparison table lists the deck stack as *"flat single-color strokes"*. It is
not. Counting per-datum colour accessors across the 23 deck drivers: **22 of 23
already map colour to a real quantity** — a diverging sign (`OcclusionDriver`
amber/blue), token or series identity (`ResidualRibbonDriver`), or magnitude
through the ramp itself (`AttentionRolloutDriver` log-normalizes attention and
calls `ramp(t)`; `GrokClockDriver`, `AblationDriver`, `LogitAttribDriver`,
`SAEPianoRollDriver` all do the same in their own units).

Colour is the **most heavily loaded channel in the whole deck stack**, not an
unused one. Laying a decorative vertical gradient over those marks would
overwrite a real encoding with a fixed pattern — the exact failure §3 exists to
prevent, and worse than the flat look it was meant to fix. So the gradient half
is refused as written.

For the record, there *is* an honest vertical gradient, and it is worth knowing
if a bar chart with a free colour channel ever appears: slice the bar in **value
space**, not pixel space, and colour each slice `rampRgb` at the value that
slice's own band represents. Then a pixel's colour is the ramp at the value its
height stands for, which is a statement about the data rather than a decoration
on top of it. Nothing in the current 25 needs it, so it is not built — writing
a helper with no caller is how the next person inherits dead code.

**What was real, and is done:** `scene/ramp.ts` exports `rampTexture()` and the
TSL `rampNode(t)`. `ChordDriver` and `BeamsLayer` were each hand-rolling the
same four lines — build a `DataTexture` from `rampTextureData()`, set
`needsUpdate`, sample at `vec2(t, 0.5)`, dispose — four chances to drift from
the canonical ramp. Both now call `rampNode(t)`; the texture is a deliberate
never-disposed singleton (1 KB, immutable, outlives any one driver, and
refcounting it would let one driver's `dispose()` blank another's colours).
The Tier-2 ports need exactly this helper.

Verified on a live WebGPU renderer rather than by inspection, since a malformed
TSL node fails at program-compile time and nowhere earlier: Atlas boots clean,
`renderer.info` reports **21 programs compiled, 17 draw calls, 100,038
triangles**, `beams` present.

### Tier 1.4 — kNN ranked panel, as built

The five neighbour chips in `SearchPanel.tsx` are now a ranked table: rank,
label, a distance bar, the stored score, and the distance. `chrome.css` gained
`.knn-*`; `.search-chip*` is gone. Data work was indeed zero — but the audit's
"Data work: zero" hid two things worth knowing.

**The score is not a similarity, and the panel used to imply it was.** The old
chip tooltip read *"similarity 0.842 (10-D cluster space)"*. `compute_point_knn`
in `backend/edges.py` ranks by exact **Euclidean** distance in 10-D `u_cluster`
and then stores `exp(-(d/sigma)^2)` — a unitless kernel value. Read as a cosine,
0.84 means something it does not mean. The column is headed `score` now, and the
provenance sits in the header tooltip.

**`sigma` was exported all along and never used.** It is one global constant (the
median neighbour distance over the whole export), which makes the kernel
invertible: `d = sigma * sqrt(-ln score)`, in `data/edges.ts` as `knnDistance`.
That recovers the quantity the search actually ranked on, comparable across every
point rather than only within one row. Ten tests in `tests/unit/knn-distance.test.ts`.

**No metric selector, deliberately.** There is nothing to select between: the
ranking happened once, offline, under one metric, and only ids and scores were
exported — the source vectors are not in the bundle. A selector would either do
nothing or re-sort a fixed list under a metric this build cannot compute. That
is a control that lies about what it changes, so it is not there.

**The bar draws distance, not score, and the data decided that.** Measured over
the bundled 49,385-point atlas: `sigma` = 0.0755, median top-1 score **0.67**,
p25 0.30, p05 0.005. The kernel saturates, so scores bunch at the top for close
neighbours and collapse for sparse ones — on a 0–1 score scale a sparse point's
entire row is empty track. Distance spreads properly. The track is a fixed
0-to-`knnDistanceFloor(sigma)` extent so a bar means the same thing after
clicking through to another point, and colour reads the same distance as length.

**9.5% of stored scores round to exactly 0.000** (`np.round(sim, 3)`), and for
those the distance is not large, it is *unrecoverable*. `knnDistance` returns
null rather than Infinity or a clamped guess; the row shows `>0.21`, and its bar
fills the track in flat `--hairline` because a lower bound should not look like a
measurement. 2.8% of points have even their best neighbour rounded away.

Verified against live data: `ment` → `ement` (0.684, d=0.047), `ements` (0.570),
`inations`, `utions`, `isations`, `izations`, bars spanning 22%→39% of the
track. The sparse case `ther` correctly saturates with four `>0.21` rows.

> If the export ever gains a decimal (`round(sim, 5)` would push the floor from
> ~2.8σ to ~3.6σ), this panel gets better for free. That is a one-character
> pipeline change, and it is the cheapest data improvement in this document.

### Tier 1.5 — tool analytics (#9), as built

`SessionsStats`'s five count-only tool chips are now per-tool outcome bars:
one row per tool, stacked, on a track shared across the card. `sessionlog.ts`
gained `toolOutcomes: ToolOutcome[]` and `unattributedErrors`.

**The spec asked for success / partial / fail. There is no partial.** A
`tool_result` either carries `is_error` or it does not — no exit code, no
severity, no third flag. Inventing "partial" means inventing a rule for which
successes are secretly half-failures, and every such rule is a guess dressed as
a measurement. What is real, and what "partial" was probably reaching for, is
**unresolved**: a `tool_use` that never received any `tool_result`, because the
call was interrupted or the log ends mid-flight. It gets its own colour rather
than being blank track, and it is deliberately *not* folded into `ok` — an
unanswered call is not a successful one, and folding it inflates the success
rate of exactly the sessions that went wrong. Two of 239 local transcripts have
unresolved calls; both had been reading as clean.

**Some failures cannot be charged to any tool.** The audit format's top-level
`tool_use_result.is_error` carries no `tool_use_id`. Those are counted in
`errorCount` and reported on their own line under the bars, because a split that
silently omits them under-reports failure — and the arithmetic
`sum(failed) + unattributedErrors == errorCount` is the invariant that keeps the
card honest with itself. Locked by test.

**Building it exposed a real over-count in numbers this page has always
shown.** A resumed session replays its earlier lines back into the same `.jsonl`
— identical `uuid`, identical `message.id`, identical `tool_use` id, a few
hundred lines later. The turn fold was already immune (replays merge into their
original accumulator by `message.id`); the tool tallies were not. Measured on a
real 36 MB transcript: **1392 tool_use blocks were 1290 actual calls, and 34
reported errors were 31 actual failures** — that session read 8% busier than it
was. `toolHistogram`, `toolTotal`, `filesTouched`, `errorCount`, `errorsByReq`
and each turn's tool list now dedupe by `tool_use` id. Verified across **all 239
local transcripts: zero disagreement** between `sum(o.total)` and `toolTotal`,
between `sum(failed) + unattributed` and `errorCount`, and between the row count
and the histogram length.

A call with no `id` at all cannot be answered and cannot be recognised as a
replay; it is counted once, as unresolved, which is precisely what it is.

**Two things the chart refuses to draw.** The tools past the six-row budget are
summed into a sentence, not a seventh bar: measured on a 23-tool session the
pooled remainder is 213 calls against the busiest tool's 40, so on a shared axis
it dominates the scale and squashes the rows the chart exists to show — while
inviting a comparison (a pool against an individual) that is a category error.
And bar length is a shared-scale count, never a per-row percentage; a row that
fills its track regardless of how many calls it represents compares nothing.

**One shipping bug, caught in the browser.** Analyses persist to IndexedDB as
raw records with no schema version, so a row written by the previous build
arrives missing `toolOutcomes` and `.length` on it blanks the whole Sessions
page. `normalizeSessionAnalysis` fills the gap at the load boundary — with
**empty, never a reconstruction**: the raw transcript is deliberately not kept,
so there is nothing to recompute from, and a plausible split derived from the
surviving histogram would be a fabricated measurement. Those cards say in words
that the record predates the split and offer the fix (load the `.jsonl` again).

**The sample fixture was wrong and the chart is what revealed it.** Its
`tool_use` blocks carried no `id` and its one `tool_result` no `tool_use_id`, so
the demo session rendered with 100% of its calls unresolved. Real transcripts
always carry both. The sample now does too, and includes a failing `npm test`
and one unanswered call — the two states a clean fixture would never exercise.

Verified live: sample reads TOOLS 6 / ERRORS 1 beside `ok 4 · failed 1 ·
unresolved 1`, bars at exactly 75/150 px for a 1-of-2 segment; a 423-call
23-tool stress session fits the stage with 365 px to spare. 217 tests pass,
`tsc` clean, production build succeeds.

### Tier 2.7 — `ChartStage`, as built

`scene/interp/chart-stage.ts` is the 3-D counterpart to `field2d.ts`: an
orbitable `three/webgpu` stage with a cage, a floor grid, a hover probe, HTML
overlay labels, and one instanced mesh of columns. It was extracted **against
#23 and only #23**, exactly as §4 required, and every decision in it traces to
something that port actually needed.

**It is a bar stage, not a heightfield.** That is the §3 constraint expressed as
an API: a stage that took a `Float32Array` grid and drew a surface would make
the interpolation between samples the easiest thing to build and the hardest to
notice. `setBars()` takes a count and per-column position/height/colour, so
there is no shape in the type system for "the value between two measurements".

Three things differ from `field2d` for reasons that are not taste:

- **Columns are opaque and depth-tested.** In a 2-D field additive blending is
  free honesty — nothing occludes anything. In 3-D, a translucent column lets
  you see a far column through a near one at an intensity that is the product of
  two unrelated weights, which reads as a third value that does not exist.
- **`NeutralToneMapping` is still mandatory** — emissive drive exceeds 1.0 here
  as it does everywhere else on this surface.
- **Picking is a real raycast against the instanced mesh**, so a column you
  cannot see is a column you cannot hover. A screen-space nearest-anchor pick
  would happily report the value of something buried behind the sink wall.

**The fit is measured, not derived.** The first version sized the camera off a
bounding sphere containing the cage, which put the chart at 39% of the canvas
width and 32% of its height. A sphere around a slab is 30–40% larger than the
slab's actual silhouette, and the cage *is* a slab. `applyFit()` now projects
the eight cage corners, takes the screen-space bounding box of the real
silhouette, and applies six ratio corrections; the same pose went to 68% width
and 93% height (distance 30.85 → 23.92). `recentre()` is deliberately pan-only
and runs separately, because orbiting must never rescale — a chart that grows
and shrinks as you turn it has no stable size to read against.

That rework exposed a sign error worth recording: moving the camera along its
own `up` axis pushes the *content* **down** the screen. `pan.x` takes the
correction's sign and `pan.y` takes its negation. With `+=` on both, the data
centred at y 1556–2230 — entirely off a 533 px canvas, with no error anywhere.

**A zero-height column is not a hidden column, it is a broken mesh.**
`InstancedMesh.raycast` inverts each instance matrix, so a zero scale is
singular and yields NaN — which silently kills picking for the *entire* mesh,
not just that instance. Heights clamp to `MIN_H = 1e-4`. `writeMatrices()` also
calls `computeBoundingSphere()` on every write, because a stale sphere culls the
raycast before it reaches a single instance.

**Bloom needs a threshold here in a way the 2-D fields never did.** `bloom()`
with a low threshold is fine when the bright pixels are a sparse minority. On
#23 at full depth the attention-sink wall is most of the frame, so the cream top
ramp stop blew the whole left half to flat white — the same failure the compare
field hit, arriving from the opposite direction. Fixed in the *data* encoding
and the look together: gold top stop instead of cream, `emissiveMax` 1.5 → 1.2,
`emissiveMin` 0.55 → 0.62, `threshold: 0.88`, and `radius` 0.4 → 0.25 because a
wide blur over that much bright area washed out the cage.

**The cage has to be visible or the empty half is ambiguous.** `frameColor`
started at `0x2b3040 @ 0.5` and was invisible against `#0e0f12`; it is now
`0x3d4560 @ 0.75`. This is load-bearing rather than decorative: the acausal half
of #23 can only read as *excluded* if you can see the lattice that is drawn over
nothing.

Labels are HTML, positioned by the driver writing the projected point into
inline `translate` while the anchoring offset lives in CSS `transform`
(`.interp-stage-lab` in `chrome.css`) — the split that the tooltip-clobber
memory exists to enforce. `project()` returns `null` behind the camera, and
labels that land off-canvas are **hidden, not clamped**: an axis label nudged
back inside the frame points at something other than what it names.

### Tier 2.8 — #23 attention rollout, as built

The first chart on this surface that is actually dimensional. Same data, same
transport, same stats — `deck.gl` `GridCellLayer` out, `ChartStage` in.

**Extrusion changes what the empty half means.** In the flat heatmap the
acausal upper triangle was cells with no colour, which is visually identical to
cells measuring zero. As columns it is bare floor inside a full T×T cage: the
lattice says the pairs exist, and their absence says they were excluded rather
than measured. `causalCells()` enumerates exactly `T(T+1)/2` pairs and nothing
else populates the grid.

**Height and colour are one quantity, not two.** Both read the same `logNorm`,
so the two channels cannot tell different stories about one weight — and because
that normalization is logarithmic, the vertical axis is a log axis and is ticked
in decades (1e-4 … 1). Linear height would leave everything except the attention
sink flat on the floor, which is precisely the structure the view exists to
show. Verified numerically in the live driver: `glow * cageHeight === height`
exactly, for every column.

The decade ticks hang off `(+half, y, −half)` — the one cage post neither token
axis occupies. They were first placed at `(−half, +half)`, where they drew on
top of the sink wall; the anchor side (`is-h` / `is-hl`) is now chosen by
projecting the post against the projected lattice centre, so orbiting never
swings them inward over the data.

**The math is now testable, and is tested.** `scene/interp/rollout.ts` holds
`computeRollouts` / `causalCells` / `logNorm` / `decadeAt` with no renderer
attached, because the recursion *is* the view's claim and a GPU-bound driver
cannot be exercised by a unit test. `tests/unit/rollout.test.ts` (15 cases)
asserts every property the paper guarantees — one matrix per layer, every row a
distribution to 12 dp at every depth, strict causality as exact zeros, `R_0` is
`Ã_0` itself against a hand-computed 2×2, `R[0][0] = 1` at every depth (the
structural constant the stats tile excludes, now recorded as a reason rather
than a habit), and a monotonically non-increasing diagonal. Plus the encoding
rules: decades land on even steps, `logNorm` and `decadeAt` agree at every
decade, and **NaN reads as 0, never as tall** — a NaN height is the exact input
that kills picking mesh-wide. The rollout product runs in float64 throughout;
in float32 the row sums visibly drift off 1 by the top of a 12-layer stack.

Verified live on the bundled gpt2 Eiffel-Tower trace (11 tokens, 12 layers,
66 columns):

- **picking** — 45 columns return themselves exactly, 21 correctly return a
  nearer occluding column, 0 misses, 0 wrong. Occlusion is reported, not faked.
- **hover** — cell (8,4) shows the probe and reads `dst "␣the" ← src "␣Tower"
  rollout 0.0014 through L11 pos 4→8`.
- **click isolation** — row 8: 9 columns lit, 57 dimmed.
- **the waterfall is real** — mean diagonal height falls 3.847 → 1.099 across
  depth while the sink wall rises 3.124 → 3.998, and 65 of 66 columns change on
  every depth step.
- **orbit is stable** — camera distance unchanged at 23.92 through a full drag,
  silhouette stays pinned on the drawable centre, and a drag is correctly not
  treated as a click.
- **labels** — 30 of 33 placed at the default pose; the three hidden ones are
  genuinely off-canvas.

Suite 217 → 232, `tsc` clean, production build succeeds.

The API is now validated by one real consumer and is ready for the remaining
ports (#8 residual ribbon, #21 weight spectrum, #18 probability simplex). #7
attention flow stays 2-D: it is a 12×12 head grid with no third quantity to
extrude, and giving it one would be decoration.

*(Outcome of that list: only #8 was ported. See Tier 2.8b.)*

### Tier 2.8b — one port, three refusals

The candidate list was re-read on the merits rather than worked through. A chart
earns `ChartStage` when it has a **third real quantity over a two-index grid** —
not when it would look better with one.

#### #8 Residual-Stream Ribbon — ported

`resid_norm[layer][token]` already *is* a (layer × token) grid of scalars, so the
extrusion adds no dimension the bundle did not have. The flat version drew eleven
polylines that crossed constantly and needed hover-focus plus a de-collision pass
on the labels to be readable at all; on the lattice each token owns a row and the
collisions are gone by construction. `.interp-rs-tok`, `.interp-rs-xtick` and
`.interp-rs-ycap` went with it — `.interp-rs-labels` survives only as the overlay
host.

**Encoding, inherited from #23 and stated in the key:**

- height is `log₁₀‖x‖₂` against an **absolute** base at ‖x‖₂ = 1, ticked in
  decades. Not the data minimum — a run whose norms happen to start high must
  look like it starts high, and a floor that moves with the data destroys
  comparison between prompts.
- hue is **the token's position in the sequence and nothing else**. Height and
  glow both carry the norm; a third channel carrying it too would add no
  information, and a third channel carrying something *else* would let two
  channels tell different stories about one column.
- discrete columns, no interpolated surface. There is no measurement between
  block 4 and block 5.

**Two traps, both paid for once:**

1. **The chart rendered nothing** — labels and stats correct, canvas black. The
   driver's `resize()` called `stage.resize()` and `positionLabels()` but not
   `stage.render()`. With `animated === false` the host gives the driver no rAF,
   and the first resize arrives *after* `setModel`, so the stage stayed empty
   forever. **Any `animated: false` driver must render on every input that
   changes pixels, and `resize` is one of them.**
2. **It looked like washed-out pastel.** Measured rather than guessed: norms run
   4.66 … 3112 on a 1 … 10000 axis, giving glow 0.167 … 0.873 with an
   interquartile band of only 0.448 … 0.548. At `emissiveMin: 0.5` the emissive
   spanned 0.61 … 1.07 — the whole chart inside a third of a stop. Now 0.22 …
   1.35 with the bloom threshold at 0.86. The narrow band is itself the finding
   and is recorded in the `LOOK` comment: residual norms are *not* spread out.

**`heightPost()` — the height axis had to stop being hand-placed.** #23 hard-coded
the one corner neither data axis occupies. That is right for a square lattice
where the free corner sits over the acausal half, which by construction has
nothing standing on it. On #8's 13×11 grid the same corner is the top of the
tallest column, and walking it outward moves it almost straight *up* the screen,
so it never escapes.

A bounding-box clearance test cannot tell those two apart — both posts project
inside the silhouette rectangle, because one tall column at the far edge stretches
that rectangle over ground it does not occupy. (Trying it regressed #23: the box's
top edge comes from the distant sink wall, so #23 fled to the front-right corner
and landed under the key card.) Clearance is now the **minimum screen distance
from the post's tick band to any column top**, which is the question actually
being asked and answers it for any lattice at any orbit. Column tops are
projected once per call, not once per candidate per step — this runs inside a
drag.

Two further corrections the first version needed:

- **Both candidates walk outward together, one step at a time; nearest that
  clears wins.** Walking one corner to exhaustion first let it succeed by sheer
  distance — #8 parked its axis 6.27 units past a 6.5-unit half-width, where the
  ticks float in open space with no post under them and measure nothing. The
  measured trace: back-right clears only at step 4 (gap 33 px) while front-right
  clears at step 0 (gap 38 px).
- **The off-canvas test checks both ends on both axes.** Checking x alone let a
  candidate win on gap while its top decade sat above the canvas, where the
  hide-not-clamp rule drops it — an axis clear of every column and missing its
  largest tick.

`logSpan`/`decadeOn` were extracted to `scene/interp/logscale.ts` because the
second chart needed a different span; `rollout.ts`'s `logNorm`/`decadeAt` now
delegate and their 15 tests pass unchanged. 8 new tests cover decade spacing,
tick agreement, clamping, `!(v > 0)` for zero/negative/**NaN** (a NaN height is
singular under the matrix inversion `InstancedMesh.raycast` performs — one of
them kills picking for the whole mesh), monotonicity, collapsed and inverted
spans, unclamped out-of-span ticks, and the delegation itself.

**Verified live, not asserted:**

- 143 columns = 11 tokens × 13 stages, matching the stat strip.
- **picking** — probing every column top: 143 self-hits, 0 occluded, 0 misses.
  (At mid-height 115 are legitimately occluded by nearer columns — a dense
  lattice at an angle, reported rather than faked.)
- **one normalization, two channels** — worst relative disagreement between
  `glow × cage` and `height` is 8.2e-8, under one Float32 ulp (1.19e-7). They
  are the same number stored twice, not two numbers that agree today.
- **click isolation** lights exactly 13 of 143 — one token's trajectory — dims
  the other 130, marks the `The` label `is-sel`, and toggles back to −1.
- **the tooltip names what it shows**: `"The" · after block 11 ‖x‖₂ 3112.42
  embed 10.1 → final 379.2 (×37.4) · peak 3112 @ L11`. Token 0's norms by layer
  are `[10.1, 132.8, 636.2, 2563.8, 2748.2, 2897.1, 2990.5, 3046.5, 3080.1,
  3100.9, 3111.4, 3112.4, 379.2]` — the final-block collapse is real, and the
  tooltip reports both the peak and the final rather than smoothing over it.
- **orbit is stable** — distance unchanged at 27.64 through a full drag, with
  `selTok` preserved and `side` correctly re-measuring at the new pose.
- **the axis fits** — all five decades and the `‖x‖₂ ↑` cap visible, ending at
  x 1016 against a legend at x 1024.

Suite 232 → 240, `tsc` clean, production build succeeds.

#### #21 Weight Spectrum — stays 2-D

Singular-value decay curves. They are **variable-length and have no common index
axis across matrix kinds** — an attention head's spectrum and an MLP's do not
share a meaningful *i*, so a (kind × i) lattice would place columns at
coordinates that mean nothing. And a single 768-wide row of columns is hairline
mush at any readable stage width; the useful shape of a spectrum is its knee,
which a line renders and an extrusion hides behind its own neighbours.

#### #18 Probability Simplex — stays 2-D

A 2-simplex is a two-dimensional object. Barycentric coordinates are three
numbers with one constraint, so there is no third axis to give it — extruding one
would mean inventing a quantity the chart does not measure.

#### #7 Attention Flow — stays 2-D

A 12×12 head grid with **no third quantity to extrude**. Height would have to
carry either the value already in the colour or nothing at all; the first is a
redundant channel and the second is decoration.

### Tier 2.9 — context utilization (#4), as built

`buildComposition()` in `sessionlog.ts`, `ContextChart` in `SessionsPage.tsx`,
12 tests in `tests/unit/session-context.test.ts`.

**What the spec asked for and why it is not available.** Segment the window into
system / user / output / retrieved. A Claude Code transcript meters tokens **per
request, not per content block**, so three of those four cannot be separated
from it, and the fourth does not exist anywhere in this repo (no RAG — the same
absence that makes #7 Retrieval Effectiveness a pipeline project). The audit's
Tier-2 item 9 called this "reducer work"; that was right that a reducer was
needed and wrong that the reducer could produce the requested split.

**What is real, and is drawn.** Every prompt decomposes exactly into three
separately-metered parts — `cache_read` (reused), `cache_creation` (newly
cached), `input_tokens` (sent uncached). That is a true partition of a measured
number, and it is the column stack: one column per main-agent turn, height =
that turn's whole window, on a scale shared across the session so growth is
visible. Colour is ordered by what a token costs: muted for cache merely
re-read, accent for cache written, hot for uncached.

**The trap, and why only real data exposed it.** The obvious second step is to
subtract the model's last output from the window's growth and call the remainder
"what the tools contributed". It is wrong twice over, and on the synthetic
fixture in the repo it merely looked odd. Run against a real 731-turn transcript
it sums to **−351,495 tokens**:

1. **Compactions dominate the signed sum.** Five of them, each dropping ~150k in
   one step.
2. **`output_tokens` is not the model's contribution to the next window.**
   Thinking is billed as output and then never re-fed. On that same transcript,
   8 non-compaction steps had the window grow by *less* than the model had just
   written.

So the field is named `residual`, not `fromOutside`, and its doc comment names
both effects it mixes. The view reports **growth and model output as two
measured totals side by side and explicitly declines to subtract them**, with
compaction steps held out of the totals (and named in a sentence) but kept in
the columns, which show the window as it really was. A test asserts no field on
a slice is named after a source the transcript cannot meter.

**Three-state honesty, twice.**

- `growth` / `priorOutput` / `residual` are `null` on the first turn — there is
  nothing to difference against, and 0 would read as "the window did not grow".
- `exact` is `boolean | null`. It is false when some response reported different
  prompt usage on different lines, so the per-field max mixes them and the three
  parts sum to an upper bound rather than to any one request's prompt. It is
  **null** for a composition rebuilt from a persisted analysis: per-turn token
  counts survive persistence so the decomposition is as real as ever, but the
  consistency check was never stored, and `false` would assert an inexactness we
  have no evidence for. This is the one place `normalizeSessionAnalysis` is
  allowed to fill a gap by *computing* rather than emptying — it runs the same
  pure function over the same real numbers.

`usageDisagrees()` deliberately ignores `output_tokens`: it varies line to line
by design (first chunk, then total) and treating that as disagreement would mark
every audit session inexact for a reason that isn't one.

**Two bugs the live check caught, neither in the arithmetic.**

1. **The tooltips were dead on arrival.** `.sessions-stats` is a click-through
   HUD (`pointer-events: none`), which the cards inherited, so a native `title`
   could never be hovered. `.sessions-stat-card` now opts back in; the empty
   track stays click-through so the scene behind still orbits.
2. **The taller cards clipped the newest session off the top.** The panel was
   `bottom`-anchored with no top bound, so it grew upward past the stage. Now
   bounded top and bottom with `overflow-y: auto`, and `margin-top: auto` on the
   first child keeps the stack hugging the bottom while it fits — the
   `justify-content: flex-end` + overflow combination would have made the first
   card unreachable.

**Verified live** by feeding a real 50-turn usage series (taken from an actual
transcript, spanning a compaction) through the real file-input parse path:

- 50 columns, exactly one marked compacted, at the right turn.
- The compaction reads as **both** a height cliff and a colour change — 215k →
  69k, of which 32k is freshly written cache, because the window was rebuilt.
  Tooltip: `turn 26 — 69k in context / 37k reused · 32k new · 2 fresh / −146k vs
  the turn before; the model had just written 2.2k`.
- Column heights track the ratio: 44 px at the 215k peak, 14.1 px at 69k.
- Growth line: *"Over 48 growing steps the window gained 52k while the model
  wrote 30k"* — 48 = 49 differenced steps minus the one compaction.
- No exactness caveat shown, correctly: transcript-format usage was consistent.
- The legend says **"at turn 50 of 50"**, because three numbers under a bare
  "reused / new / fresh" read as session totals, which they are not.

Suite 240 → 253, `tsc` clean, production build succeeds.

### Tier 2.10 — session agent graph (#2), as built

The audit row read *"A cloud, not a path. No node→edge trace with numbered
steps + per-step timing."* One third of that was already wrong: the field has
drawn a trail since the driver was written, and `showTrails` defaults on. So the
work here was the other two thirds — **numbering** and **timing** — plus one
defect the numbering exposed.

**The defect.** `SessionFieldDriver` built the trail with a single `prev`
walking `a.turns` in transcript order. A sub-agent turn interrupting the parent
therefore produced two edges — parent→sub on the way in, sub→parent on the way
out — and neither is a step either agent took. It is the same category error as
the `growth − priorOutput` attribution in Tier 2.9: an arithmetic relationship
between two adjacent records presented as a causal one.

**But measure before you weight it.** A sweep of every transcript in
`~/.claude/projects/-Users-charbelmalo-Developer-nebulai` — **239 sessions,
10,444 turns** — found **0 turns with `isSidechain` true and 0 with a non-null
`parent_tool_use_id`**. The defect is real in code and fires on nothing that has
ever been recorded here. That measurement is why the per-agent split is treated
as cheap insurance and the effort went to the timing instead.

`buildAgentGraph(turns)` in `chrome/sessionlog.ts` groups turns by `agentId`
into an `AgentPath` each, and gives every turn an `AgentStep`: `step` / `ofSteps`
within its own agent, and `gapSec` since that agent's previous step.

- **Numbered within the agent, not the session.** A sub-agent's third step is
  its third step whatever surrounded it.
- **A parent's path closes over the sub-agent that interrupted it.** The edge
  spanning the interruption is real — the parent was inside that one tool call
  the whole time — so its gap covers the interruption rather than being reset
  by it. Verified: main steps `[0, 1, 4]`, gap on step 3 = 8s across a
  sub-agent, not 6s.
- **`gapSec` reads `tMs`, never `tSec`.** `tSec` falls back to 0 on a line with
  no clock; differencing against that manufactures a gap the size of the session
  so far. Null on an agent's first step and on any pair missing a stamp.
- **A gap is elapsed time between two responses, not a step's duration.** It
  contains tool execution, model latency and — after a human prompt — the human.
  The UI says exactly that; nothing calls it "duration".

**The pace strip** (`PaceChart`, in the session stat card) is the part that
makes per-step timing readable. A tooltip alone does not answer "which steps
were slow" across 700 nodes. One cell per measured gap, in step order, tallest
cell in the accent, and a sentence naming how few gaps hold half the time.

**That strip is where the real data changed the design.** Sized on the repo's
50-turn fixture it looked fine — but the fixture's gaps are a uniform 60s, so
every cell was 100% and the chart proved nothing. Run over the 17 real sessions
with ≥100 steps:

- **Linear is unusable**: 1252 of 1254 cells in the bottom decile.
- **asinh at the existing `suggestK` default is no better**, just at a different
  height: `suggestK` pins the median at **0.45** of the axis, so p10→p90 spanned
  13.8 → 51.2 points around a wall at half height, 979 of 1254 cells in one
  decile, 7.41 of 10 deciles used.

The 0.45 target is correct for what it was written for — a 3-D axis wants the
cloud to fill the cube. A **bar strip wants the opposite**: the body low, so the
few tall bars read. `suggestK` gained a fourth parameter, `target`, defaulting
to 0.45 so every existing caller is untouched; the strip passes 0.2, chosen by
sweeping 0.45 / 0.3 / 0.2 / 0.15 / 0.12 / 0.1 / 0.08 across those 17 sessions
and taking the best decile occupancy (**8.94 of 10** vs 7.41). Measured after:
p10 13.8 → p50 20 → p90 29 → max 100, all ten deciles occupied, and the inverse
still exact to 1e-15. Only the spacing moved; every reading is still true.

**Verified live**, both halves:

- On real data (50-turn slice, one agent): tooltip gains `Since prev`, showing
  `1m` on turn 6 and **`—` on turn 1** — an unmeasurable gap is not a gap of
  zero. The per-agent step row is correctly *suppressed*, because with one agent
  it would repeat the `#N` already in the header.
- On a synthetic multi-agent transcript pushed through the **real file-drop
  path** (there being no real one to use): steps come out
  `main:1/4 main:2/4 sub:1/3 sub:2/3 sub:3/3 main:3/4 main:4/4`; the parent's
  step-3 gap is 895s, spanning the sub-agent; the sub-agent's first gap is null.
  Trail segment count **54 where the old single walk gives 55** — two spurious
  cross-agent edges removed, one real parent-continuity edge added.

**And it found a live bug in Tier 2.9's chart.** The context legend computed its
scope as `turn ${last.turn + 1} of ${n}`, mixing a position among *all* turns
with a count of *drawn* (main-agent) ones. On the first session with a sub-agent
it read **"at turn 7 of 4"**. Fixed by moving the denominator to
`n + excludedSidechain`, the session's turn count — unchanged on the
single-agent sessions that are every real transcript here.

**What is not built, and why.** Spawn edges — linking a sub-agent's first step
to the parent turn that spawned it — need the `tool_use` **ids** the fold does
not record (`SessionTurn.tools` holds names only). That is a small change, but
there is no recording anywhere on this machine that would exercise or validate
it, and shipping an unverifiable path is the mistake this document keeps
warning about. The per-agent split above is worth it at the price of a `Map`;
a new persisted field for a feature no data can test is not.

Suite 253 → 269, `tsc` clean, production build succeeds.

### Tier 3.1 — the #10 replacement, measured and refused

§6 rescoped #10 away from generic hallucination detection and toward a
seer-native grounding check over data already collected: *"Do files the final
answer cites actually exist? Did commands it claims to have run appear as tool
calls? Do reported test counts match the result lines?"* That was the right
instinct about scope. It is still the wrong feature, and the way to find out was
to count before drawing. All three checks were run over the same 239 real
transcripts (10,444 turns):

| Check | Sessions with any claim | Claims | Backed | Per-session p10 / p50 / p90 |
|---|---|---|---|---|
| Files cited in prose vs touched by a tool | 31 | 228 distinct | **88.2 %** | 60 / **100** / 100 |
| Commands the prose says it ran vs Bash inputs | **0** | **0** | — | — |
| Numbers reported vs numbers in a tool result | 26 | 148 | **98.6 %** | 100 / **100** / 100 |

Every one saturates. The median session scores **100 %** on all three, and the
file check's apparent 12-point shortfall is not grounding at all — it is path
formatting, prose citing `viewer/src/x.ts` where the tool input held
`/Users/…/viewer/src/x.ts`. A strict-equality variant reads 64.5 %, which is
worse than useless: it measures a string convention and would be read as
dishonesty. The command check has **nothing to measure** — the phrasing "I ran
`…`" does not occur once in 10,444 turns, because in an agentic transcript the
tool call *is* the narration. And of 148 numeric claims exactly **2** went
unbacked, both the same figure.

**The saturation has one structural cause, and it predicts that no variant of
this check will work.** Grounding checks find signal where generation is
decoupled from evidence — a RAG answer is written from retrieved passages the
model may drift from. In an agentic loop the prose is written *after* the tool
results are already in context, paraphrasing what was just read. Claim and
evidence are coupled by construction, so the measurement has nowhere to vary.
That is why all three saturate at once rather than one of the three being a
better-chosen probe than the others.

A tile that reads "98 % grounded" on every session forever is not a weak
measurement; it is a constant with a decorative axis, and it would manufacture
exactly the confidence §3 says this chart must not manufacture. **#10 leaves
the viewer backlog.**

**Scope of the claim, stated so nobody over-reads it:** this measures Claude
Code transcripts of one project, which is precisely the data this viewer would
chart — that is what makes the refusal decisive *here*. It is not a finding
about model grounding in general, and nothing above should be quoted as one.

**Tier 3 after this.** #7 was already not a viewer task. #10 is now refused with
numbers. #11 (coverage against a held-out activation distribution) and #12
(judge panel) both remain real work, and both are **pipeline projects that end
in a chart** — a corpus run and a live judge endpoint respectively, neither of
which is viewer work and neither of which should be started from this document.
The viewer side of the audit is done.
