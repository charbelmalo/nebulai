# Observability surface — handover

A twelve-feature LLM-observability spec was measured against what the viewer
actually ships, alongside a reference design comp. This file records the audit,
the diagnosis, and the work split into three tiers.

As of 2026-08-06. Nothing here is built yet — this is the plan, not a status
report. Read §2 before §4: half the "missing" features are missing *data*, not
missing pixels, and the tiers only make sense once that split is clear.

---

## 1. Where the code is

| | |
|---|---|
| Premium stack | `viewer/src/scene/drivers/` + `scene/sessions/` — 7 files importing `three/webgpu` |
| Flat stack | `viewer/src/scene/interp/` — 23 deck.gl drivers + `registry.ts` |
| The bridge | `viewer/src/scene/interp/field2d.ts` — WebGPU emissive field under a deck chart |
| Chart aesthetic | `viewer/src/scene/interp/chart-theme.ts` |
| Bloom | `viewer/src/scene/post/bloom.ts` |
| Design tokens | `viewer/src/styles/tokens.ts` ⟷ `tokens.css` (guarded by `tokens-sync.test.ts`) |
| Session data | `viewer/src/chrome/sessionlog.ts` — the fold that already computes tool/error tallies |

---

## 2. The audit — 12 requested features vs shipped

**2 shipped · 5 partial · 5 absent.**

| # | Feature | State | Evidence | Gap |
|---|---|---|---|---|
| 1 | Semantic Cloud Map | **shipped** | Atlas view, `three/webgpu`+TSL+bloom, 49,385 pts / 192 clusters, territories/labels/beams, 2D+3D | Points are tokens/SAE features/neurons, not prompts/responses/tool-calls/documents. Visual done; **data domain differs** |
| 2 | Session Agent Graph | partial | Sessions tab 3-D turn field (time × context × new-context), 6 tool categories, playback transport | A cloud, not a path. No node→edge trace with numbered steps + per-step timing |
| 3 | Nearest K Neighbors | partial | `knnNeighbors()` in `data/edges.ts`, 5 chips in `SearchPanel.tsx` | No ranked table, no scores surfaced, no metric selector, no chart |
| 4 | Context Utilization | partial | Y=cache-read, Z=cache-write; `CONTEXT PEAK`/`CACHE-WR`/`OUTPUT` tiles | No composition breakdown (system vs user vs output vs retrieved) |
| 5 | Attention Heatmap | partial | #7 Attention-Head Flow, #23 Attention-Rollout Waterfall — real post-softmax attn, 12×12 head picker, depth scrub | Flat 2-D grid. **Largest single style gap** |
| 6 | Token Flow Through Model | partial | #8 Residual-Stream Ribbon (log‖x‖₂ × 13 layers), #3 Logit-Lens Tunnel | No volumetric ribbon, no stage labels, no in/out token counts |
| 7 | Retrieval Effectiveness | **absent** | — | No RAG anywhere in the repo |
| 8 | Model Confidence Landscape | partial | #18 Probability Simplex, #25 Live Nebula entropy grid, #20 Tuned-Lens KL | No confidence-over-position landscape; no distribution+percentile form |
| 9 | Tool Usage Analytics | partial | `sessionlog.ts` computes `toolHistogram`, `toolTotal`, `errorCount`, per-turn `errors` — **data is all there** | Rendered as text chips only. No per-tool success/partial/fail chart |
| 10 | Hallucination Risk Map | **absent** | — | No grounding or faithfulness signal computed |
| 11 | Knowledge Coverage Map | **absent** | Atlas has density + 55.2% noise | Nothing expresses "this region is uncovered" |
| 12 | Response Quality Scorecard | **absent** | — | No eval dimensions anywhere |

The load-bearing distinction: **#3, #4, #9 already have their data** and need
only a view. **#7, #10, #11, #12 have no source in the repo at all** — they are
pipeline work that happens to end in a chart, and sizing them as chart work is
the mistake to avoid.

---

## 3. Why the charts look generic

Not taste — **two rendering stacks, and only one got the treatment.**

Seven files import `three/webgpu` (AtlasDriver, ChordDriver, CompareDriver,
SessionFieldDriver, EmbeddingConstellationDriver, NeuronFieldDriver,
`field2d.ts`). They carry TSL emissive materials and a bloom pipeline, and they
already match or beat the reference comp.

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
   `errorCount` and per-turn `errors`. Needs a grouped bar keyed by outcome.
   Data work: zero — the tallies exist, only the success/partial/fail
   *classification* needs adding to the fold.

6. **Prune stale worktrees.** Four `worktree-agent-*` branches from 2026-07-16
   and `claude/sad-wescoff-532cf5` are fully merged into `main`. Their
   worktrees under `.claude/worktrees/` are dead checkouts.

### Tier 2 — medium complexity

New rendering primitive, then port. Sequence matters — build the base class
first or you will hand-roll it five times.

7. **`ChartStage` base class.** *The missing primitive.* Every interp driver
   currently constructs its own `Deck`. `ChartStage` gives, once: perspective
   orbit camera, receding grid floor, axis cage with tick labels, TSL emissive
   material, bloom pass, depth-dim, and the WebGL2 fallback ladder. Model it on
   `field2d.ts` — same teardown rules (never dispose a Sprite's geometry), same
   `NeutralToneMapping` requirement under additive blending.

8. **Port the five highest-payoff charts**, in this order:
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

## 6. What was not done

- No implementation from §4 has started.
- The reference comp itself was not reproduced pixel-for-pixel, by design —
  see the §3 constraint on interpolated surfaces.
