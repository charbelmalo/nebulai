# SessionSeer — the live view

The Seer page can already tell you what a run *did*. It cannot yet let you
watch one *happen*: `Trajectory` and `StateBar` are re-rendered summaries of a
finished reduction, and the only genuinely live surface is a scrolling text
tail. This document is the design for the part that watches.

Five directions were on the table — a streaming lane chart, a fleet strip, a
GPU particle field, a span-tree flamegraph, and a semantic trail through the
atlas. The decision is to build all five, because they are not five features.

## 1. The claim

**They are five projections of one event stream, and four of them share an
axis.**

| projection | x | y |
|---|---|---|
| Fleet | time | one row per run |
| Score | time | action lane (9) |
| Structure | time | span depth |
| Field | *inherits the active layout* | *inherits* |
| Atlas | semantic 1 | semantic 2 |

Three of them differ only in what `y` means. That makes Fleet → Score →
Structure a **y-morph along a fixed time axis**, not three widgets — the same
layout transport that already shipped for Compare, with the same vector-origin
tweens. Zoom out to see six agents, zoom in to see one agent's lanes, drill in
to see the span tree. One continuous gesture, one legend, one time cursor.

The Field is not a fourth layout. It is an **atmospheric layer that adopts
whichever geometry is live** — particles emitted at each event's position in
the current projection, bloomed. That is why it unifies instead of competing:
it makes any of the three layouts feel alive without inventing a coordinate
system that has to be learned separately.

The Atlas is the one genuinely foreign space: no time axis, semantic position.
It gets a cross-fade rather than a morph, and the time cursor degrades to a
trail parameter. It ships last and behind the live-embedding endpoint.

## 2. Layers

```
/seer/live (SSE)
      │
      ▼
  LiveModel          viewer/src/seer/live.ts     — pure, no DOM, tested
      │              open spans · closed spans · state timeline · token
      │              series · context pressure · span tree · thought stream
      ▼
  encoding.ts        the vocabulary, shared by every projection
      │
      ├─────────────┬──────────────┐
      ▼             ▼              ▼
  field canvas   stage canvas   rails (DOM)
  (WebGPU)       (2D)           thought · tail · quality · inspector
```

Two stacked canvases, not one: the measurement layer wants crisp 2D text and
hairlines, the field wants WebGPU and bloom. This is the established viewer law
— one driver per canvas, layered DOM, camera shared through the store — and it
is what lets the field degrade to nothing on a machine without WebGPU while the
chart above it stays exactly as legible.

`LiveModel` is the piece that makes "unified" true rather than aspirational.
Every projection reads it; none of them parse SSE. It is also where the
contract's rules are enforced once instead of five times.

## 3. The encoding vocabulary

One table, one module, every projection obeys it. A bar in the Score, a
particle in the Field and a node in the Structure view must agree about what
red means.

| channel | encodes | values |
|---|---|---|
| hue | `action` | the 9 `ACTION_COLOR` entries |
| length | duration | wall clock between our two events |
| thickness | tokens or bytes | rank-normalized, never raw |
| end cap | `effect` | new_information · state_changed · failed · … |
| texture | `fidelity` | solid · hatched · outline-only · policy-gray |
| row | `SessionState` | in Fleet, the 12-state ribbon |
| overlay hatch | `stalled` / `overdue` | modifies a row, never replaces it |

Drift is guarded the way `tokens.css` and `seer-contract-sync.test.ts` already
guard theirs: a test walks the contract enums and fails if any member has no
encoding. A new `Action` that renders in the fallback grey is exactly the
silent failure the log format was built to prevent.

## 4. What the render must not do

These are the contract's rules restated as pixels, because each one is easy to
lose in a render loop:

- **`missing` is an outline-only gap.** Not a zero-height bar, not a zero.
  A viewer that cannot distinguish "nothing happened" from "we never looked"
  contradicts the data-quality panel two cards down.
- **`dropped_by_policy` gets its own policy-grey**, distinct from `missing`.
  We chose not to look; that is a different fact from not knowing.
- **Deltas animate, they never count.** A `tool.output_delta` grows the open
  bar's leading edge and updates the preview. Only the fold event commits a
  length, a token figure, or a tally.
- **An open span's leading edge is not a measurement.** It renders with a
  live cap and reads "still open" — `duration_s` for a synthetic start is `0`
  by construction, and `0` is not a duration.
- **Glow is rank-normalized, with a saturation floor.** Bloom implies
  magnitude; raw magnitude through a bloom curve is a lie with a nice finish.

## 5. Thoughts

Reasoning text is `dropped_by_policy` unless the run was captured with
`--keep-reasoning`. The thought rail therefore has two honest states, and they
look different on purpose:

- **opted in** — live reasoning text, streamed, with the same delta rule as
  everything else: the preview updates, the token counter waits for the fold.
- **default** — thought *activity* without content: a pulse per reasoning
  stream, its duration, its token cost when the agent reports one. The rail
  says which mode it is in. It never renders an empty box that could be read
  as "the agent wasn't thinking".

## 6. Order of build

| | | depends on |
|---|---|---|
| L0 | `LiveModel` + `encoding.ts` + their tests. No pixels. | — |
| L1 | Score layout, 2D canvas, follow / scrub transport | L0 |
| L2 | Fleet y-mode + the morph between it and Score | L1 |
| L3 | Structure (span-depth flame) as the third y-mode | L2 |
| L4 | Field layer, WebGPU, reusing the SessionField patterns | L1 |
| L5 | Thought rail, both states | L0 |
| L6 | Atlas cross-fade | live embedder |

L0 is the whole bet. If the model and the vocabulary are right, each later
milestone is a layout function over state that already exists — and if they are
wrong, five views will disagree with each other in five different ways.
