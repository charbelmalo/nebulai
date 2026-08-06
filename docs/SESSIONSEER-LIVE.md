# SessionSeer — the live view

The Seer page can already tell you what a run *did*. It cannot yet let you
watch one *happen*: `Trajectory` and `StateBar` are re-rendered summaries of a
finished reduction, and the only genuinely live surface is a scrolling text
tail. This document is the design for the part that watches. To run it, see
[`SESSIONSEER-LIVE-SETUP.md`](SESSIONSEER-LIVE-SETUP.md).

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
trail parameter. It was to ship last, behind the live-embedding endpoint — and
it did not ship at all. The atlas keeps its coordinates and discards the model
that produced them, so there is no projection to place a live event with. See
§7, which is the more useful half of this document's last chapter.

## 2. Layers

```
/seer/live (SSE)
      │
      ▼
  LiveModel          viewer/src/seer/live.ts     — pure, no DOM, tested
      │              the leading edge only: open spans · marks in the window ·
      │              delta previews. Never a figure. See §2.1.
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

### 2.1 The leading edge is not a fold

`SeerPage` already refuses to fold: *"The Python reducer owns every derived
number… a second implementation of the fold in TS would drift from the first,
and the drift would be invisible."* The live view does not get an exemption from
that, and the first draft of this document quietly asked for one — it had
`LiveModel` holding closed spans, a state timeline and token series, every one
of which is a figure Python already owns.

`reducer._state` makes the point sharper than the docstring does. The state
machine is driven by a `_TRANSITIONS` table and deliberately will not read
`payload["state"]`, because "an agent's words deciding our states is the one
thing the contract forbids". A TS reimplementation would have to copy that
table, and the copy would rot.

So the division is:

- **Python owns the record** — every total, every closed span, the current
  state, time in state. It arrives through the 200ms coalesced refetch that
  `markDirty` already performs.
- **`LiveModel` owns the leading edge** — how far the open spans and the current
  state have grown *since* that snapshot, the marks inside the visible time
  window, and the delta preview text. Positions, not figures.

A mark is a direct projection of one event (its `ts`, `action`, `effect`,
`fidelity`), never an accumulation, which is why it stays on the safe side of
the line. The model derives no number that anyone reads, so it has nothing to
drift *from* — and `figures()` hands back the adopted `RunView` untouched, which
a test asserts by replaying a thousand events at it and diffing.

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
  The live field (L4) resolves this the other way: it has no magnitude to rank,
  so it spends the channel on *recency* and lets crowding do the rest — see
  §4.2.
- **Depth is the reported chain, or it is nothing.** See §4.1.

### 4.1 The tree we are not allowed to draw

Structure mode (L3) puts containment on the y-axis, and containment is the one
thing in this subsystem where the tempting inference is also the wrong one.

`parent_span_id` reaches the viewer on every span — Python keeps it so
`time_decomposition` can subtract a child's seconds from its parent instead of
counting them twice. Measured against every run captured so far, it is **null
everywhere**, and no two spans' clocks overlap either. Both facts are about the
adapters, not about the agents: `claude.py` sets a parent only when a message's
own parent maps to a tool span it already opened, which in practice is a
subagent, and nothing else reports nesting at all.

So a flamegraph over this data has nothing to stack, and there are two
plausible-looking ways to manufacture something — infer a parent from one call
falling inside another's interval, or read it off nesting-shaped tool names.
Both would produce a picture with real depth in it, and both would be the
viewer asserting a relationship the record does not contain. The mode does
neither. A span whose parent we never received sits one level inside its run,
the note on the card says so in as many words, and the day an adapter reports a
chain the rows appear on their own.

What the mode draws instead is the part that *is* measured. Each run gets its
own band — its wall interval — with the intervals any call covered painted over
it, so the pale remainder is `outside_spans_s`: the model thinking and the human
reading, seconds no span accounts for. Rows below the band hold the calls, one
row per reported depth and one more per genuine overlap, because two clocks
overlapping is a measurement and deserves its own row without being called
nesting.

### 4.2 What the field is allowed to glow about

The field (L4) is a second canvas *behind* the chart: additive motes on the
exact rows the chart draws its bars on, so a busy stretch of a run glows.

An event has no magnitude. The stream says that a thing happened, what kind it
was and when — never how big. So no term in that shader reads a value:
brightness is **recency**, and everything else bright on screen is **crowding**,
which additive blending gives for free and which is a real property of the
stream. A glowing field that meant nothing would be the most persuasive lie in
the subsystem.

Two consequences that are easy to get wrong:

- **A mote per mark is the wrong rule.** `LiveModel` holds only what it ingested
  over SSE, so a run that finished before the page loaded has no marks and would
  sit in total darkness beside a live one — which reads as "this run did
  nothing". Motes come from *instants the record contains*: a closed span lights
  its start and its end, an open span lights its start and the live edge, and a
  mark lights its own moment only when it has no span of its own. A live run and
  the same run reloaded from disk then glow identically.
- **A synthetic start lights nothing.** Its start was stamped by us, not
  observed; the field may only stand where something happened.

The field is also the only part of the live view that needs a GPU and the only
part that carries no figure — the same decision twice. Without WebGPU it is
simply absent and the chart above is exactly as legible.

Positions come from `LiveDriver.field()`. The field never recomputes the
projection, though it easily could: two implementations of one mapping drift,
and a glow half a row off its bar reads as a second measurement. Same argument
as the no-fold rule, one layer down.

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

### 5.1 The third state the first two were swallowing

Building L5 against real captures turned up a state the design had not
allowed for: **the agent never carried the text at all**. Codex thread history
records *that* the model reasoned and keeps none of what it said, so
`--keep-reasoning` changes nothing for a reconciled run. Every one of the 135
reasoning fragments in the longest captured run was arriving labelled
`dropped_by_policy` with `chars: 0` — a decision nobody made, and a size of zero
for text that was never in the record. That is the "missing must never render as
`0`" rule failing one layer below the UI, where no amount of careful drawing
could have recovered it.

`reasoning_payload` now returns three answers rather than two, keyed on the
difference between `None` and `""`:

| what the agent gave us | fidelity | payload |
|---|---|---|
| text, and we asked to keep it | `native` | `text` |
| text, and we declined | `dropped_by_policy` | `chars` |
| no field at all | `missing` | `chars: null` |

`""` is deliberately not the third case. An agent that emitted a reasoning item
with no words told us something; an absent field did not.

The rail draws the third in missing ink rather than policy ink, for the same
reason the legend keeps those two swatches apart. Existing logs keep their old
labels — the log is the record, and rewriting one to match a later opinion is
the thing this subsystem exists not to do — so a run captured before this change
still shows `0 characters dropped by policy` until it is re-imported.

### 5.2 Why the rail reads the log back

`LiveModel` holds only what it ingested over SSE, so a run that finished before
the tab connected has no thoughts in it — the same trap the field hit in §4.2,
where a historical run sat dark beside a live one and read as a run that did
nothing. The rail therefore asks `/seer/run/<id>/events` for each selected run
once and folds the reasoning events out of it.

Two rules keep that from becoming a second model:

- **Thoughts only.** Marks are the leading edge, bounded by a time window and
  feeding the chart and the field. Pouring an hour of history into them would
  either be trimmed away immediately or inflate `droppedMarks`, a figure that
  means something else.
- **Folded by identity, never by addition.** Codex re-emits one `item.reasoning`
  as its text grows, so five events describe one thought; Claude emits a single
  completed event per thinking block. Summing `chars` across a stream would
  triple the first and leave the second correct — the delta rule's trap in a
  different shape. A thought is keyed by its span (or by the first event's id
  when there is no span) and the newest event *replaces* its content. That also
  makes the fold order-insensitive, which is what lets a backfilled event land
  after a live one without walking the thought backwards.

A duration needs both ends observed. Claude reports a thinking block only once
it is finished, so its start and end are the same instant and the rail prints
`—` rather than `0.0s`. An open stream on a run that has since finished reads
"no end" — the run ended before the stream was closed, which is a gap in the
capture and not a thought that is still running.

## 6. Order of build

| | | depends on |
|---|---|---|
| L0 | `LiveModel` (§2.1) + `encoding.ts` + their tests. No pixels. | — |
| L1 | Score layout, 2D canvas, follow / scrub transport | L0 |
| L2 | Fleet y-mode + the morph between it and Score | L1 |
| L3 | Structure (span-depth flame) as the third y-mode | L2 |
| L4 | Field layer, WebGPU, reusing the SessionField patterns | L1 |
| L5 | Thought rail, both states — and a third | L0 |
| L6 | Atlas cross-fade | **refused — see §7** |

L0 is the whole bet. If the model and the vocabulary are right, each later
milestone is a layout function over state that already exists — and if they are
wrong, five views will disagree with each other in five different ways.

## 7. L6, and why the trail is not drawn

Four of the five shipped. The Atlas trail did not, and the reason is worth
writing down properly, because "we ran out of time" and "the artifact cannot
express it" are different sentences and only the second one is true here.

### 7.1 The projection does not exist and cannot be recovered

`reduce_vectors` fits two UMAP models and returns their outputs:

```python
u_cluster = umap.UMAP(n_components=cluster_dim, min_dist=0.0, **common).fit_transform(vectors)
u3 = umap.UMAP(n_components=3, min_dist=0.1, **common).fit_transform(vectors)
```

Both fitted objects are locals. They are garbage-collected when the function
returns, and `reduced.npz` stores `u_cluster`, `u3` and `u2` — coordinates, not
a transform. `validate.py` says the same thing from the other side: the cache
"caches only the UMAP outputs … never the source vectors", which is why
trustworthiness has to reload the whole front-end to grade a map.

So there is no `reducer.transform(v)` to call. Not for a new map, not for an old
one, not with any flag. Placing a live event where UMAP would have placed it
requires the model that placed everything else, and that model was never saved.
Persisting it from now on would not help either: it would not retroactively give
the shipped maps one, and re-running UMAP over an existing map is the 15–30
minute stage `build_server.py` warns about.

### 7.2 What *is* possible, and why it still is not the trail

Two maps ship their source vectors beside their coordinates —
`gpt2__api-mxbai-embed-large` (86MB, `mxbai-embed-large`) and the
claude-tokenizer map (3.6MB, `all-MiniLM-L6-v2`) — because the API front-end
checkpoints its embed cache and the cache lands in the dataset directory. On
exactly those two, an event's text could be embedded with the same model,
cosined against the cache, and placed at a distance-weighted centroid of its
nearest tokens' `u3`.

That is a real operation. It is not, however, a projection, and calling the
result a trail through the atlas would be the fourth version of the same mistake
this document keeps catching:

- **It is not where UMAP would put the point.** A centroid in the source space
  and an out-of-sample UMAP embedding disagree, sometimes a lot — UMAP's whole
  job is a non-linear rearrangement. The picture would be smooth, plausible and
  unrelated to the geometry the reader has learned from the map they are looking
  at.
- **The map is of tokens.** Embedding `edit viewer/src/chrome/SeerThoughts.tsx`
  and dropping it among single-token neighbours shows where the *words of an
  action* land in token space. A reader will take it as where the agent is
  working in concept space. Those are not the same claim, and the second one is
  the one the visual makes.
- **Only two of the seventeen shipped maps could do it at all**, so the mode
  would be absent or wrong depending on which map is loaded — the mode switch
  itself would become a claim about the data.

Same shape as §4.1's flamegraph and §4.2's magnitude-free glow: a picture with
real structure in it, and the structure is ours rather than the record's. The
mode is therefore not built.

### 7.3 What would unblock it

In order, and all three are needed:

1. **Persist the fitted 3-D reducer** beside `reduced.npz` at build time, and
   rebuild the maps that should support live placement. `umap.UMAP` pickles with
   its nearest-neighbour index, so budget for a large artifact and decide
   deliberately whether it ships with the static deploy.
2. **A reachable embedding endpoint running the same model the map was built
   with.** The mxbai map's Ollama host is gone; the MiniLM map's OpenAI-compatible
   host is the M4 worker in `docs/LIVE-BUILDER.md`, which the sanitized branch
   does not name.
3. **A map whose points are the right kind of thing.** A trail over a token
   atlas answers a question nobody asked. The honest version of this feature is
   an atlas built from *session* units — actions, files, thoughts — at which
   point the trail is a projection of the same points the map is made of, and
   every objection above dissolves.

Item 3 is the interesting one, and it is a different project than a cross-fade.
