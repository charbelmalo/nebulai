# Onboarding — "Start here"

Nebul.AI currently opens on 49,857 GPT-2 token dots, 208 clusters, 55% of them
noise, a gear panel, and no statement of what any of it is. That is a fine
first screen for someone who already came for GPT-2 and a bad one for everyone
else. This document specifies the path in.

Two facts shape the whole design:

- **`nebulai probe` is the door for non-researchers.** It is the only front-end
  where a person types a word they care about instead of picking a model
  checkpoint. Everything else assumes prior interest in a specific model.
- **The honesty guardrails are the product, not a disclaimer.** A map that
  hides its 55% noise and its 0.5 seed-ARI is a prettier lie. The onboarding
  teaches each caveat *at the interaction where it bites*, never as a wall of
  text on entry.

## Shape

**One fork, five doors.** Each door is labelled by a *domain of interest* and
subtitled by the *question it answers* — so the two candidate axes collapse into
one level instead of nesting. The two-level version (domain → question) is a
pure refinement later: each door grows a sub-fork without the top level moving.

Doors live on a new `start` page in the top nav. Picking one runs a **scripted
tour** that drives the real app through ordinary store actions — the mechanism
already proven in `chrome/tours.ts` — so every step is a permalinkable state and
quitting a tour leaves you exactly where the last step put you.

| # | Door | Question | Lands in | Data |
|---|---|---|---|---|
| 1 | A word you care about | *What sits near an idea?* | Atlas | pre-baked probe clouds + live probe |
| 2 | How a model sees words | *What does a language model group together?* | Atlas + search | `gpt2` tokens (shipped) |
| 3 | Watch a model think | *Can we actually see a mechanism?* | Internals | gpt2 interp bundles (shipped) |
| 4 | How much of this is real | *Artifact or finding?* | Compare + metrics | `validation.json` (shipped) |
| 5 | Your own conversations | *What's in my data?* | Snapshot / Sessions | user-supplied |

Doors 1 and 5 need no model knowledge at all. Doors 3 and 4 are where the
existing researcher content plugs in unchanged — `tours.ts`'s induction, IOI and
SAE tours become door 3's three sub-steps verbatim.

## The tree

```mermaid
flowchart TD
    START(["Start here"]) --> FORK{"What are you curious about?"}

    FORK -->|"A word you care about"| D1
    FORK -->|"How a model sees words"| D2
    FORK -->|"Watch a model think"| D3
    FORK -->|"How much of this is real"| D4
    FORK -->|"Your own conversations"| D5

    subgraph D1 ["1 · A word you care about"]
        direction TB
        D1a["Pick a seed<br/>grief · photosynthesis · jazz · money"] --> D1b["Atlas: the cloud<br/>a dot is one concept"]
        D1b --> D1c["Hover a dot<br/>read the term"]
        D1c --> D1d["Open a cluster<br/>a neighbourhood, auto-named"]
        D1d --> H1{{"⚠ two models' opinion<br/>absent ≠ nonexistent"}}
        H1 --> D1e["Raise sensitivity<br/>watch terms drop out"]
        D1e --> H2{{"⚠ the edge is a knob,<br/>not a boundary"}}
        H2 --> D1f["Your own word →<br/>live probe"]
    end

    subgraph D2 ["2 · How a model sees words"]
        direction TB
        D2a["Atlas · gpt2 tokens<br/>49,857 dots"] --> D2b["Search a word<br/>see its neighbours"]
        D2b --> D2c["Orthographic vs semantic<br/>clusters side by side"]
        D2c --> H3{{"⚠ toggle Noise: 55%<br/>joined no cluster"}}
        H3 --> D2d["Switch model<br/>same word, different map"]
    end

    subgraph D3 ["3 · Watch a model think"]
        direction TB
        D3a["Internals"] --> D3b["Induction circuit<br/>weights → behaviour → ablation"]
        D3b --> D3c["How GPT-2 knows Mary<br/>attribution → intervention"]
        D3c --> H4{{"⚠ single-head ablation<br/>understates circuits"}}
        H4 --> D3d["What an SAE feature is"]
    end

    subgraph D4 ["4 · How much of this is real"]
        direction TB
        D4a["The claim:<br/>silhouette 0.4999"] --> D4b["The floor:<br/>shuffled null 0.3772"]
        D4b --> D4c["The margin: +0.123<br/>real, and modest"]
        D4c --> H5{{"⚠ seed ARI 0.46–0.62<br/>half the partition moves"}}
        H5 --> D4d["Compare two maps<br/>null.k ⚠ when not comparable"]
    end

    subgraph D5 ["5 · Your own conversations"]
        direction TB
        D5a["Drop a JSON log"] --> D5b["Topics<br/>topics over time"]
        D5b --> D5c["Transcripts<br/>agent transcripts"]
        D5c --> H6{{"⚠ your data stays<br/>in your browser"}}
    end

    D1f --> ANY(["Free explore<br/>tour exits in place"])
    D2d --> ANY
    D3d --> ANY
    D4d --> ANY
    H6 --> ANY

    ANY --> GUIDE["Guide<br/>math + provenance<br/>for every view"]

    classDef honest fill:#2a1420,stroke:#ea4f86,color:#ffd9e6
    classDef door fill:#141a2a,stroke:#8b3bf0,color:#e6dcff
    class H1,H2,H3,H4,H5,H6 honest
    class START,FORK,ANY,GUIDE door
```

## Honesty, wired to interactions

Each caveat fires from a specific user action, not from a splash screen. This
is the whole difference between a guardrail and a disclaimer.

| Interaction | Caveat surfaced | Source of the number |
|---|---|---|
| Open any probe cloud | "Two models' joint opinion — the generator proposed these terms, the embedder placed them. A term that is absent means the generator didn't propose it." | `meta.generator`, `meta.embed_model` |
| Raise `--sensitivity` / see drop rate | "The map's edge is a knob. `n_dropped` of `n_proposed` terms were cut for sitting too far from the seed." | `meta.n_proposed`, `meta.kept`, `meta.n_dropped` |
| Toggle **Noise** on a token map | "55% of these points joined no cluster. HDBSCAN abstained; it did not decide they're meaningless." | `meta.noise_fraction` |
| Linger on a cluster boundary | "Seed ARI is 0.46–0.62. Re-run with a different seed and roughly half this partition redraws. The gross layout is stable; this boundary is not." | `validation.json` |
| Reach the ablation step | "Knock out L7H10 alone: Δ −0.014. Zero all four induction heads: Δ +2.04 — 2.8× the sum. Redundancy is why single-head ablations understate circuits." | gpt2 interp bundle |
| Open Compare | "Silhouette rises as a partition coarsens. When the null resolved a cluster count outside 0.5–2× the map's, the margin is flagged `?` and is not evidence." | `metrics` `null.k` |
| Anywhere | "This is clustering + visualization over public micro-models. It shows how units relate geometrically — not what any unit *does* to behaviour." | project guardrail |

## Prerequisites — two blockers, one of them live

### 1. `embed_host` leaks a private LAN IP into public artifacts

`probe.py:306` and `api_tokens.py:111` both stamp the raw `--embed-host` into
exported `meta`. Two **already-shipped** datasets carry it today:

```
out/gpt2__api-mxbai-embed-large/nebulai.json          embed_host: http://<m4-host>:11434
out/Xenova__claude-tokenizer__api-.../nebulai.json    embed_host: http://<m4-host>:8040
```

Those files are served publicly. This predates the onboarding work and is a
straight regression against the `a512c21 sanitize:` intent — and pre-baking
probe clouds against the same worker would add more of them.

The host is not evidence about the map; the *model name* is, and that is stamped
separately. Fix: stop recording the host for non-loopback endpoints (keep a
`"remote"` marker so provenance still says an external service was used), then
re-export the two affected artifacts. Shared between both front-ends, so it is
one change in one place.

### 2. `build_server` can only run `nebulai tokens`

`build_cmd` hardcodes the subcommand and validates `model` against `_MODEL_ID`
(`build_server.py:93`). Door 1's live-probe branch needs a `probe` path: a seed
is free text, not a model id, so it needs its own validation — argv is a list,
never a shell string, so this is a value-sanitation question rather than an
injection one, but the seed still reaches an LLM prompt and should be length-
and character-bounded.

### 3. No generator or embedder on this machine

Ollama is not installed locally (port 11434 dead, `ollama not found`). The M4
worker's activation API at `:8100` **is** up, so pre-baking is one activation
step away — subject to blocker 1 being fixed first, or the baked clouds will
carry the LAN IP.

## Build order

1. **Sanitize `embed_host`** (shared front-end change) + re-export the two
   affected artifacts. Blocks everything else that writes a map.
2. **Pre-bake 3–5 probe clouds** — seeds spanning emotion / science / culture /
   money so door 1 has range. Add them to `out/index.json`.
3. **Generalize the tour engine** — `tours.ts` is currently gpt2-only, Internals-
   only, and typed to `InterpSelection`. It needs to also drive `page`,
   `viewMode`, `datasetId`, toggles, and search, so a tour can walk the Atlas.
4. **`start` page** — new `Page` variant, `NavPill`, mount branch, door cards.
5. **Caveat hooks** — the table above, fired from real interactions.
6. **Live probe** — `build_cmd` probe path + the door-1 "your own word" branch,
   degrading to a labelled "needs a local backend, here's the command" state
   when the build server is absent.

Steps 1–2 are back-end and unblock the demo; 3–5 are the viewer; 6 closes the
loop. Every new knob introduced along the way lands in `SettingsPage.tsx` under
the correct tab, per the project rule.

## Deliberately not in scope yet

- **Two-level tree** (domain → question). The doors are designed to grow a
  sub-fork without the top level moving; deferred by decision.
- **Rewriting the three existing Internals tours.** They are good and stay
  verbatim as door 3.
- **Mobile.** The viewer is a WebGPU desktop tool; the onboarding inherits that.
