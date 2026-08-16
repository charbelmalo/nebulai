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

## Prerequisites — one blocker left

Blocker 1 is fixed (2026-08-13) and §3 turned out to be a misdiagnosis rather
than a blocker. Only §2, the `build_server` probe path, still stands between
here and door 1's live branch.

### 1. ~~`embed_host` leaks a private LAN IP into public artifacts~~ — FIXED 2026-08-13

`probe.py` and `api_tokens.py` both stamped the raw `--embed-host` into exported
`meta`, and those files are served publicly. This predated the onboarding work
and was a straight regression against the `a512c21 sanitize:` intent.

**The blast radius was larger than this section originally recorded.** It named
two shipped datasets; a sweep of `nebulai-data/out` found **five**. The three it
missed are the `probe__*` clouds, and they carry `:11435` — the *current* port —
so they were baked *after* this warning was written. The sentence predicting
that "pre-baking probe clouds against the same worker would add more of them"
had already come true by the time anyone acted on it.

Fixed at the source: `public_embed_host()` in `backend/embed.py` passes loopback
endpoints through verbatim and collapses everything else to `"remote"`, so
provenance still records that an external service placed the points while the
address itself never reaches disk. Both front-ends call it at their single
stamping site (`api_tokens.py`, `probe.py`), so it stayed one change in one
place. It is deliberately *not* a general "is this address private" classifier —
that call fails open, and one wrong verdict publishes the address; only
loopback, which needs no judgement, survives. The evidential fields
(`embed_model`, `embed_api`) are untouched: the host was never what made a map's
vectors what they are.

The five shipped artifacts were **redacted in place, not re-exported**. A
re-export re-runs the embedder, and the GGUF build is not bit-deterministic
(measured against the July cache: ~1e-3 elementwise, cosine ≥ 0.999945), so
every coordinate in five maps would have moved to fix one metadata string. The
substitution was verified to leave each parsed document differing at exactly one
key. Originals are in `nebulai-data/.pre-redaction-backup/` — outside the served
`out/` tree, deliberately.

### 2. `build_server` can only run `nebulai tokens`

`build_cmd` hardcodes the subcommand and validates `model` against `_MODEL_ID`
(`build_server.py:93`). Door 1's live-probe branch needs a `probe` path: a seed
is free text, not a model id, so it needs its own validation — argv is a list,
never a shell string, so this is a value-sanitation question rather than an
injection one, but the seed still reaches an LLM prompt and should be length-
and character-bounded.

### 3. No generator or embedder on *this* machine — but the M4 has both

Ollama is not installed locally (`ollama not found`, nothing on 11434). That part
is still true. What was wrong, and cost an afternoon elsewhere: the M4 worker's
embedder is **not** on 11434 either — it binds **11435**, and has since
2026-08-04 (`M4-OLLAMA-HANDOVER.md`). Verified 2026-08-13:

```
GET http://<m4-host>:11435/api/tags         -> mxbai-embed-large:latest (1024-dim)
GET http://<m4-host>:11435/api/version      -> 0.23.1
GET http://<m4-host>:8100/v1/status/ollama  -> running:true, port:11435
```

So pre-baking needs **no activation step** — ollama is already running and
`KeepAlive` survives reboots. Export `NEBULAI_EMBED_HOST=http://<m4-host>:11435`
once and `tokens`/`probe`/`compare` all pick it up.

Note `:8050` on the same box is a *different* server (OpenAI-compatible `omlx`,
carrying `all-MiniLM-L6-v2` and `nomic-embed-text-v1.5`) — a different neutral
space, reached with `--embed-api openai`, not a substitute for mxbai.

Blocker 1 still applies: fix the host-stamping first, or the baked clouds carry
the LAN IP the way `gpt2__api-mxbai-embed-large` already does.

## Build order

1. ~~**Sanitize `embed_host`**~~ — done 2026-08-13 (blocker 1 above). It blocked
   everything else that writes a map, so it went first.
2. **Pre-bake 3–5 probe clouds** — seeds spanning emotion / science / culture /
   money so door 1 has range. Add them to `out/index.json`. Three already exist
   (`grief`, `glassblowing`, `tidal-ecology`) and are now clean, so this is
   nearer to done than the list implies.
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
