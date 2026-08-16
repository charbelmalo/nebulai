# Behavioral semantic divergence — research and UX plan

Status: implementation baseline, 2026-08-13. **Revised 2026-08-13** after a
methodological and visualization review; see §17 for what changed and why.

This document defines the additive Nebul.AI feature that can discover where two
model deployments produce substantially different semantic associations. GPT-2
versus a pinned Grok release is the first study. The feature is not a model
leaderboard, and a cue such as “daddy” is a drill-down example rather than the
experiment's organizing assumption.

Three things dominate everything below, and they were the review's main
findings. **First**, GPT-2 and a post-trained Grok differ in *capability* before
they differ in *semantics*, so a statistic measured against zero measures the
wrong thing (§4.2.1, §6.4.1) and needs a capability-matched reference (§5.7).
**Second**, the Behavior study needs its own pinned, in-process embedder — not
because none is reachable (one is: see the 2026-08-13 correction in §6.3.1) but
because a mutable tag cannot be a pinned revision and because raw trial outputs
must not be routed to the LAN worker (§6.3.1). **Third**, “one fixed coordinate
system” and “UMAP” are mutually exclusive given a growing cue set, and at pilot
scale UMAP is independently unsafe — this project has *measured* it inventing
clean islands from shuffled noise at n=180 (§7.1.1).

## Decision at a glance

| Question | Decision |
|---|---|
| Can Nebul.AI show GPT-2/Grok semantic differences visually? | **Yes**, by comparing repeated, controlled output distributions and visualizing only differences that survive reliability and replication checks. |
| Does this prove different internal “minds”? | **No.** It supports a narrower claim about observable association behavior under an exact protocol. GPT-2-only internals may explain its side later, but cannot make Grok's hidden internals comparable. |
| Do the existing clouds have to be removed or rebuilt? | **No.** Token, SAE, neuron, Probe, Compare, and Internals artifacts stay intact. The new study has its own data contract and can link back to them. |
| Does the current product expose any of this already? | **Partially.** Existing clouds are useful hypothesis generators; they do not yet provide repeated trials, uncertainty, false-discovery control, or held-out confirmation. |
| Where should the feature live? | Add a top-level Nebul.AI page labeled **Behavior**, with the page title **Behavior — semantic divergence**. Keep the current Map → Compare view lightweight and unchanged. |
| What is the default experience? | Open a precomputed study, see the largest **confirmed** deviations, search any cue, and inspect raw evidence. Experimental controls stay behind **New study**. |
| What is the headline number? | The **normalized contrast `Δ̂`** — between-model separation minus each model's own split-half separation — not raw MMD². Raw MMD² is displayed beside it and is never the rank key (§6.4.1). |
| How do we know an effect is not just “Grok is a better model”? | A third, local, free arm: **GPT-2-small vs GPT-2-XL** on the same cues. An effect that does not exceed its own same-family scale contrast is reported as `capability-attributable`, not as a difference in learned association (§5.7). |
| What projection does the cue landscape use? | **PCA**, because it is the only choice that makes “fixed coordinates” literally true across a growing cue set, and because UMAP at 100–300 points manufactures separation this project has already measured (§7.1.1). |

The resulting Nebul.AI navigation is:

**Semantic map · Behavior · Internals · Guide**

Behavior remains inside Nebul.AI rather than becoming a third instrument beside
Nebul.AI and Seer. It studies model behavior and needs direct links to the
semantic map and GPT-2 Internals. A separate deployment would add friction
without adding epistemic clarity.

---

## 1. Claim contract

### 1.1 What the feature may claim

The strongest default sentence in the product is:

> Under protocol P, exact model deployments A and B produced reliably different
> association distributions for cue C.

The study may also report:

- the overall correlation between the models' association structures;
- which preregistered cues have the largest confirmed divergence;
- which associates are shared or model-specific, with observed frequencies;
- how stable a result is across resampling, prompt frames, and neutral
  embedders;
- whether a selected cue's result is confirmed, suggestive, unstable,
  underpowered, or incomparable.

### 1.2 What the feature must not claim

The UI, export, and documentation must not say that:

- one model is stronger, smarter, or more correct;
- a model “believes” an association;
- a visual proves an internal concept has a particular meaning;
- GPT-2 and Grok have directly comparable internal representations;
- a provider's displayed or summarized reasoning is the model's ground-truth
  private thought process;
- an unreplicated large effect is a confirmed difference.

“Mind,” “thinking pattern,” and “reasoning style” may appear in explanatory
copy only as the user's motivating metaphor. Product labels use **behavioral
association**, **association profile**, **similarity**, and **divergence under
this protocol**.

### 1.3 Why the claim is deliberately narrow

GPT-2 is an open base completion model. A deployed Grok API model is a much
larger chat/reasoning system with different training, tokenizer, wrapper,
post-training, and safety behavior. Even identical visible prompt text does not
make those systems experimentally identical. xAI's current documentation also
states that recent Grok reasoning cannot be disabled, model aliases can move,
and recent releases do not expose token log probabilities. The shared
measurement surface is therefore repeated final-output sampling, not hidden
activations or logits.

The first release answers:

> “Where do their observable semantic associations diverge most, under a
> recorded common task?”

It does not answer:

> “Which architectural or training difference caused the divergence?”

---

## 2. What stays, what changes, and what the current product already reveals

### 2.1 Existing assets are preserved

| Existing surface | What it measures now | Decision |
|---|---|---|
| Token clouds | A checkpoint's token embedding or unembedding geometry | Preserve files, coordinates, CLI, validation, and UI. A Behavior cue may link to a matching token, but never moves it. |
| SAE clouds | SAE decoder-direction geometry | Preserve. Later accept Behavior evidence as a read-only overlay or filter. |
| Neuron clouds | MLP write-direction geometry | Preserve. Later accept the same read-only overlay contract. |
| Probe | Concepts proposed by a generator and placed by an independent embedder | Preserve as exploratory hypothesis generation. Probe results cannot be promoted to “confirmed divergence.” |
| Map → Compare | Named cluster text embedded in a neutral label space; shared/unique concepts and Jaccard overlap | Preserve exactly. Add one link: **Open behavioral comparison**. It remains a fast structural cloud comparison. |
| Internals | Real GPT-2-family forward-pass, activation, attention, logit-lens, SAE, ablation, and patching bundles | Preserve. A confirmed GPT-2 cue may deep-link to an exact prompt trace. This is an asymmetric follow-up, not pairwise proof. |
| `nebulai validate` | Geometry trustworthiness, seed stability, silhouette, and a shuffled null baseline | Preserve. Behavior receives a separate validation command because its unit of evidence is repeated trials, not map geometry. |

No existing `nebulai.json` schema changes in the first implementation. New
artifacts live under `out/behavior/`, and the normal atlas can ship even when
that directory does not exist.

### 2.2 How much the current integration exposes

The answer is **partly, but not enough for the intended claim**:

1. **Current Compare can suggest broad structural differences.** Its
   `ComparePanel` correctly says “label space, not model geometry.” It compares
   already-named cloud clusters through a third-party embedder. This is useful
   for forming hypotheses, not for proving that two models respond differently
   to the same cue.
2. **Probe can show one model-generated neighborhood around “daddy.”** The
   neighborhood is a joint product of the generator, prompt, parser, and
   independent positioning embedder. It has no repeats or uncertainty and
   cannot discover the largest deviations across a broad cue space by itself.
3. **Internals supplies strong GPT-2-side evidence.** It cannot query comparable
   Grok activations, so it cannot be the common comparison layer.
4. **Existing validation is real but answers another question.** Stable UMAP
   geometry does not establish stable cue-conditioned output behavior.

The existing clouds therefore become contextual layers around a new behavioral
experiment; none is thrown away.

---

## 3. Evidence grounding

### 3.1 Evidence ledger

| ID | Source | What it contributes | Scope / caution | Reliability |
|---|---|---|---|---|
| E1 | [The LLM World of Words, Scientific Data (2025)](https://www.nature.com/articles/s41597-025-05156-9) and its [open methods text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12084308/) | A directly relevant precedent: more than 12,000 cues, three associations, 100 repetitions per cue, weighted semantic networks, and psycholinguistic validation. | It studied Mistral, Llama 3, and Haiku, not GPT-2/Grok. Its preprocessing choices are not automatically correct for slang-heavy divergence research. | Peer-reviewed, directly relevant. |
| E2 | [Small World of Words English norms](https://doi.org/10.3758/s13428-018-1115-7) | A broad cue inventory and optional human reference with multiple responses for more than 12,000 cues. | Human norms are a reference, not truth or a strength ranking. The source data are research-only under CC BY-NC-ND and may not be redistributed or repackaged without permission. | Peer-reviewed, direct task precedent. |
| E3 | [HELM](https://arxiv.org/abs/2211.09110) | Standardized scenarios, multiple metrics, exact prompt/completion retention, and explicit disclosure of what is not measured. | A design principle rather than a ready-made metric for association divergence. | Peer-reviewed benchmark framework. |
| E4 | [How are Prompts Different in Terms of Sensitivity?](https://aclanthology.org/2024.naacl-long.325/) | Evidence that prompt form changes model behavior, motivating predeclared frames and held-out prompt replication. | The paper studies broader prompting behavior, not this exact free-association protocol. | Peer-reviewed, indirect but important. |
| E5 | [A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a.html) | Maximum mean discrepancy (MMD) as a non-parametric test of whether two samples come from different distributions. | Requires a fixed representation and kernel; those choices must be calibrated before confirmation rather than tuned per cue. | Peer-reviewed statistical method. |
| E6 | [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html) | Centered kernel alignment (CKA) for global similarity between aligned representation matrices. | Here CKA describes **behavioral association profiles**, not hidden Grok representations. | Peer-reviewed method, adapted to the behavioral matrix. |
| E7 | [Benjamini–Yekutieli false-discovery control](https://doi.org/10.1214/aos/1013699998) | Conservative false-discovery control when thousands of cue tests are dependent. | More conservative than ordinary BH; power must be measured in the pilot. | Peer-reviewed statistical method. |
| E8 | [xAI models documentation](https://docs.x.ai/developers/models), [chat API reference](https://docs.x.ai/developers/rest-api-reference/inference/chat), and [reasoning documentation](https://docs.x.ai/developers/model-capabilities/text/reasoning) | Dated releases for consistency, served model ID, response ID, usage, `system_fingerprint`, lack of current logprobs, and reasoning controls. | Provider behavior is time-varying. The implementation must run a capability audit instead of trusting this document forever. | First-party current interface documentation, accessed 2026-08-13. |
| E9 | [Grok 4.20 system card](https://data.x.ai/2026-04-07-grok-4-20-model-card.pdf) | Evidence that at least this deployed Grok generation includes supervised and reinforcement post-training plus refusal/safety interventions. | It does not describe every Grok release and does not establish that Grok is globally “adversarial.” | First-party release evidence, release-specific. |
| E10 | [Rank-biased overlap](https://doi.org/10.1145/1852102.1852106) | A top-weighted comparison for ranked lists whose lower ranks may be incomplete or unstable. | A secondary lexical-overlap metric, not a distributional significance test. | Peer-reviewed information-retrieval method. |

### 3.2 Grounded design choices

The evidence supports these choices:

- use broad, repeated free associations as the common behavioral denominator;
- keep raw prompts and outputs so every aggregate can be audited;
- compare several metrics rather than inventing one “mind distance” score;
- preregister the primary metric and prompt before collecting confirmation data;
- test many cues with false-discovery control;
- replicate on held-out prompt frames and a second independent embedder;
- record exact model releases and deployment fingerprints;
- preserve slang, emoji, and non-dictionary responses in the primary analysis,
  because filtering them through WordNet could erase exactly the modern/social
  associations the study is intended to discover.

---

## 4. Research questions and hypotheses

### 4.1 Primary research questions

1. How correlated are GPT-2 and a pinned Grok deployment's broad association
   structures under the same visible task?
2. Which cues produce the largest reliable distributional deviations beyond
   each model's own sampling variability?
3. Which deviations survive prompt paraphrase, held-out trials, and a second
   neutral semantic representation?
4. For a confirmed cue, what exact words and semantic neighborhoods account for
   the difference?

### 4.2 Predeclared hypotheses

- **H1 — global non-identity:** the two models' association structures are
  correlated but not identical.
- **H2 — localized deviation:** a subset of cues has between-model separation
  larger than calibrated same-model split-half separation.
- **H3 — robustness:** some localized deviations replicate across held-out
  prompt frames and independent embedders.
- **H4 — relationship/social register (secondary):** terms such as “daddy” may
  show larger deployment differences than matched family controls such as
  “father,” “mother,” and “parent.”

H4 is a hypothesis, not product copy. “Grok is more adversarial” is not assumed.
If “daddy” diverges, competing explanations include corpus recency, internet
slang, model scale, tokenization, chat post-training, safety policy, sampling,
or a stable semantic association difference. The study can separate some of
these explanations; it cannot identify training causality from outputs alone.

#### 4.2.1 H2 is the operational hypothesis, and §6.4 must state the same thing

H2 is written correctly and it is the load-bearing sentence in this document:
the target is between-model separation **larger than calibrated same-model
split-half separation** — a *normalized* quantity. Everything downstream has to
be that quantity, not raw MMD².

The reason is the study's largest single threat. GPT-2 is a 2019 base completion
model with no instruction tuning; the Grok side is a large post-trained
chat/reasoning system. Under a naive null — *these two produce the same
association distribution* — **the null is false everywhere for reasons that have
nothing to do with semantics.** GPT-2 format-copies, repeats, and continues
n-grams; Grok performs the task. Two consequences follow mechanically:

1. permutation `p ≈ 0` for nearly every cue, so BY has almost nothing left to
   correct and thousands of cues arrive at “confirmed”;
2. the ranked “largest confirmed deviations” list stops being a ranking of
   semantic difference and becomes **a ranking of where GPT-2 is most
   degenerate** — the cues where a base LM most fails to do the task at all.

A capability-matched reference is therefore not optional (§5.7), the primary
statistic is the normalized contrast rather than raw MMD² (§6.4), and the
confirmation gate carries a compliance-parity condition (§6.5). Raw MMD² remains
a displayed secondary value, never the rank key and never the gate.

### 4.3 Falsifiers and downgrade rules

A candidate is not confirmed if:

- it disappears in the held-out prompt frame;
- it appears only after a post-hoc prompt change;
- it is smaller than same-model sampling variability;
- the two neutral embedders disagree materially;
- the exact-word view reveals synonyms rather than a semantic shift;
- model identity or `system_fingerprint` changes mid-study;
- parsing/compliance failures leave insufficient valid trials;
- the multiple-testing-adjusted result does not pass the declared threshold.

---

## 5. Study design

### 5.1 Four separate lanes

The lanes must never be silently pooled.

| Lane | Purpose | Status in first release |
|---|---|---|
| **A. Association norms** | Same literal few-shot task, one cue per independent request, three ordered associates, repeated sampling. This is the common comparison and primary result. | Required |
| **B. Prompt robustness** | Two semantically equivalent prompt forms kept hidden during discovery, then used to confirm candidates. | Required for “confirmed” |
| **C. Context/sense frames** | Predeclared frames such as family, internet slang, or relationship context, applied to matched cue packs. Answers whether a difference is sense-specific. | Secondary |
| **D. Native deployment behavior** | Provider-typical system/personality prompt or tools, explicitly labeled as wrapper + policy behavior. | Optional and exploratory |

Tools, web search, X search, conversation history, structured output, and
provider-native personality prompts are **off** in Lane A. Structured output is
not used because GPT-2 cannot receive the same constraint. If a selected Grok
release cannot disable reasoning, the lowest supported reasoning effort is
recorded as an unavoidable deployment difference.

That last clause has a cost consequence the budget gate cannot see. Reasoning
tokens are billed, vary per request, and are **not knowable before the request
is made**, so a plan-time estimate multiplied by trial count bounds a quantity
it cannot predict — and §8.3's “immutable budget gate” inherits that error. The
mitigation is measurement plus a runtime stop:

- Phase 0 records the **empirical reasoning-token distribution** for the pinned
  release under the Lane A prompt, and the budget is computed on its **p95**,
  not its mean.
- The **runner**, not only the planner, enforces a hard spend ceiling. It
  checkpoints, halts, and reports partial coverage when actual spend crosses the
  gate — a study that stops early with an honest manifest is recoverable; one
  that silently overruns is not.

### 5.2 Common visible task

Both adapters receive the same literal few-shot text:

- a short instruction to return the first three associated words;
- one neutral input/output example;
- the target cue;
- no request for explanation.

GPT-2 receives it as a completion ending at the answer marker. Grok receives
the entire literal block in a single user message with no study-specific system
message. Each trial is a fresh request. Parsing keeps the first three
comma-separated responses while retaining the complete raw output.

Formatting compliance is reported separately. It is never imputed into semantic
distance and never used to portray a model as semantically different merely
because it followed the output format less reliably.

### 5.3 Cue suites

The built-in pilot uses a project-owned, redistributable cue list stratified by:

- frequency and part of speech;
- concrete versus abstract terms;
- affect and social relationships;
- identity and demographic terms;
- internet-era and slang terms;
- technical/scientific terms;
- ambiguous and polysemous terms;
- neutral matched controls.

The “daddy” control pack includes “father,” “mother,” “parent,” “child,” and
matched relationship/slang cues. It is guaranteed to be inspectable in the
pilot, but it receives no special statistical threshold.

SWOW cues and human norms are optional local inputs behind an explicit license
gate. Nebul.AI must not bundle or republish modified SWOW records by default.

### 5.4 Sampling and ordering

Phase 0 chooses the final trial count by empirical power and rank-stability
calibration:

1. **Calibration:** 100 cues × 40 trials × 2 models.
2. **Pilot discovery:** about 300 cues × the calibrated 20–40 trials × 2 models.
3. **Expanded discovery:** about 1,000 cues only after the pilot passes.
4. **Full norms:** up to the roughly 11,500 aligned research cues and 100
   trials only after power, license, latency, and cost gates.

Requests are randomized and interleaved in time blocks. The runner does not
collect all GPT-2 trials first and all Grok trials later. Concurrency, retries,
rate-limit waits, and response timestamps are retained.

The study has two disjoint sample partitions:

- **discovery:** ranks candidate cues using the primary frame;
- **confirmation:** new trials using held-out prompt frames and frozen analysis.

A prompt, metric, threshold, or normalizer changed after seeing discovery
results creates a new study revision. It cannot rewrite the confirmation
manifest.

**Confirmation runs two arms, not one.** As originally written, confirmation
changed two variables simultaneously — fresh trials *and* held-out frames — so a
failure could not distinguish “the discovery hit was a false positive” from “the
effect is real but specific to the discovery frame.” Yet §6.5 defines separate
outcome states (`suggestive` vs `unstable`) that claim exactly that distinction.
The two arms make it decidable:

| Arm | Frame | Sampling | Answers |
|---|---|---|---|
| **R — replication** | Same primary frame as discovery | Fresh trials, frozen analysis | Was the discovery effect real, or selection noise? |
| **G — generalization** | Held-out Lane B frames | Fresh trials, frozen analysis | Does it survive a different way of asking? |

Both arms are collected in the same interleaved time blocks. R passing and G
failing is a real, reportable finding — a frame-specific effect — and it is not
the same thing as R failing, which is a false positive. Splitting costs one
extra frame's worth of trials on the confirmation partition only, which is the
smallest partition.

**Cue-set growth interacts with the fixed landscape.** Steps 1–4 above grow the
cue set by two orders of magnitude, while §7.1 promises “one fixed coordinate
system” and §10.3 makes a cue's position permalinkable. A refit projection moves
every previously published cue and silently breaks both promises. §7.1 resolves
this by requiring a **persistable linear transform**; the constraint is recorded
here because it is this section's growth schedule that creates it.

### 5.5 Model identity and capability audit

Research mode refuses moving aliases such as `latest`. Before the cost gate,
`nebulai behavior plan` must:

1. query the provider's available model/capability surface;
2. resolve and require an exact dated Grok release;
3. record requested and served model IDs;
4. verify the supported common sampler parameters;
5. record whether reasoning can be disabled and, if not, its exact effort;
6. ensure tools/search are absent;
7. confirm that response IDs and usage are capturable, and record whether
   `system_fingerprint` is populated (see below);
8. resolve GPT-2 to a Hugging Face commit SHA and record tokenizer files.

An exploratory mode may allow a moving alias, but its results can never receive
the “confirmed” status. If a fingerprint changes during a strict study, the
runner stops or partitions the data; it never pools both deployments.

#### 5.5.1 Fingerprint is optional-if-absent, and a canary carries the load

`system_fingerprint` is a provider-controlled, frequently-omitted field on
OpenAI-compatible endpoints. A gate requiring it to be **present on every API
trial** is therefore not a study-quality gate at all — it is a bet that a
specific vendor keeps populating a specific optional key, and it is unpassable
by construction if they do not.

The requirement is restated in two parts:

- **Optional-if-absent.** The manifest records `fingerprint_available: true |
  false` once, at plan time. If `false`, the study proceeds and every artifact
  carries the flag; a missing field never fails a trial. If `true`, the field is
  captured on every trial and a *change* still stops or partitions the run, as
  above.
- **Canary drift probe (required either way).** One fixed prompt at fixed
  sampler settings is issued **once per time block** for the whole study. Its
  outputs form a time series; a distribution shift in the canary is the
  deployment-drift signal. This is strictly stronger than the fingerprint: it
  measures behavior rather than a self-reported label, it works when the field
  is absent, and it catches silent serving changes that leave the fingerprint
  untouched. Canary trials are excluded from every semantic metric and are
  reported in the study's own diagnostics.

### 5.6 GPT-2 execution backend

The repository's `GPT2Numpy` forward pass remains the reference implementation
for numerical conformance and later Internals links. A large repeated study
needs batched autoregressive sampling, so the Behavior package adds an optional
`behavior-local` dependency group using a pinned Transformers/PyTorch stack,
or an equivalently tested batched backend.

Acceptance requires its next-token logits to match `GPT2Numpy` on golden
prompts within a declared tolerance. The chosen backend, package versions,
device, precision, sampler seed, temperature, top-p, and checkpoint SHA are
recorded. This keeps the base Nebul.AI install lightweight and avoids making
the existing cloud pipelines depend on PyTorch.

### 5.7 Capability-control pair (required)

The A/A controls in §6.7 establish the **floor** — how far apart a model is from
itself. Nothing in the original design established a **reference for ordinary
between-model divergence**, so there was no basis for calling any GPT-2↔Grok
number large. Every effect was reported against zero, and against zero every
effect looks enormous.

The study therefore runs a third pair, **GPT-2-small vs GPT-2-XL**, through the
identical Lane A protocol:

- same family, same tokenizer, same pretraining corpus, same objective, both
  base completion models, neither instruction-tuned;
- the only substantial difference is **scale and therefore competence**;
- both are local — **zero API spend**, no provider ToS surface, no rate limits,
  and it can run entirely inside Phase 0/1 on the same `behavior-local` stack
  §5.6 already introduces.

This makes the capability confound measurable rather than merely acknowledged.
For each cue, the small↔XL contrast is the expected divergence when *only*
capability differs. The reportable quantity becomes:

> Is the GPT-2↔Grok contrast for this cue larger than the same-family
> capability contrast for **the same cue**?

A GPT-2↔Grok deviation that does not exceed its own small↔XL contrast is
attributable to scale and competence, and is **not** eligible to be described as
a difference in learned association. This is the single cheapest structural
improvement available to the study, and it converts §4.2.1's unbounded confound
into a measured, subtractable one.

Two honest limits. GPT-2-XL is still not instruction-tuned, so it brackets the
scale axis but not the post-training axis; and the pair shares a corpus vintage
that the Grok side does not. The arm therefore gives a **lower bound** on
capability-driven divergence — which is exactly the direction that keeps the
headline claim conservative.

---

## 6. Data treatment and statistical contract

### 6.1 Immutable raw evidence

Every trial retains:

- study, cue, frame, repeat, model, and time-block IDs;
- requested model, served model, checkpoint revision, and fingerprint;
- exact prompt text and prompt hash;
- sampler parameters and local RNG seed where supported;
- request/response IDs, timestamps, latency, usage, and estimated/actual cost;
- complete raw output;
- parsed ordered associates;
- parser version, validity flags, retry/failure reason, and moderation/redaction
  status.

Raw output is append-only. Corrections create derived normalization rows; they
do not replace what the model returned.

### 6.2 Deterministic normalization

Primary lexical normalization is deliberately minimal:

1. Unicode NFKC;
2. trim surrounding whitespace and list punctuation;
3. Unicode case-fold;
4. preserve slang, emoji, multiword responses, and out-of-vocabulary terms;
5. mark cue repetition and within-trial duplicates without silently replacing
   them.

A secondary lemma view may collapse obvious inflections, but it is never the
only evidence shown. The exact surface form always remains visible.

The claim about judging deserves precision, because §13 makes this document's
vocabulary a **test** and the loose version would fail it. “No LLM judge decides
whether two outputs mean the same thing” is not true as written: a sentence
embedder *is* a learned judge of semantic equivalence, and it is the substrate
of the primary metric (§6.3). The exact statement is:

> **No generative model adjudicates equivalence.** A pinned encoder defines the
> semantic metric, its identity and revision are declared in the manifest, and
> exact surface forms are always shown alongside every embedding-derived claim.

The distinction that matters is not “learned vs not learned” — it is that the
judge is **fixed, versioned, and inspectable** rather than a prompted model
whose verdicts drift with its own deployment.

### 6.3 Trial representation

For the semantic test, each valid trial becomes one vector:

1. embed each of its three associates with a pinned neutral text embedder;
2. L2-normalize each vector;
3. combine them using frozen reciprocal-rank weights;
4. L2-normalize the resulting trial vector.

The primary and confirmation embedders are selected before confirmation and
pinned to exact revisions. The existing project-supported MiniLM and
`mxbai-embed-large` families are candidates; Phase 0 chooses exact revisions
based on coverage and calibration, not which one produces the most exciting
GPT-2/Grok contrast.

#### 6.3.1 An embedder exists; the Behavior study still needs its own

**Correction, 2026-08-13.** An earlier draft of this section asserted that the
repository had no working embedder, that `nebulai compare` was broken, and that
standing one up was a blocking “task zero.” That was wrong. The claim traced to
`embed.py`'s 2026-08-12 note, which probed ollama's **stock port 11434** — a port
the M4 worker has never used. `docs/M4-OLLAMA-HANDOVER.md` has recorded
`OLLAMA_HOST=0.0.0.0:11435` since 2026-08-04. Verified working:

```
GET  http://<m4-host>:11435/api/tags    -> mxbai-embed-large:latest
                                           334M params, F16 GGUF, 1024-dim
GET  http://<m4-host>:11435/api/version -> 0.23.1
GET  http://<m4-host>:8100/v1/status/ollama -> running:true, port:11435
embed_texts(...)                        -> (n,1024) float32, L2-normalized,
                                           cos 0.70 related / 0.31 unrelated
```

So `compare` and the api-token front-end have a working neutral space today, and
nothing in §6 is *blocked*. `embed.py`'s docstring, the CLI defaults, and the
error path have been corrected accordingly (§16.3).

That removes the blocker. It does **not** remove the requirement, because two of
the three original arguments survive intact and one of them got stronger.

**A mutable ollama tag still cannot satisfy “pinned to exact revisions.”** The
`mxbai-embed-large` tag is an **F16 GGUF** build of
`mixedbread-ai/mxbai-embed-large-v1` — not bit-identical to the fp32 repository,
and re-pointable upstream at any time. (The earlier draft called this “quantised”
and claimed rare and slang tokens degrade most, feeding §6.4.4. At F16 — half
precision, not a 4-bit quant — that effect is small, and the honest objection is
the mutable tag and the non-reproducible artifact, not vocabulary-dependent
quantisation error. Corrected here rather than left overstated, since §13 makes
this document's own accuracy a test.)

**The LAN routing constraint is now more pressing, not less.** The M4 worker
carries a standing rule that NSFW content is never routed to it. This study
deliberately includes “daddy,” slang, and relationship/identity cues, and it
embeds **raw model outputs** — unreviewed by construction. While the box appeared
unreachable, that risk was theoretical. **It is reachable, it is the documented
host, and it is now the path of least resistance** — so an implementer will
default to it and violate the rule silently, on an unpredictable fraction of
trials, with no point at which anyone decided to. Cluster *names* (what `compare`
sends) are curated and fine; raw trial outputs are not the same thing.

**Therefore, unchanged: run the Behavior embedder in-process.**
`sentence-transformers` pinned to a Hugging Face **commit SHA**, fp32,
deterministic, under the optional `behavior-local` group §5.6 introduces for
GPT-2. The SHA, dtype, pooling mode, normalization, and library versions go in
the manifest. This satisfies the pinning requirement, keeps unreviewed content
off the LAN box, and removes a network dependency from the middle of a
long-running statistical study — a study that will issue far more embed calls
than `compare`'s few hundred cluster names.

What changes is the **status**: this is now a normal Phase 0 deliverable sized in
hours, not a blocking prerequisite. The existing LAN embedder remains available
for `compare`, for exploratory Behavior runs explicitly marked ineligible for
confirmation, and as a cross-check against the in-process vectors.

### 6.4 Per-cue metrics

There is no composite “mind distance.” Metrics stay separate.

| Metric | Role |
|---|---|
| **Normalized contrast `Δ̂`** | **Preregistered primary effect.** Between-model MMD² minus the mean of the two within-model split-half MMD². Ranks, gates, and BY correction all use this. Defined in §6.4.1. |
| **Raw MMD² over trial vectors** | Secondary, always displayed beside `Δ̂`, never the rank key and never a gate input. Gaussian kernel bandwidth is calibrated once on Phase 0 and then frozen. |
| **Within-model split-half MMD²** | Per model, the self-vs-self floor. Reported for both models; it is the subtrahend in `Δ̂` and is meaningful on its own. |
| **Capability-contrast reference** | The same `Δ̂` computed for the GPT-2-small ↔ GPT-2-XL pair on the same cue (§5.7). Provides the “is this bigger than scale alone?” comparison. |
| **Permutation p-value** | Cue-wise evidence against exchangeable model labels, using the frozen `Δ̂` statistic and permutations restricted within collection time blocks. |
| **Bootstrap interval** | Uncertainty around the primary effect and selected secondary metrics. |
| **Location/dispersion decomposition** | Splits `Δ̂` into “they centre on different things” vs “one is simply more scattered.” Defined in §6.4.2. |
| **Jensen–Shannon divergence (bias-corrected)** | Normalized associate-frequency difference, with the finite-sample correction of §6.4.3. |
| **Rank-biased overlap** | Top-weighted overlap between each model's ranked associates, at a **declared** `p`. See §6.4.3. |
| **Within-model split-half stability** | Whether each model's own profile is stable enough to compare. |
| **Informativeness** | Distinct associate types and associate entropy per model; degeneracy detectors. Gate input, not decoration — see §6.5. |
| **Valid-output/compliance rate** | Separate protocol behavior; never folded into the semantic metric — but its **between-model difference** is a gate (§6.5). |
| **Fragmentation / OOV differential** | Per-model subword-fragmentation ratio and embedder-space neighbour distance; the *differential* is flagged. See §6.4.4. |
| **Prompt-frame and embedder agreement** | Confirmation robustness, shown explicitly. |

The ranked “largest deviations” list first filters by the confirmation gate,
then sorts the surviving cues by **`Δ̂` measured on the confirmation partition**
(§6.5.3). It does not multiply effect, confidence, and lexical novelty into an
opaque score.

#### 6.4.1 The primary statistic is a normalized contrast, not raw MMD²

§4.2's H2 states the target as *“between-model separation larger than calibrated
same-model split-half separation.”* Raw MMD² does not measure that; it measures
separation against zero. Operationalizing H2 as raw MMD² makes the stated
hypothesis and the tested quantity **two different claims**, and — per §4.2.1 —
the difference is exactly where the capability confound lives.

The primary effect for a cue is therefore

```
Δ̂ = MMD²(A, B) − ½ · [ MMD²(A₁, A₂) + MMD²(B₁, B₂) ]
```

where `A₁/A₂` and `B₁/B₂` are randomized split halves of each model's own trials
at that cue, averaged over a frozen number of random splits to reduce split
noise. Equivalently, a standardized form (`MMD` divided by the pooled within-model
`MMD`) may be frozen at Phase 0; **which of the two is primary is declared in the
manifest before discovery and never changed afterwards.**

Three properties make this the right key:

1. **It is what H2 says.** A cue where both models are internally noisy and
   equally far apart scores near zero, which is the correct answer.
2. **It absorbs the capability confound at the cue level.** A base LM that is
   diffuse everywhere inflates its own within-model term, which subtracts out.
3. **It cannot be gamed by degeneracy in the direction we fear.** A degenerate
   model has a tiny within-model term, so `Δ̂` stays large — which is why
   degeneracy needs its own gate (§6.5) rather than being left to the statistic.

`Δ̂` can be negative. A negative value means the two models differ *less* than
each differs from itself, and it is reported as-is, never clipped to zero.

Raw MMD² remains in the artifact and on screen as a secondary figure, because a
reader comparing two studies will want it. It is never the rank key, never a
gate, and never the number in a headline.

#### 6.4.2 Location versus dispersion

MMD² responds to differences in **spread** as well as differences in **centre**.
Identical `temperature`/`top_p` does not equalize output entropy across a base
completion model and a post-trained chat model with mandatory reasoning and
unknown server-side penalties, so some part of every raw effect is “one model is
simply more varied.” Reported undecomposed, “they associate differently” and
“one is noisier” are indistinguishable — and they support completely different
conclusions.

Each cue therefore reports, per model:

- mean pairwise distance between that model's trial vectors (dispersion);
- distance between the two models' mean trial vectors (location);
- associate type-token ratio and associate entropy.

And the UI states which dominates. A cue whose `Δ̂` is dispersion-driven is
labeled as such and is **not** described as a difference in what the models
associate — because it is not one.

#### 6.4.3 Finite-sample bias in JSD and RBO points the same way as the confound

At 20–40 trials × 3 associates, each model contributes 60–120 associate tokens
per cue. Plug-in Jensen–Shannon divergence is **upward-biased at small `n`, and
the bias grows with the number of distinct types**. The more lexically diverse
model therefore inflates JSD for free — and under §4.2.1 that is systematically
the Grok side. RBO over unequal effective list lengths carries the same
asymmetry. Both biases push in the *same direction* as the capability confound
and as the dispersion effect in §6.4.2, so they compound rather than cancel.

Required:

- JSD uses a bias-corrected or Bayesian estimator (e.g. Miller–Madow-style
  correction, or a Dirichlet-prior plug-in), **or** matched-`n` subsampling to
  the smaller model's effective support with a bootstrap CI over subsamples.
  Which one is frozen at Phase 0.
- RBO's top-weighting parameter `p` is **declared in the manifest**. It was
  unspecified in the original design, and `p` alone can move an RBO score
  substantially, so leaving it to implementation time is a silent researcher
  degree of freedom.
- Every distribution-comparison statistic reports the effective `n` it was
  computed at, per model.

#### 6.4.4 Differential OOV and fragmentation

§3.2 and §6.2 are right to refuse to filter slang, emoji, and OOV terms — that
content is part of what is being measured. But an encoder represents exactly
those terms poorly: low-density regions, near-arbitrary directions. If one model
emits more of them, its trial vectors scatter more and the raw effect rises for
**non-semantic** reasons.

The good news, which belongs in the document because it bounds the concern: both
models are measured with the **same instrument**, so embedder bias is largely
common-mode and **cancels in a between-model contrast**. The residual risk is
specifically the *differential* — one model emitting materially more
fragmented/OOV material than the other.

So log per associate, and report per model:

- subword-fragmentation ratio under the embedder's tokenizer;
- nearest-neighbour distance in embedder space (a density proxy).

The **difference between models** is the flagged quantity. Cues above a
predeclared differential threshold carry an OOV-asymmetry flag on every surface
that shows their effect, and the flag is a documented reason a reader should
discount the cue — not a silent filter that removes it.

### 6.5 Multiple testing and confirmation states

Cue-wise discovery p-values are adjusted across the full declared discovery
family using Benjamini–Yekutieli at `q ≤ 0.05`. Confirmation uses fresh data
and applies BY again across the candidate family frozen before confirmation.
Phase 0 evaluates whether this conservative choice leaves enough power;
changing it later requires a new manifest. Bootstrap resampling is likewise
stratified by collection time block.

A cue is **confirmed deviation** only when all are true:

1. each model has the manifest's minimum number of valid independent trials;
2. within-model split-half reliability clears the calibrated floor;
3. **compliance parity** — `|parse_rate_A − parse_rate_B|` is within the
   predeclared threshold (§6.5.1);
4. **informativeness** — both models clear the degeneracy floor (§6.5.2);
5. the primary effect `Δ̂` exceeds the frozen smallest effect of interest, **and**
   exceeds the same cue's capability-contrast reference from §5.7;
6. BY-adjusted `q ≤ 0.05`;
7. **arm R** — fresh same-frame trials reproduce the effect (replication);
8. **arm G** — held-out prompt frames reproduce the effect (generalization);
9. the second embedder agrees on status and broad model-specific neighborhood,
   subject to the caveat in §6.5.4;
10. model identity, protocol, and canary drift checks stayed valid (§5.5.1).

Other states are:

- **suggestive:** discovery passed, confirmation has not;
- **frame-specific:** arm R passed and arm G failed — a real effect that does
  not survive a different way of asking. This is a finding, not a failure, and
  it is only nameable because §5.4 splits the two arms;
- **unstable:** prompt or embedder robustness failed;
- **capability-attributable:** all gates passed except 5's second clause — the
  effect is real but no larger than the same-family scale contrast, so it is not
  evidence of a difference in learned association;
- **no detected deviation:** adequately powered but below the effect/significance
  gate;
- **insufficient evidence:** too few or too-variable trials;
- **incomparable:** identity, protocol, data-integrity, or compliance-parity
  failure.

The UI never renders “same” merely because a low-powered test was not
significant. Equivalence requires a separately declared equivalence margin and
test.

#### 6.5.1 Differential compliance is a gate, not a footnote

§5.2 correctly keeps formatting compliance out of the semantic metric. That is
necessary and insufficient. If GPT-2 parses at ~40% and Grok at ~99%, then
GPT-2's surviving trials are a **non-random subsample of its behavior** — kept
precisely when a base LM happened to land in the answer format. Comparing
survivors is comparing “Grok in general” against “GPT-2 on the occasions it
behaved like an instruction-following model,” and that bias is present **even at
equal surviving `n`**. Gating on trial count, as the original design did, does
not touch it: it is missing-not-at-random, not a power problem.

Therefore:

- a predeclared maximum `|parse_rate_A − parse_rate_B|`, frozen at Phase 0;
- cues exceeding it are **`incomparable`** — never `confirmed`, and never
  silently included;
- cues near the threshold report **worst-case (Manski-style) bounds**: recompute
  the effect under the best and worst plausible assignments for the missing
  trials, and show the interval. If the bound spans the effect floor, the cue
  cannot be confirmed regardless of its point estimate;
- the parse-rate pair is displayed wherever a cue's effect is displayed.

#### 6.5.2 The reliability floor needs an informativeness floor beside it

Gate 2 requires each model's own profile to be stable. That is necessary and, on
its own, actively misleading: **a model that copies the few-shot exemplar every
single time has perfect split-half reliability and zero semantic content.**
Degeneracy is maximally reliable. §6.2 marks cue repetition and within-trial
duplicates but nothing gated on those marks, so the most degenerate cues could
pass gate 2 most easily.

The informativeness floor requires, per model per cue:

- distinct associate types above a predeclared minimum;
- associate entropy above a predeclared minimum;
- and explicit detectors, each a **gate input**, for
  - cue echo (the model returns the cue),
  - few-shot exemplar echo (the model returns the example's answers),
  - within-trial duplicates,
  - verbatim prompt copy or continuation.

A cue failing the floor for either model is `insufficient evidence`. The
detectors' rates are reported per model, because a large asymmetry in them is
itself the capability confound made visible.

#### 6.5.3 Displayed effects come from the confirmation partition only

The ranked list sorts survivors by the primary effect — but **which partition
that effect is measured on was ambiguous**, and it matters. Discovery effects
are selection-inflated by construction: a cue enters the list partly because
noise pushed it up, so its discovery magnitude is biased upward (the winner's
curse). Publishing that number overstates every headline.

Rule: **all displayed magnitudes, intervals, and rankings come from the
confirmation partition.** Discovery effects appear only as the reason a cue was
tested, are labeled as such, and are never rendered as magnitude.

#### 6.5.4 “Second embedder agrees” is a weak test, and the UI must say so

Gate 9 reads as robustness. It is close to a free pass: MiniLM and
`mxbai-embed-large` are both English sentence-transformer families trained on
heavily overlapping web data with similar objectives. Their agreement is nearly
guaranteed and carries little independent information.

Two acceptable resolutions, and the manifest must declare which is in force:

1. **Strengthen the test** — make the second representation genuinely different
   in kind: a lexical/PPMI association space, or an encoder with a different
   training objective. Then agreement is real evidence.
2. **Downgrade the claim** — keep the two similar encoders, and state the
   finding as *“not specific to one of two similar encoders,”* which is what it
   actually shows. The UI copy must match this wording; it may not say
   “embedder-independent.”

### 6.6 Global correlation

The study reports model similarity without turning it into a winner score:

- macro-averaged rank correlation of per-cue associate weights over each cue's
  union support, so a huge number of shared zero cells cannot inflate it;
- rank correlation of cue-to-cue representational dissimilarity matrices
  (behavioral RSA);
- linear CKA of aligned cue association representations;
- weighted node/edge overlap of directed association graphs;
- the same statistics by preregistered cue domain and with leave-one-domain-out
  sensitivity.

Each statistic is labeled **behavioral association correlation**. None is
called internal representational similarity for Grok.

### 6.7 Calibration controls

Phase 0 must include:

- GPT-2 A/A and Grok A/A pseudo-comparisons from randomized split halves;
- shuffled model labels to verify the nominal false-positive rate;
- known prompt perturbations as sensitivity controls;
- parser goldens covering Unicode, slang, emoji, multiword answers, refusal,
  verbosity, duplicate answers, and malformed delimiters;
- repeated analysis with both neutral embedders;
- time-block checks for provider drift;
- rank-stability curves versus trial count;
- **the achievable permutation p-floor** (§6.7.1);
- **a construct-validity positive control** (§6.7.2);
- **the capability-contrast arm** of §5.7, run end-to-end on the calibration cue
  set, since it is local and free and its output is an input to gate 5.

#### 6.7.1 Check that confirmation is reachable at all

§6.4 restricts permutations to within collection time blocks. That is the right
call for drift, and it has a consequence the original design did not check: it
**shrinks the permutation space**. With `k` blocks and small within-block `n`,
the number of distinct label assignments is bounded, so the **minimum achievable
p-value** is bounded away from zero. If that floor sits above the BY-adjusted
threshold, then **no cue can ever be confirmed, at any effect size** — the study
would run to completion and return nothing, for a purely combinatorial reason.

Phase 0 must therefore:

- predeclare a minimum number of within-block trials per model per cue;
- compute the achievable p-floor from the resulting block structure;
- verify it clears `q` with margin, and **report it in the calibration output**
  so the number is on the record rather than assumed.

If the floor does not clear, the fix is more within-block trials or fewer, larger
blocks — a design change made **before** data collection, not a threshold
loosened afterwards.

#### 6.7.2 A positive control, or the instrument is unvalidated

Human norms are correctly refused as ground truth (§3.2) and SWOW is
license-gated (§5.3). The result is that the study, as originally designed, had
**no external anchor whatsoever** — nothing established that the pipeline
measures word association at all rather than formatting, verbosity, or prompt
echo. A null result and a broken instrument would be indistinguishable.

The control is small, license-safe, and hand-authored: a set of near-universal
associations — `hot→cold`, `salt→pepper`, `cat→dog`, `king→queen`, `day→night`,
`black→white` and similar — that any competent association-producing system
should recover. The requirement is **within-model**, not cross-model: each
model's own association network must place the expected associate among its top
responses for these cues at a predeclared rate.

This is a **Phase 0 gate**. If a model fails it, that model's outputs are not
association data and no divergence number computed from them means anything. A
partial failure is informative too: if GPT-2 fails the control where Grok passes,
that is the capability confound of §4.2.1 showing up in the most direct possible
form, measured before a single expensive trial is spent.

---

## 7. Visualization contract

### 7.1 Overview: a fixed cue landscape

All cues occupy one fixed coordinate system built from the cue words themselves
with a pinned neutral embedder. The models never receive separate UMAPs that a
viewer might mistake for directly comparable internal geometry. **This is the
single best idea in the visualization design** — it sidesteps the classic trap of
comparing two independently-fit projections, which is the same trap the existing
Compare view's `native` state has to defend against at
`src/nebulai/backend/compare.py:190`.

Each cue is a paired glyph:

- location = fixed cue-space location;
- size = confirmed `Δ̂` effect, encoded as **area**, not radius;
- split color/shape = the two model identities, never good versus bad;
- ring/pattern = evidence status;
- label = cue text;
- hidden or explicitly crossed state = incomparable, never zero.

A ranked rail beside the map shows the largest confirmed deviations. Search and
domain/status filters update both surfaces. The map answers **where in the cue
landscape differences concentrate**; the rail makes exact ranking and keyboard
navigation practical.

The MVP does not draw arbitrary physical distance between two whole model
clouds. An optional later vector-field view may project pooled response
centroids in the neutral embedding, but must label that direction as a display
projection rather than the statistical test.

#### 7.1.1 “Fixed” and “UMAP” cannot both be true — the landscape is PCA

Two commitments in this document collide head-on:

- §7.1 and §13 promise **“one fixed coordinate system”** and “fixed cue
  coordinates,” and §10.3 makes a cue's position permalinkable
  (`#page=behavior&study=<id>&cue=daddy`);
- §5.4 **grows the cue set** from 100 → ~300 → ~1,000 → ~11,500.

UMAP cannot honour both. Adding cues means refitting, and a refit **moves every
existing cue** — so a permalink published at pilot scale points at a different
place after expansion, with no error and no notice. Worse, this repository has
already established that the escape hatch does not exist: `reduce_vectors` fits
UMAP as a local that is garbage-collected on return, and `reduced.npz` stores
**coordinates, not a transform**, so there is no `reducer.transform()` with which
to place a new cue out-of-sample. `docs/SESSIONSEER-LIVE.md` §7 records the
project refusing a finished, shipped-quality feature (the Atlas trail) over
exactly this, on the grounds that *“a centroid in the source space and an
out-of-sample UMAP embedding disagree, sometimes a lot — UMAP's whole job is a
non-linear rearrangement.”* That precedent is binding here.

There is a second, independent problem at the same spot, and it is measured
rather than theoretical. `src/nebulai/backend/validate.py:31-40` records:

> The null baseline is only meaningful AT THE MAP'S REAL SCALE. On a few hundred
> points UMAP manufactures clean separable islands out of shuffled noise, and the
> null can out-score the real map (measured: silhouette 0.88 on shuffled data vs
> 0.43 on three genuinely separated gaussian blobs, n=180).

Calibration ships 100 cues and the pilot ~300. **Both sit inside the regime where
this project has empirically shown UMAP produces convincing islands from pure
noise.** That is §13's own “beautiful but misleading cloud” risk arriving through
a door that row does not cover: the row guards *cross-model* geometry, and this
is *small-n* geometry.

**The cue landscape is PCA at pilot scale.** Specifically:

1. **PCA is the pilot-scale projection.** Deterministic; no seed, no
   `n_neighbors`, no `min_dist`; it cannot manufacture islands out of noise
   because it is a linear projection; and its axes are reportable by explained
   variance, so the display has something honest to say about what a direction
   *is*. Critically, **PCA is a persistable linear transform** — store the mean
   vector and the component axes and a cue added in a later study revision lands
   in the *same* space without moving anything already published. It is the only
   option that makes “fixed coordinate system” literally true rather than
   aspirationally true.
2. **Reuse the existing implementation.** `_pca_rows()` at
   `src/nebulai/backend/interp/bundles.py:146-166` already performs exact PCA by
   covariance eigendecomposition in float64 with **deterministic axis signs**,
   returning `(coords, evr, total_var)`. Adopt it rather than writing a second
   one. The Internals bundles also stamp a `"quantity"` string describing what a
   projection physically *is* (`bundles.py:221`); the cue landscape does the
   same — `"quantity": "PCA projection of cue-word embeddings in
   <embedder>@<sha>"` — and the viewer renders that string rather than inventing
   a caption.
3. **UMAP is gated on scale, not on preference.** It becomes available only at a
   cue count where `nebulai validate`'s null baseline is a genuine floor, and
   only after that cue set actually clears the null. If it ever ships, the fitted
   reducer must be **persisted** — otherwise the “fixed” promise breaks a second
   time, in the same way, for the same reason.
4. **Trustworthiness ships with the coordinates.** It is computed anyway; it
   belongs in `behavior.json` and on screen (§9.5), not only in a CLI report.
   A projection that reorders neighbourhoods should say so on the surface where
   people read neighbourhoods.

#### 7.1.2 Colour by preregistered domain, never by a discovered cluster

The temptation is to run HDBSCAN on the cue landscape and colour by cluster. It
must not happen, and this repository's own validation output says why.
`validate.py:41-46` records that seed stability is dominated by
`cluster_selection_method`: on three clean gaussian blobs, `eom` recovers 3
clusters at ARI 1.00 while `leaf` — **the project default** — splits them into 8
and drops to ARI 0.37. The README's validation table reports seed ARI of
**0.46–0.62** across the shipped maps, with the conclusion that *“roughly half of
each partition is seed-dependent. Individual cluster boundaries are not stable
findings; the gross layout is.”*

So clustering the cue landscape would generate unstable groups at exactly the `n`
where §7.1.1 already says the geometry is least reliable, and then colour would
carry the instability into every screenshot.

The plan already has the right answer and does not need a discovered one:
**§5.3's preregistered cue strata.** Colour by those. They are declared before
data collection, stable by construction, and they are what the study is actually
stratified on. §7.3's rule about labeling “a semantic group generated for
readability” then does not apply to this surface at all, because nothing on it is
generated.

#### 7.1.3 Glyph size and crowding

Two rules, because encoding an effect as size interacts badly with a projection:

- Effect is encoded as **area**, not radius. Radius-proportional encoding
  exaggerates by the square and is a standard chart defect.
- Size creates **false density**: in a crowded region, large glyphs merge into an
  apparent mass that reads as “this area of meaning is where divergence lives”
  when it may be one large cue beside many small ones. The view needs a
  declutter rule — a size cap tied to local density, or explicit aggregation at
  low zoom with the count shown — and the rail (which is exact and unaffected by
  crowding) remains the authoritative ranking.

### 7.2 Cue inspector

Selecting “daddy,” or any cue, opens a deterministic two-sided association
neighborhood:

- cue in the center;
- GPT-2-only associates on one side;
- Grok-only associates on the other;
- shared associates in the center channel;
- **indeterminate associates in a fourth, visibly de-emphasized channel**
  (§7.2.1);
- directed edge width = observed rank-weighted frequency;
- edge stability pattern = split-half reliability;
- no force-layout distance is presented as effect size.

#### 7.2.1 Model-specific is a statistical claim, so it needs a threshold

Three hard channels render a **continuous** quantity as a **categorical** one. An
associate appearing 1/40 times for one model and 0/40 for the other is noise —
but under a bare set-difference rule it lands on the “model-specific” side at
full visual weight, in the most rhetorically loaded position on the screen, and
a reader takes it as a finding.

§6.5 already refuses to render “same” from an underpowered null. This is that
same principle one level down, and it was not applied.

Classification is therefore by **frequency-difference confidence interval**
(Wilson, or bootstrap over trials) rather than by presence:

- **shared** — the CI for the difference includes zero and both counts clear the
  minimum;
- **model-specific** — the CI **excludes zero**;
- **indeterminate** — too few observations to separate the two, rendered in the
  fourth channel with reduced weight and an explicit count.

The `indeterminate` channel is not a cosmetic hedge; it is where most rare
associates will land at 20–40 trials, and showing that honestly is the point.

The same defect exists today in the shipped Compare panel's Jaccard overlap,
which is a set operation on thresholded membership with no uncertainty attached.
It is out of scope for this feature, but it is the same bug and should be fixed
with the same reasoning if that panel is ever revisited.

The inspector has three sections:

1. **Neighborhood** — the visual association graph and top exact words.
2. **Contexts** — base and predeclared sense/prompt frames.
3. **Evidence** — the primary `Δ̂` with its interval, raw MMD² and both
   within-model split-half terms beside it, the §5.7 capability-contrast
   reference for the same cue, p/q, the location/dispersion split, bias-corrected
   JSD, rank overlap at the declared `p`, **both models' parse rates**, valid
   trial counts per arm (discovery / R / G), the informativeness and OOV-
   differential flags, prompt, model IDs, fingerprint-or-canary status, and
   expandable raw trials when connected to the local study store. A static export
   says that raw trials were withheld.

For GPT-2, **Open in Internals** creates or selects the exact prompt trace. The
Grok side states **behavioral evidence only — internals unavailable**.

### 7.3 Honest visual semantics

- Divergence magnitude is never encoded only by animation.
- Confidence is not a decorative glow.
- Shared gray nodes do not imply human correctness.
- Model colors are neutral and keep the same identity everywhere.
- A semantic group generated for readability is visibly labeled as a neutral
  embedder grouping.
- Every plot has a table/list alternative with the exact values.
- **Every positioned view states what position means, on screen.** Not in a
  tooltip, not in the Guide — in the view. If a coordinate is a projection, the
  view names the projection and says which of its properties are claims
  (neighbourhood) and which are not (axis direction, absolute distance, empty
  space). If a coordinate is an *arrangement* — rank order, authoring order,
  argument order — the view says so in those words, because the default reading
  of any 2-D layout is “near means similar,” and an arrangement violates it
  silently.
- Effect magnitudes are encoded by **area**, never radius (§7.1.3).

The last two rules are not new policy invented for this feature. They generalize
what the existing viewer already does and, in three places, what it had not yet
done — the audit and the fixes are recorded in §16.

---

## 8. UX information architecture

### 8.1 Placement decision

| Placement | Advantage | Failure | Decision |
|---|---|---|---|
| Add controls to Map → Compare | No new top-nav item | Mixes label-space cloud overlap with trial-level evidence, provenance, costs, uncertainty, and confirmation. Normal cloud use becomes intimidating. | Reject |
| Put it under Internals | Sounds “research-grade” | Grok has no comparable internals; this would imply a mechanistic comparison that does not exist. | Reject |
| Create a separate third app | Maximum separation | Fragments model research, duplicates navigation/settings, and weakens Map/Internals cross-links. | Defer unless Behavior later becomes a multi-product lab. |
| Add Nebul.AI **Behavior** page | Clear promise, isolated complexity, direct cross-links, still one model-research instrument | Adds a fourth Nebul.AI nav pill | **Accept** |

The nav label is **Behavior**, not “Divergence,” because divergence is an
empirical result rather than a foregone conclusion. The page title describes the
analysis the user is opening.

One wayfinding detail: a pill reading **Behavior** and a title reading **Semantic
divergence** share no word, so nothing confirms to a user that they arrived where
they clicked. The title is therefore **“Behavior — semantic divergence”**, which
keeps the pill's promise honest and still names the analysis. The reasoning above
is preserved exactly; only the shared token is added.

### 8.2 Progressive-disclosure layout

```
┌ Semantic map | Behavior | Internals | Guide ─────────────────────────────┐
│ Behavior — semantic divergence  [study: GPT-2 ↔ Grok] [How to read] [New] │
├ global correlation ─ confirmed cues ─ protocol status ─ reliability ────┤
│ Search cues…  [domain] [status] [table/map]                              │
├────────────────────────────────────────────┬─────────────────────────────┤
│                                            │ Largest confirmed deviations│
│             fixed cue landscape            │ 1  …                        │
│                                            │ 2  daddy                    │
│                                            │ 3  …                        │
├────────────────────────────────────────────┴─────────────────────────────┤
│ Selected cue: Neighborhood | Contexts | Evidence                         │
└──────────────────────────────────────────────────────────────────────────┘
```

Default users see a precomputed, read-only study. They do not see temperature,
FDR, fingerprints, API keys, or retention settings until they ask to create or
audit a study.

Internal page structure stays small:

- **Results** is the default overview and cue inspector.
- **Runs** contains study progress, provenance, failures, cost, and exports.
- **How to read** deep-links to a new Behavior section in Guide.

“Inspect” is a selected state inside Results, not another permanent tab.
“Methods” belongs in Guide rather than creating a fourth layer of navigation.

### 8.3 Primary flows

#### Flow A — find where the models deviate

1. Open Behavior.
2. A shipped/pinned study loads with confirmed cues only by default.
3. Read global association correlation and the ranked deviations rail.
4. Select a cue to inspect exact associates and evidence.
5. Optionally reveal suggestive/unstable cues with a status filter.

Target: the first meaningful result is visible without configuration and within
one artifact load.

#### Flow B — inspect a known cue such as “daddy”

1. Search “daddy.”
2. Open its inspector regardless of status.
3. See shared and model-specific associations.
4. Open Contexts to compare the base cue with predeclared family/slang frames.
5. Open Evidence to see whether the visual is confirmed, merely suggestive, or
   underpowered.

Target: the requested cue is reachable in at most two actions. The UI never
upgrades it because the user searched for it.

#### Flow C — run a new study

1. Select **New study**.
2. Choose exact model pair; capability audit rejects aliases or marks the run
   exploratory.
3. Choose a cue suite and optional matched-control packs.
4. Choose the calibrated preset; advanced sampling/statistical settings stay
   collapsed.
5. Review exact protocol, estimated calls/tokens/cost/time, storage, and the
   immutable manifest hash.
6. Explicitly approve the paid run.
7. Monitor resumable progress in Runs.
8. Discovery finishes; the UI proposes a separate confirmation run.
9. Confirm, export, and share a compact read-only artifact.

The cost/manifest review is a full step, not a transient toast.

#### Flow D — move between existing clouds and Behavior

- Map → Compare: **Open behavioral comparison** carries selected models.
- Atlas search result: **Inspect this cue in Behavior** when a normalized cue
  match exists.
- Behavior cue: **Open GPT-2 in Internals** for the exact visible prompt.
- Behavior study: **Open structural cloud comparison** returns to existing
  Compare without changing its data.

### 8.4 Settings ownership

Add a **Behavior** tab to `viewer/src/chrome/SettingsPage.tsx`, shown only while
the Nebul.AI instrument is active and backed by the Behavior slice. It is the
canonical home for feature-wide defaults:

- local Behavior runner endpoint and health;
- XAI credential status reported by the local runner. The key itself stays in
  the runner's `XAI_API_KEY` environment and never enters browser state, local
  storage, logs, or a static export;
- default cost ceiling and concurrency;
- default trial preset and confirmation policy;
- primary/secondary pinned embedders;
- raw-output retention/redaction policy;
- reduced-motion and sensitive-content display defaults.

Run-specific choices remain visible in the New study flow and are copied into
the immutable manifest. Settings provide defaults; they never mutate an
existing study.

### 8.5 Responsive behavior

- Desktop: cue landscape and ranked rail side by side; inspector docks below or
  in a resizable sheet.
- Tablet: ranked rail becomes a collapsible side sheet.
- Mobile: ranked/table view is the default, map is optional, and cue evidence
  opens full-screen. The research workflow remains usable without a 3-D canvas.

### 8.6 Accessibility

- Model identity uses color plus label, side, and shape.
- Evidence status uses text plus ring/pattern, not opacity alone.
- Every map has a sortable table equivalent.
- Search, ranked cues, inspector tabs, raw trials, and cross-links are fully
  keyboard reachable with visible focus.
- Screen readers receive the cue, status, effect, uncertainty, and top
  associates as one concise summary.
- Reduced motion uses crossfades or instant state changes; no morph is required
  to understand a difference.
- Live run updates use a throttled polite region; errors receive focus only
  when user action is required.
- Raw sensitive outputs are collapsed with a content notice, but aggregates are
  not silently censored.

### 8.7 Required edge states

| State | User-facing behavior |
|---|---|
| No Behavior artifact | Normal map still works. Behavior offers a sample study or New study. |
| Static deployment | Results are read-only; New study explains that a local runner is required. |
| Missing XAI credential | Setup stops before model calls and links to the Behavior Settings tab. |
| Moving model alias | Strict run is refused; exploratory run is visibly ineligible for confirmation. |
| Fingerprint/model drift | Run pauses, affected block is marked incomparable, and pooling is disabled. |
| Rate limit/provider outage | Exponential backoff, visible next retry, safe pause/resume, no duplicate completed trial. |
| Partial run | Available evidence is labeled partial; significance/confirmation stays disabled until minimum n. |
| Parse failures | Raw output remains inspectable; validity counts and failure reasons are shown separately. |
| Insufficient power | “Insufficient evidence,” never “same.” Offer the precomputed number of additional trials. |
| No confirmed deviations | Say so directly, show the study's power, and allow suggestive results to be revealed intentionally. |
| Embedder disagreement | Mark unstable and show both analyses; do not choose the more dramatic one. |
| Corrupt/unknown schema | Refuse the artifact with expected/found schema versions; do not partially render. |
| Sensitive raw content | Collapsed by default, revealable only from the local study store, and excluded from the standard static export. |

---

## 9. Backend architecture and contracts

### 9.1 Additive package

Create `src/nebulai/behavior/`:

| Module | Responsibility |
|---|---|
| `contract.py` | Versioned study, model, cue, trial, metric, and export schemas. |
| `protocol.py` | Canonical prompt frames and immutable protocol hashing. |
| `adapters/base.py` | Capability and sampling interface shared by local and API models. |
| `adapters/gpt2.py` | Pinned local GPT-2 sampling and conformance checks. |
| `adapters/xai.py` | xAI model audit and Chat Completions sampling with identity/fingerprint capture. |
| `store.py` | SQLite migrations, append-only trials, leases, retries, and resumability. |
| `runner.py` | Randomized time blocks, concurrency, backoff, budget enforcement, and progress events. |
| `server.py` | Loopback-only HTTP/SSE control and status surface for the viewer; credentials remain server-side. |
| `normalize.py` | Deterministic parsing and primary/secondary normalization. |
| `embed.py` | Pinned neutral embedder adapters and cached trial representations. |
| `stats.py` | Reliability, MMD, permutation, bootstrap, BY, JSD, RBO, RSA, and CKA. |
| `analyze.py` | Discovery/confirmation orchestration and state assignment. |
| `export.py` | Compact viewer artifact, CSV/report, provenance, and redaction. |

Do not put these trial records into `Units` or route them through
`reduce → cluster → name`. A repeated behavioral experiment has a different
unit, provenance model, and validation contract.

### 9.2 CLI

Add a nested command group:

```text
nebulai behavior plan <study.yaml>
nebulai behavior run <study-id>
nebulai behavior analyze <study-id>
nebulai behavior confirm <study-id>
nebulai behavior export <study-id>
nebulai behavior inspect <study-id> --cue daddy
nebulai behavior serve
```

- `plan` resolves model identities, runs capability checks, estimates
  calls/tokens/cost/time/storage, and writes the immutable manifest.
- `run` is idempotent and resumable.
- `analyze` may produce discovery/suggestive states but not confirmation.
- `confirm` consumes the held-out partition under the frozen manifest.
- `export` writes the static contract and excludes raw outputs by default.
- `inspect` prints exact evidence without requiring the viewer.
- `serve` binds to loopback by default, exposes health/progress/control and
  local raw-evidence reads, enforces origin checks, and never returns secrets.

The existing house rules in `src/nebulai/llm.py`—no silent model substitution
and no silent budget downgrade—also apply here.

### 9.3 On-disk layout

```text
out/behavior/
  index.json
  <study-id>/
    manifest.json
    study.sqlite
    behavior.json
    report.json
    cues.csv
    provenance/
      prompt-frames.json
      environment.json
```

`study.sqlite` is local research evidence and is not copied to the static
viewer. `behavior.json` is a compact, redacted, immutable projection.

### 9.4 Manifest minimum

The manifest includes:

- schema/study ID, creation time, git commit, analysis version, and protocol
  hash;
- cue inventory hash, source, license, strata, discovery/confirmation split,
  and control packs;
- full prompt frames and hashes;
- exact requested/served model IDs, GPT-2 SHA/tokenizer, Grok dated release,
  capability result, and allowed fingerprint;
- temperature, top-p, output cap, reasoning effort, trial count, RNG rules, and
  time-block randomization;
- normalizer/parser versions;
- primary/secondary embedder IDs and **commit SHAs**, dtype, pooling, and
  normalization (§6.3.1), plus which resolution of §6.5.4 is in force;
- MMD kernel/bandwidth, permutations, bootstrap count, effect floor, reliability
  floor, multiple-testing family, and q threshold;
- **which form of `Δ̂` is primary** (difference or standardized ratio), and the
  number of random split-half draws averaged over (§6.4.1);
- **the informativeness floor** — minimum distinct types and entropy — and the
  degeneracy detector set (§6.5.2);
- **the compliance-parity threshold** (§6.5.1);
- **the OOV/fragmentation differential threshold** (§6.4.4);
- **RBO's `p`** and the chosen JSD bias correction (§6.4.3);
- **the capability-control pair identities and SHAs** (§5.7);
- `fingerprint_available`, and the canary prompt with its sampler settings
  (§5.5.1);
- the positive-control cue/associate list and its pass rate (§6.7.2);
- the measured **p95 reasoning-token** figure the budget was computed from
  (§5.1);
- budget ceiling, estimated call/token/cost/time/storage envelope;
- raw-data retention and redaction policy.

### 9.5 Static viewer contract

`behavior.json` includes:

- `meta`: schema, study, model pair, exact identities, protocol hash, dates,
  cue source, and evidence limitations;
- `projection`: **the projection's own provenance and quality** — `method`
  (`"pca"`), the `"quantity"` string described in §7.1.1, explained variance per
  retained axis, total variance, **trustworthiness**, the embedder ID+SHA the cue
  vectors came from, and whether the transform is persisted. The viewer renders
  this rather than composing its own caption. This mirrors what
  `src/nebulai/backend/interp/bundles.py:221` already does for Internals bundles,
  and it is the direct fix for a projection whose distortion was computed but
  never reached the browser;
- `global`: valid cue count, confirmation counts, association correlations,
  graph overlap, domain summaries, and reliability;
- `cues[]`: cue/domain, fixed 2-D/3-D anchor, evidence state, trial counts **per
  arm** (discovery / R / G), primary `Δ̂` with p/q/interval, raw MMD² and both
  within-model terms, the §5.7 capability-contrast value, the
  location/dispersion split, exact secondary metrics, reliability,
  informativeness, both parse rates, the OOV-differential flag, robustness, and
  top shared / model-specific / **indeterminate** associates;
- `runs`: discovery/confirmation completion, fingerprint-or-canary blocks,
  failures, token/cost totals, and export redaction;
- no API keys and no raw outputs by default.

Unknown schema versions fail closed. A cue's coordinates are meaningless without
`projection`, so an artifact carrying `cues[].anchor` without `projection` is a
**schema failure**, not a degraded render.

---

## 10. Viewer integration

### 10.1 State and navigation

Implementation touches, additively:

- `viewer/src/app/slices/shell.ts`: add `"behavior"` to `Page`; make
  `APP_PAGES.nebulai` equal
  `["map", "behavior", "interp", "guide"]`; update comments that assume three
  pages.
- `viewer/src/chrome/apps/nav.ts`: add the Behavior pill and remove
  three-pill assumptions.
- `viewer/src/chrome/apps/nebulai.tsx`: render `BehaviorPage`.
- `viewer/src/chrome/mount.tsx`: add the body page class so the atlas stage is
  hidden while Behavior owns its page.
- `viewer/tests/unit/app-pages.test.ts`: update the partition to seven pages
  and explicitly test four Nebul.AI versus three Seer pages.
- `viewer/src/app/slices/behavior.ts`: study index, selected study/cue,
  filters, display mode, run state, and defaults.
- `viewer/src/app/store.ts` and `viewer/src/chrome/state.ts`: compose and
  bridge the new slice.

### 10.2 Data and page components

Add:

- `viewer/src/data/behavior.ts`: schema validation, index/artifact loading,
  caching, and redacted raw-evidence endpoint types;
- `viewer/src/chrome/BehaviorPage.tsx`: page lifecycle and progressive
  disclosure;
- `viewer/src/chrome/BehaviorOverview.tsx`: summary, search/filter, ranked
  deviations, and table mode;
- `viewer/src/chrome/BehaviorCuePanel.tsx`: Neighborhood/Contexts/Evidence;
- `viewer/src/chrome/BehaviorRuns.tsx`: provenance, progress, cost, failure,
  confirm, export;
- `viewer/src/scene/drivers/BehaviorDriver.ts`: fixed cue landscape and exact
  picking, owned by the page like Internals owns its canvas;
- Behavior sections in `viewer/src/chrome/SettingsPage.tsx` and
  `viewer/src/chrome/GuidePage.tsx`.

The Behavior page lazy-loads its artifact and driver. A missing
`out/behavior/index.json` must not delay or break atlas boot.

### 10.3 Permalinks

Extend `viewer/src/chrome/urlState.ts` with page-scoped keys:

```text
#page=behavior&study=<id>&cue=daddy&section=evidence
```

Validate study and cue against the loaded artifact before applying them. Do not
write Behavior keys on Map, Internals, Guide, or Seer pages.

### 10.4 Cross-pipeline overlay rule

Behavior may supply a selected cue, confirmed status, and association evidence
to token, SAE, and neuron views. Those consumers may highlight or filter their
existing points. They must never recompute, align, or move existing geometry
from a behavioral score.

This rule is correct, and it is worth recording that it is **already physically
enforced**: because no fitted reducer is persisted anywhere in the pipeline
(`reduce_vectors` discards its UMAP fit — see §7.1.1 and
`docs/SESSIONSEER-LIVE.md` §7), nobody *could* place a cue onto an existing atlas
even if they decided to. A future implementer who finds this rule inconvenient
should know it is load-bearing rather than stylistic, and that the project has
already declined a completed feature over it.

Two constraints on the overlay itself:

**Tokenizer-aware matching, or no overlay.** The rule as originally written
highlighted a point “when a normalized cue match exists,” which is
underspecified in a way that produces wrong highlights. GPT-2 BPE means many cues
have **no single-token form at all**, and `" daddy"` (leading space) is a
different token from `"daddy"`. Matching must be exact and tokenizer-aware
against a declared variant list (bare, leading-space, capitalized), and when
there is no match the overlay is **absent**. Never a fuzzy match, never a
subword-prefix match — a highlight on `" dad"` because the cue was `"daddy"` is a
fabricated link presented with the same visual authority as a real one.

**Do not lend credibility to a map that has not earned it.** Overlaying
*confirmed* statistical findings onto an existing cloud implicitly borrows the
finding's confidence for the map underneath it. That map may not deserve it: the
shipped `gpt2-small · SAE features` map posts a **negative null margin
(−0.049)** — its cluster separation is not established against a column-shuffled
baseline. The overlay must therefore either be gated on the target map's own
validation margin, or must show the target map's margin alongside it. As of §16
this is directly available to the viewer, so there is no excuse for it being
implicit.

---

## 11. Implementation phases and gates

### Phase 0 — contract, capability audit, and calibration

**On the embedder.** A working LAN embedder exists (mxbai-embed-large on
`:11435`, verified 2026-08-13 — §6.3.1), so Phase 0 is **not blocked**. The
in-process, SHA-pinned, fp32 `sentence-transformers` embedder is still a Phase 0
deliverable, for pinning and content-routing reasons rather than availability
ones, and it is sized in hours. The LAN embedder stays useful as a cross-check
and for exploratory runs marked ineligible for confirmation.

Deliver:

- the in-process pinned embedder, with its SHA recorded and a golden-vector test;
- schemas and immutable manifest;
- fake adapter plus GPT-2/xAI capability prototypes;
- parser/normalizer goldens;
- `Δ̂`/permutation/BY and A/A calibration harness;
- the **capability-control arm** (§5.7) end-to-end — local, free, and an input to
  the confirmation gate;
- the **positive-control** cue pack and its per-model pass rates (§6.7.2);
- the **canary probe** wired into the runner (§5.5.1);
- reasoning-token distribution measurement for the pinned Grok release (§5.1);
- cost/time/storage estimator;
- 100-cue calibration report.

Gate:

- the embedder loads from a pinned SHA, is deterministic across runs, and no
  Behavior content is routed to a remote host;
- no paid request occurs before a human-readable estimate and approval;
- exact model identity is capturable and aliases are refused in strict mode;
- null controls meet the declared false-positive envelope;
- **both models pass the positive control** — otherwise the instrument is not
  measuring association and the study stops here;
- **the achievable permutation p-floor clears `q` with margin**, and the figure
  is printed in the calibration report (§6.7.1);
- parser and trial count produce adequate valid/reliable samples;
- the primary embedder, second embedder, bandwidth, effect floor, informativeness
  floor, compliance-parity threshold, RBO `p`, JSD correction, and the primary
  form of `Δ̂` are all frozen for the pilot.

### Phase 1 — resumable pilot backend

Deliver:

- SQLite runner, local batched GPT-2 adapter, xAI adapter, retries, and budget
  enforcement **in the runner** (§5.1);
- discovery analysis and compact export;
- about 300 preregistered cues, including the matched “daddy” pack;
- CLI inspection and reproducibility report.

Gate:

- interruption/resume produces no duplicate completed trials;
- requested/served identities are present on every API trial;
  `system_fingerprint` is captured **when the provider populates it** and its
  availability is recorded once in the manifest — an absent field never fails a
  trial (§5.5.1);
- the canary probe ran in every time block and its drift series is reportable;
- the same manifest + raw database reproduces the compact metrics;
- no cue receives “confirmed” before held-out data.

### Phase 2 — Behavior page MVP

Deliver:

- top-level Behavior navigation;
- loopback runner server, health/progress stream, and explicit paid-run approval;
- fixed cue landscape, ranked/table view, search, cue inspector, and Evidence;
- read-only sample artifact;
- Runs/provenance view and Behavior Settings;
- permalinks and Compare/Internals cross-links;
- static/offline, loading, empty, partial, error, and sensitive-content states.

Gate:

- existing cloud artifacts are byte-for-byte untouched;
- existing Map, Compare, Internals, Guide, and Seer tests pass;
- a user can find “daddy” in at most two actions and see its evidence status;
- every map encoding has an accessible text/table equivalent;
- **the cue landscape states what position means on screen**, sourced from
  `projection.quantity` and showing trustworthiness (§7.1.1, §9.5) — the view
  does not compose its own caption;
- effect is encoded by area and the declutter rule is in place (§7.1.3);
- associate channels include `indeterminate` (§7.2.1);
- no UI copy implies internal Grok access or model strength.

### Phase 3 — held-out confirmation

**Prerequisite — check the provider's terms of service.** This phase ships “a
shareable redacted artifact” comparing a named commercial model against another
model. Providers commonly restrict using their outputs to evaluate, benchmark, or
train against other models, and commonly restrict publishing comparative results.
This was absent from the original risk table and it is the class of problem that
kills a deliverable at the finish line, after the money is spent. Read the
current xAI terms, record the date and the relevant clauses in the study
manifest, and decide the publication surface **before** collecting the
confirmation partition — not after.

Deliver:

- frozen held-out prompt frames and new trial partition;
- **both confirmation arms** — R (same-frame replication) and G (held-out-frame
  generalization), reported separately (§5.4);
- second-embedder reanalysis;
- BY-adjusted confirmation states;
- global/domain correlation and robustness report;
- shareable redacted artifact.

Gate:

- a recorded ToS review exists and the artifact's distribution matches it;
- “confirmed” requires every criterion in §6.5, including compliance parity,
  informativeness, and exceeding the §5.7 capability contrast;
- all displayed magnitudes come from the confirmation partition (§6.5.3);
- post-hoc analyses are marked exploratory and cannot overwrite primary status;
- provider/model drift invalidates pooling;
- result ranking is stable enough for the declared trial count.

### Phase 4 — scale and mechanistic follow-up

Deliver only after the earlier gates:

- 1,000-cue expansion and optional licensed full-norm study;
- context/sense-frame packs;
- GPT-2 Internals links for confirmed cues;
- optional read-only Behavior overlays on token, SAE, and neuron maps, subject to
  §10.4;
- export suitable for independent analysis.

**Write the magnitude down.** The full-norms endpoint in §5.4 is 11,500 cues ×
100 trials × 2 models ≈ **2.3 million API requests**, plus reasoning tokens on
every one of them, plus roughly 1.15 million local GPT-2 generations — and, with
§5.7, another 1.15 million for the capability-control pair. The gate is right;
the number belongs in the document, because a gate is far easier to hold when
everyone can see what is on the other side of it. Realistically the study lives
at the 300–1,000 cue scale and the full-norm arm is a licensing-and-funding
decision, not an engineering one.

The GPT-2 mechanistic layer remains explicitly one-sided. It may help form a
causal hypothesis about GPT-2; it cannot explain Grok's hidden cause.

---

## 12. Verification matrix

### Backend tests

- manifest hashing and immutable revision behavior;
- exact model identity, alias refusal, fingerprint drift, and capability
  mismatch;
- GPT-2 sampler conformance against `GPT2Numpy`;
- interleaved randomization and deterministic local seeds;
- resume, retry, lease expiry, duplicate response, and budget refusal;
- Unicode/slang/emoji/multiword/refusal/verbose parser fixtures;
- known-null and known-shift synthetic distributions for MMD/permutation;
- **`Δ̂` on synthetic pairs**: equal-and-noisy → ≈ 0; one degenerate and one
  diffuse → large (proving the statistic alone does not solve degeneracy, which
  is why §6.5.2 exists); genuinely shifted → large;
- **compliance-parity gate** fires on a synthetic 40%/99% parse-rate pair, and
  the Manski bounds bracket the point estimate;
- **informativeness gate** rejects a perfectly-reliable exemplar-echoing model —
  the specific case that passes split-half reliability with zero content;
- **p-floor**: given a block structure, the computed achievable minimum p matches
  a closed-form reference, and a structure that cannot clear `q` is refused;
- **JSD bias correction** recovers the true divergence on synthetic
  distributions at small `n` where the plug-in estimator is visibly inflated;
- **embedder determinism**: same SHA + same text → bitwise-identical vectors
  across processes;
- **PCA persistence**: refitting with added cues and transforming with the stored
  axes place the original cues at identical coordinates;
- BY results against reference values;
- bootstrap and split-half reproducibility;
- discovery/confirmation partition leakage prevention;
- **R/G arm separation**: a frame-specific synthetic effect resolves to
  `frame-specific`, not to `confirmed` or to a bare failure;
- static export redaction and schema-version refusal;
- **`cues[].anchor` without `projection` is rejected** (§9.5).

### Viewer tests

- app/nav agreement for four Nebul.AI pages and three Seer pages;
- Behavior page with no index, empty study, partial study, and corrupt schema;
- cue search, filters, selection, permalink, table/map parity, and cross-links;
- every evidence status and its reason, including `frame-specific`,
  `capability-attributable`, and `indeterminate` associates;
- **the projection caption renders from `projection.quantity`** and the page
  refuses to draw the landscape when that field is missing;
- **effect radius scales with √effect** (area encoding), asserted numerically
  rather than visually;
- **cue→token overlay** produces no highlight for a cue with no exact tokenizer
  match, and the correct single highlight for one that has it;
- no raw output/API credential in the static artifact;
- keyboard flow, focus order, screen-reader summaries, contrast, reduced motion,
  and compact/mobile layouts;
- visualization labels match metric fields exactly.

### Research audit

- A/A false-positive rate and rank-stability curves;
- discovery results reproduced from raw trials by a clean environment;
- prompt-frame, embedder, time-block, and domain sensitivity;
- invalid-output and refusal analysis separated from semantics;
- complete evidence ledger and deviation from protocol report.

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Anthropomorphism | Claim contract and vocabulary lint/tests for high-risk UI phrases. |
| GPT-2/Grok interface asymmetry | Common literal task, separate compliance metric, exact wrapper provenance, and explicit limitation. |
| API/model drift | Dated releases, served ID/fingerprint capture, interleaved time blocks, fail/partition on drift. |
| Prompt sensitivity | Frozen primary frame plus held-out paraphrase confirmation. |
| Embedder bias | Exact lexical metrics always visible; two pinned embedders required for confirmation. |
| Cherry-picking “daddy” | Broad preregistered discovery, automatic ranking, matched controls, and held-out confirmation. |
| Multiple comparisons | BY-adjusted q-values and minimum effect/reliability gates. |
| Slang erased by preprocessing | Minimal primary normalization; dictionary/lemma view only secondary. |
| Cost/latency explosion | Calibration-led trial counts, immutable budget gate, resumable runs, expansion gates. |
| Sensitive/offensive outputs | Local append-only evidence, collapsed raw view, redacted static export, content notice. |
| Human norms treated as correctness | Optional third reference, never a target or strength score. |
| Licensing | Project-owned default cues; explicit local SWOW import and no repackaging. |
| Beautiful but misleading cloud | Fixed cue coordinates, deterministic local neighborhoods, exact tables, and no arbitrary cross-model geometry. |
| **Capability confound — every effect is really “GPT-2 is a base model”** | Normalized contrast `Δ̂` as the primary statistic (§6.4.1), the GPT-2-small↔XL capability-control arm as a per-cue reference (§5.7), and a `capability-attributable` outcome state. |
| **Differential attrition — comparing survivors, not models** | Compliance-parity gate with Manski-style bounds; parse rates shown wherever an effect is shown (§6.5.1). |
| **Reliable degeneracy passes the reliability floor** | Informativeness floor plus cue-echo / exemplar-echo / duplicate / prompt-copy detectors as gate inputs (§6.5.2). |
| **Small-`n` projection artifacts** — at 100–300 cues, UMAP manufactures clean islands from pure noise (measured in `validate.py:31-40`: silhouette 0.88 on shuffled data vs 0.43 on three real blobs, n=180) | The pilot landscape is PCA, which cannot produce that artifact; UMAP is gated on a cue count where the null baseline is a real floor; trustworthiness ships to the viewer (§7.1.1). |
| **“Fixed coordinates” silently broken by cue-set growth** — a refit moves every published cue and invalidates permalinks | A persistable linear transform (stored mean + axes) so added cues land in the existing space; no fitted reducer is discarded (§7.1.1, §5.4). |
| **Unstable colour** — HDBSCAN on the cue landscape would produce seed-dependent groups (shipped maps report seed ARI 0.46–0.62) | Colour by §5.3's preregistered strata only; never by a discovered cluster (§7.1.2). |
| **Overlay borrows credibility from an unvalidated map** — e.g. the shipped SAE map's null margin is −0.049 | Gate the overlay on the target map's validation margin or display that margin alongside it (§10.4, §16). |
| **Embedder identity drifts under a mutable tag** — `mxbai-embed-large:latest` is an F16 GGUF behind a re-pointable tag, so it cannot be a pinned revision | In-process, SHA-pinned, fp32 embedder as a Phase 0 deliverable, with a golden-vector test (§6.3.1). |
| **Sensitive content routed to the LAN worker** — the documented embedder host is under a standing no-NSFW rule, and this study embeds unreviewed raw outputs of deliberately provocative cues. **The host is reachable, so this is the default path, not a hypothetical one.** | In-process embedding removes the network path entirely; the constraint is stated in §6.3.1 so no implementer re-introduces it. |
| **Infrastructure declared dead on one probe** — a wrong-port check concluded “no embedder exists anywhere” and nearly added a fabricated blocking prerequisite to this plan | Probe the documented port before declaring a service down (`docs/M4-OLLAMA-HANDOVER.md` names 11435); `embed.py`'s failure message now prints the check commands and the port caveat (§16.3). |
| **Provider terms of service prohibit comparative publication** | ToS reviewed, dated, and quoted in the manifest *before* the confirmation partition is collected; publication surface decided then (Phase 3). |
| **Unbudgetable reasoning tokens** | p95-based budgeting from measured Phase 0 data, and a hard spend stop enforced by the runner, not only the planner (§5.1). |
| **Provider stops populating `system_fingerprint`, making the gate unpassable** | Fingerprint optional-if-absent, recorded once in the manifest; a behavioral canary probe per time block carries drift detection instead (§5.5.1). |
| **Broken instrument indistinguishable from a null result** | Hand-authored, license-safe positive control that each model must recover from its own associations; a Phase 0 gate (§6.7.2). |
| **Noise rendered as a categorical finding** — 1/40 vs 0/40 shown as “model-specific” | CI-based classification with an explicit `indeterminate` channel (§7.2.1). |
| **Fabricated cue→token links** — fuzzy or subword matching highlights the wrong point with full authority | Exact tokenizer-aware matching on declared variants; no match means no overlay (§10.4). |

---

## 14. Definition of done

The first research-grade release is done when:

- all current cloud pipelines and artifacts still work unchanged;
- a pinned, deterministic, in-process embedder exists and is covered by a
  golden-vector test, and no Behavior trial content has been routed off-box;
- both models clear the positive control, so the instrument is known to measure
  association at all;
- a strict GPT-2/pinned-Grok study can be planned, costed, run, paused,
  resumed, analyzed, confirmed, and exported;
- the capability-control arm has run, and no headline effect is reported that
  fails to exceed its own same-family scale contrast;
- exact prompts, outputs, model identities, fingerprints, sampler settings,
  normalizer, embedders, statistics, and code revision are auditable;
- the system discovers the largest confirmed deviations across a broad cue
  suite rather than starting from one favored keyword;
- searching “daddy” shows its two association neighborhoods, matched contexts,
  metrics, raw evidence, and honest evidence status;
- global association correlation and per-cue divergence are both visible,
  without a winner or strength score;
- results survive the declared null, false-discovery, prompt, embedder, and
  held-out confirmation gates;
- the normal Semantic map and lightweight Compare experience remain
  uncluttered;
- the Behavior page is usable in read-only static mode, keyboard/table mode,
  reduced motion, and compact layouts;
- unsupported causal/internal claims are absent from UI and exports.

## 15. Explicit non-goals for the first release

- reverse-engineering Grok weights or hidden activations;
- scoring which model is better;
- treating provider reasoning summaries as ground-truth cognition;
- automatically explaining training causes;
- replacing or reprojecting existing clouds;
- declaring no difference from a non-significant underpowered test;
- publishing or redistributing licensed human association data;
- generalizing a GPT-2/Grok result to every version in either model family;
- presenting a discovery-partition effect size as a result;
- clustering the cue landscape, or colouring it by anything other than the
  preregistered strata;
- shipping UMAP as the cue landscape at pilot scale.

---

## 16. Map-shape audit of the existing viewer

This section records an audit run against the current tree before the Behavior
work began. Its purpose was to answer a prior question — *are the existing cloud
shapes actually meaningful, or are they partly arbitrary?* — because a new page
that inherits a dishonest layout convention inherits the dishonesty with it.

### 16.1 Finding: no coordinate in this repository is random

Every front-end funnels through a single coordinate chokepoint,
`reduce_vectors` at `src/nebulai/backend/reduce.py:8-52`. Nothing else emits
positions. A sweep of `src/` and `viewer/src/` for `Math.random`, `np.random`,
jitter, noise, force simulation, golden-spiral placement, and hand-tuned offsets
found **not one line that perturbs a coordinate**. No force-layout library is
even installed. `Math.random` appears twice in the entire viewer, both times
generating an ID string.

Concretely:

- UMAP with `metric="cosine"`, `n_neighbors=30`, `min_dist=0.0` for the 10-D
  clustering space and `0.1` for the 3-D display space, `random_state=42`
  (`reduce.py:34-46`);
- the 2-D map is `PCA(u3)` — a *camera angle* on the 3-D cloud, not an
  independent fit, which is why the two views stay consistent (`reduce.py:46`);
- **clustering never happens in display space** — HDBSCAN and all similarity
  edges run on the 10-D `u_cluster` (`export.py:72-79`);
- export writes raw coordinates rounded to 4 dp, with no offset and no jitter
  (`export.py:39-40`);
- per-point opacity is HDBSCAN's `probabilities_` — real membership confidence,
  not decoration (`cluster.py:49-56`, `PointsLayer.ts:84`).

The genuinely arbitrary elements are all **categorical or cosmetic**, and
correctly so: cluster hue is a golden-ratio scramble of a discovery-order id
(identity, carrying no magnitude — `PointsLayer.ts:27-30`); halo animation phase
is a golden-angle stagger (`HaloLayer.ts:48`); Chord rim spacing and Hierarchy
leaf angle are rank-spaced by construction.

So the shapes mean something. The gaps were all in **disclosure** — places where
the code knew something the user was not told.

### 16.2 What was fixed

| Gap | Fix |
|---|---|
| The Atlas never said it was a UMAP. Edges carried provenance (`LegendCard.tsx`: *“…10-D cluster space, not screen distance”*) but the layout did not. | A Positions block in the legend naming the projection and stating that axis direction and inter-cluster distance are not claims. |
| Chord rim and Hierarchy leaves are rank-spaced at uniform `2π/n`; both drivers document it in their headers, nothing on screen did. | View-gated layout notes: the rim gap is position in the ring, the chord weight is the similarity; hierarchy radius is merge similarity, angle is leaf order. |
| Trustworthiness, seed ARI, and null margin existed only in CLI output. The shipped `gpt2-small · SAE features` map posts a **negative null margin (−0.049)** and a viewer user could not tell. | A distortion readout in the legend reading `out/compare/metrics.json`, with the `!` / `?` markers and a plain-language callout for the below-null case. The verdict is computed once in Python (`backend/metrics.py`) and only *rendered* in TypeScript, so the two cannot drift. Absence renders as “not measured”, never as “clean”. |
| The Snapshot Map keyword ring places nodes by **index in the topic preset array** — authoring order — while rendering as a network graph, where the default reading is “proximity ≈ relatedness”. It was the only positioned view in the app with no statement of what position meant. | A footer note: ring position is preset order; angle and distance carry no meaning; the measurements are the links and the node size. |
| `compare.py:190` rescales each model's native cloud separately to a fixed radius, erasing relative spread, and quadrant/palette follow **CLI argument order** — with no comment and no UI note. | A “what this destroys, deliberately” comment at the call site, and per-state notes in the Compare panel covering all four layouts. |

Nothing in `reduce.py`, no coordinate, no cluster assignment, and no existing
artifact was touched — the changes are additive text plus one metadata read
path. `nebulai.json` is byte-for-byte unchanged by construction, because the
viewer reads the separately-written `metrics.json` rather than having export
inject validation fields.

### 16.3 The embedder was never missing — it was on another port

Recorded because the wrong conclusion was more expensive than the bug.

`embed.py`'s 2026-08-12 note said “nothing serves it,” having probed
`localhost:11434`, `localhost:8050`, and the LAN box's `:11434`. From that, an
earlier draft of §6.3.1 concluded the repository had no embedder, that
`nebulai compare` was broken, and that building a replacement was a **blocking
prerequisite for the whole study**. A re-test on 2026-08-13 found:

| Probe | Result |
|---|---|
| `ping 192.168.0.200` | 100% loss — but **ARP resolves**, so the host is on the LAN |
| `:11434` (ollama stock) | filtered/timeout — the port the old note tested |
| `:5000` | open, 403 (auth-gated) |
| `:8050` | open — `omlx` OpenAI-compatible; `all-MiniLM-L6-v2`, `nomic-embed-text-v1.5`; **no mxbai** |
| **`:11435`** | **ollama 0.23.1, `mxbai-embed-large:latest`, 334M params, F16 GGUF, 1024-dim** |
| `:8100/v1/status/ollama` | `running:true, port:11435` |

`embed_texts()` against `:11435` returns `(n, 1024)` float32, L2-normalized, with
sane semantics (cos 0.70 related vs 0.31 unrelated). It has been working the
whole time.

`docs/M4-OLLAMA-HANDOVER.md` has said `OLLAMA_HOST=0.0.0.0:11435` since
2026-08-04, and explicitly flags the `11434` in an older map as *stale* — the
correct information was already in the repository, one file away from the note
that contradicted it.

Fixed, so the next probe does not repeat it:

- `embed.py`'s docstring now records the verified state, the port, the `:8050`
  vs `:11435` distinction, and what is genuinely still true (no faithful HF
  drop-in; F16 GGUF behind a mutable tag ≠ a pinned fp32 revision);
- **`NEBULAI_EMBED_HOST`** is honored by `--embed-host` across `tokens`,
  `probe`, and `compare` (explicit flag > env > `--ollama-host`), so a non-stock
  port is configured once rather than retyped per run — retyping is how the
  drift went unnoticed for a month;
- `_embed_batch`'s failure now prints the `curl` check, the 11434-vs-11435
  caveat with a pointer to the handover doc, and the env/flag/API escape hatches,
  instead of a bare `URLError`;
- `nebulai compare` catches that `RuntimeError` and exits with the guidance
  rather than an uncaught traceback.

The generalizable lesson, and the reason this sits next to a section about
honest visuals: **“I probed it and it was down” is a measurement, and it inherits
every assumption in the probe.** A single unreachable port became “no embedder
exists anywhere,” which became a fabricated blocking prerequisite in a planning
document. The same discipline this project applies to a cluster boundary —
report what was measured, and what the measurement cannot distinguish — applies
to infrastructure claims.

### 16.4 Why this section is in this document

Three of the rules the Behavior page must follow are generalizations of what the
audit found, and they are easier to hold when the precedent is visible:

1. **Every positioned view states what position means** (§7.3) — the Atlas, the
   Chord rim, the Hierarchy, and the Snapshot ring now all do.
2. **Distortion travels with the coordinates** (§9.5's `projection` block) — the
   legend readout is the same idea applied to the existing maps.
3. **An arrangement is named as an arrangement** — rank order, preset order, and
   CLI argument order are all real conventions in this codebase, and each one
   now says so where it is drawn.

And the strongest precedent is one the project set before any of this:
`docs/SESSIONSEER-LIVE.md` §7 records a finished, shipped-quality feature being
**removed** because an out-of-sample UMAP position could not be made honest. That
decision is what §7.1.1 leans on, and it is why “fixed cue coordinates” had to
become PCA rather than a promise.

---

## 17. Revision log — 2026-08-13 review

### 17.1 What changed

| § | Change | Why |
|---|---|---|
| 4.2.1 (new) | States that H2 already names the correct target and that the rest of the document must match it. | H2 said “larger than calibrated same-model separation”; §6.4 tested raw MMD². The stated hypothesis and the tested quantity were two different claims. |
| 5.1 | p95 reasoning-token budgeting; hard spend stop in the **runner**. | The “immutable budget gate” bounded a per-request quantity that is unknowable in advance. |
| 5.4 | Confirmation split into **R** (replication) and **G** (generalization) arms; cue-growth constraint recorded. | Confirmation changed two variables at once, making `suggestive` vs `unstable` undecidable despite being defined as distinct states. |
| 5.5.1 (new) | `system_fingerprint` optional-if-absent; behavioral **canary probe** per time block. | The gate required a provider-controlled optional field to be present on every trial — unpassable by construction if xAI omits it. |
| 5.7 (new) | **GPT-2-small vs GPT-2-XL** capability-control arm. | Nothing established a reference for ordinary between-model divergence, so every effect was reported against zero. Local, free, and it converts the study's largest confound into a subtractable quantity. |
| 6.2 | “No LLM judge” restated precisely as “no *generative* model adjudicates equivalence.” | A sentence embedder is a learned equivalence judge and is the primary metric's substrate. §13 makes vocabulary a test, so the document must pass its own test. |
| 6.3.1 (new, then **corrected same day**) | In-process, SHA-pinned, fp32 embedder required for Behavior; LAN-worker routing prohibited. Originally framed as a blocking “task zero” on the grounds that no embedder was reachable — **that framing was wrong and has been retracted.** | A mutable ollama tag cannot satisfy “pinned to exact revisions,” and the documented LAN host is under a standing content rule this study would violate silently. The availability argument was false: the probe used ollama's stock port 11434, but the M4 binds **11435** and has since 2026-08-04. See §16.3. |
| 6.4 | Primary statistic is `Δ̂`; added dispersion decomposition, bias-corrected JSD, declared RBO `p`, OOV differential. | Raw MMD² measured the wrong thing; and three separate small-sample biases all pushed in the *same* direction as the capability confound rather than cancelling. |
| 6.5 | Added compliance-parity gate, informativeness floor, capability-contrast clause, R/G arms, confirmation-partition-only magnitudes, second-embedder caveat. New states `frame-specific` and `capability-attributable`. | Differential attrition biases survivor comparisons even at equal `n`; a model that echoes the exemplar every time has *perfect* reliability and zero content; and discovery effects are selection-inflated by construction. |
| 6.7 | Added the achievable **p-floor** check and a **positive control**. | Restricted permutation can make confirmation combinatorially impossible regardless of effect size; and with human norms refused and SWOW gated, nothing anchored the instrument at all. |
| 7.1.1–7.1.3 (new) | Landscape is **PCA**, persistable, reusing `bundles.py:146-166`; UMAP gated on scale; trustworthiness shipped; area encoding; declutter rule. | “Fixed coordinates” and a growing cue set are incompatible under UMAP, and no fitted reducer is persisted anywhere in this codebase. Separately, `validate.py:31-40` measured UMAP producing silhouette 0.88 from *shuffled* data at n=180 — the pilot's exact regime. |
| 7.1.2 (new) | Colour by preregistered strata; never by a discovered cluster. | `validate.py:41-46` and the README's 0.46–0.62 seed ARI show cluster boundaries are not stable findings at this scale. |
| 7.2.1 (new) | `indeterminate` associate channel via frequency-difference CI. | 1/40 vs 0/40 is noise, but a set-difference rule rendered it as “model-specific” at full visual weight. |
| 7.3 | Added: every positioned view states what position means; area not radius. | Generalized from the §16 audit. |
| 9.4 / 9.5 | Manifest and artifact carry every newly-frozen parameter; `behavior.json` gains a `projection` block with trustworthiness and a `"quantity"` string. | Free parameters that are not in the manifest are researcher degrees of freedom; and coordinates without provenance are the exact defect §16 fixed in the existing viewer. |
| 10.4 | Tokenizer-aware exact matching or no overlay; do not inherit credibility from an unvalidated map. | BPE means many cues have no single-token form and `" daddy"` ≠ `"daddy"`; and the shipped SAE map posts a −0.049 null margin. |
| 11 | Phase 0 gains the embedder prerequisite, positive control, p-floor check, and capability arm; Phase 1 relaxes fingerprint and adds the canary; Phase 3 gains a **ToS review**; Phase 4 writes down the 2.3M-request magnitude. | Gates should be passable, and the ones that were unpassable or missing are the ones that fail late. |
| 13 | Fifteen new risk rows. | Each corresponds to a failure the original table did not cover. |
| 16 (new) | The map-shape audit and the viewer fixes that shipped with this revision. | The precedent the new page's honesty rules generalize from. |

### 17.2 What was left alone, deliberately

Several parts of the original design are unusually well-drawn and were not
touched:

- **The fixed cue landscape as a concept** (§7.1). One coordinate system for both
  models sidesteps the trap of comparing two independently-fit projections.
  §7.1.1 changes *how* it is built, not *whether* it is the right idea.
- **§6.5's refusal to render “same” from an underpowered null**, requiring a
  separately declared equivalence margin. This is rarely done correctly and it is
  correct here.
- **The discovery/confirmation partition with a frozen manifest**, where post-hoc
  changes create a new revision rather than rewriting history.
- **Keeping compliance out of the semantic metric** (§5.2) — the right instinct;
  §6.5.1 adds the companion gate it needed, without weakening it.
- **Permanently refusing moving aliases in strict mode** (§5.5).
- **No composite “mind distance” score** (§6.4).
- **Append-only raw evidence** (§6.1), with corrections as derived rows.
- **Every plot has a table equivalent** (§7.3, §8.6).
- **The claim contract** (§1), and §1.2's prohibition list in particular.

### 17.3 Cost of the revision

Nothing here makes the study more expensive in API spend. The capability-control
arm is local and free. The positive control is a handful of cues. The p-floor
check and the ToS review are analysis and reading. The embedder work is a
prerequisite that was already blocking `nebulai compare` regardless of this
feature. The two changes that cost trials — split R/G confirmation arms, and
minimum within-block counts — apply to the confirmation partition, which is the
smallest one, and both buy claims the study could not otherwise make.
