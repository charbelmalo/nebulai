# Behavioral semantic divergence — research and UX plan

Status: implementation baseline, 2026-08-13.

This document defines the additive Nebul.AI feature that can discover where two
model deployments produce substantially different semantic associations. GPT-2
versus a pinned Grok release is the first study. The feature is not a model
leaderboard, and a cue such as “daddy” is a drill-down example rather than the
experiment's organizing assumption.

## Decision at a glance

| Question | Decision |
|---|---|
| Can Nebul.AI show GPT-2/Grok semantic differences visually? | **Yes**, by comparing repeated, controlled output distributions and visualizing only differences that survive reliability and replication checks. |
| Does this prove different internal “minds”? | **No.** It supports a narrower claim about observable association behavior under an exact protocol. GPT-2-only internals may explain its side later, but cannot make Grok's hidden internals comparable. |
| Do the existing clouds have to be removed or rebuilt? | **No.** Token, SAE, neuron, Probe, Compare, and Internals artifacts stay intact. The new study has its own data contract and can link back to them. |
| Does the current product expose any of this already? | **Partially.** Existing clouds are useful hypothesis generators; they do not yet provide repeated trials, uncertainty, false-discovery control, or held-out confirmation. |
| Where should the feature live? | Add a top-level Nebul.AI page labeled **Behavior**, with the page title **Semantic divergence**. Keep the current Map → Compare view lightweight and unchanged. |
| What is the default experience? | Open a precomputed study, see the largest **confirmed** deviations, search any cue, and inspect raw evidence. Experimental controls stay behind **New study**. |

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

### 5.5 Model identity and capability audit

Research mode refuses moving aliases such as `latest`. Before the cost gate,
`nebulai behavior plan` must:

1. query the provider's available model/capability surface;
2. resolve and require an exact dated Grok release;
3. record requested and served model IDs;
4. verify the supported common sampler parameters;
5. record whether reasoning can be disabled and, if not, its exact effort;
6. ensure tools/search are absent;
7. confirm that response IDs, usage, and `system_fingerprint` are capturable;
8. resolve GPT-2 to a Hugging Face commit SHA and record tokenizer files.

An exploratory mode may allow a moving alias, but its results can never receive
the “confirmed” status. If a fingerprint changes during a strict study, the
runner stops or partitions the data; it never pools both deployments.

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
only evidence shown. No LLM judge decides whether two outputs “mean the same
thing.” The exact surface form always remains visible.

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

### 6.4 Per-cue metrics

There is no composite “mind distance.” Metrics stay separate.

| Metric | Role |
|---|---|
| **MMD² over trial vectors** | Preregistered primary semantic distribution effect. Gaussian kernel bandwidth is calibrated once on Phase 0 and then frozen. |
| **Permutation p-value** | Cue-wise evidence against exchangeable model labels, using the frozen MMD statistic and permutations restricted within collection time blocks. |
| **Bootstrap interval** | Uncertainty around the primary effect and selected secondary metrics. |
| **Jensen–Shannon divergence** | Exact normalized associate-frequency difference. |
| **Rank-biased overlap** | Top-weighted overlap between each model's ranked associates. |
| **Within-model split-half stability** | Whether each model's own profile is stable enough to compare. |
| **Valid-output/compliance rate** | Separate protocol behavior; never folded into semantic MMD. |
| **Prompt-frame and embedder agreement** | Confirmation robustness, shown explicitly. |

The ranked “largest deviations” list first filters by the confirmation gate,
then sorts the surviving cues by the preregistered MMD² effect. It does not
multiply effect, confidence, and lexical novelty into an opaque score.

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
3. primary effect exceeds the frozen smallest effect of interest;
4. BY-adjusted `q ≤ 0.05`;
5. the held-out prompt frame reproduces the effect;
6. the second embedder agrees on status and broad model-specific neighborhood;
7. model identity, protocol, and fingerprint stayed valid.

Other states are:

- **suggestive:** discovery passed, confirmation has not;
- **unstable:** prompt or embedder robustness failed;
- **no detected deviation:** adequately powered but below the effect/significance
  gate;
- **insufficient evidence:** too few or too-variable trials;
- **incomparable:** identity, fingerprint, protocol, or data-integrity failure.

The UI never renders “same” merely because a low-powered test was not
significant. Equivalence requires a separately declared equivalence margin and
test.

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
- rank-stability curves versus trial count.

---

## 7. Visualization contract

### 7.1 Overview: a fixed cue landscape

All cues occupy one fixed coordinate system built from the cue words themselves
with a pinned neutral embedder. The models never receive separate UMAPs that a
viewer might mistake for directly comparable internal geometry.

Each cue is a paired glyph:

- location = fixed cue-space location;
- size = confirmed MMD² effect;
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

### 7.2 Cue inspector

Selecting “daddy,” or any cue, opens a deterministic two-sided association
neighborhood:

- cue in the center;
- GPT-2-only associates on one side;
- Grok-only associates on the other;
- shared associates in the center channel;
- directed edge width = observed rank-weighted frequency;
- edge stability pattern = split-half reliability;
- no force-layout distance is presented as effect size.

The inspector has three sections:

1. **Neighborhood** — the visual association graph and top exact words.
2. **Contexts** — base and predeclared sense/prompt frames.
3. **Evidence** — MMD², interval, p/q, JSD, rank overlap, valid trial counts,
   prompt, model IDs, fingerprint, and expandable raw trials when connected to
   the local study store. A static export says that raw trials were withheld.

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
empirical result rather than a foregone conclusion. The page title may be
**Semantic divergence** because it describes the analysis the user is opening.

### 8.2 Progressive-disclosure layout

```
┌ Semantic map | Behavior | Internals | Guide ─────────────────────────────┐
│ Semantic divergence   [study: GPT-2 ↔ Grok …] [How to read] [New study] │
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
- primary/secondary embedder IDs and revisions;
- MMD kernel/bandwidth, permutations, bootstrap count, effect floor, reliability
  floor, multiple-testing family, and q threshold;
- budget ceiling, estimated call/token/cost/time/storage envelope;
- raw-data retention and redaction policy.

### 9.5 Static viewer contract

`behavior.json` includes:

- `meta`: schema, study, model pair, exact identities, protocol hash, dates,
  cue source, and evidence limitations;
- `global`: valid cue count, confirmation counts, association correlations,
  graph overlap, domain summaries, and reliability;
- `cues[]`: cue/domain, fixed 2-D/3-D anchor, evidence state, trial counts,
  primary effect/p/q/interval, exact metrics, reliability, robustness, and top
  shared/model-specific associates;
- `runs`: discovery/confirmation completion, fingerprint blocks, failures,
  token/cost totals, and export redaction;
- no API keys and no raw outputs by default.

Unknown schema versions fail closed.

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

---

## 11. Implementation phases and gates

### Phase 0 — contract, capability audit, and calibration

Deliver:

- schemas and immutable manifest;
- fake adapter plus GPT-2/xAI capability prototypes;
- parser/normalizer goldens;
- MMD/permutation/BY and A/A calibration harness;
- cost/time/storage estimator;
- 100-cue calibration report.

Gate:

- no paid request occurs before a human-readable estimate and approval;
- exact model identity is capturable and aliases are refused in strict mode;
- null controls meet the declared false-positive envelope;
- parser and trial count produce adequate valid/reliable samples;
- the primary embedder, second embedder, bandwidth, and effect floor are frozen
  for the pilot.

### Phase 1 — resumable pilot backend

Deliver:

- SQLite runner, local batched GPT-2 adapter, xAI adapter, retries, and budget
  enforcement;
- discovery analysis and compact export;
- about 300 preregistered cues, including the matched “daddy” pack;
- CLI inspection and reproducibility report.

Gate:

- interruption/resume produces no duplicate completed trials;
- requested/served identities and fingerprints are present on every API trial;
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
- no UI copy implies internal Grok access or model strength.

### Phase 3 — held-out confirmation

Deliver:

- frozen held-out prompt frames and new trial partition;
- second-embedder reanalysis;
- BY-adjusted confirmation states;
- global/domain correlation and robustness report;
- shareable redacted artifact.

Gate:

- “confirmed” requires every criterion in §6.5;
- post-hoc analyses are marked exploratory and cannot overwrite primary status;
- provider/model drift invalidates pooling;
- result ranking is stable enough for the declared trial count.

### Phase 4 — scale and mechanistic follow-up

Deliver only after the earlier gates:

- 1,000-cue expansion and optional licensed full-norm study;
- context/sense-frame packs;
- GPT-2 Internals links for confirmed cues;
- optional read-only Behavior overlays on token, SAE, and neuron maps;
- export suitable for independent analysis.

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
- BY results against reference values;
- bootstrap and split-half reproducibility;
- discovery/confirmation partition leakage prevention;
- static export redaction and schema-version refusal.

### Viewer tests

- app/nav agreement for four Nebul.AI pages and three Seer pages;
- Behavior page with no index, empty study, partial study, and corrupt schema;
- cue search, filters, selection, permalink, table/map parity, and cross-links;
- every evidence status and its reason;
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

---

## 14. Definition of done

The first research-grade release is done when:

- all current cloud pipelines and artifacts still work unchanged;
- a strict GPT-2/pinned-Grok study can be planned, costed, run, paused,
  resumed, analyzed, confirmed, and exported;
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
- generalizing a GPT-2/Grok result to every version in either model family.
