# Generative variance — grounding audit, research plan, and UX plan

Status: **approved 2026-08-16; the two unblockable build items have landed, and
the study itself has not started.** No story has been generated, no scorer has
been run, **no paid call has been made**, and the question set is not frozen —
so Phase 0's substance (items 1–7, §10) is still ahead, and it is gated on human
annotation time rather than on code (§8.4).

What exists as of this revision is the scaffolding that had to precede any
spending, both landed with tests:

- **The §8.3 cost gate**, all three requirements — `cost_gate(...,
  require_price=True)` fails closed on an unpriceable model, `llm.RunBudget`
  enforces a *cumulative* run ceiling, and spending requires a printed
  pre-flight estimate followed by an explicit `approve()` step.
- **The §6.5 instrument freeze** — `backend/instrument.py` gives the question
  set a file format, a content hash over exactly the fields that determine what
  a score means, load-time tamper detection, and a refusal to pool trials across
  instruments. It ships **no questions**: the template in `docs/instruments/`
  documents the format and cannot be frozen, because choosing the real questions
  is the study's central scientific act and is gated on §6.2–6.4.

Everything else below remains a proposal to be amended or rejected.

This document plans two workstreams that ask whether a language model's
*story architecture* — not its prose style — varies from run to run, and
whether it varies less than the gap between models.

- **W1 (self-variance baseline).** One model, ~30 prompts, ~40 trials each,
  scored into a fixed architecture vector. Measures how much one model's
  narrative structure moves when nothing changes but the sampler. This is the
  priority and it is a prerequisite, not a warm-up.
- **W2 (cross-model architecture divergence).** The same instrument across
  several models. **Gated on W1**, because a between-model distance is
  uninterpretable until the within-model distance is known.

Three findings dominate everything below, and all three came from the audit
rather than from the brief.

**First, the source study did not measure what W1 measures, and this is
the strongest argument for doing W1 at all.** StoryScope generated one story
per prompt per source and validated the *scorer's* repeatability across five
runs (Krippendorff's α = 0.88). It never generated the same prompt twice from
the same model. Its dispersion claim — "AI-generated stories cluster in a
shared region of narrative space, while human-authored stories exhibit greater
diversity" — is therefore a statement about a between-source geometry with no
within-source repeat measurement underneath it (§2.2). W1 is the missing
denominator.

**Second, the scorer-validity plan has a measured ceiling, and it is lower than
the floor the brief implies.** StoryScope's own human annotators agreed with
*each other* at Cohen's κ = 0.7385 — **below** their mean agreement with the
model (κ = 0.8390). Any floor we declare must be read against 0.74, not against
1.0, or we will discard questions for failing to beat human beings (§6.2).

**Third, two of the brief's premises about this repo are false, and a third is
false in a way that removes work rather than adding it.** The probe path *has*
landed and the embed-host leak *is* closed (§0.1 A4, A5). And because W1's trial
vector is the architecture score itself, the entire embedder apparatus that
`BEHAVIORAL-DIVERGENCE-PLAN.md` needs — pinned in-process model, second-embedder
replication, OOV handling — **does not apply to this study at all** (§0.2).

---

## Decision at a glance

| Question | Decision |
|---|---|
| Is W1 worth running? | **Yes — go.** Its premise survived the audit intact: the source study measured scorer repeatability, never generator self-variance, so W1's number does not exist in the literature we grounded against (§2.2). |
| Is W1 blocked on anything? | **Yes, one thing: a validated scorer.** Every downstream number is a function of the scorer's output, so Phase 0 is scorer validation and nothing else ships until it passes (§6). |
| Can GPT-2 participate? | **No, and the reason is a wall rather than a cost.** Measured `n_ctx = 1024` caps it near 750 words, so a ~5,000-word story is categorically unrepresentable; there is also no `generate` path at all. This removes the free local arm that `BEHAVIORAL-DIVERGENCE-PLAN.md` relies on and makes **every** arm a paid call (§0.1 A12, §8). |
| What does that do to the cost gate? | Moves it from optional to **mandatory and pre-flight**. `cost_gate` failed OPEN on any model without a corpus price (§0.1 A7), which is exactly the configuration that would let an overnight run bill unbounded. **Fixed 2026-08-16** — opt-in fail-closed, a cumulative run ceiling, and an explicit approval step (§8.3). |
| What is the headline number? | **The variance ratio `ρ̂` = between-model MMD² ÷ mean within-model MMD²**, with a CI. Not a raw between-model distance. A between-distance with no within-denominator is the exact error §2.2 attributes to the source study. |
| Does W1 need an embedder? | **No.** The architecture scores *are* the trial vector. This deletes four sections' worth of machinery the sibling plan needs (§0.2). |
| Does this get a new viewer page? | **No.** Recommend a CLI report + static JSON + a Guide section. The "Behavior" pill the brief assumes is **unbuilt**, and the six pages are a pinned partition across two instruments (§9). |
| What is the scorer floor? | Per-question κ whose **lower CI bound** clears a floor set *relative to the human–human ceiling of 0.7385*, not an absolute 0.8 (§6.2). |
| Biggest one-way door? | The **question set**. Scores are only comparable within a fixed instrument; changing a question after data collection invalidates every trial already scored (§6.5). |
| Biggest reversible choice? | Model, trial count, and prompt set. All three can be extended in a later run without invalidating earlier data, provided the question set and scorer are pinned. |

---

## 0. Grounding audit

The brief asked for this first, and asked that it be reported as
claim → actual → consequence. Every row was checked against the tree on
2026-08-16 at the cited path and line. Rows marked **FALSE** are cases where
the brief's stated premise does not match the repository.

### 0.1 Audit table

| ID | Claim as briefed | Actual, with citation | Consequence for this brief |
|---|---|---|---|
| **A1** | Nebul.AI deploys statically with zero server-side compute, three build targets. | **True, with a correction to the count.** `viewer/package.json:8-12` defines *four* vite builds — `build` (combined), `build:nebulai`, `build:seer`, `build:hub`. The *deploy* is three: `build:deploy` (:12) runs nebulai + seer + hub. Each sets `VITE_BASE` to a sub-path and blanks `VITE_LIVE_URL`/`VITE_BUILD_URL`/`VITE_EMBED_HOST`/`VITE_SEER_URL` (:9-10), so a deployed build has no backend to call. | Any W1 output must be a **static artifact**, computed offline and shipped as JSON. No W1 feature may require a live endpoint. Confirms §9's recommendation shape. |
| **A2** | The viewer reads precomputed artifacts and never computes. | **True.** The blanked env vars at `viewer/package.json:9-10` are the enforcement: the deployed bundle is handed empty URLs for live, build, embed, and seer-capture, so the code paths that would call out have nowhere to go. | W1's statistics — MMD², CIs, permutation p-values — are computed in Python and **serialized**. The viewer renders numbers it is given. No in-browser statistics. |
| **A3** | `Units` is the only front-end contract. | **True.** `src/nebulai/units.py:6-30`: `ids: list[int]`, `vectors: np.ndarray` (n,d) float32, `labels: list[str]`, `meta: dict`, with a `__post_init__` length check (:20-26). Every front-end reduces to this and the whole back-end sees only this (docstring :8-13). | W1 **can** reuse the existing reduce → cluster → name → export → viz pipeline by emitting a `Units` whose vectors are architecture-score rows and whose `ids` are trial indices. This is the single largest reuse win (§5). |
| **A4** | The probe front-end has not landed in the build server. | **FALSE.** It has landed. `src/nebulai/backend/build_server.py:146-147` validates `source not in ("hf","api","probe")`; the probe branch is :150-165; `dataset_id_for` delegates to `probe_dataset_id` (:129-132); `--force` is suppressed for probe because "probe always grows a fresh cloud; it has no cache" (:197-198). | The brief's plan to "follow probe's pattern once it lands" is **available now**, not deferred. W1's runner should be modelled on the probe branch, which is the existing precedent for a front-end with **no model weights** — exactly W1's shape. Removes a blocker. |
| **A5** | Exported artifacts leak the private embed host. | **FALSE — closed.** `src/nebulai/backend/embed.py:91-129` `public_embed_host()` collapses every non-loopback endpoint to `PUBLISHED_REMOTE_HOST = "remote"` (:88). It deliberately is *not* a general private-address classifier because that call "fails OPEN" (:104-108). Two stamping sites only (`frontends/probe.py:642`, `frontends/api_tokens.py:112`); tests at `tests/test_api_tokens.py:195-228`. All five shipped `out/` artifacts read `"embed_host": "remote"`. | No remediation work for W1. **But** the pattern is a requirement to inherit: if W1 stamps provenance, any host field goes through `public_embed_host`. Also note the fix was applied *in place* on shipped artifacts, so W1 must not assume artifacts are immutable once published. |
| **A6** | Corpus has four models with pinned endpoints and prices. | **True.** `src/nebulai/corpus.py:52-155`. muse-glimmer-30b ($0.35/$1.50 per M), gemma-4-26b (**$0.00/$0.00**, tied embeddings, the W_E/W_U control), ling-2.6-flash ($0.010/$0.030), mistral-nemo ($0.019/$0.030). `DEFAULT_MAX_COST_USD = 1.00` at :173. `estimate_naming_cost` at :186-195 documents its own shape as an **upper bound** ("Rounded up per batch"). | Gives W1 real prices for §8 rather than guesses. **gemma-4-26b at $0.00 is the obvious W1 subject** — it makes the priority workstream nearly free. But it is an MoE with tied embeddings (:88, :96), which is a *different* generator from the dense untied models, so W1-on-gemma does not generalize to W2's arms without saying so. |
| **A7** | House rules forbid silent substitution and silent budget downgrade. | **True as written, but per-caller in practice, and the budget gate FAILS OPEN.** `llm.py:55` `IdentityError`, `:64` `BudgetError` are centrally defined. `cost_gate` (:175) returns `None` — permitting the call — when the model has no corpus price, printing that "the ceiling cannot be enforced for it" (:198). It is invoked from only four sites: `backend/name.py:782,858` and `frontends/probe.py:420,432` (via `_probe_cost_gate`, :261-269). `chat_openai` (:448) has **no gate of its own**. | **This is the single most important operational finding for W1.** A new W1 runner calling `chat_openai` directly inherits *no* budget protection, and a model outside `CORPUS` silently bypasses the ceiling. Since GPT-2 cannot participate (A12) every W1 arm is paid, so this is a live financial risk. §8.3 makes closing it a Phase 0 gate. **Closed 2026-08-16**: `require_price=True` fails closed, `RunBudget` adds the cumulative ceiling the per-call gate never had, and `chat_openai`'s gap is addressed by routing responses through `RunBudget.charge_response` rather than by adding a fifth per-call gate. |
| **A8** | Projection-honesty validators exist and are applied. | **True, and still applied.** `backend/validate.py:91` `trustworthiness_score`, `:134` `seed_stability`, `:210` `null_baseline`. `scale_cluster_kwargs` (:66) is genuinely called at :401. The docstring at :33-46 records the two measured caveats: silhouette **0.88 on shuffled noise vs 0.43 on real data** at n=180, and leaf-mode ARI 0.37 vs eom 1.00. | W1 inherits a working honesty harness for free. The shuffled-noise result is a **direct warning for W1's own scale**: ~30 prompts × 40 trials = ~1,200 points is above the n=180 danger zone, but any *per-prompt* view (40 points) is far below it. Per-prompt projections must be null-tested or not shipped. |
| **A9** | The feature would be "a mode inside the Behavior page." | **FALSE — the Behavior page does not exist.** Nebul.AI's nav is **three** pills: `viewer/src/chrome/apps/nav.ts:APP_CHROME.nebulai.nav` = Semantic map / Internals / Guide. `viewer/src/app/slices/shell.ts:36` types exactly six pages; `:46-49` partitions them `nebulai: ["map","interp","guide"]`, `seer: ["seer","sessions","snapshot"]`. `viewer/tests/unit/app-pages.test.ts:29` hardcodes `ALL_PAGES` as those six and pins pills == `APP_PAGES` in order, no page owned twice, none orphaned. There is no `BehaviorPage.tsx` in `viewer/src/chrome/`. | The brief's integration target is a mode inside an **unbuilt page**. `BEHAVIORAL-DIVERGENCE-PLAN.md` §8.1 already reserves that same unbuilt pill for a *different* study. Adding one for W1 means editing four files *and* a pinned test invariant, to fight a sibling plan for a slot neither has built. **§9 recommends not doing it.** |
| **A10** | A PCA projection is persistable for a fixed coordinate system. | **FALSE as implemented.** `backend/interp/bundles.py:146-166` `_pca_rows` computes the mean at :152 and the axes at :158-164 as **locals**, then returns only `(coords, evr, total_var)` at :166. The transform is discarded. | `BEHAVIORAL-DIVERGENCE-PLAN.md` §7.1.1 point 2 assumes this function yields a persistable transform. It does not. **If** W1 ever ships a fixed prompt-landscape, `_pca_rows` needs a variant returning `(mean, axes)` too. Since §9 recommends no projection for W1, this is *deferred, not fixed* — flagged so the sibling plan does not inherit the wrong assumption. |
| **A11** | An embedder is reachable on **port 11435** (not stock 11434). | **True per the tree's own corrected record; not re-probed by me.** `backend/embed.py:17-31` documents the 2026-08-13 correction verbatim, including that the earlier "nothing serves it" verdict was wrong *because it probed :11434*. It records mxbai-embed-large, 334M params, F16 GGUF, 1024-dim on :11435. **I did not send a probe**, so I am reporting the file's record, not a measurement. Note :45-51: the ollama tag is a **mutable** F16 GGUF, so it cannot satisfy "pinned to an exact revision". | Moot for W1 — §0.2 shows W1 needs no embedder. Recorded so the next reader does not repeat the :11434 mistake. If a prompt-landscape is ever added, the mutable-tag problem at :45-51 applies and must be solved, not ignored. |
| **A12** | GPT-2 could serve as a free local arm. | **FALSE, on three grounds — two now measured.** (1) *No generation path*: `backend/interp/gpt2_numpy.py` has `forward` (:129) and `logit_lens` (:183) and **no `generate`, no sampling, no KV cache**; `__init__` (:88) refuses any non-gpt2 id. (2) **Hard architectural cap, measured**: the loaded model reports **`n_ctx = 1024`**. A ~5,000-word story is ~6,700 tokens, so GPT-2 **cannot represent one at all** — it tops out near 750 words. This is a wall, not a slowdown. (3) **Speed, measured**: forward-pass wall clock on this machine was 0.384–0.412 s at T=16–64, 0.557 s at T=256, and **1.052 s at T=512**, with the materialized trace reaching **349.9 MB at T=512** and growing quadratically in the attention term. With no KV cache, generating *k* tokens costs *k* full forward passes over the whole prefix — so even a maximum-length 1024-token (~750-word) story is **roughly 12–15 minutes**, and 1,200 of them is on the order of **250+ hours** single-threaded. | **Every W1 and W2 arm is a paid API call**, and the reason is now stronger than "too slow": at `n_ctx = 1024` GPT-2 is *categorically incapable* of the task, so no amount of compute rescues it. This promotes the cost gate from a nicety to a Phase 0 blocker (§8.3) and makes gemma-4-26b's $0.00 endpoint (A6) strategically important rather than incidental. |

### 0.2 What the audit changes in the brief

Two of the brief's own framing assumptions did not survive, and both changes
*reduce* scope.

**The embedder apparatus does not apply to this study.** The brief carries over
§6.3.1 (a pinned in-process embedder), §6.5.4 (a second-embedder replication
gate), §4.3 (an embedder-disagreement downgrade rule), and §6.4.4 (differential
OOV and fragmentation) from `BEHAVIORAL-DIVERGENCE-PLAN.md`. All four exist
because that study's trial vector is *an embedding of free-text output*. W1's
trial vector is **the architecture score vector itself** — a fixed-length row of
answers to a fixed question set. No text is embedded anywhere in the primary
analysis. **Difference this makes:** four sections of machinery, one mutable-tag
pinning problem (A11), and one whole class of "did the embedder change the
answer" failure modes drop out of scope. The honesty burden does not vanish —
it *moves*, entirely, onto the scorer (§6).

An embedder becomes necessary again **only** if a prompt-landscape projection
ships. §9 recommends it does not.

**The integration target does not exist.** The brief says "a mode inside
Behavior." A9 shows Behavior is unbuilt and contested. **Difference this makes:**
§9 argues for a CLI report plus static JSON plus a Guide section, which leaves
`viewer/tests/unit/app-pages.test.ts` untouched and does not force a
land-grab on a pill that a sibling plan has already claimed.

### 0.3 What I could not verify

Stated plainly, per the brief's instruction not to present inference as
measurement:

- **A12 — now measured, superseding an earlier estimate.** Forward-pass times
  and trace sizes above are real measurements from this machine (weight load
  itself took ~1020 s over an unauthenticated HF download, which is why the
  number arrived late). What remains **derived rather than measured** is the
  per-story generation time: it is `k` forward passes × the measured per-pass
  cost, since no `generate` exists to time end-to-end. The `n_ctx = 1024` cap
  is read directly from the loaded model and is not an estimate.
- **A11 endpoint liveness.** Not probed. Reported from `embed.py`'s own
  2026-08-13 record. I did not send a request to :11435.
- **`nebulai-data/.pre-redaction-backup/`.** `docs/ONBOARDING.md:170` claims
  pre-redaction originals live there. Neither `find` nor `mdfind` located any
  `nebulai-data/` directory on this machine. Either the path is stale or the
  directory is elsewhere. **Consequence:** do not rely on those backups
  existing when planning any re-export.
- **Live pricing.** `corpus.py` prices were verified 2026-08-12 *by its own
  header comment* (:50-51). I did not re-query OpenRouter. All §8 figures
  inherit that staleness and must be re-checked at Phase 0.

---

## 1. Claim contract

### 1.1 What W1 may claim

The strongest default sentence:

> Under protocol P, exact model deployment M, sampling at fixed settings S,
> produced story-architecture vectors whose within-prompt dispersion was `d`
> on instrument Q.

W1 may also report:

- the ratio of between-prompt to within-prompt dispersion, with a CI;
- which individual architecture questions carry the variance and which are
  effectively constant for this model;
- per-question scorer reliability, as a first-class number displayed beside
  every result that depends on it.

### 1.2 What W1 must not claim

- **Not** "this model is creative/uncreative." Dispersion on a 30-question
  instrument is dispersion on a 30-question instrument.
- **Not** "AI stories cluster more tightly than human stories." W1 has no
  human arm. That is StoryScope's claim, made on StoryScope's design, and
  §2.2 explains why even *they* did not establish it in the form it is usually
  quoted.
- **Not** a comparison to any number in the StoryScope paper. Different
  models, different prompts, different question set, different scorer. The
  instruments are not interchangeable and the scores are not on a common scale.
- **Not** anything about *why* variance is high or low. W1 is descriptive.

### 1.3 Why the claim is deliberately narrow

Because the entire result is a function of a scorer we built. If the scorer
is unreliable on a question, every downstream statistic on that question is
noise wearing a number's clothes. The claim contract is narrow so that the
scorer-validity section (§6) is load-bearing rather than decorative — a wider
claim would let a reader skip it.

---

## 2. Evidence grounding

### 2.1 Evidence ledger

| ID | Source | What it contributes | Scope / caution | Reliability |
|---|---|---|---|---|
| G1 | *StoryScope: Investigating idiosyncrasies in AI fiction* — Russell, Rajendhran, Pham, Iyyer (UMD), Wieting (Google DeepMind). [PDF](https://jenna-russell.github.io/assets/pdf/storyscope.pdf) | The direct precedent. 10,272 prompts × (1 human + 5 LLMs) = 61,608 stories averaging **4,753 words**; 304 features across 10 dimensions; a compact **30 core + 75 fingerprint** feature set; LLM-based structured extraction; released code, prompts, and 51,336 AI narratives. | **"Preprint. Under review."** Not peer-reviewed. Its headline numbers are *classification* metrics, not dispersion metrics (§2.2). | Primary source, unrefereed. |
| G2 | G1's Table 7, human validation on **240 story-features across 12 stories** | The measured scorer-agreement prior: Annotator 1 vs model 91.67% / κ=0.9056; Annotator 2 vs model 79.86% / κ=0.7724; mean human vs model 85.76% / κ=**0.8390**; **human vs human 76.85% / κ=0.7385**. | 240 judgments spread over 304 features is **~0.8 judgments per feature** — the κ is pooled *across* features and establishes no per-question reliability (§6.1). | Measured, small-n, pooled. |
| G3 | G1's repeatability check: Gemini-3 over **5 independent runs**, Krippendorff's **α = 0.88**, 300 extraction outputs | Evidence that an LLM feature-extractor is stable enough to be an instrument. | This is **scorer** repeatability, not **generator** dispersion. The distinction is the whole of §2.2. | Measured. |
| G4 | `src/nebulai/backend/validate.py:33-46` — this project's own measurements | Silhouette **0.88 on shuffled noise vs 0.43 on real data** at n=180; leaf ARI 0.37 vs eom 1.00. | Specific to that dataset and n. Directional, not a universal threshold. | First-party measurement. |
| G5 | [A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a.html) | MMD² as a non-parametric test of whether two samples share a distribution. | Needs a fixed representation and a frozen kernel, chosen before confirmation rather than tuned per prompt. | Peer-reviewed. |
| G6 | [Benjamini–Yekutieli](https://doi.org/10.1214/aos/1013699998) | FDR control valid under arbitrary dependence — required because the ~30 questions are correlated by construction. | More conservative than BH; power must be measured in the pilot, not assumed. | Peer-reviewed. |
| G7 | Cohen's κ / Krippendorff's α, and the prevalence–κ paradox | The agreement statistics §6 is built on, and the known failure where high raw agreement yields low κ on skewed marginals. | κ is not comparable across questions with different base rates. This is why §6.3 pairs κ with PABAK/AC1 rather than replacing it. | Standard method. |
| G8 | `src/nebulai/corpus.py:52-155` | Real, pinned endpoint prices for the four corpus models. | Verified 2026-08-12 by the file's own header (:50-51); **not re-verified by me** (§0.3). | First-party, possibly stale. |

### 2.2 What StoryScope did and did not establish

This subsection is the load-bearing justification for W1, so it states the
distinction as precisely as the source permits.

**What it established.** That stories from five named LLMs (Claude, DeepSeek,
Gemini, GPT, Kimi — at dated releases, e.g. "Claude Sonnet 4.6 (Anthropic,
2026)") can be separated from human stories at **93.2% macro-F1**, and
attributed to one of six sources at **68.4% macro-F1**, using narrative
structure alone; and that this holds after length-matching (macro-F1 91.6 /
94.3 / 93.7 across short/medium/long bands). That an LLM extractor is a usable
instrument (G3). That a compact **30-feature** core carries much of the signal
— which is where the brief's "~30 questions" sizing comes from, and it comes
from the paper rather than from intuition.

**What it did not establish, and what W1 therefore supplies.**

*Separability is not relative dispersion.* 93.2% macro-F1 says the human and
AI regions have separable *locations*. Two clouds with different centroids are
separable at any dispersion ratio. The abstract's "AI-generated stories cluster
in a shared region of narrative space, while human-authored stories exhibit
greater diversity" is stated qualitatively; the quantitative backing in the
paper is classification performance plus group-level centroid/dispersion
statistics, and the paper itself concedes those "do not directly measure
whether individual human stories are more structurally unusual" — which is why
it adds a separate per-story rarity analysis (mean Euclidean distance to k=25
nearest neighbours). **Difference this makes:** the widely-quoted "AI is more
homogeneous" reading is a claim about dispersion resting substantially on
evidence about location, and W1 measures dispersion directly.

*The α = 0.88 repeatability figure is about the scorer, not the generator.*
It was computed over five independent runs **of Gemini-3 extracting features
from fixed stories**. It says the ruler is stable. It says nothing about
whether the thing being measured moves. I searched the extracted text for
temperature, sampling settings, and any repeated generation per prompt and
found **none** — the design is one story per prompt per source.

**Therefore:** the within-model, across-trial dispersion of story architecture
is not measured by the source study. W1's premise survives the audit intact,
and W1's number is the denominator that the source study's dispersion language
implicitly assumes but never measures.

---

## 3. Research questions and hypotheses

### 3.1 Primary questions

- **RQ1 (W1).** Holding model, prompt, and sampling settings fixed, how much
  does story architecture vary across independent trials?
- **RQ2 (W1).** Is that within-prompt variance small relative to the
  between-*prompt* variance for the same model? (If not, the instrument is
  not resolving prompt differences and W2 is pointless.)
- **RQ3 (W2, gated).** Does between-*model* distance exceed within-model
  distance, and by how much?

### 3.2 Predeclared hypotheses

- **H1.** Within-prompt dispersion is **non-zero and non-trivial** — the model
  does not produce one architecture per prompt.
- **H2.** Between-prompt dispersion **exceeds** within-prompt dispersion for
  the same model. *This is the operational hypothesis*: it is the instrument's
  own validity check, and it must be tested before RQ3 is asked.
- **H3 (W2 only).** Between-model dispersion exceeds within-model dispersion.

### 3.3 Falsifiers and downgrade rules

| Observation | Verdict |
|---|---|
| A question's κ lower CI bound falls below the floor (§6.2) | Question **dropped** from the primary vector, reported as `insufficient evidence`, and retained in raw form. |
| H2 fails — within ≈ between across prompts | **W2 is cancelled**, not postponed. The instrument does not resolve the thing W2 would compare. |
| Within-prompt dispersion ≈ 0 on nearly all questions | H1 falsified. Report it. A null that survives a power check is a finding, not a failed run. |
| Dispersion is driven by ≤ 2 questions | Report as `frame-specific`. A 30-question instrument carrying one question's signal is a one-question instrument with decoration. |
| Scorer intra-rater agreement (same input twice) < inter-rater agreement | Instrument **unstable**; halt. The ruler is moving. |

Underpowered null ≠ "the same". Any null result ships with the effect size the
design could have detected, or it is not reported as a null.

---

## 4. Study design

### 4.1 W1 — self-variance baseline (priority)

| Parameter | Value | Rationale / reversibility |
|---|---|---|
| Model | **1**, pinned by exact endpoint id from `corpus.py` | `gemma-4-26b` at $0.00/$0.00 (`corpus.py:94-95`) is the strong default: it makes the priority workstream nearly free. Caveat recorded: it is MoE with tied embeddings (:88, :96), so it is not representative of the dense untied arms W2 would use. **Reversible.** |
| Prompts | ~30, preregistered, spanning genre/length/constraint | Sized from G1's own 30-core-feature result. **Reversible** (extendable). |
| Trials per prompt | ~40 | Gives a usable within-prompt distribution and a split-half at 20/20. **Reversible upward.** |
| Sampling settings | Fixed, recorded verbatim, identical across all trials | The independent variable is *the sampler's own randomness*. Anything else varying makes the number mean something else. |
| Story length | Target ~5,000 words, per G1's 4,753-word mean | Comparability of *design*, never of *scores*. **Reversible** but changes the cost envelope linearly. |
| Question set | ~30 architecture questions, fixed **before** collection | **ONE-WAY DOOR** — see §6.5. |
| Scorer | One model, pinned, minimal reasoning, structured output | Follows G1's production choice (Gemini 3 Flash, minimal thinking). **One-way door once collection starts.** |

Total: ~1,200 stories. Ordering randomized and time-blocked so any provider-side
drift is a nuisance factor that can be tested for, not a confound baked into
prompt order.

### 4.2 W2 — cross-model divergence (gated)

Runs **only if** H2 holds. Same prompts, same question set, same scorer,
several models from `corpus.py`. The headline is `ρ̂` = between-model MMD² ÷
mean within-model MMD², because W1 will by then have supplied the denominator.

**Capability confound, stated up front:** the corpus models differ in size,
architecture, and post-training. A between-model difference in story
architecture is not evidence of a difference in "narrative preference" if the
weaker model simply writes worse. `BEHAVIORAL-DIVERGENCE-PLAN.md` §5.7 solves
this with a same-family scale pair. **W2 has no such pair available in
`corpus.py`** — the four entries are four different families. Until one exists,
W2 results are reported as `capability-attributable` unless a same-family
control is added. This is a real limitation, not a caveat to bury.

---

## 5. Reuse map

What W1 takes from the existing system, and what it must build.

| Need | Reuse or build | Grounding |
|---|---|---|
| Trial representation | **Reuse `Units`** — `ids` = trial index, `vectors` = (n_trials, n_questions) float32 score rows, `labels` = "prompt/trial", `meta` = provenance | `units.py:6-30`. The whole back-end sees only this shape (:8-13). Largest single reuse win. |
| Reduce → cluster → name → export | **Reuse unchanged** | Guaranteed by A3: the pipeline is defined over `Units`, not over embeddings. |
| Projection honesty | **Reuse `validate.py`** — `trustworthiness_score` (:91), `seed_stability` (:134), `null_baseline` (:210) | A8. Free harness. Mandatory for any per-prompt view (n≈40, well inside the regime where G4 measured silhouette 0.88 on pure noise). |
| Runner / job shape | **Reuse the probe branch pattern** — `build_server.py:150-165` | A4. Probe is the existing precedent for a front-end with **no model weights**, which is W1's exact shape. Includes the "no cache, always fresh" rule (:197-198). |
| Provenance stamping | **Reuse `public_embed_host`** if any host is stamped | `embed.py:91-129`. A5. |
| Cost ceiling | **Reuse `cost_gate`, after fixing its fail-open branch** — **built 2026-08-16** as `cost_gate(..., require_price=True)` plus `llm.RunBudget` for the cumulative run ceiling | `llm.py:175,198`. A7. See §8.3 — this was a build, not a pure reuse. |
| Budget/identity errors | **Reuse `BudgetError` / `IdentityError`** | `llm.py:55,64`. |
| Instrument freeze | **Built 2026-08-16** — `backend/instrument.py`: file format, content hash, `require_frozen()`, load-time tamper check, `require_compatible()`. Ships no questions. | §6.5's one-way door, mechanised. The trial matrix it emits is exactly the `Units.vectors` shape above, so the reuse win is realised rather than asserted. |
| Persistable PCA transform | **Build, only if a landscape ships** | A10 — `bundles.py:146-166` discards mean and axes. §9 recommends deferring. |
| Story generation runner | **Build** | Nothing in the tree generates long-form fiction. |
| Architecture scorer | **Build** | The core new component, and the only one §6 governs. |
| Embedder integration | **Do not build** | §0.2 — not needed for this study. |

Net: W1 builds **three** things (generation runner, scorer, cost-gate fix) and
reuses the rest — plus the instrument freeze, which the original count missed
because §6.5 described it as a policy rather than as code. Policies that are
only written down are not mechanisms, and this one guards the study's most
expensive irreversible step.

**As of 2026-08-16 the cost-gate fix and the instrument freeze are built and
tested; the generation runner and the scorer are not.** Both remaining items sit
behind Phase 0 items 1–6, which are human work.

---

## 6. Scorer validity plan

Everything W1 claims is a function of the scorer. This section is the reason
the claim contract in §1 is narrow.

### 6.1 Why the source study's validation is not sufficient for us

G2 is a real validation and it is better than nothing, but it does not license
skipping our own, for two reasons.

**It is pooled.** 240 story-feature judgments spread across 304 features is
roughly 0.8 judgments per feature. The reported κ = 0.8390 is an *aggregate
across heterogeneous questions*. It establishes that the extractor is broadly
sane. It establishes **no per-question reliability whatsoever** — and W1's
primary output is precisely a per-question variance decomposition (RQ1, and
the `frame-specific` downgrade in §3.3). A pooled κ cannot tell us which of our
~30 questions to trust.

**Our instrument is not theirs.** Different questions, different model,
different prompts. Agreement does not transfer across instruments.

**Difference this makes:** we must run our own per-question validation, and it
will be roughly **6× larger** than the source study's (50 stories × 30
questions = 1,500 judgments vs their 240). That is a defensible position to be
in — our validation would exceed the precedent's — and it should be stated
plainly rather than treated as an embarrassment of cost.

### 6.2 The floor must be set against the human ceiling, not against 1.0

This is the finding that most changes the brief's §4.

G2 measured **human vs human at κ = 0.7385** — *below* mean human-vs-model
agreement at κ = 0.8390. Two trained annotators, on the paper's own instrument,
agreed with each other less than the model agreed with them.

**Therefore a scorer at κ ≈ 0.75 is at the ceiling, not failing it.** A naive
"κ ≥ 0.8" floor would discard questions on which the scorer is performing as
well as a human being can be expected to. **Difference this makes:** the floor
is expressed *relative to a measured human–human ceiling on our own gold set*,
not as an absolute constant imported from a rubric.

Concretely:

1. Two annotators independently label the gold set.
2. Compute human–human κ **per question**. This is the ceiling, `κ_H`.
3. Compute mean human–model κ per question, `κ_M`.
4. A question passes if the **lower bound of the CI on `κ_M`** clears the floor.
5. The floor is the lesser of a fixed absolute minimum (a question nobody can
   score reliably is useless regardless of why) and a fraction of `κ_H` for
   that question.

Questions where `κ_H` is itself very low are **dropped as badly specified** —
that is a defect in our question, not in the scorer, and rewriting the question
is the fix.

### 6.3 Statistics, and the prevalence trap

- **Cohen's κ** per question, for two-rater agreement.
- **Krippendorff's α** where a question is ordinal or has missing values, and
  for the repeatability check (matching G3's method so the numbers are at
  least methodologically comparable).
- **PABAK / Gwet's AC1 reported alongside κ**, because several architecture
  questions will have skewed marginals (most stories will not contain a dream
  sequence), and skewed marginals drive κ toward 0 even at 95% raw agreement.
  Reporting κ alone would delete exactly the rare-feature questions that carry
  the most information. **Difference this makes:** a rare-but-reliably-detected
  feature survives the cut instead of being discarded as noise.
- **Raw percent agreement** always shown, because it is the number that
  reveals a prevalence paradox at a glance.

### 6.4 Sample size, and the reason it is not smaller

κ's standard error at n = 50 is roughly **±0.15–0.20**. A point estimate of
0.80 with that SE has a lower bound near 0.60. **This is why §6.2 step 4 tests
the lower bound rather than the point estimate** — testing the point estimate
would pass questions whose true reliability is unknown.

It also means the gold set cannot be much smaller than ~50 stories per question
without the CI becoming too wide to make any decision at all. **The binding
constraint on W1 is not dollars — it is human labelling hours** (§8.4).

### 6.5 Intra-rater and the one-way door

**Intra-rater check:** the scorer re-scores a held-out subset it has already
seen, blind. If intra-rater agreement is *below* inter-rater agreement, the
instrument is unstable and the run halts (§3.3). This is the check that catches
a scorer whose output depends on sampling noise rather than on the story — and
it is the direct analogue of G3, which is the one validation the source study
did that we must replicate rather than merely cite.

**The one-way door.** Scores are only comparable within a fixed instrument.
Changing, adding, or reordering a question after collection begins invalidates
**every trial already scored** — there is no rescue short of re-scoring the
whole corpus. **Therefore:** the question set is frozen at the end of Phase 0,
in a file, with a hash recorded in every artifact's provenance. Everything else
in §4.1 (model, trial count, prompt set) is reversible and may be extended
later without invalidating earlier data.

---

## 7. Data treatment and statistical contract

- **Immutable raw evidence.** Every generated story and every raw scorer
  response is written once and never mutated. Aggregates are always
  recomputable from raw. (Noting A5: shipped artifacts in this project *have*
  been edited in place before, so "immutable" is a rule to enforce, not a
  property to assume.)
- **Trial vector.** The ~30 question scores, in frozen order. Missing is
  encoded as `missing`, never as `0` — a question the scorer declined is not a
  question answered "no", and collapsing the two would manufacture agreement.
- **Primary statistic.** `ρ̂` = between-group MMD² ÷ mean within-group MMD²,
  with a frozen Gaussian kernel whose bandwidth is fixed on **pilot data
  only**, before confirmation. A raw between-distance is displayed beside `ρ̂`
  and is never the rank key — that is the §2.2 error in statistical form.
- **Uncertainty.** Bootstrap CIs over trials; permutation p-values computed
  **within time blocks**, so provider-side drift cannot masquerade as signal.
- **Multiple testing.** Benjamini–Yekutieli at q ≤ 0.05 across questions (G6).
  BY rather than BH because the questions are correlated by construction and
  BH's independence assumption is not defensible here.
- **Split-half reliability** on every reported effect: 20/20 within each
  prompt's 40 trials.
- **Evidence states**, carried through to every display:
  `confirmed` · `suggestive` · `frame-specific` · `unstable` ·
  `capability-attributable` · `insufficient evidence`.
  *Terminology note:* the brief also listed `incomparable`. In this tree that
  word appears only in prose (`src/nebulai/seer/compare.py:133`,
  `seer/analysis.py:44`) and is **not** a state name — Seer's `Refusal`
  dataclass carries the grounds instead. Using it as a state here would invent
  a status the codebase does not have.
- **Absence has ink.** A question dropped for low κ is *displayed as dropped*,
  with its κ and CI. A refusal is printed as the finding. An underpowered null
  ships with its detectable effect size or is not called a null.

---

## 8. Cost envelope

All figures use `corpus.py` prices (G8), which were verified 2026-08-12 by that
file's own header and **not re-verified by me** (§0.3). They are estimates from
token arithmetic, not measurements, and must be re-derived at Phase 0.

### 8.1 W1 generation

Assume ~6,700 output tokens per ~5,000-word story, short prompts, 1,200 trials:
≈ **0.36M input / 8.04M output tokens**.

| Model | $/M in / out | Estimated W1 generation cost |
|---|---|---|
| **gemma-4-26b** | 0.00 / 0.00 | **$0.00** |
| ling-2.6-flash | 0.010 / 0.030 | ≈ $0.245 |
| mistral-nemo | 0.019 / 0.030 | ≈ $0.248 |
| muse-glimmer-30b | 0.35 / 1.50 | ≈ **$12.19** |

**muse-glimmer-30b alone exceeds `DEFAULT_MAX_COST_USD = 1.00` by ~12×**
(`corpus.py:173`). At frontier pricing (~$3/$15) the same run is ≈ $121.68.

### 8.2 W1 scoring

Each story must be read by the scorer: ≈ **9.0M input / 0.48M output tokens**.
At ling-flash pricing ≈ **$0.104**; at $1/$5 ≈ $11.40; at $3/$15 ≈ $34.20.

**Wall clock** is the underrated cost: ~6h+ of generation at 4 concurrent
requests. W1 is an overnight run, which is precisely the situation in which an
unattended fail-open budget gate is dangerous.

**Storage** is negligible: ~36 MB of text, ~100 MB SQLite.

### 8.3 The cost gate is a Phase 0 blocker

A7 established that `cost_gate` (`llm.py:175`) **returns `None` — permitting
the call — for any model with no corpus price** (:198), that it is invoked from
only four sites (`name.py:782,858`; `probe.py:420,432`), and that
`chat_openai` (:448) has no gate of its own.

A12 established that **every W1 arm is a paid call**, because GPT-2 cannot
generate.

Those two facts together are the risk: an overnight, unattended, 1,200-call run
against a model that happens not to be in `CORPUS` would bill without a
ceiling, and the only signal would be a printed line nobody is awake to read.

**Required before Phase 1** (all three; **reversible** implementation choices,
non-negotiable outcome). **All three landed 2026-08-16** — this blocker is
cleared:

1. `cost_gate` **fails closed** for W1 — an unknown price raises `BudgetError`
   rather than printing and continuing. The existing fail-open behaviour may
   remain for existing callers if changing it is judged too disruptive, but W1
   must not inherit it.
   → **Done:** `cost_gate(..., require_price=True)` (`llm.py`). The default
   stays `False`, so the four existing call sites are untouched; the refusal
   message lists priceable alternatives rather than dead-ending, matching the
   over-budget branch's existing convention.
2. The W1 runner routes **every** call through the gate, with a **run-level**
   cumulative ceiling, not a per-call one. A per-call ceiling never trips on
   1,200 individually-cheap calls.
   → **Done:** `llm.RunBudget`. `charge_response(model_id, payload)` takes one
   provider response, accumulates its reported usage, re-prices the *cumulative*
   total through the existing `actual_cost`, and raises `BudgetError` when the
   run passes its ceiling. It charges **measured** usage, not the estimate —
   the estimate is an upper bound by construction, so charging it would trip the
   ceiling on a run that was in fact affordable. An unpriceable model raises on
   charge rather than accumulating as $0, closing the same fail-open hole from
   the other end. The regression test is the literal §8.3 scenario: eight calls
   that each clear a per-call ceiling nine times over, and a ninth that trips
   the run.
3. A pre-flight estimate is printed and **confirmed** before the first paid
   token, following the upper-bound convention `estimate_naming_cost`
   already sets (`corpus.py:186-195`).
   → **Done:** `RunBudget.preflight()` prints the estimate and labels it an
   upper bound; `approve()` records confirmation and **cannot be called before**
   `preflight()`; `charge*()` raises until it has been. Approval is a step, not
   a flag — which is what the brief asked for ("Not a toast — a step"). An
   estimate that already exceeds the ceiling raises at pre-flight, before
   anything is sent.

**Still open, and deliberately so:** wiring these into a W1 runner. There is no
runner yet, because Phase 0 items 1–7 come first and none of them generate a
story. The gate is built before the thing it gates, which is the correct order.

### 8.4 The real constraint

Dollars are not the binding constraint for W1 — **human gold-set labelling time
is**. Two annotators × 50 stories × 30 questions = 3,000 human judgments, on
~5,000-word stories. If that budget does not exist, W1 does not run, because
§6 is not optional. Choosing gemma-4-26b makes generation free and throws this
into sharper relief rather than softening it.

---

## 9. Viewer integration

**Recommendation: no new page and no new nav pill. Ship a CLI report, a static
JSON artifact, and a Guide section.**

This is argued, not assumed, because the brief assumed the opposite.

**The premise is false.** The brief places W1 as "a mode inside the Behavior
page." A9 established that Behavior does not exist: Nebul.AI's nav is three
pills (`chrome/apps/nav.ts`), `shell.ts:36` types exactly six pages, and
`:46-49` partitions them across the two instruments.

**The cost of adding one is higher than it looks.**
`viewer/tests/unit/app-pages.test.ts:29` hardcodes `ALL_PAGES` as those six and
pins four invariants: pills equal `APP_PAGES` in order, each app boots on its
first pill, no page is owned twice, none is orphaned. Adding a page means
editing `shell.ts`, `nav.ts`, a new page component, the `renderPage` switch,
**and** a test that deliberately exists to make exactly this change loud.

**And the slot is contested.** `BEHAVIORAL-DIVERGENCE-PLAN.md` §8.1 already
claims the "Behavior" pill for a *different, also-unbuilt* study. Two unbuilt
studies competing for one unbuilt pill is a coordination problem, and W1 —
which is a **measurement**, not an interactive instrument — is the weaker
claimant.

**What W1 actually needs to show** is a ranked table of questions by variance
contribution, each with κ, CI, and evidence state. That is a table. It does not
need a 3-D scene, and it does not need the shared bundle.

**Difference this makes:** the recommendation leaves `app-pages.test.ts`
untouched, ships W1's findings without blocking on a page neither plan has
built, and keeps the "Behavior" pill free for whichever study earns it. It is
**fully reversible** — if W1's results justify a page later, the static JSON is
already the data contract that page would read.

**Corollary — no projection.** A prompt-landscape view would drag in the
persistable-PCA gap (A10) and, at n≈40 per prompt, sit squarely in the regime
where this project *measured* silhouette 0.88 on shuffled noise (G4). Not
worth it for a table. If it is ever built, `_pca_rows` must first be extended
to return `(mean, axes)`.

---

## 10. Phases and gates

Each gate has a stop condition. A gate that cannot fail is not a gate.

### Phase 0 — instrument definition and scorer validation

*No stories are generated in this phase.*

1. Draft ~30 architecture questions with explicit scoring rubrics. — *open;
   authored by a human, not by the tooling that validates them.*
2. Draft ~30 prompts. — *open.*
3. Generate a **small** pilot set of stories for the gold set only. — *open.*
4. Two annotators independently label 50 stories × 30 questions. — *open, and
   this is the binding constraint (§8.4), not a code task.*
5. Compute `κ_H` (human–human), `κ_M` (human–model), α, PABAK/AC1, raw
   agreement — **per question**, with CIs. — *open.*
6. Run the intra-rater repeatability check (§6.5). — *open.*
7. Freeze the question set and record its hash. — **mechanism done
   2026-08-16, freeze not performed and must not be.** `backend/instrument.py`
   provides `QuestionSet.freeze()`, a content hash over `(id, text, scale)` in
   order, `require_frozen()` gating collection, load-time tamper detection, and
   `require_compatible()` refusing to pool across instruments. Editorial fields
   are excluded from the hash so notes stay writable after the door closes.
   **No questions ship**: the template in `docs/instruments/` uses the reserved
   `example_` id namespace, which `freeze()` refuses outright — so the format
   can be read without the template becoming an instrument. The freeze itself
   waits on items 1–6, by design.
8. Fix `cost_gate` per §8.3. — **done 2026-08-16.** See §8.3 for what landed.
9. Re-verify `corpus.py` prices against live pricing (§0.3). — *open; requires
   checking live provider pricing, and it must be redone immediately before
   Phase 1 regardless, since a price verified weeks early is not verified.*

**GATE 0 — stop conditions.** Halt and report if: fewer than ~20 questions
clear the §6.2 floor on their lower CI bound (an instrument too thin to carry a
variance decomposition); **or** intra-rater agreement is below inter-rater
agreement (§3.3 — the ruler is moving); **or** the human labelling budget is
unavailable. Halting here costs ~$0 in API spend. **This is the cheapest gate
in the plan and the only one that can prevent a wasted overnight run.**

### Phase 1 — W1 collection

Generate ~1,200 stories, randomized and time-blocked, under the run-level cost
ceiling. Score with the frozen instrument. Write raw evidence immutably.

**GATE 1 — stop conditions.** Halt if the run-level ceiling trips; if the
provider serves a different model id than requested (`IdentityError`,
`llm.py:55` — same rule, applied to the generator); or if the scorer's refusal
rate on any question exceeds a preregistered threshold.

### Phase 2 — W1 analysis

Compute within- and between-prompt MMD², `ρ̂`, bootstrap CIs, within-block
permutation p-values, BY correction, split-half reliability. Assign evidence
states. Emit static JSON + CLI report.

**GATE 2 — the W2 gate.** **If H2 fails — within-prompt dispersion is not
meaningfully below between-prompt dispersion — W2 is cancelled, not
postponed.** Report W1's null with its detectable effect size. Do not proceed
to spend money comparing models on an instrument that cannot separate prompts.

### Phase 3 — W2 (only if Gate 2 passes)

Same instrument, several models. Report `ρ̂` against W1's measured denominator.
All results carry `capability-attributable` unless a same-family scale control
has been added (§4.2).

---

## 11. Risks

| Risk | Mitigation | Residual |
|---|---|---|
| Scorer is the whole instrument and could be systematically wrong in a way both annotators share | Human–human ceiling measured independently (§6.2); intra-rater check (§6.5) | Shared human bias is not detectable by this design. Stated as a limitation. |
| Fail-open cost gate + unattended overnight paid run | §8.3, all three items, before Phase 1 | Provider-side price changes mid-run. |
| Prevalence paradox deletes rare-feature questions | PABAK/AC1 beside κ (§6.3) | Genuinely rare features still have wide CIs. |
| Provider drift over a 6h run | Time-blocked randomization; within-block permutation (§7) | Undetectable drift faster than one block. |
| `corpus.py` prices stale since 2026-08-12 | Re-verify at Phase 0 (§0.3) | — |
| Choosing gemma-4-26b (MoE, tied) makes W1 unrepresentative of W2's dense arms | Stated in §4.1; W2 must not assume W1's denominator transfers across families | Real. May require a second W1 arm before W2. |
| Question set frozen too early | Phase 0 pilot exists precisely to iterate before freezing | One-way door by nature (§6.5). |

---

## 12. Non-goals

- Not a creativity benchmark or a model leaderboard.
- Not a replication of StoryScope, and not comparable to its numbers.
- Not a human-vs-AI study — there is no human arm.
- Not an explanation of *why* variance is high or low.
- Not a new viewer page (§9).
- Not an embedder project (§0.2).

---

## 13. Part B — ranked recommendations

Ranked by value per unit of risk. Includes a rejection, as requested.

1. **Fix the fail-open `cost_gate` regardless of whether W1 proceeds.**
   `llm.py:198` permits unbounded spend for any model absent from `CORPUS`, and
   `chat_openai` (:448) has no gate at all. This is a standing financial defect
   in the repo, not a W1-specific one. Cheapest, highest-value item here.

2. **Run Phase 0 alone and treat it as its own deliverable.** A per-question
   reliability table for a 30-question narrative-architecture instrument,
   validated against a measured human–human ceiling, is publishable on its own
   and is ~6× the validation the source study performed. It costs almost no API
   spend and it de-risks everything downstream.

3. **Adopt "floor relative to measured human ceiling" as a house rule.** G2 is
   direct evidence that an LLM scorer can beat human–human agreement. Any
   future plan in this repo that declares an absolute κ floor is one measurement
   away from discarding a scorer for outperforming its ground truth.

4. **Give `_pca_rows` a persistable variant now, while it is cheap.**
   `bundles.py:146-166` discards mean and axes;
   `BEHAVIORAL-DIVERGENCE-PLAN.md` §7.1.1 already assumes it does not. Returning
   `(mean, axes)` is a small additive change today and a data-migration
   tomorrow.

5. **Resolve the "Behavior" pill contention before either study builds
   anything.** Two plans claim one unbuilt slot (A9). Decide now whether
   Behavior is one page with modes or two separate surfaces — the
   `app-pages.test.ts` partition invariant makes this expensive to get wrong.

6. **Locate or write off `nebulai-data/.pre-redaction-backup/`.**
   `docs/ONBOARDING.md:170` claims it holds pre-redaction originals; I could not
   find it anywhere on this machine (§0.3). Either fix the path or correct the
   doc, because a recovery plan that points at a missing directory is worse than
   no recovery plan.

7. **REJECTED — crossing W1 with Seer to measure "agent run variance."**
   Superficially the same shape (repeated trials, structured scoring,
   dispersion) and it would reuse Seer's existing capture. It fails on two
   independent grounds. *Statistically:* agent runs **mutate a shared
   workspace**, so trials are not independent draws — run *n* changes the
   initial conditions of run *n+1*, and MMD² over dependent samples does not
   mean what §7 says it means. *Architecturally:* Seer already has a refusal
   engine (`seer/compare.py`) whose `Refusal` dataclass declines comparison on
   category mismatch, fidelity mismatch, and capture gap — it exists because
   this project already concluded that agent runs are frequently not comparable.
   Building a variance metric that ignores that engine's own grounds would be
   arguing with the codebase's settled position. Revisit only with genuinely
   isolated per-run environments, which is a much larger project.

---

## 14. Revision log

- **2026-08-16 — initial draft.** Grounding audit performed against the tree at
  the cited paths. A4 and A5 independently confirmed FALSE as the brief
  predicted; A9, A10, and A12 found FALSE and not predicted. StoryScope traced
  to primary source and its validation table extracted (G1–G3). No code written,
  no paid call made, nothing committed.
- **2026-08-16 — A12 upgraded from estimate to measurement.** A local GPT-2
  timing run completed after the first draft. It replaced the estimated
  "too slow" argument with a stronger measured one: `n_ctx = 1024` makes
  ~5,000-word stories *impossible*, not merely expensive. §0.1 A12, §0.3, and
  the glance table were corrected. No other conclusion changed — GPT-2 was
  already excluded, and it is now excluded for a better reason.
- **2026-08-16 — approved; the two unblockable build items landed.** §8.3's
  cost gate (all three requirements) and §6.5's instrument freeze were
  implemented with tests, and the status header, §5, §8.3 and §10 Phase 0 were
  corrected to say so. **No question set was authored or frozen, no story was
  generated, and no paid call was made** — those are Phase 0 items 1–6, they are
  gated on human annotation time (§8.4), and item 7 in particular is the
  one-way door §6.5 exists to protect. The two items built are precisely the
  ones that must exist *before* spending is possible, so building them first is
  the order the plan asks for rather than an early start on the study.
- **2026-08-16 — A10 corrected in the sibling plan.** `BEHAVIORAL-DIVERGENCE-PLAN.md`
  §7.1.1 told the reader to "reuse" `_pca_rows` and treated its
  fixed-coordinate promise as thereby satisfied. It is not: `_pca_rows`
  (`bundles.py:146-166`) discards both the mean vector and the axes as locals,
  which is the same "coordinates, not a transform" defect that section holds
  against `reduce_vectors` two paragraphs earlier. The section now specifies
  the additive change, the three unaffected callers, and the round-trip test
  that would make the promise real. This is A10 in this document's audit,
  propagated to the plan that depends on it.
