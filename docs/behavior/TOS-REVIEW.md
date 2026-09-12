# xAI Terms of Service review — recorded review form

> **THIS IS AN EMPTY TEMPLATE. NO REVIEW HAS BEEN PERFORMED.**
>
> Every field below is deliberately blank. A blank field is not an oversight to
> be tidied up; it is the accurate current state of this record, and it is the
> only honest thing this file can say until a person fills it in.
>
> **This review requires a human.** It cannot be performed, drafted, guessed,
> summarised or "pre-filled pending confirmation" by an agent, and no agent may
> fill any field below on a human's behalf. The reason is not ceremony. The
> output of this review is a legal reading that decides whether a named
> commercial model's outputs may be published in a comparison, and a plausible
> paraphrase of a terms document is indistinguishable on the page from a
> reading of the actual document. The gate exists precisely to make the
> difference visible, so the only thing that can satisfy it is a named person
> who has opened the current terms and read them.
>
> **Until this file is completed, signed and dated, the xAI arm of the
> behavioural divergence study must not run.** That means: no Grok trials of
> any lane, no confirmation partition, and no shareable artifact. The GPT-2
> local arms and the §5.7 GPT-2-small↔XL capability-control pair are
> unaffected — they are local, spend nothing, and have no provider terms
> surface (§5.7).

## Why this file exists

`docs/BEHAVIORAL-DIVERGENCE-PLAN.md` §11 Phase 3 opens with a prerequisite
before the deliverables, not after them:

> Read the current xAI terms, record the date and the relevant clauses in the
> study manifest, and decide the publication surface **before** collecting the
> confirmation partition — not after.

Phase 3 ships "a shareable redacted artifact" that compares a named commercial
model against another model. Providers commonly restrict using their outputs to
evaluate, benchmark or train against other models, and commonly restrict
publishing comparative results. Phase 3's first gate condition is therefore
"a recorded ToS review exists and the artifact's distribution matches it", and
§13's risk table commits to the terms being "reviewed, dated, and quoted in the
manifest *before* the confirmation partition is collected".

This file is that record. Its completed contents are what §9.4's manifest
minimum refers to when it requires the terms review alongside the dated Grok
release and the capability result; the manifest should carry the same values,
not a second independent reading of them.

Nothing in this document states or implies what xAI's terms say. Deciding that
is the whole job of the review, and a template that guessed at the answer would
have already done the damage it exists to prevent.

---

## 1. Reviewer identity

**Reviewer name:**

> A single named human being who personally opened and read the terms. A team
> name, a role ("legal"), a handle with no person behind it, or an agent or
> model name is invalid. If more than one person read the terms, name the one
> accountable for this record and list the others under notes.

**Role / relationship to the project:**

> Enough for a later reader to know in what capacity this person signed, e.g.
> principal investigator, project owner, counsel. Invalid if it does not
> identify a real accountable position.

**Date of review (ISO 8601, YYYY-MM-DD):**

> The date the terms were actually read, not the date this file was edited or
> committed. A range, "circa", or a date copied from an earlier review is
> invalid.

**Reviewed under legal advice?  yes / no:**

> A plain yes or no. If yes, say who advised under notes. "Unclear" is not an
> answer; the reviewer either took advice or did not.

## 2. The document that was reviewed

**Exact URL of the terms reviewed:**

> The full URL of the specific document, as fetched. A provider's top-level
> site, a docs landing page, a search result, or a link that merely redirects
> somewhere is invalid — the record must identify the page whose sentences were
> read.

**Document title as published:**

> The title as it appears on the page itself, copied, not normalised. Invalid if
> it is the reviewer's description of the document rather than its own name.

**Version, revision, or "last updated" line as published:**

> Whatever version marker the document carries, transcribed exactly. If the
> document publishes no version marker at all, write that it publishes none —
> that fact is itself part of the record and is materially different from
> leaving the field blank.

**Other documents incorporated by reference that were also read:**

> Terms routinely pull in an acceptable use policy, a developer or API
> agreement, a service-specific addendum, or a privacy policy. List each with
> its own URL and version line. Invalid if it names a document that was not
> actually opened, and invalid if the reviewer knows a referenced document
> exists and omitted it.

**Local snapshot of the reviewed text (path or archive URL):**

> A copy of the page as read, so a later dispute can be settled against the text
> the decision was actually made on. Terms change silently and a live URL is not
> evidence of what it said today. Invalid if it points at a live URL only.

**Retrieval hash of that snapshot:**

> A content hash of the stored snapshot. Invalid if it was computed over a
> different file than the one at the path above.

## 3. The clauses that govern this study

For each item, quote the governing clause verbatim with its section number, then
state what it permits or prohibits *for this study specifically*. A summary
without a quote is invalid — the plan asks for clauses to be "quoted in the
manifest", because a paraphrase cannot be re-checked when the terms change.
"Nothing in the terms addresses this" is a valid and useful answer when it is
true, and must be written out rather than left blank.

**3.1 — Using model outputs to evaluate or benchmark the provider's model:**

> Quote plus verdict. Lane A collects free-association outputs from a pinned
> Grok release for measurement. If this is restricted, the study as designed
> does not run and no later field can rescue it.

**3.2 — Comparing the provider's model against another named model:**

> Quote plus verdict. The study's entire output is a GPT-2↔Grok contrast; a
> restriction here is fatal to the comparison, not merely to its publication.

**3.3 — Publishing comparative results, including aggregate statistics:**

> Quote plus verdict, and say explicitly whether aggregate or de-identified
> statistics are treated differently from raw outputs. Invalid if it answers
> only for raw outputs when the artifact ships aggregates.

**3.4 — Redistributing raw or redacted model outputs:**

> Quote plus verdict. Phase 3's deliverable is a redacted artifact; this field
> decides whether redaction is sufficient or whether outputs cannot leave the
> local evidence store at all.

**3.5 — Using outputs to train, fine-tune, distil, or improve another model:**

> Quote plus verdict. The study does not train on outputs, but embedding them
> with a third-party embedder is an adjacent activity and the clause language
> often sweeps wider than "training". Say whether the clause reaches §6.3.1's
> in-process embedding.

**3.6 — Attribution, naming, trademark, and disclosure requirements:**

> Quote plus verdict, including any required wording. Invalid if it states that
> attribution is required without recording the exact form required.

**3.7 — Rate limits, automation, and programmatic access constraints:**

> Quote plus verdict. §5.4's interleaved time blocks and the canary probe in
> §5.5.1 are automated, high-volume and long-running; the review must confirm
> the protocol is permitted as designed.

**3.8 — Restrictions on prompt content (the §5.3 cue suites include identity,
slang and deliberately provocative terms):**

> Quote plus verdict. This is the field most likely to be skipped and most
> likely to matter: the cue suites are provocative by design and an acceptable
> use policy may prohibit exactly them.

**3.9 — Data retention, logging, and confidentiality of outputs:**

> Quote plus verdict against §6.1's immutable append-only local evidence store.
> Invalid if it does not address retention of the raw outputs specifically.

**3.10 — Anything else in the reviewed documents that constrains this study:**

> The reviewer's own catch-all. Write "none found" only if the documents were
> read end to end; blank is not the same statement.

## 4. Decisions that follow

**Does the xAI arm run at all?  yes / no / conditionally:**

> A decision, not an assessment. If conditional, the conditions belong in the
> next field and each must be checkable by someone who was not present for the
> review.

**Conditions or modifications required before it runs:**

> Concrete and verifiable, e.g. a lane removed, a cue pack dropped, a rate
> ceiling, a required disclosure. "Proceed with care" is invalid.

**Publication surface decided (§11 Phase 3 requires this decided BEFORE the
confirmation partition is collected):**

> Exactly where the Phase 3 artifact may go — private to the project, shared
> with named parties, published publicly, or not distributed at all. Invalid if
> it is deferred, because deferring it is the failure the prerequisite exists to
> prevent.

**What the artifact may contain at that surface:**

> Raw outputs, redacted outputs, aggregate statistics only, or metrics with no
> outputs. Must be specific enough that the exporter can be configured from this
> line alone.

**Required attribution or disclaimer wording, verbatim:**

> The exact string that must appear, if any. A description of the required
> wording is invalid; the artifact will carry a string, and this is where that
> string is fixed.

**Re-review trigger and expiry date:**

> Terms change. State the date by which this review must be redone, and the
> events that invalidate it early — at minimum a change to the reviewed
> documents' "last updated" line, a change of Grok release, or a widening of the
> publication surface. A review with no expiry is invalid.

## 5. Manifest linkage

**Study ID(s) this review authorises:**

> The specific study manifest IDs (§9.4). A review is scoped to the studies it
> names; a blanket authorisation for future studies is invalid.

**Grok dated release this review was performed against:**

> The exact dated release resolved by §5.5, not a moving alias. Research mode
> already refuses aliases such as `latest`, and a review against an alias
> authorises nothing.

**Manifest fields updated to carry these findings:**

> Which keys in the study manifest now hold the date, quoted clauses and
> publication-surface decision recorded above. Invalid if this file and the
> manifest disagree.

## 6. Sign-off

By signing, the named reviewer states that they personally opened and read the
documents listed in §2 on the date given in §1, that the quotations in §3 are
transcribed from those documents rather than recalled or summarised, and that
the decisions in §4 are theirs.

**Signature (name, typed):**

**Date signed (YYYY-MM-DD):**

**Sign-off status:**

> One of: `NOT REVIEWED` (the state of this file as written), `REVIEWED —
> ARM MAY RUN`, `REVIEWED — ARM MAY RUN UNDER §4 CONDITIONS`, or `REVIEWED —
> ARM MUST NOT RUN`. Any other value, or a blank, means the Phase 3 gate is
> not satisfied.

**Current status of this file: `NOT REVIEWED`.**

## 7. Notes

> Anything the fields above did not fit: who else read the terms, what advice
> was taken, ambiguities the reviewer could not resolve, and questions left open
> for counsel. Ambiguity recorded here is worth more than a confident field
> above it.
