---
name: composable-product-design
description: "Product-shape and information-architecture decisions for tools, apps, and platforms. Covers reducing a domain to a small set of composable primitives instead of many one-off features, using the IKEA effect for engagement and retention, defeating blank-canvas paralysis with scaffolds and templates, layering complexity across a user spectrum (residents, gardeners, builders, architects) via progressive disclosure, constraining open-ended power features to their core job, and guarding against the tyranny of the marginal user. Use this whenever deciding what a product should BE: its primitives, feature scope, onboarding shape, product-level empty states, or 'what should we build next.' Applies even when the user frames it as UX strategy, IA, feature prioritization, scoping an MVP, or simplifying a bloated app. Reach for it BEFORE writing UI code when the open question is structural rather than visual."
---

# Composable Product Design

This skill governs the *shape* of a product — the layer that is expensive to reverse once shipped. Use it when the question is "what should this be and how should it be organized," not "how should this button look." When the question is visual, use `visual-craft`; when it's about a specific interaction, use `interaction-craft`; when it's about process or what-to-build-next mechanics, use `design-process`.

Consult `references/frameworks-and-research.md` for the primitive-audit worksheet, the what-to-build heuristics, the IKEA boundary conditions, and sources.

## The five moves

### 1. Ship composable primitives, not features

Don't start from a specific user problem and build a bespoke feature for it. Extract the primitives common to the whole domain and expose them as a small set of configurable blocks the user assembles. Notion reduced "every productivity tool" to blocks, lists, tables/databases, boards, and links; Notion Mail noticed that pin, star, flag, important, folders, and nested labels are all the same object and collapsed them into one primitive — the **label** — from which views compose.

Apply it:
- Run a **primitive audit** (worksheet in the reference file): list every feature, then ask which are secretly the same underlying object with different chrome. Target ~5–7 irreducible primitives.
- Prefer **one generic, configurable element** over many purpose-built ones.
- Enforce **total interop**: every new primitive must compose with every existing one. "Some of them just don't fit together" is the recurring failure mode — treat "does it compose with all N primitives?" as a merge gate.

Why it matters beyond elegance: it triggers move #2's engagement effect, and it keeps surface area small while expressivity grows.

### 2. Use the IKEA effect — but only if the user finishes

People overvalue what they build themselves (Norton, Mochon & Ariely, 2011 — subjects paid ~63% more for self-assembled furniture). A system a user assembled is switching-cost they built with their own hands. The load-bearing caveat: the effect **only holds when the build is completed successfully** — participants who failed to finish, or built and then dismantled, lost the bump entirely.

Apply it:
- Design the **completion moment** deliberately. Instrument first-run for the point where the user reaches a *working artifact*, and strip friction before it.
- Editing an opinionated default still counts as "self-built," so defaults and templates are IKEA-effect accelerators, not shortcuts around it.

### 3. Scaffold the blank canvas

The tax on move #1 is blank-canvas paralysis: maximum flexibility, maximum "what could I possibly build?" Never let a first surface be genuinely empty and inert — empty + inert = churn.

Four escalating scaffolds, cheapest first: ghost/scaffold content that shows the *shape* of a populated state; opinionated defaults the user edits; pre-assembled templates; generative setup (describe intent, get an editable structure). Keep generation as an *accelerator into an editable result*, never a black box — the user still needs the shaping moment from move #2.

### 4. Layer complexity across the user spectrum; design to the edges

Users sit on a spectrum: **residents** (~60–70%: "give me the tool, don't make me tweak"), **gardeners** (will learn configuration), **builders**, and a small **architect** tier. Check every feature against both ends: can a resident still use it, and can an architect still discover the depth?

The anti-pattern this defends against is designing for the *average* user. In 1950 Lt. Gilbert Daniels measured 4,063 pilots on 10 dimensions; not one fell within the "average" range on all ten. Averaging your users yields an interface that fits no one. The fix, like the Air Force's, is to **design to the edges** and make the experience adjustable per tier.

Apply it:
- Default path = resident path: fully usable with zero configuration. No knob a resident *must* touch to get value.
- Reveal depth **incrementally** and behind prerequisite understanding (you learn "page," then "database," then "customize layout"), not as a settings dump on screen one.
- For each decision, sanity-check the resident *and* the architect explicitly. A feature that's great for the median but blocks residents or caps architects is a net loss.

### 5. Constrain open-ended power features to their core job

An open-ended, do-anything feature is maximally expressive and maximally undiscoverable — "you don't know where the boundaries are until you hit them." Notion Mail's Autopilot (a free-text prompt that could label, archive, translate, extract) became **Auto Label**: they found labeling was the dominant real use, narrowed the whole surface to it, then removed the prompt entirely and replaced typing with suggestions and one-click buttons grounded in the user's real emails. The general engine still exists underneath — it just stopped being the entry point.

Apply it:
- Ship the power feature, watch which single job dominates, and re-cut the *default* surface around that job.
- Remove the input where you can: if the 80% case can be expressed as buttons/suggestions over the user's real data, that beats a prompt box for discoverability and speed.
- Keep the general capability as an escape hatch for the architect tier, not the default.
- Tell that you've over-generalized: users can't articulate what the feature *can't* do.

## The guardrail: the tyranny of the marginal user

A large product optimizes not for its existing users but for the *marginal* (next) user — and chasing the lowest-context newcomer is why software trends toward less agency and lower quality even as budgets grow (Ivan Vendrov). Moves #4 and #5 only protect depth if the team refuses to sand it off for the lowest-context user.

- **Define who the product is NOT for**, explicitly.
- When simplifying for onboarding, add scaffolding/defaults (#3, #5) rather than removing capability.
- If you measure engagement, separate *meaningful* engagement (task completion, returned-for-real-work) from raw time-on-app — optimizing the latter is exactly the pressure that degrades products.

## How to respond with this skill

Diagnose which move(s) the situation calls for, then give a concrete structural recommendation — the primitive set, the onboarding scaffold, the disclosure layering, the scope cut — with the *why* attached. When the user is scoping features, run the what-to-build heuristics from the reference file. Keep the advice grounded in the user's actual domain rather than restating Notion examples; the examples are illustrations, not the deliverable.
