# Frameworks & Research (composable-product-design)

Read this when running a primitive audit, scoping features, or when the user wants the evidence behind a recommendation.

## Primitive-audit worksheet

Goal: collapse a sprawling feature list into ~5–7 irreducible, mutually-composable primitives.

1. **Dump every feature/object** the product has or is asked for, one per line.
2. **Cluster by "is this secretly the same thing?"** For each cluster, ask: do these differ in *substance* or only in *chrome/label/default*? (Pin vs. star vs. flag vs. important = one thing: a label. Table vs. board vs. calendar vs. gallery = one thing: a view over a database.)
3. **Name the primitive** behind each cluster. Prefer the most general honest name — but not so abstract it confuses (Notion kept "database" instead of "table" on purpose, because the concept is genuinely richer than a table).
4. **Interop matrix**: draw an N×N grid of primitives. For each cell, confirm the two compose. Any cell that doesn't is either a missing capability or a sign one "primitive" is actually a bespoke feature in disguise.
5. **Configurable-element test**: for each primitive, is there one generic element the user configures, or many purpose-built variants? Collapse variants into configuration wherever the variants share substance.
6. **Completion path**: identify the shortest route from empty state to a working artifact built from these primitives. That path is your onboarding spine (see scaffolds).

Output: the primitive set, the interop matrix with any gaps flagged, and the completion path.

## "What should we build next?" heuristics

Use when a user-generated-content surface can always absorb "one more view / one more option."

- **The one-vs-five test.** A feature that nails a single use case vs. one that serves five adequately — decide explicitly which you're optimizing. Don't let it be implicit. Lean toward expressivity across use cases *only if* you can still distill the main idea simply.
- **Distillability gate.** Flexibility that can't be distilled into a simple mental model is complexity debt. If you can't explain the feature to a resident in one sentence, redesign or cut.
- **Interop cost is first-class.** Does it compose with every existing primitive? If not, the coherence/maintenance cost usually outweighs the feature.
- **Both-edges check.** Resident-usable *and* architect-discoverable. If it's fine for the median but blocks residents or caps architects, it's a no.
- **Marginal-user check.** Are you adding this to serve existing power users, or to chase the lowest-context newcomer at their expense? Name it.
- **Reversibility.** Structural/primitive changes are expensive to unwind; visual and interaction changes are cheap. Spend deliberation proportional to reversibility.

## IKEA effect — boundary conditions

- The valuation bump requires **successful completion**. Failed or dismantled builds show no effect (and can show negative affect). Design for a reachable finished state.
- It applies to novices, not only self-identified DIYers.
- Editing/customizing a provided default still engages it — the labor need not be from-scratch.
- Over-personalization can *lower* value to *other* people (idiosyncratic customizations read as noise to a second user) — relevant when the artifact is meant to be shared or handed off.

## User spectrum (working definitions)

- **Resident** (~60–70%): wants the tool to work out of the box; will not tweak. Default path must serve them with zero configuration.
- **Gardener**: will learn properties/configuration to make the tool fit them.
- **Builder**: assembles new systems from primitives.
- **Architect** (small): reverse-engineers the deepest capabilities (formula languages, layout builders). Serve them via discoverable depth and escape hatches, never by putting that depth on screen one.

Design to the edges (5th–95th percentile thinking), not the mean. The mean fits no one.

## Sources

- IKEA effect — Norton, Mochon & Ariely, "The IKEA Effect: When Labor Leads to Love," *Journal of Consumer Psychology* (2012). ~63% higher willingness-to-pay for self-assembled items; completion is a necessary condition. Lineage: effort justification (Festinger 1957; Aronson & Mills 1959).
- The "average" trap — Lt. Gilbert S. Daniels, Wright Air Force Base (1950); 4,063 pilots, 10 dimensions, zero within the average range → "design to the edges" / adjustable cockpits. Popularized in Todd Rose, *The End of Average* (2016).
- Marginal user — Ivan Vendrov, "The Tyranny of the Marginal User" (2023).
- Primitives / block model & feature-scope tension — Notion and Notion Mail design practice (composable primitives; Autopilot → Auto Label narrowing).
