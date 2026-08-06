---
name: design-process
description: "How high-craft product teams actually work and decide, plus a shippable quality bar. Covers designing interactions in real code rather than static mockups (because focus, timing, keyboard, and resize only reveal themselves running), dogfooding and building for a specific named user to earn strong taste, the compounding-care thesis (perceived quality is the sum of a thousand micro-decisions), a decision framework for what to build next (the one-vs-five use-case test, expressivity with distillation, interop cost, both-edges check), and a concrete component review gate covering targets, keyboard path, focus, concentric radii, tabular numbers, empty states, reduced motion, and touch. Use this for design-process questions, prototyping strategy, deciding what to build or prioritize, setting up a design/fidelity review checklist or CI gate, or judging whether a UI is done. Reach for it when the user asks how to prioritize, how to review, how to raise the quality bar, or whether something is ready to ship."
---

# Design Process

The process-and-judgment layer: how to work so the craft actually happens, and how to check that it did. Pair it with the other three skills — `composable-product-design` (what to build), `interaction-craft` (interaction mechanics), `visual-craft` (visual fidelity). This skill is where they get prioritized, prototyped, and reviewed.

The bundled `references/component-review-gate.md` is a ready checklist — copy it into a repo or PR template.

## 1. Design interactions in real code, not just mockups

Static design tools capture the *picture*; the *design* of anything stateful is its behavior — keyboard handling, focus management, scroll padding, submenu geometry, resize behavior, timing — none of which exist in an artboard. Menus are "just a square" in Figma. Prototype the interaction, not the picture: for menus, comboboxes, drag, and motion, the first real artifact should be code you can *feel*. Real teams have burned months polishing a mockup only to discover, once it was running and lived-in, that it was distracting or awkward — a discovery no static file surfaces.

Apply it: when the question is about a stateful interaction, build a quick running version early rather than iterating in a design tool. Use it before judging it.

## 2. Dogfood / build for a specific named user

Building for a specific, real user (yourself, a named persona you actually observe) yields a fast, coherent intuition that lets you make thousands of micro-calls a survey-by-committee can't. Notion Mail's dense default layout came from founders who triage a lot of mail — a decision from *use*, not a poll.

The caveat, imported from `composable-product-design`: pair it with the user spectrum so "built for me" doesn't collapse into "built *only* for me." Dogfood for taste and speed; check the resident and architect edges before shipping.

## 3. The compounding-care thesis

No single micro-detail matters; the aggregate is the moat. Perceived quality is the *integral* of micro-decisions, and what users ultimately sense is that the builders cared. Two complementary engineering stances belong on the team at once: simplify and don't over-complicate the *architecture*, and be willing to write ugly code if it makes the *experience* measurably better. Simplify the internals; splurge on the experience. Budget effort accordingly — cheap, reversible experience wins are worth disproportionate care; structural changes get proportionally more deliberation.

## 4. Decide what to build next

When a surface can always absorb "one more option," use these tests:
- **One-vs-five:** a feature that nails a single use case vs. one that serves five adequately — decide explicitly which you're optimizing. Lean toward expressivity across cases only if you can still distill the main idea simply.
- **Distillability:** flexibility that can't be reduced to a simple mental model is complexity debt.
- **Interop cost is first-class:** does it compose with every existing primitive?
- **Both edges:** resident-usable *and* architect-discoverable.
- **Marginal-user check:** are you serving existing power users or chasing the lowest-context newcomer at their expense?
- **Reversibility:** structural changes are expensive to unwind; visual/interaction changes are cheap. Deliberate in proportion.

## 5. The review gate

Treat craft as an enforced bar, not a thing to remember. Run `references/component-review-gate.md` on any new interactive component before it ships — targets, keyboard path, focus retention, concentric radii, tabular numbers, adaptive corners, empty states, safe-triangle submenus, reduced-motion, and touch. It maps cleanly onto a CI/fidelity gate: the same checks can be automated or made a PR checklist so these decisions are caught rather than recalled.

## How to respond with this skill

For prioritization questions, run the "what to build next" tests and give a ranked, reasoned answer. For "is this done / review this" questions, run the review gate and report specific pass/fail items with fixes. For prototyping questions, push toward a running artifact over more mockup iteration when the thing is stateful. Always attach the *why* — these are heuristics to reason with, not rules to obey blindly.
