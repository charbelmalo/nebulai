---
name: visual-craft
description: "Visual and typographic fidelity details that separate polished UI from templated defaults. Covers squircles and continuous corners via the superellipse (CSS corner-shape plus SVG/library fallbacks, and the three parameterizations you must not conflate), concentric nested corner radii (outer radius equals inner radius plus padding), tabular figures for aligning numbers, timers, prices, and ordered lists, and optical alignment of icons and rows. Use this whenever tuning the look of cards, buttons, modals, toolbars, avatars, badges, tables, numbered lists, counters, or icon sets — anything where corner geometry, nested rounding, number alignment, or icon centering is off. Ships drop-in CSS tokens in assets/. Reach for it when something looks 'slightly off' but the user can't say why, when matching an iOS/Apple-grade finish, or when a designer flags uneven gaps, jittering numbers, or misaligned icons."
---

# Visual Craft

The visual/typographic fidelity layer. These details are individually tiny; the eye registers their absence as "unpolished" even when it can't name why. Encode them as tokens/defaults (see `assets/craft-tokens.css`) rather than re-deciding per component.

For interaction mechanics use `interaction-craft`; for product structure use `composable-product-design`.

Drop-in: `assets/craft-tokens.css` — a radius scale with concentric derivation, tabular-nums utilities, and squircle progressive enhancement.

## 1. Squircles / continuous corners (superellipse)

A normal `border-radius` corner is a circular arc that meets the straight edge with an abrupt curvature jump — a G1 discontinuity the eye reads as a slightly hard break. A **superellipse** (Lamé curve; popularized by Piet Hein) flows the edge into the corner with continuous curvature, which reads as "premium." This is why Apple icons and hardware use squircles, not rounded rectangles.

**Three parameterizations — do NOT conflate them:**
- **figma-squircle / iOS "corner smoothing," 0–1:** Apple's UI uses ~0.6; a radius of ~22.37% of icon width reproduces the iOS icon.
- **Raw Lamé exponent n:** a classic squircle is n=4; Apple's hardware curve is closer to n≈5.
- **CSS `corner-shape: superellipse(K)`:** here the `squircle` keyword equals `superellipse(2)`; `round`=`superellipse(1)`, `square`=`superellipse(∞)`, `bevel`=`superellipse(0)`, `scoop`=`superellipse(-1)`. The CSS K is NOT the Lamé n.

Implement with progressive enhancement (in `craft-tokens.css`): keep a plain `border-radius` baseline, add `corner-shape: superellipse(2)` behind `@supports`. As of early 2026 `corner-shape` is Chromium-only (M139+); for Safari/Firefox parity today use an SVG clip-path/mask or a library (figma-squircle, squircle.js, CornerKit). `corner-shape` also animates (it interpolates via `superellipse()`), which is a cheap, distinctive motion accent.

**When it's worth it:** large surfaces, brand icons, hero cards, hardware-adjacent framing. **When it isn't:** tiny elements — on a small box with a large radius the smoothing has no room to express and looks identical to a plain arc, so don't pay the cost there.

## 2. Concentric (nested) radius

Nest a rounded element in a rounded container with the *same* radius and the gap looks fatter at the corners than along the edges, because the curves don't share a center. The golden formula: **outer_radius = inner_radius + padding** (equivalently inner = outer − padding). Store one radius token and derive the rest with `calc()`; clamp at 0 so `padding ≥ radius` yields square inner corners rather than a negative radius. Works with squircle corners too. (SwiftUI shipped a `ConcentricRectangle` API at WWDC 2025 for exactly this — a signal of how fundamental it is.) CSS in `craft-tokens.css`.

## 3. Tabular figures

For ordered lists, tables, counters, timers/durations, prices, and leaderboards, use `font-variant-numeric: tabular-nums` (fallback `font-feature-settings: "tnum" 1`) so digits share a fixed advance and columns align — "10" lines up with single-digit rows, and timers stop jittering as digits change. Add `slashed-zero` where 0/O ambiguity matters and `lining-nums` if the face defaults to old-style figures in UI chrome. Confirm the chosen webfont actually ships `tnum` — not all do.

## 4. Optical alignment

Metric (bounding-box) alignment often looks *wrong* because glyphs carry different visual weight and whitespace; the eye wants **optical** alignment. Put every icon in a shared fixed box and align the *optical center* of the mark, not its bounding box (triangles, chevrons, and circles each need small manual nudges). Align text to a shared baseline/leading grid so bullets, numbers, and checkboxes share one left edge and one baseline regardless of list type. Keep an icon-grid spec (target size, safe area, stroke weight) so mixed icon sources sit consistently — that spec belongs in a design-token/codegen contract, not re-decided per screen.

## How to respond with this skill

When something "looks off," check these four in order — uneven nested gaps (concentric radius), jittering/misaligned numbers (tabular figures), hard-feeling corners on large surfaces (squircle), and miscentered icons (optical alignment) — and fix with the token layer so the fix generalizes. State the *why* (curvature continuity, shared-center gaps, fixed advance, optical vs. metric) so it's not cargo-culted. Don't apply squircles to tiny elements just because you can.
