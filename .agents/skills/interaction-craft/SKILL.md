---
name: interaction-craft
description: "Micro-interaction engineering that makes a UI feel considered rather than templated. Covers pointer-forgiving submenus (the safe-triangle / safe-polygon aim algorithm), forgiving hit targets that beat Fitts's Law, keyboard navigation and focus that never breaks (the aria-activedescendant combobox pattern, scroll-padding, non-wrapping arrow nav, Escape closing innermost-first), state-aware adaptive corner radii on grouped and selected rows, and empty states that teach via ghost scaffolds. Use this whenever building or reviewing menus, dropdowns, context menus, submenus, comboboxes, autocompletes, mention pickers, command palettes, selectable lists, hover states, tab/focus order, or any interaction where cursor forgiveness, keyboard access, or focus management matters. Ships drop-in JS and CSS in assets/. Reach for it for the 'why does this menu keep closing on me,' 'the keyboard doesn't work here,' or 'this list feels janky' class of problems — even if the user just says 'make this menu feel better.'"
---

# Interaction Craft

This skill is the hands-on interaction layer: the specific mechanics that produce the "feels good" quality. Each mechanic below is small; the payoff is cumulative, so apply them together and treat them as a review bar (see the checklist in `design-process`), not one-offs.

For strategy/IA use `composable-product-design`; for corners/typography/icons use `visual-craft`.

Drop-in code lives in `assets/`:
- `assets/safe-polygon.js` — pointer-forgiving submenus
- `assets/combobox-keyboard.js` — keyboard + focus for lists/autocompletes
- `assets/interaction.css` — hit targets, adaptive corners, active-scroll padding, coarse-pointer handling

Lift these and adapt them to the user's framework; they're framework-agnostic on purpose.

## 1. Pointer-forgiving submenus (safe triangle / safe polygon)

Problem: moving the cursor diagonally from a parent item toward a submenu clips through a sibling item and closes the menu. Fix: while the pointer is over the trigger, remember its position; build a triangle from that point to the two near corners of the open submenu; suppress the close while the cursor stays inside that triangle (the user is plausibly *aiming* at the submenu). Add a small grace timeout so a brief overshoot doesn't slam it shut.

Provenance so you use the canon: the technique is from Tognazzini and Batson at Apple; popularized by Amazon's mega-dropdown and Ben Kamens' jQuery-menu-aim; today the production-grade version is Floating UI's `safePolygon()` (used by Radix, React Aria).

**In production, prefer `safePolygon` from Floating UI.** Ship `assets/safe-polygon.js` only when you can't add that dependency — it exists so the mechanic is understood, not to reinvent it. On touch (`pointer: coarse`), drop hover-open entirely: tap to open.

## 2. Forgiving hit targets (Fitts's Law)

Fitts's Law: acquisition time falls as target width grows (`T = a + b·log2(1 + D/W)`). Expand the *effective* target beyond the visual bounds — the glyph stays small, the hittable area grows — so "clicking slightly off still works." Keep an invisible enlarged hit area via a pseudo-element (in `assets/interaction.css`).

Minimum baselines: 44×44 pt (Apple HIG), 48×48 dp (Material), 44 CSS px for WCAG 2.5.5 (AAA), 24 CSS px for WCAG 2.5.8 (AA). Enlarge further on coarse pointers.

## 3. Keyboard navigation + focus that never breaks

The behavior that makes a mention/autocomplete menu feel seamless: arrow keys move a highlight through the list while the user keeps typing and editing text, and only an action that truly leaves the field dismisses the menu.

The key architectural choice is **`aria-activedescendant`, not roving `tabindex`**. DOM focus stays in the input the whole time, so the caret, backspace, and left/right editing keep working; ArrowUp/Down move a *virtual* highlight by pointing `aria-activedescendant` at a different option id. That single decision is what lets left/right-arrow edit the query without losing the menu. Use `assets/combobox-keyboard.js`.

Non-negotiable details (they're what separate polished from janky):
- Arrow navigation **clamps at the ends — do not wrap.**
- Active-item scroll uses `scrollIntoView({ block: "nearest" })` plus `scroll-padding` so the active row never sits flush against the edge (the "autoscroll keeps a padding" feel).
- Hover and keyboard share **one** active index; hovering an option updates it.
- Escape closes the **innermost** layer first (submenu before menu).
- Placeholder/typeahead text can autofill as the user types, without stealing the caret.

## 4. State-aware adaptive corners

Grouped or selectable rows should restitch their corners by position and state: the first row rounds only on top, the last only on bottom, a single row fully rounds, and a *selected* middle row can promote to a fully-rounded detached block. Do it with a data attribute or pure-CSS adjacency (`:first-of-type` / `:last-of-type`, or `:has()`), sharing the radius token with the visual-craft scale so nested content stays concentric. CSS in `assets/interaction.css`.

## 5. Empty states that teach

A brand-new list/table should render dimmed ghost rows that show the *shape* of the populated state, plus an obvious dashed "＋ new" affordance so it reads as fillable, not broken. A clearable sample-data toggle is the strongest teacher and feeds the IKEA-effect completion moment (see `composable-product-design`). The tradeoff is real — ghosts can imply stale content — so dim them and label the zero-state clearly.

## How to respond with this skill

Identify which mechanic the surface needs, apply it with the bundled code adapted to the user's stack, and explain the *why* (the aim geometry, the Fitts tradeoff, the focus model) so the choice is legible. When reviewing existing UI, run the relevant subset as a checklist and name specific failures ("this menu has no aim-forgiveness," "this list wraps at the ends and traps focus") with the fix.
