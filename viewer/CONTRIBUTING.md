# Contributing to the viewer

Chrome (anything under `src/chrome/`) ships through the **component review
gate** below — it is the same checklist as
`.claude/skills/design-process/references/component-review-gate.md`, kept in
sync by hand. Run it on every PR that adds or changes an interactive
component; record the result in the PR description. Items that don't apply
to a component (e.g. submenus in a viewer with none) are marked N/A, not
skipped silently.

## Settings-home rule

Any new user-facing knob — driver setting, chrome preference, probing
option, dataset filter — MUST land in
[`src/chrome/SettingsPage.tsx`](src/chrome/SettingsPage.tsx) under the
matching tab (General / Appearance / Model Probing / Data / About) and be
backed by a slice of `appStore` (`settings`, `appearance[<graph>]`,
`probing`). The compact left `Sidebar` is a quick-access subset, never the
only home for an option. See the `nebulai` skill's "Settings home" section
for the full rule.

The automated half of the bar lives in `npm run test` (vitest unit),
`npm run e2e` (Playwright: goldens per view × gpu rung, perf budgets, axe,
keyboard reachability, reduced motion) and `npm run typecheck`. All three
must be green before the manual gate below means anything.

## Component Review Gate

### Interaction (interaction-craft)

- [ ] **Targets** meet 44pt / 48dp (>=24 CSS px AA; 44 CSS px AAA) on coarse pointers. Visual size may be smaller via an expanded hit area.
- [ ] **Keyboard path is complete**: arrow nav (clamped, non-wrapping), Enter to commit, Escape to dismiss; Escape closes the innermost layer first (submenu before menu).
- [ ] **Focus never traps or drops**: text-editing surfaces (autocomplete, mention picker, command palette) use `aria-activedescendant`, not roving `tabindex`, so editing keeps working while the highlight moves.
- [ ] **Active-item scroll** uses `scrollIntoView({ block: "nearest" })` + `scroll-padding` — no edge-flush rows, no jump.
- [ ] **Submenus are pointer-forgiving** (safe-triangle / `safePolygon`), so diagonal cursor moves don't close them.
- [ ] **Adaptive corners** on grouped/selected rows restitch correctly by position and state.
- [ ] **Hover and keyboard share one active index**; hovering updates it.

### Visual (visual-craft)

- [ ] **Concentric radii**: every nested rounded element derives from `outer − padding` (clamped at 0), not the same radius as its parent.
- [ ] **Tabular figures** on any aligned numbers — ordered lists, tables, counters, timers/durations, prices.
- [ ] **Squircle** used only where it has room to express (large surfaces / brand marks), with an `@supports` fallback; not forced onto tiny elements.
- [ ] **Icons** share a fixed box and are optically centered (not bounding-box aligned); text shares one baseline/left edge across list types.

### Product & state (composable-product-design)

- [ ] **Empty state teaches**: ghost scaffold showing the populated shape + a clear creation affordance; never inert. Dimmed/labeled so it doesn't read as stale data.
- [ ] **Completion path exists**: there is a short route from empty to a working artifact (the IKEA-effect completion moment).
- [ ] **Default path = resident path**: usable with zero configuration; advanced depth is discoverable but not on screen one.

### Robustness

- [ ] **`prefers-reduced-motion`** honored for any motion/transitions.
- [ ] **Touch**: hover-open replaced by tap; no hover-only affordances.
- [ ] **Long content / resize**: verified at multiple widths and with overflowing content, not just the happy-path size.
