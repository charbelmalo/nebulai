/**
 * combobox-keyboard.js — keyboard + focus for lists, autocompletes, mention
 * pickers, and command palettes, using the aria-activedescendant pattern.
 *
 * WHY activedescendant instead of roving tabindex: DOM focus never leaves the
 * input, so the caret, backspace, and left/right editing keep working while
 * ArrowUp/Down move a *virtual* highlight. That is what lets the user edit the
 * query without dismissing the menu — the detail that makes mention menus feel
 * seamless. Roving tabindex moves real focus onto options and breaks text editing.
 *
 * Markup contract:
 *   <input role="combobox" aria-expanded="true" aria-controls="opts"
 *          aria-activedescendant="" />
 *   <ul id="opts" role="listbox">
 *     <li id="opt-0" role="option">…</li> ...
 *   </ul>
 * Pair with the scroll-padding CSS in interaction.css so the active row never
 * sits flush against the scroll edge.
 */

export function attachComboboxKeyboard({ input, listbox, onCommit, onDismiss }) {
  const options = () => Array.from(listbox.querySelectorAll('[role="option"]'));
  let activeIndex = -1;

  function setActive(index) {
    const opts = options();
    if (opts.length === 0) { activeIndex = -1; input.removeAttribute('aria-activedescendant'); return; }
    // Clamp — do NOT wrap. Wrapping at the ends is a common jank tell.
    activeIndex = Math.max(0, Math.min(index, opts.length - 1));
    opts.forEach((el, i) => el.setAttribute('aria-selected', String(i === activeIndex)));
    const el = opts[activeIndex];
    input.setAttribute('aria-activedescendant', el.id);
    el.scrollIntoView({ block: 'nearest' }); // no jump; respects scroll-padding
  }

  function onKeydown(e) {
    switch (e.key) {
      case 'ArrowDown': e.preventDefault(); setActive(activeIndex + 1); break;
      case 'ArrowUp':   e.preventDefault(); setActive(activeIndex - 1); break;
      case 'Enter': {
        const el = options()[activeIndex];
        if (el) { e.preventDefault(); onCommit?.(el, activeIndex); }
        break;
      }
      case 'Escape': e.preventDefault(); onDismiss?.(); break;
      // ArrowLeft / ArrowRight / Backspace: intentionally NOT handled here.
      // Default text editing runs and the menu stays open because focus never
      // left the input. Let the surrounding logic dismiss only when the caret
      // truly exits the trigger context.
      default: break;
    }
  }

  // Hover and keyboard must share ONE active index.
  function onPointerOver(e) {
    const el = e.target.closest('[role="option"]');
    if (!el) return;
    setActive(options().indexOf(el));
  }

  input.addEventListener('keydown', onKeydown);
  listbox.addEventListener('pointerover', onPointerOver);

  // Call after the option list re-renders (filtered results): reset to the top.
  function refresh() { setActive(options().length ? 0 : -1); }

  return {
    refresh,
    teardown() {
      input.removeEventListener('keydown', onKeydown);
      listbox.removeEventListener('pointerover', onPointerOver);
    },
  };
}
