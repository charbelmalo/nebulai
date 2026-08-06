/**
 * safe-polygon.js — pointer-forgiving submenus (the "safe triangle").
 *
 * Keeps a submenu open while the pointer is plausibly *aiming* at it: while the
 * cursor stays inside the triangle formed by its last position over the trigger
 * and the two near corners of the submenu, suppress the close. A small grace
 * timeout absorbs brief overshoots.
 *
 * PRODUCTION NOTE: prefer Floating UI's `safePolygon()` (used by Radix, React
 * Aria) when you can add the dependency. This file is the dependency-free
 * fallback and a reference for what that abstraction is doing.
 *
 * Assumes the submenu opens to the RIGHT of the trigger. For a left-opening
 * submenu, use `rect.right` instead of `rect.left` for the near edge.
 */

// Canonical sign-based point-in-triangle test (barycentric sign method).
function sign(a, b, c) {
  return (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y);
}

export function pointInTriangle(p, v1, v2, v3) {
  const d1 = sign(p, v1, v2);
  const d2 = sign(p, v2, v3);
  const d3 = sign(p, v3, v1);
  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
  const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(hasNeg && hasPos); // inside iff all three have the same sign
}

/**
 * Wire aim-forgiveness onto a trigger + submenu pair.
 *
 * @param {Object}   opts
 * @param {Element}  opts.trigger      The parent menu item.
 * @param {Element}  opts.submenu      The submenu element (must be positioned).
 * @param {Function} opts.close        Called to close the submenu.
 * @param {number}   [opts.grace=6]    px of slack added above/below the submenu.
 * @param {number}   [opts.delay=80]   ms grace before closing once off the aim path.
 * @param {'right'|'left'} [opts.side='right'] Which side the submenu opens on.
 * @returns {Function} teardown() to remove listeners.
 */
export function attachSafePolygon({ trigger, submenu, close, grace = 6, delay = 80, side = 'right' }) {
  let anchor = null;       // pointer position captured while over the trigger
  let closeTimer = null;

  const captureAnchor = (e) => { anchor = { x: e.clientX, y: e.clientY }; };

  const onMove = (e) => {
    const p = { x: e.clientX, y: e.clientY };
    const rect = submenu.getBoundingClientRect();
    const nearX = side === 'right' ? rect.left : rect.right;
    const top = { x: nearX, y: rect.top - grace };
    const bottom = { x: nearX, y: rect.bottom + grace };
    const aiming = anchor && pointInTriangle(p, anchor, top, bottom);

    clearTimeout(closeTimer);
    if (aiming) return;                       // still heading for the submenu → stay open
    closeTimer = setTimeout(close, delay);    // wandered off → close, softly
  };

  trigger.addEventListener('pointermove', captureAnchor);
  document.addEventListener('pointermove', onMove);

  return function teardown() {
    clearTimeout(closeTimer);
    trigger.removeEventListener('pointermove', captureAnchor);
    document.removeEventListener('pointermove', onMove);
  };
}

/*
 * Touch: skip aim-forgiveness entirely and use tap-to-open.
 *   if (window.matchMedia('(pointer: coarse)').matches) { /* attach tap handlers, not this */ }
 *
 * Pure-CSS alternative (no JS): overlay an invisible clip-path triangle on the
 * parent with pointer-events toggled while the submenu is open. Works, but the
 * triangle can't follow a repositioning submenu as cleanly as this does.
 */
