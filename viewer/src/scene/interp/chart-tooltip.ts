/** Shared hover tooltip for the interp drivers. Before this, 25 drivers each
 *  hand-rolled the identical dance: create `<div class="point-tooltip
 *  interp-tooltip">`, `innerHTML = ""`, append `point-tooltip-label` /
 *  `point-tooltip-conf` rows, then clamp the position with a hardcoded guess
 *  (`this.cssW - 330`). This collapses all of that to one measured, glassmorphic
 *  helper (req 2) with a structured row API — and, crucially, a *measured*
 *  clamp (offsetWidth/Height) so the tooltip never clips regardless of content.
 *
 *  The DOM shape stays exactly `point-tooltip interp-tooltip` so the existing
 *  CSS keeps applying; the obsidian-frosted upgrade lives in chrome.css on
 *  `.interp-tooltip`. Rows can carry a color swatch (a small chip matching the
 *  series/data color) so a value reads against its own hue — the glass tooltip
 *  as a legend key, not just text. */

import type { RGB } from "./chart-theme";

export interface TipRow {
  text: string;
  /** "label" = the bold mono headline (one per tooltip, first); "conf" = a dim
   *  supporting line. Defaults to "conf". */
  kind?: "label" | "conf";
  /** Optional [r,g,b] swatch chip rendered before the text — use it to tie a
   *  value to the exact color the driver drew for it. */
  swatch?: RGB;
  /** Optional right-aligned value, turning the row into a monospace key→value
   *  pair (`text` becomes the dim key). Tabular-aligned so a stack of them reads
   *  like a readout. */
  value?: string;
  /** Render the value in the hot/danger color — the focused or anomalous datum
   *  the cursor is locked onto (borrowed from the "one red value" tooltip idiom
   *  where the active reading pops out of the stack). */
  hot?: boolean;
}

export class InterpTooltip {
  private el: HTMLElement;

  constructor(overlay: HTMLElement) {
    this.el = document.createElement("div");
    this.el.className = "point-tooltip interp-tooltip";
    this.el.style.visibility = "hidden";
    // ARIA: hover readouts are live status, mirrored for AT even though the
    // pointer path is mouse-first (keyboard focus path lives in the driver).
    this.el.setAttribute("role", "status");
    this.el.setAttribute("aria-live", "polite");
    overlay.appendChild(this.el);
  }

  /** Replace the tooltip content and reveal it. Pass rows top-to-bottom. */
  show(rows: TipRow[]): void {
    this.el.textContent = "";
    for (const r of rows) {
      const row = document.createElement("div");
      row.className = r.kind === "label" ? "point-tooltip-label" : "point-tooltip-conf";
      if (r.swatch) {
        const sw = document.createElement("span");
        sw.className = "point-tooltip-swatch";
        sw.style.background = `rgb(${r.swatch[0]},${r.swatch[1]},${r.swatch[2]})`;
        sw.style.color = `rgb(${r.swatch[0]},${r.swatch[1]},${r.swatch[2]})`; // drives the glow
        row.appendChild(sw);
      }
      if (r.value !== undefined) {
        // key → value readout row: dim key on the left, value right-aligned
        row.classList.add("point-tooltip-kv");
        const key = document.createElement("span");
        key.className = "point-tooltip-key";
        key.textContent = r.text;
        const val = document.createElement("span");
        val.className = r.hot ? "point-tooltip-val is-hot" : "point-tooltip-val";
        val.textContent = r.value;
        row.append(key, val);
      } else {
        row.appendChild(document.createTextNode(r.text));
      }
      this.el.appendChild(row);
    }
    this.el.style.visibility = "visible";
  }

  /** Position near (x, y) in overlay-local pixels, clamped inside a boundsW ×
   *  boundsH box using the tooltip's MEASURED size (not a guess) so it never
   *  clips at the edges. `gap` is the cursor offset.
   *
   *  Positioning writes the independent `translate` property, NOT `transform`:
   *  the glass entrance (`interp-tip-in`) animates `transform`, and a CSS
   *  animation outranks an inline style for the property it animates — so
   *  writing position into `transform` would be clobbered by the animation's
   *  filled `translateY(0)`, pinning every tooltip to the overlay's top-left
   *  (over the header). `translate` composes with the animation's `transform`
   *  instead of fighting it, so the tooltip tracks the cursor and still rises. */
  move(x: number, y: number, boundsW: number, boundsH: number, gap = 14): void {
    const w = this.el.offsetWidth;
    const h = this.el.offsetHeight;
    const px = Math.min(Math.max(x + gap, 6), Math.max(6, boundsW - w - 6));
    const py = Math.min(Math.max(y + gap, 6), Math.max(6, boundsH - h - 6));
    this.el.style.translate = `${px.toFixed(1)}px ${py.toFixed(1)}px`;
  }

  hide(): void {
    this.el.style.visibility = "hidden";
  }

  get visible(): boolean {
    return this.el.style.visibility === "visible";
  }

  dispose(): void {
    this.el.remove();
  }
}
