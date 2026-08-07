/** #15 Embedding Constellation — every token embedding W_E[i] as a star, placed
 *  at its exact PCA score on the top-2 principal axes of the mean-centered
 *  embedding matrix. Nothing here is a synthetic layout: positions are real PC
 *  scores, star size is the real row L2 norm ‖W_E[i]‖₂, and color is a real
 *  orthographic property — whether the token string begins with a space —
 *  decoded per token. The honest finding it makes visible: GPT-2's leading
 *  embedding axes organize tokens largely by SURFACE FORM (leading space, case,
 *  digits, the commonest function words at the PC1 extreme), not meaning — and
 *  the top 2 PCs explain only ~2.6% of the variance, so this is deliberately
 *  shown as a low-dimensional shadow of a genuinely high-dimensional space.
 *
 *  Axes are drawn to a single isometric scale (equal px per PC unit) so on-screen
 *  distances are faithful; PC1 simply spans a wider range than PC2 because it
 *  carries more variance.
 *
 *  RENDERING: three/webgpu + TSL emissive field (`field2d.ts`), not a deck.gl
 *  scatter. 50,257 hard-edged discs overplot into one flat silhouette — the
 *  density that this view is largely ABOUT was being thrown away. Additive
 *  soft sprites let overlap sum, so the crowded PC1 spine reads as bright and
 *  the sparse periphery as dark, which is the real distribution. Two encodings
 *  now carry the norm: size is linear in ‖W_E‖₂ exactly as before, and glow is
 *  its RANK. Rank is monotone in the norm, so the two channels can never
 *  disagree about which star is bigger; rank rather than magnitude because the
 *  norm distribution is long-tailed and a linear ramp would hand the whole
 *  visible range to a few outliers. Hover chrome stays in the DOM overlay —
 *  anything in the scene blooms, and a marker that blooms reads as data.
 *
 *  Camera off, static (redraw on hover).
 *
 *  Source: embed.json → PCA of W_E computed offline in float64 (eigendecomp of
 *  the 768×768 covariance; coords = Wc·V). */

import type { GpuTier } from "../../app/capabilities";
import { type EmbedBundle, loadEmbed } from "../../data/interp";
import { EmissiveField2D, FieldMarker, rankNormalize, type Field2DLook } from "./field2d";
import { InterpTooltip, type TipRow } from "./chart-tooltip";
import type { InterpDriver } from "./InterpDriver";

const GL = 60; // px gutters (axis captions + anchor labels)
const GR = 60;
const GT = 78;
const GB = 64;
const FIT = 0.94; // leave a little breathing room around the data bbox

const SPACE: [number, number, number] = [245, 190, 92]; // leading-space token (warm)
const NOSPACE: [number, number, number] = [92, 198, 236]; // non-space token (cool)

/** Tuned by measurement against the DENSEST region — the core where tens of
 *  thousands of common tokens pile up — not by eye.
 *
 *  The metric is the fraction of LIT pixels (max channel > 0.12) whose
 *  saturation has collapsed below 0.15, i.e. gone white: that fraction IS the
 *  fraction of the frame where the warm/cool leading-space split — the whole
 *  colour encoding, and the finding this view exists to show — has been
 *  destroyed. First honest numbers, at the values ported over from the compare
 *  field: 76% of the frame lit, 40% of it desaturated — one uniform white blob.
 *  Landing at 40% lit / 14% desaturated took two things:
 *
 *    - `pointPx` down to 1.4. 50,257 additive sprites is an enormous amount of
 *      energy; per-point size is the term that multiplies it.
 *    - A bloom `threshold` of 1.2, well above the module default of 0.55. The
 *      threshold reads the scene BEFORE tone mapping, where the core's summed
 *      value is far past 1.0, so at the default the entire core bloomed and
 *      smeared across half the stage. Only genuine spikes should bloom here.
 *
 *  Exposure was the wrong knob and is left at 1: it darkens the isolated
 *  periphery stars just as much as the core, and those stars are data. */
const LOOK: Field2DLook = {
  pointPx: 1.4,
  emissiveMin: 0.12,
  emissiveMax: 1.45,
  glowGamma: 2.5,
  moteFloor: 0.1,
  dimLevel: 0.28,
  bloom: { strength: 0.45, radius: 0.32, threshold: 1.2 },
};

/** Global brightness while a star is focused — the field steps back so the
 *  marker is findable, exactly what the old layer-opacity dip did. */
const HOVER_DIM = 0.42;

interface Star {
  x: number;
  y: number;
  z: number; // PC3 (hover only)
  norm: number;
  lead: number; // 1 = leading space
  str: string;
  id: number;
}

export class EmbeddingConstellationDriver implements InterpDriver {
  readonly animated = false;
  private field = new EmissiveField2D(LOOK);
  private canvas!: HTMLCanvasElement;
  private tooltip!: InterpTooltip;
  private marker!: FieldMarker;
  private labelRoot!: HTMLElement;

  private bundle: EmbedBundle | null = null;
  private stars: Star[] = [];
  private anchors: Star[] = []; // a few extreme tokens, labelled to orient the eye
  private minX = 0;
  private maxX = 1;
  private minY = 0;
  private maxY = 1;
  private normMin = 0;
  private normMax = 1;
  private hover: Star | null = null;

  private cssW = 1;
  private cssH = 1;
  private disposers: Array<() => void> = [];

  async init(canvas: HTMLCanvasElement, tier: GpuTier, overlay: HTMLElement): Promise<void> {
    this.canvas = canvas;
    await this.field.init(canvas, tier);

    this.tooltip = new InterpTooltip(overlay);
    this.marker = new FieldMarker(overlay);
    this.labelRoot = document.createElement("div");
    this.labelRoot.className = "interp-embed-labels";
    overlay.appendChild(this.labelRoot);

    const onMove = (e: PointerEvent) => this.onPointerMove(e);
    const onLeave = () => this.onLeave();
    // this view has no click-driven selection — the tap's only job is
    // standing in for hover on a device that has none, so its readout can
    // actually be read
    const onClick = (e: PointerEvent) => this.onClick(e);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("click", onClick);
    this.disposers.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("click", onClick);
    });
  }

  async setModel(model: string, _trace?: string): Promise<void> {
    const b = await loadEmbed(model);
    this.bundle = b;
    const n = b.n;
    const c = b.coords; // flat 2n
    const stars: Star[] = new Array(n);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    let nmin = Infinity;
    let nmax = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = c[i * 2] ?? 0;
      const y = c[i * 2 + 1] ?? 0;
      const nm = b.norm[i] ?? 0;
      stars[i] = { x, y, z: b.z[i] ?? 0, norm: nm, lead: b.lead_space[i] ?? 0, str: b.strs[i] ?? "", id: i };
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (nm < nmin) nmin = nm;
      if (nm > nmax) nmax = nm;
    }
    this.stars = stars;
    this.minX = minX;
    this.maxX = maxX;
    this.minY = minY;
    this.maxY = maxY;
    this.normMin = nmin;
    this.normMax = nmax;
    this.hover = null;
    this.marker.hide();
    this.field.setDim(1);

    // orient the eye with a handful of REAL extremes (no cherry-picking of the
    // interior): the tokens at the PC1/PC2 range ends and the largest-norm star.
    const set = new Map<number, Star>();
    if (stars.length) {
      const extreme = (f: (s: Star) => number, sign: number): Star => {
        let best = stars[0] as Star;
        for (const s of stars) if (sign * f(s) > sign * f(best)) best = s;
        return best;
      };
      for (const s of [
        extreme((s) => s.x, +1),
        extreme((s) => s.x, -1),
        extreme((s) => s.y, +1),
        extreme((s) => s.y, -1),
        extreme((s) => s.norm, +1),
      ]) {
        set.set(s.id, s);
      }
    }
    this.anchors = [...set.values()];

    this.pushField();
    this.field.fitInset(minX, minY, maxX, maxY, this.inset(), FIT);
    this.field.render();
    this.positionLabels();
  }

  private radiusOf(norm: number): number {
    const t = (norm - this.normMin) / Math.max(1e-6, this.normMax - this.normMin);
    return 1.2 + t * 2.4; // 1.2 .. 3.6 × base, ∝ real embedding norm
  }

  private inset() {
    return { left: GL, right: GR, top: GT, bottom: GB };
  }

  /** Upload the field once per dataset. Unlike the deck.gl version there is no
   *  per-hover layer rebuild: hover only moves DOM chrome and nudges one
   *  brightness uniform, so 50k attributes are built exactly once. */
  private pushField(): void {
    const n = this.stars.length;
    if (!n) return;
    const pos = new Float32Array(n * 2);
    const color = new Float32Array(n * 3);
    const radius = new Float32Array(n);
    const norms = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const s = this.stars[i] as Star;
      pos[i * 2] = s.x;
      pos[i * 2 + 1] = s.y;
      const [r, g, b] = s.lead ? SPACE : NOSPACE;
      color[i * 3] = r / 255;
      color[i * 3 + 1] = g / 255;
      color[i * 3 + 2] = b / 255;
      radius[i] = this.radiusOf(s.norm);
      norms[i] = s.norm;
    }
    this.field.setData({ count: n, pos, color, rank: rankNormalize(norms), radius });
  }

  private dataBoxPx(): { x0: number; y0: number; x1: number; y1: number } {
    const [x0, y1] = this.field.worldToScreen(this.minX, this.minY);
    const [x1, y0] = this.field.worldToScreen(this.maxX, this.maxY);
    return { x0, y0, x1, y1 };
  }

  private positionLabels(): void {
    this.labelRoot.textContent = "";
    if (!this.bundle) return;
    const evr = this.bundle.explained_variance_ratio;
    const pc1 = ((evr[0] ?? 0) * 100).toFixed(1);
    const pc2 = ((evr[1] ?? 0) * 100).toFixed(1);

    const cap = (cls: string, html: string, sx: number, sy: number) => {
      const el = document.createElement("div");
      el.className = cls;
      el.innerHTML = html;
      el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px)`;
      this.labelRoot.appendChild(el);
    };
    // axis captions at the data extremes, on the isometric frame
    const dataCY = (this.minY + this.maxY) / 2;
    const dataCX = (this.minX + this.maxX) / 2;
    const [rx, ry] = this.field.worldToScreen(this.maxX, dataCY);
    cap("interp-embed-axis", `PC1 → · ${pc1}% var`, rx - 96, ry - 22);
    const [tx, ty] = this.field.worldToScreen(dataCX, this.maxY);
    cap("interp-embed-axis is-v", `PC2 ↑ · ${pc2}% var`, tx + 8, ty + 2);

    // anchor labels: the real extreme tokens, so the cloud has landmarks
    for (const s of this.anchors) {
      const [sx, sy] = this.field.worldToScreen(s.x, s.y);
      cap(
        `interp-embed-anchor${s.lead ? " is-space" : ""}`,
        escapeHtml(fmtTok(s.str)),
        sx + 6,
        sy - 8,
      );
    }
  }

  private onPointerMove(e: PointerEvent): void {
    // a tap-pinned tooltip (touch has no hover) survives a stray move
    if (this.tooltip.pinned) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const hit = this.field.pickAt(x, y, 6);
    const s = hit >= 0 ? ((this.stars[hit] as Star) ?? null) : null;
    if ((s?.id ?? -1) !== (this.hover?.id ?? -1)) {
      this.hover = s;
      this.field.setDim(s ? HOVER_DIM : 1);
      if (s) {
        const [sx, sy] = this.field.worldToScreen(s.x, s.y);
        this.marker.show(sx, sy, this.radiusOf(s.norm) + 2, this.dataBoxPx());
      } else {
        this.marker.hide();
      }
      this.field.render();
    }
    if (!s) {
      this.tooltip.hide();
      this.canvas.style.cursor = "";
      return;
    }
    this.showTooltipFor(s, x, y);
    this.canvas.style.cursor = "crosshair";
  }

  /** Row-building + placement, shared by the hover path and a touch tap-to-pin. */
  private showTooltipFor(s: Star, x: number, y: number): void {
    const [cr, cg, cb] = s.lead ? SPACE : NOSPACE;
    const rows: TipRow[] = [
      { kind: "label", text: `token “${fmtTok(s.str)}”`, swatch: [cr, cg, cb] },
      {
        text: `PC1 ${s.x.toFixed(2)} · PC2 ${s.y.toFixed(2)} · PC3 ${s.z.toFixed(2)}`,
      },
      {
        text: `‖W_E‖ = ${s.norm.toFixed(2)} · ${s.lead ? "leading space" : "no leading space"}`,
      },
    ];
    this.tooltip.show(rows);
    this.tooltip.move(x, y, this.cssW, this.cssH);
  }

  private onClick(e: PointerEvent): void {
    // this view has no click-driven selection — a tap stands in for the
    // hover this device doesn't have, both for the marker and the tooltip
    if (e.pointerType !== "touch") return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const hit = this.field.pickAt(x, y, 6);
    const s = hit >= 0 ? ((this.stars[hit] as Star) ?? null) : null;
    this.hover = s;
    this.field.setDim(s ? HOVER_DIM : 1);
    if (s) {
      const [sx, sy] = this.field.worldToScreen(s.x, s.y);
      this.marker.show(sx, sy, this.radiusOf(s.norm) + 2, this.dataBoxPx());
      this.tooltip.pinned = true;
      this.showTooltipFor(s, x, y);
    } else {
      this.marker.hide();
      this.tooltip.pinned = false;
      this.tooltip.hide();
    }
    this.field.render();
  }

  private onLeave(): void {
    // a tap-pinned tooltip must survive the pointer leaving — touch has no
    // hover state to "leave" in the first place
    if (this.tooltip.pinned) return;
    if (this.hover) {
      this.hover = null;
      this.marker.hide();
      this.field.setDim(1);
      this.field.render();
    }
    this.tooltip.hide();
    this.canvas.style.cursor = "";
  }

  frame(_dt: number, _t: number): void {
    // static — the constellation is one fixed projection, no data-bearing motion
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = width;
    this.cssH = height;
    this.field.resize(width, height, dpr); // re-applies the stored fit itself
    // the marker and tooltip are anchored in screen px, so a resize invalidates
    // both — drop the hover rather than leave chrome pointing at nothing
    this.hover = null;
    this.marker.hide();
    this.tooltip.hide();
    this.field.setDim(1);
    this.field.render();
    this.positionLabels();
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.tooltip?.dispose();
    this.marker?.dispose();
    this.labelRoot?.remove();
    this.field.dispose();
  }
}

function fmtTok(s: string): string {
  return s.replace(/^ /, "␣").replace(/\n/g, "⏎") || "∅";
}
function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
