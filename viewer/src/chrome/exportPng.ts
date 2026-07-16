/** Save-as-PNG for the active view. Canvas pixels alone are NOT the view —
 *  every interp driver draws its token text, axis labels, and legend as HTML
 *  overlay (deck's ASCII font atlas silently drops ␣/⏎/… glyphs), so a
 *  canvas-only capture would export a chart missing its own labels — a
 *  truthfulness bug. Instead we composite the whole stage element:
 *
 *    1. snapshot every <canvas> inside the stage to a data URI,
 *    2. clone the stage subtree with each canvas swapped for that <img>,
 *    3. inline all same-origin stylesheet rules (the app has no external
 *       fonts — system stacks only — so text rasterizes faithfully),
 *    4. serialize into <svg><foreignObject> and draw it onto a 2-D canvas at
 *       the device pixel ratio, on the app's real background color.
 *
 *  No screenshot library, no network: everything comes from the live DOM. */

import { appStore } from "../app/store";

/** The element that owns the active page's visualization. */
function stageFor(page: string): HTMLElement | null {
  const sel: Record<string, string> = {
    map: "#stage",
    interp: ".interp-stage",
    snapshot: ".snapshot-stage",
    sessions: ".sessions-stage",
  };
  const s = sel[page];
  return s ? document.querySelector<HTMLElement>(s) : null;
}

function collectCss(): string {
  let css = "";
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      for (const rule of Array.from(sheet.cssRules)) css += `${rule.cssText}\n`;
    } catch {
      // cross-origin sheet — none in this app, but never let one break export
    }
  }
  // SVG-image rasterization samples animations at t=0, so any entrance
  // animation with a from{opacity:0} (e.g. .interp-canvas's fade-in) would
  // export its element invisible — freeze everything at its settled state
  css += "*{animation:none !important;transition:none !important}\n";
  return css;
}

function cloneWithCanvasImages(root: HTMLElement): HTMLElement {
  const clone = root.cloneNode(true) as HTMLElement;
  const liveCanvases = root.querySelectorAll("canvas");
  const cloneCanvases = clone.querySelectorAll("canvas");
  cloneCanvases.forEach((c, i) => {
    const live = liveCanvases[i];
    if (!live) return;
    const img = document.createElement("img");
    try {
      img.src = live.toDataURL("image/png");
    } catch {
      return; // tainted/unreadable canvas: leave the (blank) canvas in place
    }
    // pin the CSS box the canvas occupied; the backing store is often DPR-scaled
    const r = live.getBoundingClientRect();
    img.style.width = `${r.width}px`;
    img.style.height = `${r.height}px`;
    img.className = c.className;
    (c as HTMLElement).replaceWith(img);
  });
  // inputs lose their typed value on cloneNode — copy it into the attribute so
  // the live prompt (etc.) shows in the export
  const liveInputs = root.querySelectorAll("input");
  clone.querySelectorAll("input").forEach((inp, i) => {
    const live = liveInputs[i];
    if (live) inp.setAttribute("value", live.value);
  });
  return clone;
}

/** Render the active page's stage to a PNG blob. Resolves null when the page
 *  has no capturable stage (e.g. the Guide). */
export async function captureStagePng(): Promise<{ blob: Blob; name: string } | null> {
  const st = appStore.getState();
  const stage = stageFor(st.page);
  if (!stage) return null;

  const rect = stage.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  const clone = cloneWithCanvasImages(stage);
  clone.style.width = `${w}px`;
  clone.style.height = `${h}px`;
  clone.style.margin = "0";

  const bg = getComputedStyle(document.body).backgroundColor || "#07080d";
  const wrap = document.createElement("div");
  wrap.appendChild(clone);
  const xhtml = new XMLSerializer().serializeToString(wrap);
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">` +
    `<foreignObject width="100%" height="100%">` +
    `<div xmlns="http://www.w3.org/1999/xhtml"><style>${collectCss()}</style>${xhtml}</div>` +
    `</foreignObject></svg>`;

  const img = new Image();
  // MUST be a data: URI — Chrome taints the destination canvas when a
  // foreignObject SVG is drawn from a blob: URL, which would make toBlob throw
  const url = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  {
    await new Promise<void>((res, rej) => {
      img.onload = () => res();
      img.onerror = () => rej(new Error("stage rasterization failed"));
      img.src = url;
    });
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const out = document.createElement("canvas");
    out.width = Math.round(w * dpr);
    out.height = Math.round(h * dpr);
    const g = out.getContext("2d");
    if (!g) return null;
    g.fillStyle = bg;
    g.fillRect(0, 0, out.width, out.height);
    g.scale(dpr, dpr);
    g.drawImage(img, 0, 0);
    const blob = await new Promise<Blob | null>((res) => out.toBlob(res, "image/png"));
    if (!blob) return null;
    return { blob, name: pngName() };
  }
}

function pngName(): string {
  const st = appStore.getState();
  const parts = ["nebulai", st.page];
  if (st.page === "map") parts.push(st.viewMode, st.datasetId ?? "");
  if (st.page === "interp") parts.push(st.interp.featureId, st.datasetId ?? "");
  const ts = new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");
  return `${parts.filter(Boolean).join("-")}-${ts}.png`;
}

/** Capture + trigger a download. Returns false when nothing was capturable. */
export async function downloadStagePng(): Promise<boolean> {
  const shot = await captureStagePng();
  if (!shot) return false;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(shot.blob);
  a.download = shot.name;
  a.click();
  URL.revokeObjectURL(a.href);
  return true;
}
