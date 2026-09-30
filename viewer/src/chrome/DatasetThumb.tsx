/** DatasetThumb.tsx — the small preview image of a published map.
 *
 *  `scripts/make_thumbs.py` renders `out/<id>/thumb.webp` from the map's own
 *  2-D layout and cluster ids. The choosers show it so a visitor sees the
 *  shape of a map before asking for its file. Decorative (the card's text
 *  names the map), and a server without the image just shows the frame. */

import { useState } from "preact/hooks";
import { DATA_BASE } from "../data/base";

export function thumbUrl(datasetPath: string, base = DATA_BASE): string {
  const dir = datasetPath.slice(0, datasetPath.lastIndexOf("/"));
  return `${base}/${dir}/thumb.webp`;
}

export function DatasetThumb(props: { path: string; caption?: string }) {
  const [failed, setFailed] = useState(false);
  return (
    <span class="dataset-thumb" aria-hidden="true">
      {!failed && (
        <img
          src={thumbUrl(props.path)}
          alt=""
          width={240}
          height={150}
          loading="lazy"
          decoding="async"
          onError={() => setFailed(true)}
        />
      )}
      {props.caption && <span class="dataset-thumb-caption">{props.caption}</span>}
    </span>
  );
}

/** A power spectrum drawn from a verified Fourier bundle: log power against
 *  frequency, the same numbers the task's table reads. */
export function SpectrumThumb(props: { freqs: readonly number[]; power: readonly number[]; caption: string }) {
  const W = 240;
  const H = 150;
  const pts: string[] = [];
  const n = props.freqs.length;
  let lo = Infinity;
  let hi = -Infinity;
  const logs: number[] = [];
  for (let i = 1; i < n; i++) {
    const v = Math.log10(Math.max(props.power[i] ?? 0, 1e-12));
    logs.push(v);
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const span = Math.max(hi - lo, 1e-9);
  // log frequency axis: the low frequencies carry the structure
  const lx = (i: number) => Math.log(i) / Math.log(n - 1);
  for (let i = 1; i < n; i++) {
    const x = 10 + lx(i) * (W - 20);
    const y = H - 22 - ((logs[i - 1]! - lo) / span) * (H - 40);
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  const line = pts.join(" ");
  return (
    <span class="dataset-thumb is-spectrum" aria-hidden="true">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <defs>
          <linearGradient id="spectrum-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="var(--gold, #f2b84b)" stop-opacity="0.35" />
            <stop offset="1" stop-color="var(--gold, #f2b84b)" stop-opacity="0" />
          </linearGradient>
        </defs>
        <polygon points={`10,${H - 22} ${line} ${W - 10},${H - 22}`} fill="url(#spectrum-fill)" />
        <polyline points={line} fill="none" stroke="var(--gold, #f2b84b)" stroke-width="1.4" vector-effect="non-scaling-stroke" />
      </svg>
      <span class="dataset-thumb-caption">{props.caption}</span>
    </span>
  );
}
