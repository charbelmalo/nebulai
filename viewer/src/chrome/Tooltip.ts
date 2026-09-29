/** Hover tooltip for individual points. Plain DOM for M1 (Preact chrome lands
 *  with M2); token labels render repr-style so whitespace and byte fragments
 *  stay visible — " the" and "the" are different tokens and must look it. */

/** One measured scalar for the hovered point, printed exactly (R8).
 *
 *  `value: null` is "not measured", and it renders as an em dash with the word
 *  beside it. It must never fall back to 0: the glitch lens is a claim about
 *  small numbers, so a zero in this readout is the most misleading value the
 *  tooltip could possibly print. */
export interface TooltipChannel {
  label: string;
  value: number | null;
  units: string;
}

export interface TooltipContent {
  label: string;
  clusterTitle: string | null; // null = noise
  confidence: number; // 0–1
  /** measured per-point scalars, in the order they should be read */
  channels?: TooltipChannel[];
}

/** How many scalar lines the hover box will carry before it stops being a
 *  tooltip and starts being a table. The rail is where a long list belongs. */
const MAX_CHANNEL_LINES = 4;

export class Tooltip {
  private el: HTMLElement;
  private labelEl: HTMLElement;
  private clusterEl: HTMLElement;
  private confEl: HTMLElement;
  private chanEl: HTMLElement;

  constructor(container: HTMLElement) {
    this.el = document.createElement("div");
    this.el.className = "point-tooltip";
    this.el.style.visibility = "hidden";

    this.labelEl = document.createElement("div");
    this.labelEl.className = "point-tooltip-label";
    this.clusterEl = document.createElement("div");
    this.clusterEl.className = "point-tooltip-cluster";
    this.confEl = document.createElement("div");
    this.confEl.className = "point-tooltip-conf";
    this.chanEl = document.createElement("div");
    this.chanEl.className = "point-tooltip-channels";

    this.el.append(this.labelEl, this.clusterEl, this.confEl, this.chanEl);
    container.appendChild(this.el);
  }

  show(sx: number, sy: number, content: TooltipContent): void {
    this.labelEl.textContent = JSON.stringify(content.label);
    this.clusterEl.textContent = content.clusterTitle ?? "noise (unclustered)";
    this.confEl.textContent = `confidence ${(content.confidence * 100).toFixed(0)}%`;
    this.renderChannels(content.channels ?? []);

    this.el.style.visibility = "visible";
    // measure after content so the clamp uses the real size
    const w = this.el.offsetWidth;
    const h = this.el.offsetHeight;
    const vw = this.el.parentElement?.clientWidth ?? window.innerWidth;
    const vh = this.el.parentElement?.clientHeight ?? window.innerHeight;
    const x = Math.min(Math.max(sx + 14, 8), vw - w - 8);
    const y = Math.min(Math.max(sy + 14, 8), vh - h - 8);
    this.el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
  }

  private renderChannels(channels: TooltipChannel[]): void {
    this.chanEl.textContent = "";
    const shown = channels.slice(0, MAX_CHANNEL_LINES);
    this.chanEl.hidden = shown.length === 0;
    for (const c of shown) {
      const row = document.createElement("div");
      row.className = "point-tooltip-channel";
      const k = document.createElement("span");
      k.className = "point-tooltip-channel-k";
      k.textContent = c.label;
      const v = document.createElement("span");
      v.className = "point-tooltip-channel-v";
      if (c.value === null || !Number.isFinite(c.value)) {
        v.textContent = "— not measured";
        v.classList.add("is-missing");
      } else {
        v.textContent = c.units ? `${c.value.toFixed(3)} ${c.units}` : c.value.toFixed(3);
      }
      row.append(k, v);
      this.chanEl.appendChild(row);
    }
  }

  hide(): void {
    this.el.style.visibility = "hidden";
  }

  dispose(): void {
    this.el.remove();
  }
}
