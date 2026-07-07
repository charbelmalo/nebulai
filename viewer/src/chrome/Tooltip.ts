/** Hover tooltip for individual points. Plain DOM for M1 (Preact chrome lands
 *  with M2); token labels render repr-style so whitespace and byte fragments
 *  stay visible — " the" and "the" are different tokens and must look it. */

export interface TooltipContent {
  label: string;
  clusterTitle: string | null; // null = noise
  confidence: number; // 0–1
}

export class Tooltip {
  private el: HTMLElement;
  private labelEl: HTMLElement;
  private clusterEl: HTMLElement;
  private confEl: HTMLElement;

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

    this.el.append(this.labelEl, this.clusterEl, this.confEl);
    container.appendChild(this.el);
  }

  show(sx: number, sy: number, content: TooltipContent): void {
    this.labelEl.textContent = JSON.stringify(content.label);
    this.clusterEl.textContent = content.clusterTitle ?? "noise (unclustered)";
    this.confEl.textContent = `confidence ${(content.confidence * 100).toFixed(0)}%`;

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

  hide(): void {
    this.el.style.visibility = "hidden";
  }

  dispose(): void {
    this.el.remove();
  }
}
