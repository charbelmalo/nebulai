/** Rotated count pills riding the beams ("4.9K", "25K" in the video). HTML,
 *  not GPU text — ≤64 badges, projected through the same Camera2D as the
 *  scene so they can't drift. Badges sit at the beam midpoint, rotated to the
 *  beam's screen angle, flipped when upside-down so text stays readable. */

import type { Camera2D } from "../scene/camera2d";

const MIN_BEAM_PX = 90; // hide badges on beams too short to label

export interface BadgeSpec {
  start: [number, number]; // world
  end: [number, number];
  text: string;
}

export class BeamBadges {
  private root: HTMLDivElement;
  private els: HTMLDivElement[] = [];
  private specs: BadgeSpec[] = [];

  constructor(container: HTMLElement) {
    this.root = document.createElement("div");
    this.root.className = "beam-badges";
    container.appendChild(this.root);
  }

  setBadges(specs: BadgeSpec[]): void {
    this.specs = specs;
    // grow the pool as needed; never shrink (cheap, avoids churn)
    while (this.els.length < specs.length) {
      const el = document.createElement("div");
      el.className = "beam-badge";
      this.root.appendChild(el);
      this.els.push(el);
    }
    for (let i = 0; i < this.els.length; i++) {
      const el = this.els[i]!;
      if (i < specs.length) {
        el.textContent = specs[i]!.text;
      } else {
        el.style.display = "none";
      }
    }
  }

  update(cam: Camera2D): void {
    for (let i = 0; i < this.specs.length; i++) {
      const s = this.specs[i]!;
      const el = this.els[i]!;
      const [x1, y1] = cam.worldToScreen(s.start[0], s.start[1]);
      const [x2, y2] = cam.worldToScreen(s.end[0], s.end[1]);
      const dx = x2 - x1;
      const dy = y2 - y1;
      if (Math.hypot(dx, dy) < MIN_BEAM_PX) {
        el.style.display = "none";
        continue;
      }
      let deg = (Math.atan2(dy, dx) * 180) / Math.PI;
      if (deg > 90) deg -= 180; // keep text upright
      if (deg < -90) deg += 180;
      const mx = (x1 + x2) / 2;
      const my = (y1 + y2) / 2;
      el.style.display = "";
      el.style.transform = `translate(-50%,-50%) translate(${mx.toFixed(1)}px,${my.toFixed(1)}px) rotate(${deg.toFixed(1)}deg)`;
    }
  }

  clear(): void {
    this.setBadges([]);
  }

  set visible(v: boolean) {
    this.root.style.display = v ? "" : "none";
  }

  dispose(): void {
    this.root.remove();
  }
}
