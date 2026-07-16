/** Cluster label pills — HTML, not GPU text. Only ~200 clusters ever get a
 *  persistent pill (point labels appear on hover only), which is what makes
 *  DOM labels viable at 50K points. Greedy collision culling by cluster size,
 *  zoom-band hiding below a minimum projected radius, and per-frame transform
 *  updates that project through the same Camera2D the GPU renders with. */

import { hullRadius, type ClusterHull } from "../../data/hulls";
import type { NebulaiCluster } from "../../data/schema";
import type { Camera2D } from "../camera2d";
import { clusterColor } from "../layers/PointsLayer";

/** hide a pill when its cluster projects smaller than this radius (px) */
const MIN_PROJECTED_RADIUS = 24;
const SCREEN_MARGIN = 60;

interface Pill {
  el: HTMLElement;
  clusterId: number;
  anchor: [number, number];
  /** cluster centroid in pos3 (u3 display) space — the 3D anchor */
  anchor3: [number, number, number];
  worldRadius: number;
  size: number;
  w: number;
  h: number;
  shown: boolean;
}

/** Projects a morph-space world position through the render camera;
 *  null = outside the clip range (hide the pill). */
export type ProjectWorld = (x: number, y: number, z: number) => [number, number] | null;

export class LabelOverlay {
  private pills: Pill[] = [];
  private root: HTMLElement;

  constructor(
    container: HTMLElement,
    hulls: ClusterHull[],
    clusters: NebulaiCluster[],
    onSelect: (clusterId: number) => void,
  ) {
    this.root = document.createElement("div");
    this.root.className = "label-overlay";
    container.appendChild(this.root);

    const titles = new Map(clusters.map((c) => [c.id, c.title]));
    const centroids = new Map(clusters.map((c) => [c.id, c.centroid]));

    for (const hull of hulls) {
      const worldRadius = hullRadius(hull);

      const el = document.createElement("button");
      el.className = "cluster-pill";
      el.type = "button";
      el.textContent = titles.get(hull.clusterId) ?? `cluster ${hull.clusterId}`;
      const [r, g, b] = clusterColor(hull.clusterId);
      el.style.setProperty(
        "--dot",
        `rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)})`,
      );
      el.style.visibility = "hidden";
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        onSelect(hull.clusterId);
      });
      this.root.appendChild(el);

      const c3 = centroids.get(hull.clusterId);
      this.pills.push({
        el,
        clusterId: hull.clusterId,
        anchor: hull.anchor,
        anchor3: c3 ? [c3[0], c3[1], c3[2]] : [hull.anchor[0], hull.anchor[1], 0],
        worldRadius,
        size: hull.size,
        w: 0,
        h: 0,
        shown: false,
      });
    }

    // measure once — pills are fixed-content, width never changes afterward
    for (const p of this.pills) {
      p.w = p.el.offsetWidth || 80;
      p.h = p.el.offsetHeight || 22;
    }

    // biggest clusters win collisions
    this.pills.sort((a, b) => b.size - a.size);
  }

  set visible(v: boolean) {
    this.root.style.display = v ? "" : "none";
  }

  /** Fade the whole overlay (pointer events cut below 0.6 so half-faded
   *  pills can't steal clicks). The dimension morph no longer calls this —
   *  pills persist through the flythrough. */
  setFade(f: number): void {
    this.root.style.opacity = String(f);
    this.root.style.pointerEvents = f < 0.6 ? "none" : "";
  }

  /** Call when the camera moved (driver keeps a dirty flag). At morph 0 the
   *  fast top-down Camera2D path is used unchanged; mid-morph and in 3D each
   *  pill anchors to mix(anchor2, centroid3, morph) projected through the
   *  render camera, so labels track their clusters through the flythrough. */
  update(cam: Camera2D, morph = 0, projectWorld?: ProjectWorld): void {
    const kept: { x0: number; y0: number; x1: number; y1: number }[] = [];
    const pad = 4;
    const in3d = morph > 0.02 && projectWorld !== undefined;

    for (const p of this.pills) {
      const projRadius = p.worldRadius / cam.wpp;
      let show = projRadius >= MIN_PROJECTED_RADIUS;

      let sx = 0;
      let sy = 0;
      if (show) {
        if (in3d) {
          const proj = projectWorld!(
            p.anchor[0] * (1 - morph) + p.anchor3[0] * morph,
            p.anchor[1] * (1 - morph) + p.anchor3[1] * morph,
            p.anchor3[2] * morph,
          );
          if (proj) [sx, sy] = proj;
          else show = false; // outside the camera's clip range
        } else {
          [sx, sy] = cam.worldToScreen(p.anchor[0], p.anchor[1]);
        }
        show =
          show &&
          sx > -SCREEN_MARGIN &&
          sx < cam.viewportW + SCREEN_MARGIN &&
          sy > -SCREEN_MARGIN &&
          sy < cam.viewportH + SCREEN_MARGIN;
      }

      if (show) {
        const rect = {
          x0: sx - p.w / 2 - pad,
          y0: sy - p.h / 2 - pad,
          x1: sx + p.w / 2 + pad,
          y1: sy + p.h / 2 + pad,
        };
        for (const k of kept) {
          if (rect.x0 < k.x1 && rect.x1 > k.x0 && rect.y0 < k.y1 && rect.y1 > k.y0) {
            show = false;
            break;
          }
        }
        if (show) kept.push(rect);
      }

      if (show) {
        p.el.style.transform = `translate(-50%, -50%) translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px)`;
      }
      if (show !== p.shown) {
        p.el.style.visibility = show ? "visible" : "hidden";
        p.shown = show;
      }
    }
  }

  setSelected(clusterId: number | null): void {
    for (const p of this.pills) {
      p.el.classList.toggle("is-selected", p.clusterId === clusterId);
    }
  }

  dispose(): void {
    this.root.remove();
    this.pills = [];
  }
}
