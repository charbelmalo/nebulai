/** The "US map" analogue: every cluster hull as a dark raised territory with a
 *  1px hairline coast. All ~200 fills merge into one geometry (one draw call),
 *  all outlines into one LineSegments — the layer costs two draws total.
 *  Sits at z = -0.05 so points float above it. */

import * as THREE from "three/webgpu";
import type { ClusterHull } from "../../data/hulls";
import { BG_RAISE, TEXT } from "../../styles/tokens";

/** Minimal position+index merge. The addons BufferGeometryUtils would pull a
 *  second copy of three core (it imports bare `three`, we use `three/webgpu`),
 *  and flat fills only need positions anyway. */
function mergeFills(geometries: THREE.BufferGeometry[]): THREE.BufferGeometry {
  let vertCount = 0;
  let indexCount = 0;
  for (const g of geometries) {
    vertCount += g.attributes.position!.count;
    indexCount += g.index!.count;
  }
  const positions = new Float32Array(vertCount * 3);
  const indices = new Uint32Array(indexCount);
  let vo = 0;
  let io = 0;
  for (const g of geometries) {
    const pos = g.attributes.position!;
    positions.set(pos.array as Float32Array, vo * 3);
    const idx = g.index!;
    for (let i = 0; i < idx.count; i++) indices[io + i] = idx.getX(i) + vo;
    vo += pos.count;
    io += idx.count;
  }
  const merged = new THREE.BufferGeometry();
  merged.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  merged.setIndex(new THREE.BufferAttribute(indices, 1));
  return merged;
}

const FILL_Z = -0.05;
const FILL_ALPHA = 0.85;
const STROKE_ALPHA = 0.1; // --hairline

export class TerritoriesLayer {
  readonly group = new THREE.Group();

  private disposables: { dispose(): void }[] = [];
  private fillMat: THREE.MeshBasicMaterial | null = null;
  private strokeMat: THREE.LineBasicMaterial | null = null;
  private shown = true;
  private fade = 1;

  constructor(hulls: ClusterHull[]) {
    const fills: THREE.BufferGeometry[] = [];
    const strokeVerts: number[] = [];

    for (const hull of hulls) {
      const m = hull.ring.length / 2;
      if (m < 3) continue; // degenerate hulls get no territory

      const shape = new THREE.Shape();
      shape.moveTo(hull.ring[0]!, hull.ring[1]!);
      for (let j = 1; j < m; j++) {
        shape.lineTo(hull.ring[j * 2]!, hull.ring[j * 2 + 1]!);
      }
      shape.closePath();
      fills.push(new THREE.ShapeGeometry(shape));

      for (let j = 0; j < m; j++) {
        const k = (j + 1) % m;
        strokeVerts.push(
          hull.ring[j * 2]!, hull.ring[j * 2 + 1]!, FILL_Z + 0.001,
          hull.ring[k * 2]!, hull.ring[k * 2 + 1]!, FILL_Z + 0.001,
        );
      }
    }

    if (fills.length > 0) {
      const merged = mergeFills(fills);
      for (const g of fills) g.dispose();
      const fillMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(BG_RAISE),
        transparent: true,
        opacity: FILL_ALPHA,
        depthWrite: false,
      });
      this.fillMat = fillMat;
      const fillMesh = new THREE.Mesh(merged, fillMat);
      fillMesh.position.z = FILL_Z;
      fillMesh.frustumCulled = false;
      this.group.add(fillMesh);
      this.disposables.push(merged, fillMat);
    }

    if (strokeVerts.length > 0) {
      const strokeGeo = new THREE.BufferGeometry();
      strokeGeo.setAttribute("position", new THREE.Float32BufferAttribute(strokeVerts, 3));
      const strokeMat = new THREE.LineBasicMaterial({
        color: new THREE.Color(TEXT),
        transparent: true,
        opacity: STROKE_ALPHA,
        depthWrite: false,
      });
      this.strokeMat = strokeMat;
      const stroke = new THREE.LineSegments(strokeGeo, strokeMat);
      stroke.frustumCulled = false;
      this.group.add(stroke);
      this.disposables.push(strokeGeo, strokeMat);
    }
  }

  set visible(v: boolean) {
    this.shown = v;
    this.apply();
  }

  /** Territories are flat 2-D hulls — the dimension morph fades them out. */
  setFade(f: number): void {
    this.fade = f;
    if (this.fillMat) this.fillMat.opacity = FILL_ALPHA * f;
    if (this.strokeMat) this.strokeMat.opacity = STROKE_ALPHA * f;
    this.apply();
  }

  private apply(): void {
    this.group.visible = this.shown && this.fade > 0.005;
  }

  dispose(): void {
    for (const d of this.disposables) d.dispose();
    this.disposables = [];
  }
}
