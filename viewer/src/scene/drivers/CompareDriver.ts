/** Cross-model comparison — raw WebGPU, no three.js. A line-for-line port of
 *  the retired backend/viewer.py template: instanced quad sprites whose vertex
 *  shader interpolates between four layout states (native clouds → unified
 *  semantic space → per-model columns → concept knots) on a smoothstepped
 *  uniform t, premultiplied-alpha blending, orbit camera.
 *
 *  This driver is deliberately not three-TSL: it exists as the bespoke-WGSL
 *  rung of the multi-toolkit architecture (see nebulai-viz skill). It owns its
 *  own canvas — one driver per canvas, no shared GPU contexts — and is
 *  WebGPU-only by nature; the sidebar capability-gates the view mode.
 *
 *  At ~840 points the CPU projection hover loop is exact and cheap; that
 *  pattern does NOT scale to the atlas's 50K (which is why AtlasDriver uses
 *  kdbush / GPU id-buffer picking). */

import { appStore, type CompareUI } from "../../app/store";
import { COMPARE_FLOATS, compareInstances, type CompareData } from "../../data/compare";
import { BG, hexToRgb01 } from "../../styles/tokens";

const STATE_TWEEN_MS = 900;
const REDUCED_TWEEN_MS = 150;

/** view+proj (128) + params (16) + visA (16) + visB (16) + flags (16) */
const UBO_SIZE = 192;

const SHADER = /* wgsl */ `
struct U {
  view: mat4x4<f32>, proj: mat4x4<f32>,
  params: vec4<f32>,  // fromState, toState, t, pointScale
  visA: vec4<f32>, visB: vec4<f32>,
  flags: vec4<f32>,   // sharedOnly
};
@group(0) @binding(0) var<uniform> u: U;

struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>,
  @location(1) col: vec3<f32>, @location(2) shrd: f32 };

fn statePos(i: f32, p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>) -> vec3<f32> {
  if (i < 0.5) { return p0; } else if (i < 1.5) { return p1; }
  else if (i < 2.5) { return p2; } return p3;
}
fn visOf(si: f32) -> f32 {
  if (si < 0.5) { return u.visA.x; } else if (si < 1.5) { return u.visA.y; }
  else if (si < 2.5) { return u.visA.z; } else if (si < 3.5) { return u.visA.w; }
  else if (si < 4.5) { return u.visB.x; } else if (si < 5.5) { return u.visB.y; }
  else if (si < 6.5) { return u.visB.z; } return u.visB.w;
}

@vertex
fn vs(@location(0) corner: vec2<f32>,
      @location(1) nat: vec3<f32>, @location(2) sem: vec3<f32>,
      @location(3) bym: vec3<f32>, @location(4) byc: vec3<f32>,
      @location(5) col: vec3<f32>, @location(6) size: f32,
      @location(7) srcIdx: f32, @location(8) shrd: f32) -> VSOut {
  let tt = smoothstep(0.0, 1.0, u.params.z);
  let pf = statePos(u.params.x, nat, sem, bym, byc);
  let pt = statePos(u.params.y, nat, sem, bym, byc);
  let world = mix(pf, pt, tt);

  var vis = visOf(srcIdx);
  if (u.flags.x > 0.5 && shrd < 0.5) { vis = 0.0; }
  let sz = size * u.params.w * vis;

  var vp = u.view * vec4<f32>(world, 1.0);
  vp.x += corner.x * sz;
  vp.y += corner.y * sz;
  var o: VSOut;
  o.pos = u.proj * vp;
  o.uv = corner;
  o.col = col;
  o.shrd = shrd;
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  let edge = smoothstep(1.0, 0.72, d);
  let shade = 0.62 + 0.38 * (1.0 - d);
  var c = in.col * shade;
  if (in.shrd > 0.5) { c = mix(c, vec3<f32>(1.0), 0.14); } // shared points glow lighter
  return vec4<f32>(c * edge, edge);
}
`;

// ── column-major mat4 helpers (ported from the template) ───────────────────
function perspective(fovy: number, asp: number, n: number, f: number): Float32Array {
  const t = 1 / Math.tan(fovy / 2);
  const o = new Float32Array(16);
  o[0] = t / asp;
  o[5] = t;
  o[10] = (f + n) / (n - f);
  o[11] = -1;
  o[14] = (2 * f * n) / (n - f);
  return o;
}

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

function lookAt(e: V3, c: V3, up: V3): Float32Array {
  const z = norm(sub(e, c));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  const o = new Float32Array(16);
  o[0] = x[0]; o[1] = y[0]; o[2] = z[0];
  o[4] = x[1]; o[5] = y[1]; o[6] = z[1];
  o[8] = x[2]; o[9] = y[2]; o[10] = z[2];
  o[12] = -dot(x, e); o[13] = -dot(y, e); o[14] = -dot(z, e);
  o[15] = 1;
  return o;
}

export class CompareDriver {
  private canvas!: HTMLCanvasElement;
  private device: GPUDevice | null = null;
  private ctx: GPUCanvasContext | null = null;
  private pipeline!: GPURenderPipeline;
  private bind!: GPUBindGroup;
  private quadBuf!: GPUBuffer;
  private instBuf: GPUBuffer | null = null;
  private ubo!: GPUBuffer;
  private uboData = new Float32Array(UBO_SIZE / 4);
  private depthTex: GPUTexture | null = null;

  private data: CompareData | null = null;
  private count = 0;

  // orbit camera (spherical around target)
  private theta = 0.7;
  private phi = 0.35;
  private radius = 46;

  // state tween
  private curState = 1;
  private fromState = 1;
  private toState = 1;
  private tAnim = 1;
  private animStart = 0;
  private animing = false;

  private visible = [1, 1, 1, 1, 1, 1, 1, 1];
  private sharedOnly = 0;
  private pointScale = 1;

  private view: Float32Array | null = null;
  private proj: Float32Array | null = null;
  private cssW = 2;
  private cssH = 2;
  private dpr = 1;

  private tooltip: HTMLElement | null = null;
  private unsubscribe: (() => void) | null = null;
  private abort = new AbortController();
  private reducedMotion = false;

  async init(canvas: HTMLCanvasElement): Promise<void> {
    this.canvas = canvas;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("compare view needs WebGPU (no adapter)");
    const device = await adapter.requestDevice();
    this.device = device;
    const ctx = canvas.getContext("webgpu");
    if (!ctx) throw new Error("compare view: no webgpu canvas context");
    this.ctx = ctx;
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format: fmt, alphaMode: "premultiplied" });

    const quad = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
    this.quadBuf = device.createBuffer({
      size: quad.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.quadBuf, 0, quad);

    this.ubo = device.createBuffer({
      size: UBO_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const shader = device.createShaderModule({ code: SHADER });
    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: shader,
        entryPoint: "vs",
        buffers: [
          { arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] },
          {
            arrayStride: COMPARE_FLOATS * 4,
            stepMode: "instance",
            attributes: [
              { shaderLocation: 1, offset: 0, format: "float32x3" },
              { shaderLocation: 2, offset: 12, format: "float32x3" },
              { shaderLocation: 3, offset: 24, format: "float32x3" },
              { shaderLocation: 4, offset: 36, format: "float32x3" },
              { shaderLocation: 5, offset: 48, format: "float32x3" },
              { shaderLocation: 6, offset: 60, format: "float32" },
              { shaderLocation: 7, offset: 64, format: "float32" },
              { shaderLocation: 8, offset: 68, format: "float32" },
            ],
          },
        ],
      },
      fragment: {
        module: shader,
        entryPoint: "fs",
        targets: [
          {
            format: fmt,
            blend: {
              color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    this.bind = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubo } }],
    });

    this.reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.initInput();
    this.initTooltip();

    // mirror store → driver (chrome writes the store; we only read)
    const apply = (c: CompareUI) => {
      this.sharedOnly = c.sharedOnly ? 1 : 0;
      for (let i = 0; i < 8; i++) this.visible[i] = c.hiddenModels.includes(i) ? 0 : 1;
      if (c.state !== (this.animing ? this.toState : this.curState)) this.goto(c.state);
    };
    apply(appStore.getState().compare);
    let prev = appStore.getState().compare;
    this.unsubscribe = appStore.subscribe((s) => {
      if (s.compare !== prev) {
        prev = s.compare;
        apply(s.compare);
      }
      this.pointScale = s.settings.pointScale;
    });
  }

  setData(data: CompareData): void {
    if (!this.device) throw new Error("init first");
    this.data = data;
    this.count = data.points.length;
    this.instBuf?.destroy();
    const inst = compareInstances(data);
    this.instBuf = this.device.createBuffer({
      size: inst.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.instBuf, 0, inst);
  }

  resize(width: number, height: number, dpr: number): void {
    if (!this.device) return;
    this.cssW = Math.max(width, 1);
    this.cssH = Math.max(height, 1);
    this.dpr = Math.min(dpr, 2);
    const w = Math.max(Math.floor(this.cssW * this.dpr), 1);
    const h = Math.max(Math.floor(this.cssH * this.dpr), 1);
    if (this.canvas.width === w && this.canvas.height === h) return;
    this.canvas.width = w;
    this.canvas.height = h;
    this.depthTex?.destroy();
    this.depthTex = this.device.createTexture({
      size: [w, h],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  frame(_dt: number, _t: number): void {
    const device = this.device;
    if (!device || !this.ctx || !this.instBuf || !this.depthTex || this.count === 0) return;

    const now = performance.now();
    if (this.animing) {
      const dur = this.reducedMotion ? REDUCED_TWEEN_MS : STATE_TWEEN_MS;
      this.tAnim = Math.min(1, (now - this.animStart) / dur);
      if (this.tAnim >= 1) {
        this.animing = false;
        this.curState = this.toState;
        this.fromState = this.toState;
      }
    }

    const eye: V3 = [
      this.radius * Math.cos(this.phi) * Math.cos(this.theta),
      this.radius * Math.sin(this.phi),
      this.radius * Math.cos(this.phi) * Math.sin(this.theta),
    ];
    this.view = lookAt(eye, [0, 0, 0], [0, 1, 0]);
    this.proj = perspective(1.0, this.canvas.width / this.canvas.height, 0.1, 600);

    const u = this.uboData;
    u.set(this.view, 0);
    u.set(this.proj, 16);
    u[32] = this.animing ? this.fromState : this.curState;
    u[33] = this.animing ? this.toState : this.curState;
    u[34] = this.animing ? this.tAnim : 1;
    u[35] = this.pointScale;
    for (let i = 0; i < 8; i++) u[36 + i] = this.visible[i]!;
    u[44] = this.sharedOnly;
    device.queue.writeBuffer(this.ubo, 0, u);

    const [br, bg, bb] = hexToRgb01(BG);
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: this.ctx.getCurrentTexture().createView(),
          clearValue: { r: br, g: bg, b: bb, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthTex.createView(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind);
    pass.setVertexBuffer(0, this.quadBuf);
    pass.setVertexBuffer(1, this.instBuf);
    pass.draw(6, this.count);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  private goto(s: number): void {
    if (s === this.curState && !this.animing) return;
    this.fromState = this.animing ? this.toState : this.curState;
    this.curState = this.fromState;
    this.toState = s;
    this.animStart = performance.now();
    this.animing = true;
  }

  // ── input: orbit drag + wheel zoom + CPU hover ────────────────────────────
  private initInput(): void {
    let dragging = false;
    let lx = 0;
    let ly = 0;
    const sig = this.abort.signal;

    this.canvas.addEventListener(
      "pointerdown",
      (e) => {
        dragging = true;
        lx = e.clientX;
        ly = e.clientY;
        this.canvas.setPointerCapture(e.pointerId);
      },
      { signal: sig },
    );
    this.canvas.addEventListener("pointerup", () => (dragging = false), { signal: sig });
    this.canvas.addEventListener(
      "pointermove",
      (e) => {
        if (dragging) {
          this.theta -= (e.clientX - lx) * 0.006;
          this.phi = Math.max(-1.5, Math.min(1.5, this.phi + (e.clientY - ly) * 0.006));
          lx = e.clientX;
          ly = e.clientY;
        }
        const rect = this.canvas.getBoundingClientRect();
        this.updateTooltip(e.clientX - rect.left, e.clientY - rect.top);
      },
      { signal: sig },
    );
    this.canvas.addEventListener("pointerleave", () => this.hideTooltip(), { signal: sig });
    this.canvas.addEventListener(
      "wheel",
      (e) => {
        this.radius = Math.max(8, Math.min(180, this.radius * Math.exp(e.deltaY * 0.0012)));
      },
      { passive: true, signal: sig },
    );
  }

  /** world → CSS px through the same matrices the GPU rendered with. */
  private project(w: [number, number, number]): [number, number] | null {
    const v = this.view;
    const p = this.proj;
    if (!v || !p) return null;
    const [x, y, z] = w;
    const vx = v[0]! * x + v[4]! * y + v[8]! * z + v[12]!;
    const vy = v[1]! * x + v[5]! * y + v[9]! * z + v[13]!;
    const vz = v[2]! * x + v[6]! * y + v[10]! * z + v[14]!;
    // perspective matrix: only [0], [5], [10], [11]=-1, [14] populated
    const cx = p[0]! * vx;
    const cy = p[5]! * vy;
    const cw = -vz;
    if (cw <= 0) return null;
    return [((cx / cw) * 0.5 + 0.5) * this.cssW, (1 - ((cy / cw) * 0.5 + 0.5)) * this.cssH];
  }

  private initTooltip(): void {
    const overlay = document.getElementById("overlay-html");
    if (!overlay) return;
    this.tooltip = document.createElement("div");
    this.tooltip.className = "point-tooltip compare-tooltip";
    this.tooltip.style.visibility = "hidden";
    overlay.appendChild(this.tooltip);
  }

  private updateTooltip(mx: number, my: number): void {
    const data = this.data;
    if (!data || !this.tooltip) return;
    const stateName = data.states[this.animing ? this.toState : this.curState]!;
    let best = -1;
    let bd = 14;
    for (let i = 0; i < this.count; i++) {
      const pt = data.points[i]!;
      if (!this.visible[pt.source_idx]) continue;
      if (this.sharedOnly && !pt.shared) continue;
      const sc = this.project(pt.positions[stateName]!);
      if (!sc) continue;
      const d = Math.hypot(sc[0] - mx, sc[1] - my);
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    if (best < 0) {
      this.hideTooltip();
      return;
    }
    const pt = data.points[best]!;
    this.tooltip.textContent = "";
    const t = document.createElement("div");
    t.className = "point-tooltip-label";
    t.textContent = pt.title;
    const m = document.createElement("div");
    m.className = "point-tooltip-cluster";
    m.textContent = `${pt.source} · ${pt.shared ? "shared concept" : "unique"} · ${pt.size} tokens`;
    this.tooltip.append(t, m);
    this.tooltip.style.visibility = "visible";
    const w = this.tooltip.offsetWidth;
    const h = this.tooltip.offsetHeight;
    const x = Math.min(Math.max(mx + 14, 8), this.cssW - w - 8);
    const y = Math.min(Math.max(my + 14, 8), this.cssH - h - 8);
    this.tooltip.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
  }

  private hideTooltip(): void {
    if (this.tooltip) this.tooltip.style.visibility = "hidden";
  }

  dispose(): void {
    this.abort.abort();
    this.unsubscribe?.();
    this.tooltip?.remove();
    this.instBuf?.destroy();
    this.depthTex?.destroy();
    this.quadBuf?.destroy();
    this.ubo?.destroy();
    this.device?.destroy();
    this.device = null;
  }
}
