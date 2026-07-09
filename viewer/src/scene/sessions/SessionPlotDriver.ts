/** SessionPlotDriver — a true-3D plot of agent-mode sessions as trajectories.
 *
 *  Each analysed session (from chrome/sessionlog.ts) becomes a flight path
 *  through a REAL quantity cube, one node per model response (folded by message
 *  id / requestId, so token counts are honest — never per-JSONL-line):
 *
 *    X  = wall-clock time            (tSec, seconds since the session's start)
 *    Y  = context window size        (cache_read_input_tokens — the memory load,
 *                                      vertical because it's the resource that grows)
 *    Z  = new context THIS turn      (cache_creation_input_tokens — the fresh
 *                                      material read/ingested this turn; spikes mark
 *                                      big file-reads / tool results entering context)
 *    node size  ∝ tools invoked that turn   (activity)
 *    node color = dominant tool category    (orient/plan/edit/exec/deliver/reflect)
 *    path color = per-session hue            (to tell overlaid sessions apart)
 *    dimmed/smaller node = sub-agent (sidechain) turn
 *
 *  Z uses cache-write, not output tokens, on purpose: the SDK *audit* format
 *  streams only partial per-line output_tokens (they sum far below the true
 *  session total), whereas per-turn cache-write is exact — its per-turn sum
 *  reconciles to the authoritative result-line total. Plotting output per turn
 *  would be a misleading encoding on real large sessions.
 *
 *  Axes are normalised to a shared 100-unit cube by the GLOBAL max across the
 *  visible sessions, so magnitudes stay comparable when several are overlaid
 *  (a longer/heavier session genuinely reads bigger). Tick labels carry the
 *  real units. deck.gl OrbitView (WebGL2) with an orbit/zoom controller; GPU
 *  picking gives exact per-turn hover values. Nothing is faked or smoothed.
 */

import type { Deck, OrbitView, OrbitViewState, PickingInfo } from "@deck.gl/core";
import type { SessionAnalysis, SessionTurn, ToolCategory } from "../../chrome/sessionlog";

type LayersModule = typeof import("@deck.gl/layers");
type CoreModule = typeof import("@deck.gl/core");

const S = 100; // cube side in world units

/** Node color per dominant tool category — MUST match the page legend. */
export const CATEGORY_RGB: Record<ToolCategory, [number, number, number]> = {
  orient: [92, 198, 236], // cyan — reading / searching
  plan: [245, 190, 92], // gold — task lifecycle
  edit: [126, 222, 150], // green — writing files
  exec: [198, 130, 240], // violet — running commands
  deliver: [240, 120, 150], // pink — presenting / publishing
  reflect: [150, 158, 180], // grey — pure text / thinking
};

interface Node {
  sessionId: string;
  sessionName: string;
  index: number; // turn index within its session
  position: [number, number, number];
  floor: [number, number, number]; // same X/Z at Y=0 (drop-line foot)
  color: [number, number, number];
  radius: number;
  turn: SessionTurn;
}

interface Seg {
  source: [number, number, number];
  target: [number, number, number];
  color: [number, number, number];
}

interface TickLabel {
  position: [number, number, number];
  text: string;
  color: [number, number, number];
  anchor: "start" | "middle" | "end";
}

interface PathEntry {
  path: [number, number, number][];
  color: [number, number, number];
  isSub: boolean; // sub-agent excursion (own context window) vs main spine
}

/** golden-angle hue → rgb, for per-session path tinting. */
function sessionHue(i: number): [number, number, number] {
  const h = (i * 137.508) % 360;
  return hslToRgb(h / 360, 0.66, 0.62);
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  if (s === 0) {
    const v = Math.round(l * 255);
    return [v, v, v];
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const f = (t: number) => {
    let tt = t;
    if (tt < 0) tt += 1;
    if (tt > 1) tt -= 1;
    if (tt < 1 / 6) return p + (q - p) * 6 * tt;
    if (tt < 1 / 2) return q;
    if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
    return p;
  };
  return [
    Math.round(f(h + 1 / 3) * 255),
    Math.round(f(h) * 255),
    Math.round(f(h - 1 / 3) * 255),
  ];
}

function fmtTokens(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`;
  return `${Math.round(n)}`;
}
function fmtSecs(n: number): string {
  if (n >= 90) return `${(n / 60).toFixed(1)}m`;
  return `${Math.round(n)}s`;
}

export class SessionPlotDriver {
  private deck: Deck<OrbitView[]> | null = null;
  private core!: CoreModule;
  private layers!: LayersModule;
  private canvas!: HTMLCanvasElement;
  private overlay!: HTMLElement;
  private tooltip!: HTMLElement;

  private nodes: Node[] = [];
  private segs: Seg[] = [];
  private drops: Seg[] = [];
  private axes: Seg[] = [];
  private grid: Seg[] = [];
  private ticks: TickLabel[] = [];

  private cssW = 1;
  private cssH = 1;
  private dpr = 1;
  private viewState: OrbitViewState;
  private disposers: Array<() => void> = [];

  constructor() {
    this.viewState = {
      target: [S / 2, S * 0.42, S / 2],
      zoom: 2.4,
      rotationX: 24,
      rotationOrbit: -32,
      minZoom: 0.5,
      maxZoom: 6,
    };
  }

  async init(canvas: HTMLCanvasElement, overlay: HTMLElement): Promise<void> {
    this.canvas = canvas;
    this.overlay = overlay;
    const [core, layers] = await Promise.all([
      import("@deck.gl/core"),
      import("@deck.gl/layers"),
    ]);
    this.core = core;
    this.layers = layers;

    this.deck = new core.Deck({
      canvas,
      views: [new core.OrbitView({ id: "orbit", orbitAxis: "Y", fovy: 50 })],
      viewState: { orbit: this.viewState },
      controller: { inertia: 220, scrollZoom: { speed: 0.02, smooth: true } },
      useDevicePixels: Math.min(this.dpr, 2),
      onViewStateChange: ({ viewState }: { viewState: OrbitViewState }) => {
        this.viewState = viewState;
        this.deck?.setProps({ viewState: { orbit: viewState } });
      },
      onHover: (info: PickingInfo) => this.onHover(info),
      layers: [],
      width: this.cssW,
      height: this.cssH,
    }) as unknown as Deck<OrbitView[]>;

    this.tooltip = document.createElement("div");
    this.tooltip.className = "point-tooltip session-tooltip";
    this.tooltip.style.visibility = "hidden";
    overlay.appendChild(this.tooltip);
  }

  /** Rebuild the geometry for the given (already-active) sessions. */
  setSessions(analyses: SessionAnalysis[]): void {
    this.nodes = [];
    this.segs = [];
    this.drops = [];

    // global maxes across visible sessions → shared, comparable axes
    let xMax = 0;
    let yMax = 0;
    let zMax = 0;
    for (const a of analyses) {
      for (const t of a.turns) {
        if (t.tSec > xMax) xMax = t.tSec;
        if (t.cacheRead > yMax) yMax = t.cacheRead;
        if (t.cacheWrite > zMax) zMax = t.cacheWrite;
      }
    }
    // guard against degenerate (single point / all-zero) axes
    const sx = xMax > 0 ? S / xMax : 0;
    const sy = yMax > 0 ? S / yMax : 0;
    const sz = zMax > 0 ? S / zMax : 0;
    let maxTools = 1;
    for (const a of analyses) for (const t of a.turns) if (t.tools.length > maxTools) maxTools = t.tools.length;

    analyses.forEach((a, ai) => {
      const hue = sessionHue(ai);
      const pts: Node[] = [];
      for (const t of a.turns) {
        const pos: [number, number, number] = [t.tSec * sx, t.cacheRead * sy, t.cacheWrite * sz];
        const base = 3 + Math.sqrt(t.tools.length / maxTools) * 8;
        const radius = t.isSidechain ? base * 0.62 : base;
        const node: Node = {
          sessionId: a.id,
          sessionName: a.name,
          index: t.index,
          position: pos,
          floor: [pos[0], 0, pos[2]],
          color: CATEGORY_RGB[t.category],
          radius,
          turn: t,
        };
        pts.push(node);
        this.nodes.push(node);
        this.drops.push({ source: pos, target: [pos[0], 0, pos[2]], color: hue });
      }
      // trajectory: connect consecutive turns in time order
      for (let i = 1; i < pts.length; i++) {
        const prev = pts[i - 1];
        const cur = pts[i];
        if (prev && cur) this.segs.push({ source: prev.position, target: cur.position, color: hue });
      }
    });

    this.buildFrame(xMax, yMax, zMax);
    this.render();
  }

  /** Axis lines, floor grid, and tick labels carrying real units. */
  private buildFrame(xMax: number, yMax: number, zMax: number): void {
    const axisCol: [number, number, number] = [120, 132, 160];
    const gridCol: [number, number, number] = [70, 78, 100];
    this.axes = [
      { source: [0, 0, 0], target: [S, 0, 0], color: axisCol },
      { source: [0, 0, 0], target: [0, S, 0], color: axisCol },
      { source: [0, 0, 0], target: [0, 0, S], color: axisCol },
    ];
    this.grid = [];
    const DIV = 4;
    for (let i = 1; i <= DIV; i++) {
      const p = (i / DIV) * S;
      this.grid.push({ source: [p, 0, 0], target: [p, 0, S], color: gridCol });
      this.grid.push({ source: [0, 0, p], target: [S, 0, p], color: gridCol });
    }

    const tickCol: [number, number, number] = [178, 186, 210];
    const ticks: TickLabel[] = [];
    for (let i = 0; i <= DIV; i++) {
      const f = i / DIV;
      // X (time) — along floor front edge
      ticks.push({
        position: [f * S, 0, -6],
        text: fmtSecs(f * xMax),
        color: tickCol,
        anchor: "middle",
      });
      // Y (context) — up the vertical axis
      if (i > 0)
        ticks.push({
          position: [-6, f * S, 0],
          text: fmtTokens(f * yMax),
          color: tickCol,
          anchor: "end",
        });
      // Z (output) — along the left floor edge
      if (i > 0)
        ticks.push({
          position: [-6, 0, f * S],
          text: fmtTokens(f * zMax),
          color: tickCol,
          anchor: "end",
        });
    }
    // axis captions
    const capCol: [number, number, number] = [225, 230, 244];
    ticks.push({ position: [S * 0.5, 0, -16], text: "TIME →", color: capCol, anchor: "middle" });
    ticks.push({ position: [-16, S * 0.55, 0], text: "CONTEXT ↑", color: capCol, anchor: "end" });
    ticks.push({ position: [-16, 0, S * 0.5], text: "NEW CTX ↗", color: capCol, anchor: "end" });
    this.ticks = ticks;
  }

  private render(): void {
    if (!this.deck) return;
    const { LineLayer, ScatterplotLayer, PathLayer, TextLayer } = this.layers;
    const layers = [
      new LineLayer({
        id: "session-grid",
        data: this.grid,
        getSourcePosition: (d: Seg) => d.source,
        getTargetPosition: (d: Seg) => d.target,
        getColor: (d: Seg) => [...d.color, 60] as [number, number, number, number],
        getWidth: 1,
      }),
      new LineLayer({
        id: "session-axes",
        data: this.axes,
        getSourcePosition: (d: Seg) => d.source,
        getTargetPosition: (d: Seg) => d.target,
        getColor: (d: Seg) => [...d.color, 200] as [number, number, number, number],
        getWidth: 1.6,
      }),
      new LineLayer({
        id: "session-drops",
        data: this.drops,
        getSourcePosition: (d: Seg) => d.source,
        getTargetPosition: (d: Seg) => d.target,
        getColor: (d: Seg) => [...d.color, 40] as [number, number, number, number],
        getWidth: 1,
      }),
      new PathLayer({
        id: "session-paths",
        data: this.pathData(),
        getPath: (d: PathEntry) => d.path,
        // sub-agent excursions get a dimmer, thinner line so the main trajectory
        // reads as the session's real context spine (each agent is its own window)
        getColor: (d: PathEntry) => [...d.color, d.isSub ? 70 : 165] as [number, number, number, number],
        getWidth: (d: PathEntry) => (d.isSub ? 1.2 : 2),
        widthUnits: "pixels",
        capRounded: true,
        jointRounded: true,
        billboard: true,
        updateTriggers: { getColor: this.nodes.length, getWidth: this.nodes.length },
      }),
      new ScatterplotLayer({
        id: "session-nodes",
        data: this.nodes,
        getPosition: (d: Node) => d.position,
        getFillColor: (d: Node) =>
          [...d.color, d.turn.isSidechain ? 130 : 235] as [number, number, number, number],
        getRadius: (d: Node) => d.radius,
        radiusUnits: "pixels",
        billboard: true,
        stroked: true,
        getLineColor: [12, 14, 22, 220],
        lineWidthUnits: "pixels",
        getLineWidth: 1,
        pickable: true,
        radiusMinPixels: 2,
      }),
      new TextLayer({
        id: "session-ticks",
        data: this.ticks,
        getPosition: (d: TickLabel) => d.position,
        getText: (d: TickLabel) => d.text,
        getColor: (d: TickLabel) => [...d.color, 235] as [number, number, number, number],
        getSize: 11,
        sizeUnits: "pixels",
        getTextAnchor: (d: TickLabel) => d.anchor,
        getAlignmentBaseline: "center",
        billboard: true,
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
        characterSet: "auto",
      }),
    ];
    this.deck.setProps({ layers, viewState: { orbit: this.viewState } });
  }

  /** One polyline PER AGENT, not per session. A session's main agent and each
   *  spawned sub-agent have SEPARATE context windows; connecting them with one
   *  line would draw a false sawtooth (main context "collapsing" to a sub-agent's
   *  fresh window and back). We split by (sessionId, agentId) so each agent's
   *  turns are joined only to its own, in time order. Hue is per session so all
   *  its agents share a colour; sub-agent lines are flagged for dimmer styling. */
  private pathData(): PathEntry[] {
    const byAgent = new Map<string, PathEntry>();
    const sessionHueIdx = new Map<string, number>();
    const agentOrder: string[] = [];
    for (const n of this.nodes) {
      if (!sessionHueIdx.has(n.sessionId)) sessionHueIdx.set(n.sessionId, sessionHueIdx.size);
      const agentId = n.turn.agentId ?? "main"; // tolerate pre-agentId persisted records
      const key = `${n.sessionId}::${agentId}`;
      let e = byAgent.get(key);
      if (!e) {
        e = {
          path: [],
          color: sessionHue(sessionHueIdx.get(n.sessionId) ?? 0),
          isSub: agentId !== "main",
        };
        byAgent.set(key, e);
        agentOrder.push(key);
      }
      e.path.push(n.position);
    }
    return agentOrder.map((k) => byAgent.get(k)!).filter((e) => e.path.length >= 2);
  }

  private onHover(info: PickingInfo): void {
    const obj = info.object as Node | undefined;
    if (!obj || info.layer?.id !== "session-nodes") {
      this.tooltip.style.visibility = "hidden";
      return;
    }
    const t = obj.turn;
    const tools = t.tools.length
      ? t.tools.map((n) => n.split("__").pop()).join(" · ")
      : "—";
    this.tooltip.innerHTML = `
      <div class="session-tt-head">
        <span class="session-tt-name">${escapeHtml(obj.sessionName)}</span>
        <span class="session-tt-turn">response ${obj.index + 1}${t.isSidechain ? " · sub-agent" : ""}</span>
      </div>
      <div class="session-tt-cat" data-cat="${t.category}">${t.category}</div>
      <dl class="session-tt-grid">
        <dt>t+</dt><dd>${fmtSecs(t.tSec)}</dd>
        <dt>context</dt><dd>${t.cacheRead.toLocaleString()} tok</dd>
        <dt>new&nbsp;ctx</dt><dd>${t.cacheWrite.toLocaleString()} tok</dd>
        <dt>tools</dt><dd>${escapeHtml(tools)}</dd>
      </dl>`;
    this.tooltip.style.visibility = "visible";
    const x = info.x ?? 0;
    const y = info.y ?? 0;
    this.tooltip.style.left = `${x + 14}px`;
    this.tooltip.style.top = `${y + 14}px`;
  }

  resize(width: number, height: number, dpr: number): void {
    this.cssW = Math.max(1, width);
    this.cssH = Math.max(1, height);
    this.dpr = dpr;
    this.deck?.setProps({
      width: this.cssW,
      height: this.cssH,
      useDevicePixels: Math.min(dpr, 2),
    });
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
    this.deck?.finalize();
    this.deck = null;
    this.tooltip?.remove();
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    c === "&" ? "&amp;" : c === "<" ? "&lt;" : c === ">" ? "&gt;" : c === '"' ? "&quot;" : "&#39;",
  );
}
