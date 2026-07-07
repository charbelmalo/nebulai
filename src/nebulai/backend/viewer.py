"""Self-contained WebGPU viewer for the cross-model comparison.

DEPRECATED: the unified viewer app (`viewer/`, CompareDriver) supersedes this
standalone page — it renders the same compare.json with the shared chrome,
model legend, and honesty stats. This module keeps writing index.html for one
release as a fallback, then retires.

`write_viewer` injects the comparison JSON into an HTML template. Each point
carries its position in every layout state; switching state animates a uniform
`t` and the vertex shader interpolates on the GPU, so the whole cloud morphs
smoothly (native clouds -> unified semantic space -> per-model columns ->
collapsed concept knots). Colors encode the source model; shared concepts can
be isolated. Opens directly in a WebGPU-capable browser (Chrome/Edge).
"""

import json
from pathlib import Path

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Nebul.AI — model comparison</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: #05060a; color: #e8ecf4;
    font: 13px/1.5 -apple-system, "SF Pro Text", Segoe UI, Roboto, sans-serif; overflow: hidden; }
  canvas { position: fixed; inset: 0; width: 100vw; height: 100vh; display: block; }
  .panel { position: fixed; background: rgba(12,16,26,.82); backdrop-filter: blur(10px);
    border: 1px solid rgba(120,140,180,.18); border-radius: 12px; padding: 12px 14px; }
  #top { top: 16px; left: 16px; max-width: 340px; }
  #top h1 { margin: 0 0 2px; font-size: 15px; letter-spacing: .2px; }
  #top .sub { color: #93a0b8; font-size: 11.5px; margin-bottom: 10px; }
  .states { display: flex; flex-wrap: wrap; gap: 6px; }
  .states button { cursor: pointer; border: 1px solid rgba(120,140,180,.25); background: rgba(30,38,56,.6);
    color: #cdd6e8; padding: 6px 10px; border-radius: 8px; font-size: 12px; transition: .15s; }
  .states button:hover { border-color: rgba(120,160,220,.6); }
  .states button.active { background: #2a6cf0; border-color: #2a6cf0; color: #fff; }
  #side { top: 16px; right: 16px; width: 250px; }
  .legend-row { display: flex; align-items: center; gap: 8px; padding: 3px 0; cursor: pointer; user-select: none; }
  .legend-row.off { opacity: .35; }
  .swatch { width: 12px; height: 12px; border-radius: 3px; flex: none; }
  .legend-row .n { color: #8b97ad; margin-left: auto; font-variant-numeric: tabular-nums; }
  hr { border: 0; border-top: 1px solid rgba(120,140,180,.15); margin: 10px 0; }
  .stat { display: flex; justify-content: space-between; padding: 2px 0; color: #b9c3d6; }
  .stat b { color: #eef2fa; font-variant-numeric: tabular-nums; }
  .jac { color: #8b97ad; font-size: 11.5px; }
  label.toggle { display: flex; align-items: center; gap: 7px; margin-top: 8px; color: #cdd6e8; cursor: pointer; }
  #hint { bottom: 14px; left: 50%; transform: translateX(-50%); color: #7c8aa2; font-size: 11.5px;
    background: rgba(10,13,20,.6); padding: 6px 12px; }
  #tip { position: fixed; pointer-events: none; background: rgba(8,11,18,.95); border: 1px solid rgba(120,140,180,.3);
    border-radius: 8px; padding: 7px 10px; font-size: 12px; display: none; max-width: 260px; z-index: 20; }
  #tip .t { font-weight: 600; margin-bottom: 2px; }
  #tip .m { color: #9aa6bd; }
  #err { position: fixed; inset: 0; display: none; place-items: center; text-align: center; padding: 40px; }
  #err div { max-width: 420px; color: #c7d0e2; }
</style>
</head>
<body>
<canvas id="c"></canvas>

<div id="top" class="panel">
  <h1>Nebul.AI — model comparison</h1>
  <div class="sub" id="subtitle"></div>
  <div class="states" id="states"></div>
</div>

<div id="side" class="panel">
  <div id="legend"></div>
  <label class="toggle"><input type="checkbox" id="sharedOnly"> shared concepts only</label>
  <hr>
  <div class="stat"><span>shared concepts</span><b id="s-shared">–</b></div>
  <div id="s-unique"></div>
  <hr>
  <div class="jac" id="s-jac"></div>
</div>

<div id="hint" class="panel">drag to orbit · scroll to zoom · hover a point for its concept</div>
<div id="tip"></div>
<div id="err"><div><h2>WebGPU unavailable</h2><p>Open this file in Chrome or Edge (WebGPU enabled). In Chrome you can also visit <code>chrome://flags</code> → enable “Unsafe WebGPU”.</p></div></div>

<script type="module">
const DATA = /*__NEBULAI_DATA__*/;

const err = document.getElementById('err');
if (!navigator.gpu) { err.style.display = 'grid'; throw new Error('no webgpu'); }

// ---------- math ----------
const mul = (a,b)=>{const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;};
function perspective(fovy,asp,n,f){const t=1/Math.tan(fovy/2);const o=new Float32Array(16);
  o[0]=t/asp;o[5]=t;o[10]=(f+n)/(n-f);o[11]=-1;o[14]=(2*f*n)/(n-f);return o;}
function lookAt(e,c,u){const z=norm(sub(e,c)),x=norm(cross(u,z)),y=cross(z,x);const o=new Float32Array(16);
  o[0]=x[0];o[1]=y[0];o[2]=z[0];o[4]=x[1];o[5]=y[1];o[6]=z[1];o[8]=x[2];o[9]=y[2];o[10]=z[2];
  o[12]=-dot(x,e);o[13]=-dot(y,e);o[14]=-dot(z,e);o[15]=1;return o;}
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const norm=a=>{const l=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/l,a[1]/l,a[2]/l];};

// ---------- data → instance buffer ----------
const STATES = DATA.states;
const pts = DATA.points;
const N = pts.length;
const models = DATA.meta.models;
let maxSize = 1; for (const p of pts) maxSize = Math.max(maxSize, p.size);
const FLOATS = 18; // 4 states*3 + color3 + size + srcIdx + shared
const inst = new Float32Array(N * FLOATS);
for (let i=0;i<N;i++){
  const p = pts[i], o = i*FLOATS;
  const st = s => p.positions[s];
  inst.set(st('native'), o+0);
  inst.set(st('semantic'), o+3);
  inst.set(st('by_model'), o+6);
  inst.set(st('by_concept'), o+9);
  inst.set(p.color, o+12);
  inst[o+15] = 0.28 + 0.95*Math.sqrt(p.size/maxSize);
  inst[o+16] = p.source_idx;
  inst[o+17] = p.shared ? 1 : 0;
}

// ---------- gpu ----------
const canvas = document.getElementById('c');
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const ctx = canvas.getContext('webgpu');
const fmt = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({ device, format: fmt, alphaMode: 'premultiplied' });

const quad = new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]);
const quadBuf = device.createBuffer({ size: quad.byteLength, usage: GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(quadBuf, 0, quad);
const instBuf = device.createBuffer({ size: inst.byteLength, usage: GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(instBuf, 0, inst);

const UBO_SIZE = 128 + 16 + 16 + 16 + 16; // view+proj + params + visA + visB + flags
const ubo = device.createBuffer({ size: UBO_SIZE, usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });

const shader = device.createShaderModule({ code: `
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
` });

const pipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: { module: shader, entryPoint: 'vs', buffers: [
    { arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] },
    { arrayStride: FLOATS*4, stepMode: 'instance', attributes: [
      { shaderLocation:1, offset:0,  format:'float32x3' },
      { shaderLocation:2, offset:12, format:'float32x3' },
      { shaderLocation:3, offset:24, format:'float32x3' },
      { shaderLocation:4, offset:36, format:'float32x3' },
      { shaderLocation:5, offset:48, format:'float32x3' },
      { shaderLocation:6, offset:60, format:'float32' },
      { shaderLocation:7, offset:64, format:'float32' },
      { shaderLocation:8, offset:68, format:'float32' },
    ]},
  ]},
  fragment: { module: shader, entryPoint: 'fs', targets: [{
    format: fmt,
    blend: { color: { srcFactor:'one', dstFactor:'one-minus-src-alpha' },
             alpha: { srcFactor:'one', dstFactor:'one-minus-src-alpha' } },
  }]},
  primitive: { topology: 'triangle-list' },
  depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
});
const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding:0, resource:{ buffer: ubo } }] });

let depthTex;
function resize(){
  const dpr = Math.min(devicePixelRatio||1, 2);
  canvas.width = Math.floor(innerWidth*dpr); canvas.height = Math.floor(innerHeight*dpr);
  if (depthTex) depthTex.destroy();
  depthTex = device.createTexture({ size:[canvas.width,canvas.height], format:'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
}
addEventListener('resize', resize); resize();

// ---------- camera ----------
let theta = 0.7, phi = 0.35, radius = 46, target = [0,0,0];
let dragging = false, lx=0, ly=0;
canvas.addEventListener('pointerdown', e=>{ dragging=true; lx=e.clientX; ly=e.clientY; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointerup', ()=> dragging=false);
canvas.addEventListener('pointermove', e=>{
  if (dragging){ theta -= (e.clientX-lx)*0.006; phi += (e.clientY-ly)*0.006;
    phi = Math.max(-1.5, Math.min(1.5, phi)); lx=e.clientX; ly=e.clientY; }
  updateTip(e.clientX, e.clientY);
});
addEventListener('wheel', e=>{ radius *= Math.exp(e.deltaY*0.0012); radius = Math.max(8, Math.min(180, radius)); }, {passive:true});

// ---------- state machine ----------
let curState = 1, fromState = 1, toState = 1, tAnim = 1, animStart = 0, animing = false;
const DUR = 900;
function goto(s){ if (s===curState && !animing) return; fromState = curState; toState = s; animStart = performance.now(); animing = true; setActive(s); }

const visible = models.map(()=>1);
let sharedOnly = 0;

// ---------- uniforms ----------
const ubuf = new Float32Array(UBO_SIZE/4);
function writeUBO(view, proj, from, to, t){
  ubuf.set(view, 0); ubuf.set(proj, 16);
  ubuf[32]=from; ubuf[33]=to; ubuf[34]=t; ubuf[35]=pointScale;
  ubuf[36]=visible[0]||0; ubuf[37]=visible[1]||0; ubuf[38]=visible[2]||0; ubuf[39]=visible[3]||0;
  ubuf[40]=visible[4]||0; ubuf[41]=visible[5]||0; ubuf[42]=visible[6]||0; ubuf[43]=visible[7]||0;
  ubuf[44]=sharedOnly;
  device.queue.writeBuffer(ubo, 0, ubuf);
}
let pointScale = 1.0;

// ---------- render loop ----------
let curView, curProj;
function frame(now){
  if (animing){ tAnim = Math.min(1, (now-animStart)/DUR); if (tAnim>=1){ animing=false; curState=toState; fromState=toState; } }
  const eye = [ target[0] + radius*Math.cos(phi)*Math.cos(theta),
               target[1] + radius*Math.sin(phi),
               target[2] + radius*Math.cos(phi)*Math.sin(theta) ];
  curView = lookAt(eye, target, [0,1,0]);
  curProj = perspective(1.0, canvas.width/canvas.height, 0.1, 600);
  writeUBO(curView, curProj, animing?fromState:curState, animing?toState:curState, animing?tAnim:1);

  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue:{r:0.02,g:0.024,b:0.04,a:1}, loadOp:'clear', storeOp:'store' }],
    depthStencilAttachment: { view: depthTex.createView(), depthClearValue:1, depthLoadOp:'clear', depthStoreOp:'store' },
  });
  pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.setVertexBuffer(0, quadBuf); pass.setVertexBuffer(1, instBuf);
  pass.draw(6, N);
  pass.end();
  device.queue.submit([enc.finish()]);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---------- hover picking (CPU project) ----------
const tip = document.getElementById('tip');
function projPoint(w){
  const v = curView, p = curProj;
  const x=w[0],y=w[1],z=w[2];
  const vx=v[0]*x+v[4]*y+v[8]*z+v[12], vy=v[1]*x+v[5]*y+v[9]*z+v[13], vz=v[2]*x+v[6]*y+v[10]*z+v[14], vw=v[3]*x+v[7]*y+v[11]*z+v[15];
  const cx=p[0]*vx+p[4]*vy+p[8]*vz+p[12]*vw, cy=p[1]*vx+p[5]*vy+p[9]*vz+p[13]*vw, cw=p[3]*vx+p[7]*vy+p[11]*vz+p[15]*vw;
  if (cw<=0) return null;
  return [ (cx/cw*0.5+0.5)*innerWidth, (1-(cy/cw*0.5+0.5))*innerHeight ];
}
function stateName(){ return STATES[animing?toState:curState]; }
function updateTip(mx,my){
  let best=-1, bd=14;
  for (let i=0;i<N;i++){
    const p=pts[i];
    if (!visible[p.source_idx]) continue;
    if (sharedOnly && !p.shared) continue;
    const sc=projPoint(p.positions[stateName()]); if(!sc) continue;
    const d=Math.hypot(sc[0]-mx, sc[1]-my); if(d<bd){bd=d;best=i;}
  }
  if (best<0){ tip.style.display='none'; return; }
  const p=pts[best];
  tip.innerHTML = `<div class="t">${esc(p.title)}</div><div class="m">${esc(p.source)} · ${p.shared?'shared concept':'unique'} · ${p.size} tokens</div>`;
  tip.style.display='block'; tip.style.left=(mx+14)+'px'; tip.style.top=(my+14)+'px';
}
const esc=s=>String(s).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));

// ---------- UI ----------
document.getElementById('subtitle').textContent =
  `${N} clusters from ${models.length} models · embedded in ${DATA.meta.embed_model}`;
const statesEl = document.getElementById('states');
const LABELS = { native:'Native clouds', semantic:'Semantic space', by_model:'By model', by_concept:'By concept' };
STATES.forEach((s,i)=>{ const b=document.createElement('button'); b.textContent=LABELS[s]||s; b.dataset.i=i;
  if(i===curState) b.classList.add('active'); b.onclick=()=>goto(i); statesEl.appendChild(b); });
function setActive(i){ [...statesEl.children].forEach(b=> b.classList.toggle('active', +b.dataset.i===i)); }

const legend = document.getElementById('legend');
models.forEach((m,i)=>{ const c=DATA.colors[m].map(v=>Math.round(v*255));
  const row=document.createElement('div'); row.className='legend-row';
  const n=pts.filter(p=>p.source_idx===i).length;
  row.innerHTML=`<span class="swatch" style="background:rgb(${c[0]},${c[1]},${c[2]})"></span><span>${esc(m)}</span><span class="n">${n}</span>`;
  row.onclick=()=>{ visible[i]=visible[i]?0:1; row.classList.toggle('off', !visible[i]); };
  legend.appendChild(row); });

document.getElementById('sharedOnly').onchange = e=> sharedOnly = e.target.checked?1:0;

document.getElementById('s-shared').textContent = DATA.stats.n_shared_concepts;
const su = document.getElementById('s-unique');
for (const [m,n] of Object.entries(DATA.stats.n_unique_per_model))
  su.insertAdjacentHTML('beforeend', `<div class="stat"><span>unique · ${esc(m)}</span><b>${n}</b></div>`);
const jac = document.getElementById('s-jac');
jac.innerHTML = '<b style="color:#b9c3d6">concept overlap (Jaccard)</b>';
for (const [k,v] of Object.entries(DATA.stats.jaccard))
  jac.insertAdjacentHTML('beforeend', `<div class="stat"><span>${esc(k)}</span><b>${v}</b></div>`);
</script>
</body>
</html>
"""


def write_viewer(out_path: Path, comparison: dict) -> None:
    html = _TEMPLATE.replace("/*__NEBULAI_DATA__*/", json.dumps(comparison))
    out_path.write_text(html)
