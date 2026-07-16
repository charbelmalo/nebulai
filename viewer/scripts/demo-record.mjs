/**
 * Automated browse-plan recorder for the Nebul.AI viewer.
 *
 * Drives the REAL app (no mockups): gpt2 Atlas hero → Internals gallery →
 * Live Prompt Nebula, where each keystroke runs a genuine GPT-2 forward pass
 * against the local probe server (127.0.0.1:8123). Playwright records the page
 * to WebM; a sibling ffmpeg step transmuxes to an optimized, social-ready MP4.
 *
 * "Cursor tracking" = an injected DOM cursor + click-ripple (recording chrome,
 * not a claimed app feature) that follows Playwright's synthetic mouse, so the
 * pointer path is visible in the capture.
 *
 * Run (from viewer/):  node scripts/demo-record.mjs
 * Requires: the vite dev server on :5173 and the live probe server on :8123.
 */
import { chromium } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const BASE = process.env.DEMO_BASE ?? "http://localhost:5173";
const OUT = process.env.DEMO_OUT ?? resolve(process.cwd(), "..", "scratchpad-demo");
const RAW = resolve(OUT, "raw");
mkdirSync(RAW, { recursive: true });

const W = 1920;
const H = 1080;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── on-theme prompts (verified real GPT-2 completions, agentic → coding) ──
const PROMPT_A = "The AI agent completed the assigned"; // → " task"  (0.67)
const PROMPT_B = "The developer opened a new pull"; //      → " request" (0.89)


// injected caption lower-third + brand end card (recording chrome). Captions are
// baked into the capture (this ffmpeg build has no drawtext), centered so they
// survive a 1:1 social crop. Exposes window.__cap / __capHide / __endcard.
const CAPTION_JS = `
(() => {
  if (window.__capReady) return;
  window.__capReady = true;
  const boot = () => {
    // Hover tooltips duplicate our baked captions in a capture, so hide them —
    // nothing model-derived is suppressed, only a redundant readout.
    const st = document.createElement('style');
    st.textContent = '.interp-tooltip{display:none!important}';
    (document.head || document.documentElement).appendChild(st);

    // ── lower-third caption ──
    const wrap = document.createElement('div');
    wrap.id = '__cap';
    Object.assign(wrap.style, {
      position:'fixed', left:'50%', bottom:'8.5%', transform:'translateX(-50%) translateY(8px)',
      maxWidth:'900px', textAlign:'center', pointerEvents:'none', zIndex:'2147483640',
      opacity:'0', transition:'opacity .42s ease, transform .42s ease',
      fontFamily:'Helvetica, Arial, sans-serif',
    });
    const panel = document.createElement('div');
    Object.assign(panel.style, {
      display:'inline-block', padding:'15px 30px 17px', borderRadius:'16px',
      background:'rgba(8,10,14,0.56)', backdropFilter:'blur(7px)', WebkitBackdropFilter:'blur(7px)',
      border:'1px solid rgba(245,195,59,0.20)', boxShadow:'0 10px 40px rgba(0,0,0,0.5)',
    });
    const kick = document.createElement('div');
    Object.assign(kick.style, {
      color:'rgb(245,195,59)', fontSize:'15px', fontWeight:'700', letterSpacing:'0.24em',
      textTransform:'uppercase', marginBottom:'9px', textShadow:'0 1px 8px rgba(0,0,0,0.7)',
    });
    const head = document.createElement('div');
    Object.assign(head.style, {
      color:'#f4f6fb', fontSize:'33px', fontWeight:'600', lineHeight:'1.24',
      textShadow:'0 2px 20px rgba(0,0,0,0.85)',
    });
    panel.append(kick, head); wrap.appendChild(panel); document.body.appendChild(wrap);
    window.__cap = (k, h) => {
      kick.textContent = k || ''; head.textContent = h || '';
      wrap.style.opacity = '1'; wrap.style.transform = 'translateX(-50%) translateY(0)';
    };
    window.__capHide = () => {
      wrap.style.opacity = '0'; wrap.style.transform = 'translateX(-50%) translateY(8px)';
    };

    // ── brand end card ──
    const card = document.createElement('div');
    Object.assign(card.style, {
      position:'fixed', inset:'0', zIndex:'2147483641', pointerEvents:'none',
      display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
      gap:'22px', opacity:'0', transition:'opacity .6s ease',
      background:'radial-gradient(120% 90% at 50% 42%, rgba(20,16,30,0.86) 0%, rgba(6,7,11,0.97) 62%)',
      fontFamily:'Helvetica, Arial, sans-serif', textAlign:'center',
    });
    const mark = document.createElement('div');
    mark.innerHTML = 'nebul<span style="color:rgb(245,195,59)">.ai</span>';
    Object.assign(mark.style, { color:'#f6f8fc', fontSize:'76px', fontWeight:'700', letterSpacing:'-0.01em', textShadow:'0 4px 40px rgba(0,0,0,0.6)' });
    const tag = document.createElement('div');
    tag.textContent = 'Watch a language model think — one real forward pass at a time.';
    Object.assign(tag.style, { color:'#c7ccd8', fontSize:'26px', fontWeight:'400', maxWidth:'880px' });
    const feats = document.createElement('div');
    feats.textContent = 'Semantic token maps · live logit-lens · 25 interpretability views';
    Object.assign(feats.style, { color:'rgb(245,195,59)', fontSize:'16px', fontWeight:'600', letterSpacing:'0.14em', textTransform:'uppercase', opacity:'0.92' });
    card.append(mark, tag, feats); document.body.appendChild(card);
    window.__endcard = () => { window.__capHide(); card.style.opacity = '1'; };
  };
  if (document.body) boot(); else window.addEventListener('DOMContentLoaded', boot);
})();
`;

// injected cursor + click ripple, present from document start
const CURSOR_JS = `
(() => {
  if (window.__demoCursorReady) return;
  window.__demoCursorReady = true;
  const boot = () => {
    const c = document.createElement('div');
    c.id = '__democursor';
    Object.assign(c.style, {
      position:'fixed', left:'0', top:'0', width:'26px', height:'26px',
      marginLeft:'-13px', marginTop:'-13px', borderRadius:'50%',
      border:'2px solid rgba(245,195,59,0.9)',
      boxShadow:'0 0 14px 4px rgba(245,195,59,0.45), inset 0 0 6px rgba(245,195,59,0.35)',
      pointerEvents:'none', zIndex:'2147483647',
      transition:'width .12s ease, height .12s ease, background .12s ease',
      background:'rgba(255,255,255,0.04)',
      willChange:'transform,left,top', mixBlendMode:'screen',
    });
    const dot = document.createElement('div');
    Object.assign(dot.style, {
      position:'absolute', left:'50%', top:'50%', width:'6px', height:'6px',
      marginLeft:'-3px', marginTop:'-3px', borderRadius:'50%',
      background:'#fff', boxShadow:'0 0 8px 2px rgba(255,255,255,0.9)',
    });
    c.appendChild(dot);
    document.body.appendChild(c);
    const move = (x, y) => { c.style.left = x + 'px'; c.style.top = y + 'px'; };
    move(window.innerWidth/2, window.innerHeight/2);
    window.addEventListener('mousemove', (e) => move(e.clientX, e.clientY), true);
    const ripple = (x, y) => {
      const r = document.createElement('div');
      Object.assign(r.style, {
        position:'fixed', left:x+'px', top:y+'px', width:'14px', height:'14px',
        marginLeft:'-7px', marginTop:'-7px', borderRadius:'50%',
        border:'2px solid rgba(234,79,134,0.85)', pointerEvents:'none',
        zIndex:'2147483646', mixBlendMode:'screen',
      });
      document.body.appendChild(r);
      r.animate(
        [ { transform:'scale(1)', opacity:0.9 }, { transform:'scale(6)', opacity:0 } ],
        { duration:600, easing:'cubic-bezier(.2,.7,.3,1)' }
      ).onfinish = () => r.remove();
      c.style.width='18px'; c.style.height='18px'; c.style.background='rgba(245,195,59,0.18)';
      setTimeout(()=>{ c.style.width='26px'; c.style.height='26px'; c.style.background='rgba(255,255,255,0.04)'; },120);
    };
    window.addEventListener('mousedown', (e) => ripple(e.clientX, e.clientY), true);
  };
  if (document.body) boot();
  else window.addEventListener('DOMContentLoaded', boot);
})();
`;

// ── cursor motion state (viewport CSS px) ──
let cur = { x: W / 2, y: H / 2 };
const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

async function glide(page, x, y, ms = 750) {
  const from = { ...cur };
  const steps = Math.max(12, Math.round(ms / 16));
  for (let i = 1; i <= steps; i++) {
    const e = easeInOut(i / steps);
    const px = from.x + (x - from.x) * e;
    const py = from.y + (y - from.y) * e;
    await page.mouse.move(px, py);
    await sleep(ms / steps);
  }
  cur = { x, y };
}

async function drift(page, ms = 1200, r = 22) {
  // gentle idle motion so holds aren't frozen
  const from = { ...cur };
  const steps = Math.round(ms / 16);
  for (let i = 1; i <= steps; i++) {
    const a = (i / steps) * Math.PI * 2;
    await page.mouse.move(from.x + Math.cos(a) * r, from.y + Math.sin(a) * r * 0.6);
    await sleep(16);
  }
  await page.mouse.move(from.x, from.y);
  cur = { ...from };
}

async function clickAt(page, x, y) {
  await glide(page, x, y, 700);
  await sleep(120);
  await page.mouse.down();
  await sleep(70);
  await page.mouse.up();
  await sleep(180);
}

async function centerOf(locator) {
  const b = await locator.boundingBox();
  if (!b) throw new Error("no bounding box for target");
  return { x: b.x + b.width / 2, y: b.y + b.height / 2, box: b };
}

async function humanType(page, text, { thinkAt = -1 } = {}) {
  let i = 0;
  for (const ch of text) {
    await page.keyboard.type(ch);
    let d = 55 + Math.random() * 105; // 55–160ms base
    if (ch === " ") d += 30 + Math.random() * 80; // pause between words
    if (Math.random() < 0.07) d += 160 + Math.random() * 200; // occasional think
    await sleep(d);
    if (i === thinkAt) await sleep(1100); // deliberate mid-prompt pause (partial forward)
    i++;
  }
}

async function smoothScrollRail(page, sel, deltaY, ms = 900) {
  const steps = Math.max(10, Math.round(ms / 16));
  for (let i = 1; i <= steps; i++) {
    await page.evaluate(
      ([s, dy]) => {
        const el = document.querySelector(s);
        if (el) el.scrollTop += dy;
      },
      [sel, deltaY / steps],
    );
    // tiny cursor wobble over the rail while scrolling
    await page.mouse.move(cur.x + (Math.random() * 4 - 2), cur.y + deltaY / steps);
    cur.y = Math.min(H - 40, cur.y + deltaY / steps * 0.15);
    await sleep(ms / steps);
  }
}

async function waitInterpReady(page, ms = 6000) {
  try {
    await page.waitForFunction(
      () => {
        const el = document.querySelector(".interp-status.is-loading");
        return !el; // loading overlay gone
      },
      undefined,
      { timeout: ms, polling: 120 },
    );
  } catch { /* proceed regardless — non-fatal */ }
}

async function clickFeature(page, label) {
  const btn = page.locator(".interp-feature", { hasText: label }).first();
  await btn.scrollIntoViewIfNeeded().catch(() => {});
  const { x, y } = await centerOf(btn);
  await clickAt(page, x, y);
}

// ── main ──
const marks = { beats: {} };
const contextStart = Date.now();

// caption + beat helpers (beat offsets feed the social reframe in demo-encode)
const cap = (page, kicker, head) =>
  page.evaluate(([k, h]) => window.__cap?.(k, h), [kicker, head]);
const capHide = (page) => page.evaluate(() => window.__capHide?.());
const endcard = (page) => page.evaluate(() => window.__endcard?.());
const beat = (label) => {
  if (marks.tourStart) marks.beats[label] = (Date.now() - marks.tourStart) / 1000;
};

const browser = await chromium.launch({
  headless: true,
  channel: "chromium", // full build → real GPU headless (ANGLE Metal)
  args: [
    "--enable-unsafe-webgpu",
    "--enable-gpu",
    "--use-angle=metal",
    "--ignore-gpu-blocklist",
    "--enable-features=Vulkan",
  ],
});

const context = await browser.newContext({
  viewport: { width: W, height: H },
  deviceScaleFactor: 1,
  reducedMotion: "no-preference",
  recordVideo: { dir: RAW, size: { width: W, height: H } },
});
await context.addInitScript(CAPTION_JS);
await context.addInitScript(CURSOR_JS);
const page = await context.newPage();
const video = page.video();
page.on("console", (m) => {
  if (m.text().startsWith("[REC]")) console.log(m.text());
});

try {
  // 1) boot + switch to gpt2 (has the interp export); wait until fully loaded
  await page.goto(`${BASE}/`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => window.__store && window.__perf?.bootMs !== undefined && window.__store.getState().dataset !== null,
    undefined,
    { timeout: 45000 },
  );
  // switch dataset via the settings select (robust: DOM change fallback)
  await page
    .locator("#sel-dataset")
    .selectOption("gpt2")
    .catch(async () => {
      await page.evaluate(() => {
        const s = document.querySelector("#sel-dataset");
        if (s) {
          s.value = "gpt2";
          s.dispatchEvent(new Event("input", { bubbles: true }));
          s.dispatchEvent(new Event("change", { bubbles: true }));
        }
      });
    });
  await page.waitForFunction(
    () => {
      const st = window.__store.getState();
      return st.datasetId === "gpt2" && !st.loading?.active && st.dataset !== null;
    },
    undefined,
    { timeout: 60000 },
  );
  await sleep(1500); // let the atlas fly-in settle

  // chrome-free hero: collapse the settings sidebar so the nebula reads full-bleed
  await page.evaluate(() => {
    const b = document.querySelector('[aria-label="Collapse settings"]');
    if (b) b.click();
  });
  await sleep(1000);

  marks.tourStart = Date.now();
  await page.evaluate(() => console.log("[REC] tour-start"));
  beat("hero");

  // 2) HERO — drift across the semantic map
  await cap(page, "Semantic map", "Every token GPT-2 knows — as a living galaxy");
  await glide(page, W * 0.62, H * 0.42, 1000);
  await drift(page, 1000, 26);
  await glide(page, W * 0.4, H * 0.55, 1000);
  await drift(page, 900, 20);

  // 3) → Internals
  {
    const tab = page.getByRole("button", { name: "Internals", exact: true }).first();
    const { x, y } = await centerOf(tab);
    await clickAt(page, x, y);
    await waitInterpReady(page);
    await sleep(500);
  }

  // 4) Weight Spectrum (default) — brief; it's the darkest view
  beat("spectrum");
  await cap(page, "Internals · weights", "The singular-value spectrum of every layer");
  await glide(page, W * 0.6, H * 0.5, 800);
  await drift(page, 700, 22);

  // 5) Embedding Constellation — a visually rich weights view
  await clickFeature(page, "Embedding Constellation");
  await waitInterpReady(page);
  await sleep(700);
  beat("embedding");
  await cap(page, "Token geometry", "Embeddings cluster by meaning, not spelling");
  await glide(page, W * 0.58, H * 0.52, 800);
  await drift(page, 900, 22);

  // 6) → Live Prompt Nebula (scroll rail to reveal it)
  await glide(page, W * 0.13, H * 0.55, 700);
  await smoothScrollRail(page, ".interp-rail-scroll", 700, 1000);
  await clickFeature(page, "Live Prompt Nebula");
  await waitInterpReady(page);
  await sleep(700);
  beat("liveOpen");
  await cap(page, "Live prompt nebula", "Type a prompt — run a real GPT-2 forward pass, live");

  // clear the generic default so only our on-theme prompt is ever shown
  const input = page.locator('input[aria-label="live prompt"]').first();
  {
    const { x, y } = await centerOf(input);
    await glide(page, x, y, 700);
    await sleep(120);
    await page.mouse.down();
    await sleep(60);
    await page.mouse.up();
    await page.keyboard.press("Control+A");
    await page.keyboard.press("Meta+A");
    await page.keyboard.press("Backspace");
    await sleep(500);
  }

  // 7) TYPE prompt A (agentic) from empty — mid-prompt pause forces a partial
  //    forward so the lens grid visibly builds, then completes → " task"
  beat("typeA");
  await cap(page, "Live · agentic", "“The AI agent completed the assigned …”");
  await humanType(page, PROMPT_A, { thinkAt: PROMPT_A.indexOf("completed") + "completed".length - 1 });
  await sleep(2200); // let the forward compute + render
  // rest on the winning answer bar (top-right) — informative, and clear of the
  // top-left description band that the cell tooltip used to collide with
  beat("resolveA");
  await cap(page, "Next-token readout", "It resolves to “ task” — p ≈ 0.67");
  await glide(page, W * 0.86, H * 0.23, 900);
  await drift(page, 1400, 10);

  // 8) EDIT → prompt B (coding) — whole lens grid recomputes live → " request"
  {
    const { x, y } = await centerOf(input);
    await glide(page, x, y, 700);
    await sleep(100);
    await page.mouse.down();
    await sleep(60);
    await page.mouse.up();
    await page.keyboard.press("Control+A");
    await page.keyboard.press("Meta+A");
    await page.keyboard.press("Backspace");
    await sleep(350);
  }
  beat("typeB");
  await cap(page, "Live · coding", "“The developer opened a new pull …”");
  await humanType(page, PROMPT_B);
  await sleep(2200);
  beat("resolveB");
  await cap(page, "Next-token readout", "…and to “ request” — p ≈ 0.89");
  await glide(page, W * 0.86, H * 0.23, 900);
  await drift(page, 1500, 10);

  // 9) brand end card
  beat("endcard");
  await capHide(page);
  await sleep(300);
  await endcard(page);
  await sleep(2600);
  beat("end");
  marks.tourEnd = Date.now();
} finally {
  await context.close(); // finalizes the WebM
  await browser.close();
}

const webmPath = await video.path();
marks.contextStart = contextStart;
marks.trimSec = (marks.tourStart - contextStart) / 1000;
writeFileSync(resolve(OUT, "marks.json"), JSON.stringify(marks, null, 2));
console.log("WEBM:" + webmPath);
console.log("TRIM_SEC:" + marks.trimSec.toFixed(2));
console.log("TOUR_SEC:" + ((marks.tourEnd - marks.tourStart) / 1000).toFixed(2));
