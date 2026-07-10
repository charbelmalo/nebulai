# Nebul.AI — 25 real interp features (build spine)

Goal: 25 SceneDrivers in `viewer/`, each a *hyper-visual* view of a **real
computed quantity** from a micro model — no placeholder data, no fake motion, no
misleading encodings. Every feature shares the store/camera/picking spine and
exposes exact hover numbers. This doc is the durable plan the `/loop` iterates
against and the source for the `/guide` route.

## Architecture decision (locked)

The Phase-1 export (`nebulai.json`) is *only* the token-embedding UMAP map. It
has no attention/activations/logits/SAE. The Python env has **no torch /
transformer_lens / sae_lens**. So:

> Compute every quantity from the model's **actual weights** in pure numpy,
> offline, and ship the results as static **interp bundles** the viewer loads.
> `src/nebulai/backend/interp/gpt2_numpy.py` is a real GPT-2 forward pass
> (validated: correct next-token predictions, causal attention, logit-lens
> identity). Weight-only analyses (SVD, DFT, write directions) need no forward
> pass. SAE features load a pretrained SAE's weights (safetensors) and encode in
> numpy. The one non-GPT-2 feature (Grokking Clock) trains a tiny modular-add
> transformer with numpy SGD.

Result: everything is real, cheap to precompute, and Netlify-static. Provenance
(`model`, `weight_key`, prompt set, seeds, exact formula) is stamped into every
bundle — the honesty guardrail made literal.

### Data flow
```
model.safetensors ──(numpy)──> interp/*.py ──> out/<model>/interp/<feature>.json
                                                   │
viewer: data/interp.ts loader ──> <Feature>Driver (SceneDriver) ──> #scene-canvas
```
Bundles are keyed to `nebulai.json` point ids where a feature overlays the atlas
(SAE aurora, write-direction field, direction compass), else self-contained
(logit lens, attention graph). New CLI: `nebulai interp --model <id> --features ...`.

### Viewer wiring (per feature)
- New `ViewMode` value + `NavPill`/menu entry (see `store.ts`, `TopBar.tsx`).
- New driver in `viewer/src/scene/drivers/` implementing `SceneDriver`.
- Any knob → `SettingsPage.tsx` + `appearance.<feature>` slice in `store.ts`
  (project rule: no bespoke inputs, reuse `controls.tsx`).
- Hover writes exact values through the shared `Tooltip`.

## The 25 — honest data-source classification

Status: ⬜ todo · 🟨 data-ready (bundle computes) · 🟩 rendered+3 passes done.
Existing drivers: Atlas/Chord/Hierarchy/Compare already ship (adapt, don't rebuild).

| # | Feature | Real quantity | Source | Bundle | Status |
|--|--|--|--|--|--|
| 9 | Concept Atlas | W_E rows → UMAP | existing export | nebulai.json | 🟩 (AtlasDriver) |
| 10 | Hierarchy Dendrogram | HDBSCAN condensed tree | existing | nebulai.json | 🟨 verify tree is real |
| 11 | Compare Morph | cross-model cluster embed | existing | compare.json | 🟩 (CompareDriver) |
| 21 | Weight Spectrum | SVD σ of weight matrices | weight-only | weights.json | 🟨 SVD verified |
| 1 | Fourier Atlas | DFT of W_pe (+attn) | weight/forward | fourier.json | 🟨 DFT verified |
| 6 | Neuron Write-Direction Field | mlp.c_proj rows (W_out) | weight-only | neurons.json | ⬜ |
| 15 | Embedding Constellation | W_E rows, ortho/semantic | weight-only | nebulai.json | ⬜ |
| 12 | Cosine-Similarity Web | cosine-kNN in raw W_E | weight-only | knn_raw.json | ⬜ |
| 22 | Direction Compass | concept dirs in W_E | weight-only | directions.json | ⬜ |
| 3 | Logit-Lens Tunnel | ln_f∘unembed per layer | forward | trace_<p>.json | 🟨 lens verified |
| 7 | Attention-Head Flow Graph | post-softmax attention | forward | trace_<p>.json | 🟨 attn verified |
| 8 | Residual-Stream Ribbon | resid[L] trajectory | forward | trace_<p>.json | 🟨 |
| 18 | Probability Simplex | next-token softmax | forward | trace_<p>.json | 🟨 |
| 23 | Attention-Rollout Waterfall | cumulative attn rollout | forward | trace_<p>.json | ⬜ |
| 19 | Semantic Vignette | rollout/occlusion importance | forward | trace_<p>.json | ⬜ |
| 2 | Attribution Ink | occlusion Δlogit / grad·input | forward | attrib_<p>.json | ⬜ |
| 4 | Causal-Trace Heatmap | activation patching Δ | forward×patch | patch_<p>.json | ⬜ |
| 17 | Ablation Ghosts | neuron/head ablation Δlogit | forward×ablate | ablate_<p>.json | ⬜ |
| 14 | Tuned-Lens Delta | regression translator vs lens | forward+fit | tuned.json | ✅ shipped as #20 (caveat stated) |
| 5 | SAE Firing Aurora | SAE encode of resid | SAE weights | sae_<p>.json | ⬜ (download SAE) |
| 20 | Feature Piano-Roll | SAE feature × position | SAE+forward | sae_<p>.json | ⬜ |
| 13 | Superposition Prism | SAE decoder geometry | SAE weights | sae_geom.json | ⬜ |
| 24 | Polysemantic Venn | SAE feature co-firing | SAE+corpus | sae_cofire.json | ⬜ |
| 16 | Grokking Clock | Fourier features of trained toy | numpy train | grok.json | ⬜ (trains tiny model) |
| 25 | Live Prompt Nebula | live forward on typed text | forward (local server) | live_server.py | ✅ capstone — probe-server, NOT a JS port (0.5 GB weights stay local) |

> **Status source of truth:** `viewer/src/scene/interp/registry.ts` (rendered at
> `/guide`). All **25 of 25** are live as of 2026-07-10; the per-row boxes above
> are the original roadmap and the doc's numbering drifted from the shipped
> `#n` ids — trust the registry.

Honesty caveats to surface in `/guide` and in-view:
- **Tuned lens** here is a least-squares affine translator, not the full trained
  lens — label it as such.
- **Attribution**: occlusion (leave-one-token-out Δlogit) is exact and needs no
  autograd; if grad·input is added, note the numpy backward pass.
- **Grokking Clock** uses a *separately trained* toy transformer — it is NOT
  GPT-2; the view must say so (GPT-2 has no clean modular-arithmetic circuit).
- **No causal claims** beyond what patching/ablation actually measures.

## Build order (loop milestones)
1. ✅ Keystone: numpy GPT-2 forward + hooks, validated.
2. Interp export layer: `interp/bundles.py` + `nebulai interp` CLI; ship
   weights.json (Weight Spectrum) + fourier.json + trace bundle for a curated
   prompt set. Add `viewer/src/data/interp.ts` loader.
3. Driver scaffolding: register new ViewModes + a driver base; port one
   weight-only feature end-to-end (Weight Spectrum) through all 3 review passes.
4. Weight-only batch (6, 12, 15, 22) → forward batch (3,7,8,18,23,19,2,4,17,14).
5. SAE batch (5,20,13,24) after downloading a gpt2-small SAE.
6. ✅ Grokking Clock (numpy training) + Live Prompt Nebula (local probe server —
   the JS-port idea was dropped: 0.5 GB float32 weights don't belong in a tab).
7. ✅ `/guide` route (math + source per feature). Netlify deploy still open —
   the live feature degrades to its honest offline banner on a static host.

## Review-pass checklist (every feature, ≥3 passes)
- **P1 numerical**: tensor shapes asserted; units/normalization correct; formula
  matches the cited definition; edge cases (T=1, all-noise, empty).
- **P2 truthfulness**: colormap perceptually honest (no rainbow on ordered data);
  no hidden smoothing; decorative motion ≠ data; legend states the quantity+unit.
- **P3 perf/interaction**: GPU mem bounded; 60fps target; hover picking exact;
  WebGPU→WebGL fallback; reduced-motion + mobile.
