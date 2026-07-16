"""Interp bundle producers — real quantities → static JSON for the viewer.

Each producer returns a JSON-safe dict with a `meta` block that stamps the exact
provenance (model, formula, params) so a viewer feature can display where its
numbers came from. Nothing here fabricates or smooths data; arrays are rounded
only for transport (documented per bundle) and the rounding precision is stated.

Bundles written by `write_bundles`:
  weights.json    — SVD spectra of every weight matrix           (#21 Weight Spectrum)
  fourier.json    — DFT of positional embeddings W_pe            (#1  Fourier Atlas)
  embed.json      — PCA projection of the token embedding W_E    (#15 Embedding Constellation)
  neurons.json    — PCA of MLP-neuron write directions W_out     (#6  Neuron Write-Direction Field)
  heads.json      — per-head OV/QK circuit stats + behavior      (#2  Head Fingerprints)
  ov_eigs.json    — every head's full complex OV spectrum        (#2b OV Eigenvalue Constellation)
  comp.json       — Q/K/V composition between cross-layer heads  (#2c Composition Web)
  sae_acts.json   — SAE encoder activations on the bundled prompts (#5 Firing Piano-Roll)
  attrib.json     — direct logit attribution of the final margin (#13 Logit Attribution)
  patch.json      — residual-stream activation patching grids    (#14 Causal Patching Map)
  induction.json  — repeated-sequence induction-head diagnostic  (#2d Induction Microscope)
  ablation.json   — per-head ablation Δ-loss on that sequence    (#17 Ablation Ghosts)
  occlusion.json  — leave-one-token-out Δ log-prob per prompt    (#19 Occlusion Vignette)
  trace_<slug>.json — one real forward pass per curated prompt   (#3/7/8/18/23 …)

The trace bundle deliberately does NOT store full logits (T×50257 is ~26 MB);
it stores what the drivers actually render: full attention, per-layer residual
norms, and logit-lens top-k readouts — all real, all small.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from .gpt2_numpy import GPT2Numpy, Trace, _gelu_new, _layernorm, _softmax

# Curated prompts chosen for legible circuits, not cherry-picked outputs:
#  - factual recall (capital / landmark)
#  - the IOI name-mover setup (classic attention-head circuit)
#  - an induction/counting pattern
#  - an antonym mapping
DEFAULT_PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "When Mary and John went to the store, John gave a drink to",
    "1, 2, 3, 4, 5,",
    "The opposite of hot is",
    "The capital of France is",
]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _spectrum(W: np.ndarray) -> dict:
    """Singular-value spectrum + honest rank summaries of one matrix.
    stable_rank = ||W||_F^2 / ||W||_2^2 ; effective_rank = exp(entropy of
    normalized singular values) (Roy & Vetterli 2007)."""
    s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    s = s[s > 0]
    fro2 = float((s * s).sum())
    smax = float(s[0])
    p = s / s.sum()
    eff = float(np.exp(-(p * np.log(p)).sum()))
    keep = min(len(s), 256)
    return {
        "shape": list(W.shape),
        "n_sv": int(len(s)),
        "singular_values": [round(float(v), 5) for v in s[:keep]],
        "sigma_max": round(smax, 5),
        "sigma_min": round(float(s[-1]), 6),
        "stable_rank": round(fro2 / (smax * smax), 3),
        "effective_rank": round(eff, 3),
        "condition": round(smax / float(s[-1]), 2),
    }


def compute_weights(m: GPT2Numpy) -> dict:
    """#21 — SVD spectrum of wte, wpe, and every per-layer attn/mlp matrix."""
    mats: list[dict] = []

    def add(name: str, key: str, kind: str, layer):
        d = _spectrum(m._g(key))
        d.update({"name": name, "kind": kind, "layer": layer})
        mats.append(d)

    add("W_E (wte)", "wte.weight", "embed", None)
    add("W_pos (wpe)", "wpe.weight", "pos", None)
    for L in range(m.n_layer):
        p = f"h.{L}."
        add(f"attn.c_attn L{L}", p + "attn.c_attn.weight", "attn_qkv", L)
        add(f"attn.c_proj L{L}", p + "attn.c_proj.weight", "attn_out", L)
        add(f"mlp.c_fc L{L}", p + "mlp.c_fc.weight", "mlp_in", L)
        add(f"mlp.c_proj L{L}", p + "mlp.c_proj.weight", "mlp_out", L)

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "singular values of raw weight matrices (float64 SVD)",
            "formula": "s = svd(W); stable_rank=||W||_F^2/sigma_max^2; "
            "effective_rank=exp(-sum p_i ln p_i), p_i=s_i/sum s",
            "note": "top 256 singular values stored per matrix",
            "n_layer": m.n_layer,
            "d": m.d,
        },
        "matrices": mats,
    }


def compute_fourier(m: GPT2Numpy) -> dict:
    """#1 (weight mode) — DFT of positional embeddings across positions.
    Real periodic structure: GPT-2's learned W_pe has low-frequency components."""
    Wpe = m.wpe.astype(np.float64)  # (n_ctx, d)
    n_ctx = Wpe.shape[0]
    Wc = Wpe - Wpe.mean(axis=0, keepdims=True)
    F = np.fft.rfft(Wc, axis=0)  # (n_ctx//2+1, d)
    powspec = (np.abs(F) ** 2)  # per-dim power
    power_mean = powspec.mean(axis=1)  # averaged over dims
    freqs = np.arange(powspec.shape[0])  # cycles per full context window
    per_dim_dominant = (1 + np.argmax(powspec[1:], axis=0)).astype(int)  # skip DC
    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "power spectrum of mean-centered W_pe along position axis",
            "formula": "P(f) = mean_d |rfft(W_pe - mean)[f, d]|^2 ; "
            "freq unit = cycles per context window",
            "n_ctx": int(n_ctx),
            "d": m.d,
        },
        "freqs": [int(x) for x in freqs],
        "power_mean": [round(float(x), 4) for x in power_mean],
        "per_dim_dominant": [int(x) for x in per_dim_dominant],
    }


def _pca_rows(rows: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact PCA of a row matrix via the (d×d) covariance eigendecomposition —
    float64, deterministic axis signs (largest-|loading| positive). Returns
    (coords (n,dims) PC scores, explained-variance ratio per PC, total variance).
    Shared by compute_embed / compute_neurons / compute_sae so the three
    constellations are the SAME math on different row sets."""
    R = rows.astype(np.float64)
    Rc = R - R.mean(axis=0)
    cov = Rc.T @ Rc
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    axes = evecs[:, :dims]
    for j in range(dims):
        k = int(np.argmax(np.abs(axes[:, j])))
        if axes[k, j] < 0:
            axes[:, j] = -axes[:, j]
    coords = Rc @ axes
    evr = evals[:dims] / evals.sum()
    return coords, evr, float(evals.sum())


def _unembed_readout(
    rows32: np.ndarray, m: GPT2Numpy, chunk: int = 2048
) -> tuple[list[str], list[float], list[str], list[float]]:
    """Direct-path logit readout of residual-stream write directions through the
    model's own final LN + tied unembedding: ℓ = ((w − mean(w)) ⊙ γ_f)·W_Eᵀ.
    Folds in ln_f's centering and gain; drops only the per-input 1/σ — a
    positive scalar, so token RANKING is preserved. Direct path only (no
    downstream-layer effects), positive activation assumed — callers must state
    that caveat in their meta."""
    g_f = m._g("ln_f.weight")  # (d,)
    v = (rows32 - rows32.mean(axis=1, keepdims=True)) * g_f
    n = v.shape[0]
    top_tok: list[str] = []
    top_val: list[float] = []
    bot_tok: list[str] = []
    bot_val: list[float] = []
    for s in range(0, n, chunk):
        lg = v[s : s + chunk] @ m.wte.T  # (chunk, V) logit deltas
        hi = np.argmax(lg, axis=1)
        lo = np.argmin(lg, axis=1)
        r = np.arange(lg.shape[0])
        top_tok += [m.decode1(int(t)) for t in hi]
        top_val += [round(float(x), 2) for x in lg[r, hi]]
        bot_tok += [m.decode1(int(t)) for t in lo]
        bot_val += [round(float(x), 2) for x in lg[r, lo]]
    return top_tok, top_val, bot_tok, bot_val


def compute_embed(m: GPT2Numpy, dims: int = 3) -> dict:
    """#15 Embedding Constellation — PCA of the token-embedding matrix W_E.

    Projects every token embedding onto the top principal axes of the mean-
    centered embedding matrix. The honest structure this exposes: the leading
    PCs of GPT-2's W_E organize tokens largely by SURFACE FORM (leading space,
    case, digits) rather than meaning — so we ship a real orthographic property
    (leading-space, decoded from the token) as the color and let the geometry
    speak for itself. Positions are exact PC scores; per-token size = exact row
    L2 norm. No smoothing, no synthetic layout — this is the model's own W_E.

    PCA is done via the 768×768 covariance eigendecomposition (cheap and exact),
    not a giant thin-U SVD, so it runs in seconds and stays float64.
    """
    coords, evr, total_var = _pca_rows(m.wte, dims)  # (V, dims) exact PC scores
    norms = np.linalg.norm(m.wte.astype(np.float64), axis=1)  # exact per-token magnitude
    strs = [m.decode1(i) for i in range(m.V)]
    lead = [1 if s[:1] == " " else 0 for s in strs]
    xy = coords[:, :2].reshape(-1)  # flat [pc1_0, pc2_0, pc1_1, pc2_1, …]
    z = coords[:, 2] if dims >= 3 else np.zeros(m.V)
    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "PCA projection of the token embedding matrix W_E",
            "formula": "Wc = W_E - mean_row(W_E); eig(WcᵀWc) → top-k axes V; "
            "coords = Wc·V (exact PC scores). size = ‖W_E[i]‖₂.",
            "note": "color = leading-space (orthographic), decoded per token; "
            "coords rounded to 3 dp for transport",
            "d": m.d,
            "n_tokens": int(m.V),
        },
        "n": int(m.V),
        "dims": dims,
        "explained_variance_ratio": [round(float(x), 5) for x in evr],
        "total_variance": round(total_var, 3),
        "coords": [round(float(v), 3) for v in xy],  # flat 2N (PC1, PC2)
        "z": [round(float(v), 3) for v in z],  # PC3 (hover only)
        "norm": [round(float(v), 3) for v in norms],
        "lead_space": lead,  # 1 if the token string starts with a space
        "strs": strs,
    }


def compute_neurons(m: GPT2Numpy, dims: int = 3) -> dict:
    """#6 Neuron Write-Direction Field — PCA of every MLP neuron's write direction.

    MLP neuron i of layer L contributes h_i · W_out[i] to the residual stream,
    where W_out = mlp.c_proj.weight (HF Conv1D: shape (d_mlp, d), applied as
    h @ W — so ROW i is neuron i's write direction). We stack all
    n_layer × d_mlp rows, mean-center, and project onto the top principal axes
    of the 768×768 covariance (exact PC scores, float64, no layout synthesis).
    Neurons are stored in layer order, so layer = floor(i / d_mlp) — the layer
    array is implicit, not shipped.

    Per neuron we also ship a direct-path logit readout: the token its write
    direction most promotes and most suppresses through the tied unembedding,
        ℓ = ((w − mean(w)) ⊙ γ_f) · W_Eᵀ
    This folds in ln_f's centering (the LN Jacobian projects out the mean) and
    its gain γ_f, and drops only the per-input 1/σ — a positive scalar, so token
    RANKING is preserved. Values are logit deltas per unit of (positive) neuron
    activation via the direct path only (no downstream-layer effects). That
    caveat is stated in meta and in the viewer.
    """
    d, d_mlp = m.d, 4 * m.d
    blocks = []
    for L in range(m.n_layer):
        W = m._g(f"h.{L}.mlp.c_proj.weight")
        assert W.shape == (d_mlp, d), f"c_proj L{L} shape {W.shape} != ({d_mlp},{d})"
        blocks.append(W)
    rows32 = np.concatenate(blocks, axis=0)  # (n_layer*d_mlp, d) float32
    n = rows32.shape[0]

    coords, evr, total_var = _pca_rows(rows32, dims)
    norms = np.linalg.norm(rows32.astype(np.float64), axis=1)  # exact ‖w_out‖₂
    top_tok, top_val, bot_tok, bot_val = _unembed_readout(rows32, m)

    xy = coords[:, :2].reshape(-1)
    z = coords[:, 2] if dims >= 3 else np.zeros(n)
    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "PCA of MLP-neuron write directions (rows of mlp.c_proj) "
            "+ direct-path logit readout per neuron",
            "formula": "rows = stack_L(c_proj_L) (n_layer·d_mlp × d); "
            "Rc = rows − mean_row; eig(RcᵀRc) → top-k axes V; coords = Rc·V. "
            "readout ℓ = ((w − mean(w)) ⊙ γ_f)·W_Eᵀ (drops only the positive "
            "1/σ scalar — rank-preserving); size = ‖w_out‖₂.",
            "note": "layer implicit: layer = floor(i/d_mlp), neurons stored in "
            "layer order. readout = direct path only, assumes positive "
            "activation. coords rounded to 3 dp for transport.",
            "n_layer": m.n_layer,
            "d_mlp": d_mlp,
            "d": m.d,
        },
        "n": int(n),
        "dims": dims,
        "explained_variance_ratio": [round(float(x), 5) for x in evr],
        "total_variance": round(total_var, 3),
        "coords": [round(float(x), 3) for x in xy],  # flat 2n (PC1, PC2)
        "z": [round(float(x), 3) for x in z],  # PC3 (hover only)
        "norm": [round(float(x), 3) for x in norms],
        "top_tok": top_tok,
        "top_val": top_val,
        "bot_tok": bot_tok,
        "bot_val": bot_val,
    }


# The open-source GPT-2-small residual SAEs (Joseph Bloom's "res-jb" release,
# trained on 300M activations, the set Neuronpedia indexes). Layer 8 is the
# canonical mid-late hook where features are most semantic.
SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
SAE_HOOK = "blocks.8.hook_resid_pre"


def compute_sae(m: GPT2Numpy, repo: str = SAE_REPO, hook: str = SAE_HOOK, dims: int = 3) -> dict:
    """#5 SAE Decoder Constellation — PCA of every SAE feature's decoder direction.

    A sparse-autoencoder feature i reconstructs its share of the residual stream
    as a_i · W_dec[i] — ROW i of W_dec (d_sae × d_in in the sae_lens format,
    shape-asserted) is feature i's write direction, exactly analogous to an MLP
    neuron's c_proj row. Same exact-PCA treatment as compute_embed /
    compute_neurons (shared _pca_rows), same direct-path unembedding readout
    (shared _unembed_readout) — so the three constellations are directly
    comparable: W_E rows vs W_out rows vs W_dec rows.

    The release ships each feature's measured log10 firing sparsity (fraction of
    tokens the feature fires on, over the evaluation set) — a REAL activation
    statistic, exported per feature. Features that never fire ("dead") sit at
    the sparsity floor. NOTE the readout caveat is stronger here than for
    neurons: these directions enter at the layer-8 residual and pass through
    blocks 8-11 before the unembedding, so the direct path skips even more of
    the model. Stated in meta and in the viewer.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    cfg = json.loads(open(hf_hub_download(repo, f"{hook}/cfg.json")).read())
    t = load_file(hf_hub_download(repo, f"{hook}/sae_weights.safetensors"))
    sp = load_file(hf_hub_download(repo, f"{hook}/sparsity.safetensors"))

    W_dec = t["W_dec"]
    d_sae, d_in = int(cfg["d_sae"]), int(cfg["d_in"])
    assert W_dec.shape == (d_sae, d_in), f"W_dec shape {W_dec.shape} != ({d_sae},{d_in})"
    assert d_in == m.d, f"SAE d_in {d_in} != model d {m.d}"
    sparsity = sp["sparsity"].astype(np.float64)  # log10 firing fraction
    assert sparsity.shape == (d_sae,), f"sparsity shape {sparsity.shape} != ({d_sae},)"

    coords, evr, total_var = _pca_rows(W_dec, dims)
    norms = np.linalg.norm(W_dec.astype(np.float64), axis=1)  # exact ‖W_dec[i]‖₂
    top_tok, top_val, bot_tok, bot_val = _unembed_readout(W_dec.astype(np.float32), m)

    n = d_sae
    xy = coords[:, :2].reshape(-1)
    z = coords[:, 2] if dims >= 3 else np.zeros(n)
    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "sae_repo": repo,
            "hook_point": hook,
            "quantity": "PCA of SAE decoder directions (rows of W_dec) + measured "
            "log10 firing sparsity + direct-path logit readout per feature",
            "formula": "Rc = W_dec − mean_row; eig(RcᵀRc) → top-k axes V; "
            "coords = Rc·V. readout ℓ = ((w − mean(w)) ⊙ γ_f)·W_Eᵀ "
            "(rank-preserving). sparsity = log10(firing fraction), measured "
            "by the SAE release over its evaluation set.",
            "note": "readout = direct path only — decoder directions enter at "
            f"{hook} and pass through the remaining blocks before the "
            "unembedding, so this skips more of the model than the neuron "
            "readout. coords rounded to 3 dp for transport.",
            "d_sae": d_sae,
            "d_in": d_in,
            "l1_coefficient": cfg.get("l1_coefficient"),
            "training_tokens": cfg.get("training_tokens"),
        },
        "n": int(n),
        "dims": dims,
        "explained_variance_ratio": [round(float(x), 5) for x in evr],
        "total_variance": round(total_var, 3),
        "coords": [round(float(x), 3) for x in xy],  # flat 2n (PC1, PC2)
        "z": [round(float(x), 3) for x in z],  # PC3 (hover only)
        "norm": [round(float(x), 4) for x in norms],
        "log_sparsity": [round(float(x), 3) for x in sparsity],
        "top_tok": top_tok,
        "top_val": top_val,
        "bot_tok": bot_tok,
        "bot_val": bot_val,
    }


def load_sae_weights(
    repo: str = SAE_REPO, hook: str = SAE_HOOK
) -> tuple[dict, dict, np.ndarray]:
    """Fetch the res-jb SAE release for one hook point: (cfg, tensors, sparsity).

    tensors holds W_enc (d_in, d_sae), b_enc, W_dec (d_sae, d_in), b_dec.
    Shared by the offline bundle writer and the live server (which keeps the
    result resident so /live/sae pays the download/mmap once)."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    cfg = json.loads(open(hf_hub_download(repo, f"{hook}/cfg.json")).read())
    t = load_file(hf_hub_download(repo, f"{hook}/sae_weights.safetensors"))
    sp = load_file(hf_hub_download(repo, f"{hook}/sparsity.safetensors"))
    d_sae, d_in = int(cfg["d_sae"]), int(cfg["d_in"])
    assert t["W_enc"].shape == (d_in, d_sae), f"W_enc shape {t['W_enc'].shape}"
    return cfg, t, sp["sparsity"].astype(np.float64)


def sae_trace_for_prompt(
    m: GPT2Numpy,
    prompt: str,
    t: dict,
    sparsity: np.ndarray,
    layer: int,
    top_k: int = 32,
) -> dict:
    """One prompt's SAE piano-roll trace — the loop body of compute_sae_acts,
    extracted so /live/sae produces byte-identical rows for typed prompts."""
    W_enc, b_enc = t["W_enc"], t["b_enc"]
    W_dec, b_dec = t["W_dec"], t["b_dec"]

    tr: Trace = m.forward(prompt)
    x = tr.resid[layer].astype(np.float32)  # (T, d) = blocks.L.hook_resid_pre
    xc = x - x.mean(axis=1, keepdims=True)  # TL center_writing_weights basis
    acts = np.maximum((xc - b_dec) @ W_enc + b_enc, 0.0)  # (T, d_sae)
    recon = acts @ W_dec + b_dec
    cos = (recon * xc).sum(axis=1) / (
        np.linalg.norm(recon, axis=1) * np.linalg.norm(xc, axis=1) + 1e-12
    )
    l0 = (acts > 0).sum(axis=1)

    def rows_for(ids: np.ndarray) -> list[dict]:
        top_tok, top_val, _, _ = _unembed_readout(W_dec[ids].astype(np.float32), m)
        return [
            {
                "id": int(fid),
                "log_sparsity": round(float(sparsity[fid]), 3),
                "top_tok": top_tok[k],
                "top_val": top_val[k],
                "max": round(float(acts[:, fid].max()), 3),
                "acts": [round(float(a), 3) for a in acts[:, fid]],
            }
            for k, fid in enumerate(ids)
        ]

    # Main board: ranked by peak over positions ≥ 1. The first position
    # carries GPT-2's massive-activation outlier (‖x‖ ≈ 3000 vs ~100) and a
    # handful of SAE features fire 60–100× everything else there and ~0
    # elsewhere; ranked raw they'd fill the board with redundant rows.
    # They're real, so they ship too — as a separate labeled band.
    peak = acts[1:].max(axis=0) if acts.shape[0] > 1 else acts.max(axis=0)
    ids = np.argsort(-peak)[:top_k]
    ids = ids[peak[ids] > 0]
    sink_ids = np.argsort(-acts[0])[:4]
    sink_ids = sink_ids[acts[0][sink_ids] > 0]
    sink_ids = sink_ids[~np.isin(sink_ids, ids)]

    return {
        "slug": _slug(prompt),
        "prompt": prompt,
        "token_strs": tr.token_strs,
        "T": len(tr.tokens),
        "l0": [int(v) for v in l0],
        "cos": [round(float(v), 4) for v in cos],
        "features": rows_for(ids),
        "sink_features": rows_for(sink_ids),
    }


def compute_sae_acts(
    m: GPT2Numpy,
    prompts: list[str] | None = None,
    repo: str = SAE_REPO,
    hook: str = SAE_HOOK,
    top_k: int = 32,
) -> dict:
    """#5 SAE Firing Piano-Roll — the res-jb encoder run on real residuals.

    The behavioral counterpart of compute_sae: instead of where a feature's
    decoder direction POINTS, this measures which features FIRE on each token
    of the bundled prompts, with exact activation values.

    Basis correction (exact, not a heuristic): the SAE was trained on
    TransformerLens activations with center_writing_weights, which centers
    every matrix writing to the residual stream (W_E, W_pos, every c_proj).
    LayerNorm is invariant to the mean component, so the model function is
    unchanged, and the TL residual equals our HF-basis residual minus its
    per-position mean: x̄ = x − mean(x). Feeding uncentered x gives garbage
    (L0 ≈ 2700, cos ≈ 0.76); centered gives the published regime
    (L0 ≈ 30–100, cos ≈ 0.93–0.9999) — verified empirically both ways.

    Encoder/decoder are the original mats_sae_training architecture (cfg has
    no apply_b_dec_to_input flag; subtraction confirmed by reconstruction
    quality): acts = ReLU((x̄ − b_dec)·W_enc + b_enc), recon = acts·W_dec + b_dec.

    Per prompt: per-position L0 and reconstruction cosine (the honesty metric —
    how much of the stream the SAE actually explains), plus the top_k features
    by peak activation with their FULL activation rows, each labeled with its
    direct-path top token and the release's measured global log-sparsity."""
    prompts = prompts or DEFAULT_PROMPTS
    cfg, t, sparsity = load_sae_weights(repo, hook)
    d_sae, d_in = int(cfg["d_sae"]), int(cfg["d_in"])
    layer = int(cfg["hook_point_layer"])
    assert d_in == m.d, f"SAE d_in {d_in} != model d {m.d}"

    traces = [sae_trace_for_prompt(m, p, t, sparsity, layer, top_k) for p in prompts]

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "sae_repo": repo,
            "hook_point": hook,
            "quantity": "SAE encoder activations on real forward passes: which "
            "features fire on which tokens, per-position L0, and reconstruction "
            "cosine",
            "formula": "x̄ = x − mean(x) per position (TransformerLens "
            "center_writing_weights basis — LayerNorm-invariant, exact); "
            "acts = ReLU((x̄ − b_dec)·W_enc + b_enc); recon = acts·W_dec + b_dec; "
            "cos = cosine(recon, x̄) per position.",
            "note": f"top {top_k} features per prompt by peak activation over "
            "positions ≥ 1; the first position's massive-activation outlier "
            "(‖x‖≈3000) drives a few features 60–100× everything else, exported "
            "separately as sink_features. Rows rounded to 3 dp. Activations are "
            "for these bundled prompts only — not a global feature ranking. "
            "log_sparsity is the release's measured global firing fraction; "
            "top_tok is the direct-path readout (skips blocks 8–11).",
            "d_sae": d_sae,
            "d_in": d_in,
            "hook_layer": layer,
            "top_k": top_k,
        },
        "traces": traces,
    }


def compute_heads(m: GPT2Numpy, prompts: list[str] | None = None) -> dict:
    """Per-attention-head fingerprints: weight-circuit stats + measured behavior.

    Weight circuits (per head h of layer L, ln_1 gain folded, biases excluded —
    they don't depend on the input):
      OV map  A_ov = diag(γ₁)·W_V^h·W_O^h  (d×d, rank ≤ d_head) — the linear map
        from a (normalized) residual direction the head reads to the direction
        it writes back. Its nonzero eigenvalues equal eig(W_O^h·diag(γ₁)·W_V^h)
        (d_head×d_head — exact, cheap). copying = Σ Re λᵢ / Σ |λᵢ| ∈ [−1, 1]:
        +1 means every direction is written back with positive sign (a copying
        head), negative means the head systematically inverts what it reads.
        (Eigenvalue copying analysis after Elhage et al. 2021, applied in
        residual space rather than the full vocab circuit.)
      QK map  A_qk = diag(γ₁)·W_Q^h·W_K^hᵀ·diag(γ₁)/√d_head — the bilinear form
        that scores key against query directions; σ_max bounds how sharp the
        head's attention logits can get per unit of (normalized) residual.
    Behavior (measured, NOT derived from weights): real forward passes over the
    bundled prompts, unrounded post-softmax attention. Per head, averaged over
    every query position i ≥ 1 of every prompt: attention to the previous token
    a[i,i−1], to the first token a[i,0] (the "sink"), to itself a[i,i], and the
    normalized entropy H(a[i,:i+1])/ln(i+1) ∈ [0,1]."""
    prompts = prompts or DEFAULT_PROMPTS
    H, dh, nL = m.n_head, m.d_head, m.n_layer
    n = nL * H

    copying = np.zeros(n)
    eig1_re = np.zeros(n)
    eig1_im = np.zeros(n)
    fro_ov = np.zeros(n)
    sigma_qk = np.zeros(n)

    for L in range(nL):
        p = f"h.{L}."
        g1 = m._g(p + "ln_1.weight").astype(np.float64)  # (d,)
        W = m._g(p + "attn.c_attn.weight").astype(np.float64)  # (d, 3d), x@W
        Wq, Wk, Wv = np.split(W, 3, axis=1)  # each (d, d)
        Wo = m._g(p + "attn.c_proj.weight").astype(np.float64)  # (d, d), rows=head slices
        for h in range(H):
            i = L * H + h
            s = slice(h * dh, (h + 1) * dh)
            A = g1[:, None] * Wv[:, s]  # (d, dh) — read map, γ₁ folded
            B = Wo[s, :]  # (dh, d) — write map
            # nonzero eig(A·B) = eig(B·A): d_head×d_head, exact
            lam = np.linalg.eigvals(B @ A)
            denom = float(np.abs(lam).sum())
            copying[i] = float(lam.real.sum()) / denom if denom > 0 else 0.0
            k = int(np.argmax(np.abs(lam)))
            eig1_re[i], eig1_im[i] = float(lam[k].real), float(lam[k].imag)
            # ‖A·B‖_F² = tr((AᵀA)(BBᵀ)) — no d×d intermediate needed
            fro_ov[i] = float(np.sqrt(np.trace((A.T @ A) @ (B @ B.T))))
            # σ(Q̃K̃ᵀ)² = nonzero eig((K̃ᵀK̃)(Q̃ᵀQ̃)); real ≥ 0 up to roundoff
            Q = g1[:, None] * Wq[:, s]
            K = g1[:, None] * Wk[:, s]
            ev = np.linalg.eigvals((K.T @ K) @ (Q.T @ Q))
            sigma_qk[i] = float(np.sqrt(max(ev.real.max(), 0.0)) / np.sqrt(dh))

    # measured behavior: fresh unrounded forward passes (trace bundles round
    # attention to 4 dp for transport; these stats come from the raw values)
    prev_s = np.zeros(n)
    sink_s = np.zeros(n)
    self_s = np.zeros(n)
    ent_s = np.zeros(n)
    n_rows = 0
    for prompt in prompts:
        a = m.forward(prompt).attn.astype(np.float64)  # (nL, H, T, T)
        T = a.shape[-1]
        if T < 2:
            continue
        rows = a[:, :, 1:, :]  # query positions i ≥ 1 (row 0 is trivially 1.0)
        idx = np.arange(1, T)
        prev_s += rows[:, :, idx - 1, idx - 1].sum(axis=2).reshape(-1)
        sink_s += rows[:, :, :, 0].sum(axis=2).reshape(-1)
        self_s += rows[:, :, idx - 1, idx].sum(axis=2).reshape(-1)
        plogp = rows * np.log(np.where(rows > 0, rows, 1.0))  # 0·log0 → 0
        ent = -plogp.sum(axis=3) / np.log(idx + 1)[None, None, :]  # ∈ [0,1]
        ent_s += ent.sum(axis=2).reshape(-1)
        n_rows += T - 1

    prev_s /= n_rows
    sink_s /= n_rows
    self_s /= n_rows
    ent_s /= n_rows

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "per-head OV/QK circuit stats (weights, ln_1 gain "
            "folded, biases excluded) + attention behavior measured over the "
            "bundled prompts (unrounded forward passes)",
            "formula": "copying = Σ Re λ / Σ |λ| over eig(W_O^h·diag(γ₁)·W_V^h); "
            "σ_qk = σ_max(diag(γ₁)·W_Q^h·W_K^hᵀ·diag(γ₁))/√d_head; "
            "prev/sink/self = mean over query rows i≥1 of a[i,i−1] / a[i,0] / "
            "a[i,i]; entropy = mean H(a[i,·])/ln(i+1)",
            "note": "behavior is a sample: "
            f"{len(prompts)} prompts, {n_rows} query rows total — stated, not "
            "hidden. copying is the residual-space OV eigenvalue score "
            "(Elhage et al. 2021), not the full vocab circuit.",
            "n_layer": nL,
            "n_head": H,
            "d_head": dh,
            "prompts": prompts,
            "n_rows": n_rows,
        },
        "n": n,
        "copying": [round(float(x), 4) for x in copying],
        "eig1_re": [round(float(x), 3) for x in eig1_re],
        "eig1_im": [round(float(x), 3) for x in eig1_im],
        "fro_ov": [round(float(x), 3) for x in fro_ov],
        "sigma_qk": [round(float(x), 3) for x in sigma_qk],
        "prev": [round(float(x), 4) for x in prev_s],
        "sink": [round(float(x), 4) for x in sink_s],
        "self": [round(float(x), 4) for x in self_s],
        "entropy": [round(float(x), 4) for x in ent_s],
    }


def compute_ov_eigs(m: GPT2Numpy) -> dict:
    """The FULL OV spectra behind compute_heads' copying scores: every complex
    eigenvalue of every head's residual-space OV map, d_head per head.

    A_ov = diag(γ₁)·W_V^h·W_O^h (d×d, rank ≤ d_head); its nonzero eigenvalues
    equal eig(W_O^h·diag(γ₁)·W_V^h) at d_head×d_head — exact (verified against
    the full d×d eigendecomposition when compute_heads was built). The matrix
    is real, so the spectrum is conjugate-symmetric; both halves are exported
    (the symmetry in the plot is a property of the math, not decoration).
    An eigenvalue λ means: along its eigendirection, the head writes back λ×
    what it reads — |λ|>1 amplifies, positive real copies, negative inverts.
    copying (Σ Re λ / Σ |λ|) is re-exported per head so the bundle is
    self-contained."""
    H, dh, nL = m.n_head, m.d_head, m.n_layer
    n = nL * H
    re = np.zeros((n, dh))
    im = np.zeros((n, dh))
    copying = np.zeros(n)
    for L in range(nL):
        p = f"h.{L}."
        g1 = m._g(p + "ln_1.weight").astype(np.float64)
        Wv = np.split(m._g(p + "attn.c_attn.weight").astype(np.float64), 3, axis=1)[2]
        Wo = m._g(p + "attn.c_proj.weight").astype(np.float64)
        for h in range(H):
            i = L * H + h
            s = slice(h * dh, (h + 1) * dh)
            A = g1[:, None] * Wv[:, s]
            B = Wo[s, :]
            lam = np.linalg.eigvals(B @ A)
            lam = lam[np.argsort(-np.abs(lam))]  # descending |λ| within a head
            re[i] = lam.real
            im[i] = lam.imag
            denom = float(np.abs(lam).sum())
            copying[i] = float(lam.real.sum()) / denom if denom > 0 else 0.0

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "all complex eigenvalues of every attention head's "
            "residual-space OV map (d_head per head)",
            "formula": "λ = eig(W_O^h·diag(γ₁)·W_V^h) — the nonzero eigenvalues "
            "of A_ov = diag(γ₁)·W_V^h·W_O^h since eig(AB) = eig(BA). ln_1 gain "
            "folded, biases excluded. copying = Σ Re λ / Σ |λ| per head.",
            "note": "real matrix → conjugate-symmetric spectrum (both halves "
            "exported). λ within each head sorted by descending |λ|. Rounded "
            "to 4 dp for transport.",
            "n_layer": nL,
            "n_head": H,
            "d_head": dh,
        },
        "n": n,
        "d_head": dh,
        "re": [round(float(x), 4) for x in re.reshape(-1)],  # flat n·d_head
        "im": [round(float(x), 4) for x in im.reshape(-1)],
        "copying": [round(float(x), 4) for x in copying],
    }


def compute_comp(m: GPT2Numpy, baseline_n: int = 200, seed: int = 0) -> dict:
    """Q/K/V composition scores between every cross-layer head pair (#2c
    Composition Web) — the weights-only quantity that reveals multi-head
    CIRCUITS (Elhage et al. 2021, A Mathematical Framework).

    With row-vector convention (x @ W) and ln_1 gain folded per layer:
      M_ov^h = diag(γ₁)·W_V^h·W_O^h   (what the head writes given what it reads)
      M_qk^h = diag(γ₁)·W_Q^h·W_K^hᵀ·diag(γ₁)  (how it scores query vs key)
    Head 1 (earlier layer) composes into head 2 (later layer) when head 2
    reads what head 1 wrote:
      Q-comp = ‖M_ov¹·M_qk²‖_F / (‖M_ov¹‖_F·‖M_qk²‖_F)  (h1 feeds h2's query)
      K-comp = ‖M_qk²·M_ov¹ᵀ‖_F / (same norms)           (h1 feeds h2's key)
      V-comp = ‖M_ov¹·M_ov²‖_F / (‖M_ov¹‖_F·‖M_ov²‖_F)   (h1 feeds h2's value)
    All matrices are rank ≤ d_head, so every norm is computed from d_head-sized
    Gram matrices (‖X·M·Y‖_F² = tr(Mᵀ(XᵀX)M(YYᵀ)) — exact, no d×d products).

    Honesty: raw Frobenius composition has a POSITIVE floor for unrelated maps.
    The exported `baseline` is that floor measured over `baseline_n` random
    Gaussian factor pairs of the same shapes (mean ± std, seeded) — scores
    should be read relative to it, and the viewer says so. Same-layer pairs
    are excluded: heads in one layer act in parallel and cannot compose. The
    LayerNorm between the layers is folded as its gain only (the per-input
    1/σ normalizer is not a weight); same convention as heads.json."""
    H, dh, nL, d = m.n_head, m.d_head, m.n_layer, m.d
    rng = np.random.default_rng(seed)

    # per-head factors + Gram matrices (all float64)
    P_ov = np.zeros((nL, H, d, dh))  # diag(γ₁)·W_V^h
    Q_ov = np.zeros((nL, H, dh, d))  # W_O^h
    P_q = np.zeros((nL, H, d, dh))  # diag(γ₁)·W_Q^h
    K_q = np.zeros((nL, H, d, dh))  # diag(γ₁)·W_K^h
    for L in range(nL):
        p = f"h.{L}."
        g1 = m._g(p + "ln_1.weight").astype(np.float64)
        W = m._g(p + "attn.c_attn.weight").astype(np.float64)
        Wq, Wk, Wv = np.split(W, 3, axis=1)
        Wo = m._g(p + "attn.c_proj.weight").astype(np.float64)
        for h in range(H):
            s = slice(h * dh, (h + 1) * dh)
            P_ov[L, h] = g1[:, None] * Wv[:, s]
            Q_ov[L, h] = Wo[s, :]
            P_q[L, h] = g1[:, None] * Wq[:, s]
            K_q[L, h] = g1[:, None] * Wk[:, s]

    def gram(X: np.ndarray) -> np.ndarray:
        return np.einsum("...ij,...ik->...jk", X, X)  # XᵀX per head

    gP_ov = gram(P_ov)  # (nL,H,dh,dh)
    gQ_ov = np.einsum("...ij,...kj->...ik", Q_ov, Q_ov)  # Q·Qᵀ
    gP_q = gram(P_q)
    gK_q = gram(K_q)

    # ‖M‖_F per head: ‖X·Yᵀ‖_F² = tr((XᵀX)(YᵀY)); ‖P·Q‖_F² = tr((PᵀP)(QQᵀ))
    fro_ov = np.sqrt(np.einsum("lhij,lhji->lh", gP_ov, gQ_ov))
    fro_qk = np.sqrt(np.einsum("lhij,lhji->lh", gP_q, gK_q))

    def score(M: np.ndarray, G1: np.ndarray, G2: np.ndarray) -> float:
        """‖X·M·Y‖_F where G1 = XᵀX, G2 = Y·Yᵀ (all d_head×d_head)."""
        return float(np.sqrt(max(0.0, np.trace(M.T @ G1 @ M @ G2))))

    layer_pairs = [(i, j) for i in range(nL) for j in range(i + 1, nL)]
    q_s = np.zeros((len(layer_pairs), H, H))
    k_s = np.zeros((len(layer_pairs), H, H))
    v_s = np.zeros((len(layer_pairs), H, H))
    for pi, (i, j) in enumerate(layer_pairs):
        # cross terms, one batched contraction per type: (H1,dh,d)×(H2,d,dh)
        m_q = np.einsum("aud,bdv->abuv", Q_ov[i], P_q[j])  # Q_ov1·P_q2
        m_kq = np.einsum("aud,bdv->abuv", Q_ov[i], K_q[j])  # Q_ov1·K_q2
        m_v = np.einsum("aud,bdv->abuv", Q_ov[i], P_ov[j])  # Q_ov1·P_ov2
        for h1 in range(H):
            for h2 in range(H):
                nrm_qk = fro_ov[i, h1] * fro_qk[j, h2]
                # Q-comp: M_ov1·M_qk2 = P_ov1·(Q_ov1·P_q2)·K_q2ᵀ
                q_s[pi, h1, h2] = score(m_q[h1, h2], gP_ov[i, h1], gK_q[j, h2]) / nrm_qk
                # K-comp: M_qk2·M_ov1ᵀ = P_q2·(K_q2ᵀ·Q_ov1ᵀ)·P_ov1ᵀ
                k_s[pi, h1, h2] = score(m_kq[h1, h2].T, gP_q[j, h2], gP_ov[i, h1]) / nrm_qk
                # V-comp: M_ov1·M_ov2 = P_ov1·(Q_ov1·P_ov2)·Q_ov2
                v_s[pi, h1, h2] = score(m_v[h1, h2], gP_ov[i, h1], gQ_ov[j, h2]) / (
                    fro_ov[i, h1] * fro_ov[j, h2]
                )

    # measured random-matrix floor: same shapes, iid Gaussian factors
    base = []
    for _ in range(baseline_n):
        A = [rng.standard_normal((d, dh)) for _ in range(4)]
        M = A[0] @ (A[1].T @ A[2]) @ A[3].T
        base.append(
            np.sqrt((M * M).sum())
            / (np.linalg.norm(A[0] @ A[1].T) * np.linalg.norm(A[2] @ A[3].T))
        )
    base = np.array(base)

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "Q/K/V composition scores between every cross-layer "
            "attention-head pair (weights only, ln_1 gain folded)",
            "formula": "Q = ‖M_ov¹·M_qk²‖_F/(‖M_ov¹‖‖M_qk²‖); "
            "K = ‖M_qk²·M_ov¹ᵀ‖_F/(same); V = ‖M_ov¹·M_ov²‖_F/(‖M_ov¹‖‖M_ov²‖) "
            "with M_ov = diag(γ₁)W_V W_O, M_qk = diag(γ₁)W_Q W_Kᵀ diag(γ₁) "
            "(Elhage et al. 2021)",
            "note": "raw Frobenius composition has a positive floor for "
            f"unrelated maps: baseline = {base.mean():.4f} ± {base.std():.4f} "
            f"measured over {baseline_n} random Gaussian factor pairs of the "
            "same shapes (seeded). Read scores relative to it. Same-layer "
            "pairs excluded (parallel heads cannot compose). The inter-layer "
            "LayerNorm's per-input 1/σ is not a weight and is not folded. "
            "Rounded to 4 dp.",
            "n_layer": nL,
            "n_head": H,
            "d_head": dh,
            "baseline_mean": round(float(base.mean()), 4),
            "baseline_std": round(float(base.std()), 4),
            "baseline_n": baseline_n,
        },
        # pair index = pair_of(i,j) in layer_pairs order, then h1·n_head + h2
        "layer_pairs": [[i, j] for i, j in layer_pairs],
        "q": [round(float(x), 4) for x in q_s.reshape(-1)],
        "k": [round(float(x), 4) for x in k_s.reshape(-1)],
        "v": [round(float(x), 4) for x in v_s.reshape(-1)],
    }


def compute_logit_attrib(m: GPT2Numpy, prompts: list[str] | None = None) -> dict:
    """Direct logit attribution: which components wrote the final prediction?

    The final residual at the last position decomposes exactly into everything
    ever added to the stream:
      x = emb + Σ_L ( Σ_h head_out_{L,h} + b_o^L + mlp_out_L )
    Per-head attention outputs and per-layer MLP outputs are recomputed from
    the same weights and (unrounded) attention patterns the forward pass used;
    the attention out-projection bias b_o belongs to no head, so it is exported
    as its own per-layer bucket rather than smeared across heads.

    Each component v is projected into the margin between the model's top-1
    and runner-up next-token logits through the final LayerNorm with its
    normalizer FROZEN at the actual value from this forward pass (standard DLA
    linearization — the one thing not attributed is how a component shifts the
    normalizer itself):
      contrib(v) = ((v − mean(v)) ⊙ γ_f) · (W_U[c1] − W_U[c2]) / σ(x)
    Frozen σ makes the decomposition exact and additive: the exported pieces
    sum to the true margin up to float32 forward accumulation, which is
    measured and exported per trace (`sum_check` next to `margin`) — never
    hidden. The ln_f bias β contributes β·(W_U[c1]−W_U[c2]) once (`lnf_bias`).

    Per head, the argmax-attended token at the final query row (and its
    weight) is exported for hover context: it is what the head read, not a
    causal claim about why it wrote what it wrote."""
    prompts = prompts or DEFAULT_PROMPTS
    nL, H, dh, d = m.n_layer, m.n_head, m.d_head, m.d
    g_f = m._g("ln_f.weight").astype(np.float64)
    b_f = m._g("ln_f.bias").astype(np.float64)

    traces = []
    for prompt in prompts:
        tr = m.forward(prompt)
        T = len(tr.tokens)
        tau = T - 1
        lg = tr.logits[tau].astype(np.float64)  # ground truth for this prompt
        order = np.argsort(-lg)
        c1, c2 = int(order[0]), int(order[1])
        pr = np.exp(lg - lg.max())
        pr /= pr.sum()
        margin = float(lg[c1] - lg[c2])
        dirv = (m.wte[c1] - m.wte[c2]).astype(np.float64)  # tied unembedding

        x_final = tr.resid[nL, tau].astype(np.float64)
        sigma = float(np.sqrt(((x_final - x_final.mean()) ** 2).mean() + 1e-5))

        def contrib(v: np.ndarray) -> float:
            return float(((v - v.mean()) * g_f) @ dirv / sigma)

        emb_v = (m.wte[tr.tokens[tau]] + m.wpe[tau]).astype(np.float64)
        heads = np.zeros((nL, H))
        mlps = np.zeros(nL)
        biases = np.zeros(nL)
        attend_tok: list[str] = []
        attend_w: list[float] = []
        recon = emb_v.copy()  # rebuilt stream — checked against resid[-1]

        for L in range(nL):
            p = f"h.{L}."
            x = tr.resid[L].astype(np.float64)  # (T, d) block input
            ln1, _ = _layernorm(x, m._g(p + "ln_1.weight").astype(np.float64),
                                m._g(p + "ln_1.bias").astype(np.float64))
            W = m._g(p + "attn.c_attn.weight").astype(np.float64)
            bqkv = m._g(p + "attn.c_attn.bias").astype(np.float64)
            v_all = ln1 @ W[:, 2 * d :] + bqkv[2 * d :]  # (T, d) value vectors
            Wo = m._g(p + "attn.c_proj.weight").astype(np.float64)
            bo = m._g(p + "attn.c_proj.bias").astype(np.float64)
            a = tr.attn[L].astype(np.float64)  # (H, T, T) unrounded
            attn_out_tau = bo.copy()
            for h in range(H):
                s = slice(h * dh, (h + 1) * dh)
                ho = (a[h, tau] @ v_all[:, s]) @ Wo[s, :]  # (d,) head write
                heads[L, h] = contrib(ho)
                attn_out_tau += ho
                recon += ho
                j = int(np.argmax(a[h, tau]))
                attend_tok.append(tr.token_strs[j])
                attend_w.append(float(a[h, tau, j]))
            biases[L] = contrib(bo)
            recon += bo
            # MLP is position-wise: only the final row is needed
            x_mid = x[tau] + attn_out_tau
            ln2, _ = _layernorm(x_mid[None, :], m._g(p + "ln_2.weight").astype(np.float64),
                                m._g(p + "ln_2.bias").astype(np.float64))
            hid = _gelu_new(ln2[0] @ m._g(p + "mlp.c_fc.weight").astype(np.float64)
                            + m._g(p + "mlp.c_fc.bias").astype(np.float64))
            mo = hid.astype(np.float64) @ m._g(p + "mlp.c_proj.weight").astype(np.float64) \
                + m._g(p + "mlp.c_proj.bias").astype(np.float64)
            mlps[L] = contrib(mo)
            recon += mo

        # honesty checks, exported: stream reconstruction + margin additivity
        recon_rel = float(np.linalg.norm(recon - x_final) / np.linalg.norm(x_final))
        emb_c = contrib(emb_v)
        lnf_bias = float(b_f @ dirv)
        total = emb_c + lnf_bias + float(heads.sum() + mlps.sum() + biases.sum())

        traces.append({
            "slug": _slug(prompt),
            "prompt": prompt,
            "token_strs": tr.token_strs,
            "T": T,
            "top1": [m.decode1(c1), round(float(lg[c1]), 4), round(float(pr[c1]), 4)],
            "top2": [m.decode1(c2), round(float(lg[c2]), 4), round(float(pr[c2]), 4)],
            "margin": round(margin, 4),
            "sum_check": round(total, 4),
            "recon_rel": round(recon_rel, 6),
            "emb": round(emb_c, 4),
            "lnf_bias": round(lnf_bias, 4),
            "heads": [round(float(v), 4) for v in heads.reshape(-1)],  # layer-major
            "mlp": [round(float(v), 4) for v in mlps],
            "bias": [round(float(v), 4) for v in biases],
            "attend_tok": attend_tok,  # flat n_layer·n_head, layer-major
            "attend_w": [round(float(v), 3) for v in attend_w],
        })

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "direct logit attribution: every component's exact "
            "contribution to the top-1 vs runner-up next-token logit margin "
            "at the final position",
            "formula": "x_final = emb + Σ_L(Σ_h head_out + b_o + mlp_out); "
            "contrib(v) = ((v − mean(v)) ⊙ γ_f)·(W_U[c1] − W_U[c2]) / σ(x_final) "
            "with σ frozen at the forward pass's actual final-LN normalizer",
            "note": "frozen-σ linearization (standard DLA): contributions are "
            "exact and additive given this forward's normalizer; what is NOT "
            "attributed is each component's effect on σ itself. sum_check vs "
            "margin exposes the float32 accumulation error per trace. "
            "attend_tok/attend_w = argmax attention at the final query row — "
            "what the head read, not a causal claim. Rounded to 4 dp.",
            "n_layer": nL,
            "n_head": H,
        },
        "traces": traces,
    }


# Matched clean/corrupt prompt pairs for activation patching, with designated
# single-token answers (the published convention: the metric contrasts the two
# candidate answers, whether or not either is the run's argmax — each answer's
# rank in its own run is exported so nothing is overstated). Each pair must
# tokenize to the same length so residual rows align position-for-position.
PATCH_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "When Mary and John went to the store, John gave a drink to",
        "When Mary and John went to the store, Mary gave a drink to",
        " Mary",
        " John",
    ),
    ("The capital of France is", "The capital of Italy is", " Paris", " Rome"),
    ("The opposite of hot is", "The opposite of cold is", " cold", " hot"),
]


def _forward_from(m: GPT2Numpy, x: np.ndarray, layer_start: int) -> np.ndarray:
    """Resume the forward pass from a (possibly patched) residual state.

    `x` is the residual stream entering block `layer_start` (i.e. resid[i]
    from a Trace); runs blocks layer_start..n_layer-1 then final LN + tied
    unembedding, replicating GPT2Numpy.forward's float32 op order exactly so
    an unpatched resume reproduces the original logits bit-for-bit. Returns
    final-position logits (V,)."""
    T, d, H, dh = x.shape[0], m.d, m.n_head, m.d_head
    cmask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
    for L in range(layer_start, m.n_layer):
        p = f"h.{L}."
        xn, _ = _layernorm(x, m._g(p + "ln_1.weight"), m._g(p + "ln_1.bias"))
        qkv = xn @ m._g(p + "attn.c_attn.weight") + m._g(p + "attn.c_attn.bias")
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(T, H, dh).transpose(1, 0, 2)
        k = k.reshape(T, H, dh).transpose(1, 0, 2)
        v = v.reshape(T, H, dh).transpose(1, 0, 2)
        a = _softmax((q @ k.transpose(0, 2, 1)) / np.sqrt(dh) + cmask, axis=-1)
        ctx = (a @ v).transpose(1, 0, 2).reshape(T, d)
        x = x + (ctx @ m._g(p + "attn.c_proj.weight") + m._g(p + "attn.c_proj.bias"))
        xn2, _ = _layernorm(x, m._g(p + "ln_2.weight"), m._g(p + "ln_2.bias"))
        h = _gelu_new(xn2 @ m._g(p + "mlp.c_fc.weight") + m._g(p + "mlp.c_fc.bias"))
        x = x + (h @ m._g(p + "mlp.c_proj.weight") + m._g(p + "mlp.c_proj.bias"))
    xf, _ = _layernorm(x, m._g("ln_f.weight"), m._g("ln_f.bias"))
    return xf[-1] @ m.wte.T


def compute_patching(m: GPT2Numpy, pairs: list[tuple[str, str]] | None = None) -> dict:
    """Activation patching (causal tracing) over the residual stream.

    For each matched (clean, corrupt) prompt pair: run both forwards, then for
    every (layer i, position p) copy the CLEAN residual row resid[i][p] into
    the corrupt forward's state and resume the corrupt forward from block i.
    The effect is measured as the logit difference between the clean answer
    and the corrupt answer at the final position, normalized:

        r = (LD_patched − LD_corrupt) / (LD_clean − LD_corrupt)

    r = 0 means the patch did nothing; r = 1 means that single residual row
    carries everything needed to flip the corrupt run to the clean answer.
    Unlike attribution, this IS an intervention — r is a causal quantity
    (with the usual caveat that single-site patches can miss redundancy).

    Row i = the residual stream entering block i (i = 0 is token+position
    embeddings; i = n_layer is the final residual, where only the last
    position can still matter — kept, honestly showing exactly that). Raw
    patched LDs are exported alongside r so nothing hides in normalization."""
    pairs = pairs or PATCH_PAIRS
    nL = m.n_layer
    out_pairs = []
    for clean, corrupt, ans_c_str, ans_r_str in pairs:
        tr_c = m.forward(clean)
        tr_r = m.forward(corrupt)
        T = len(tr_c.tokens)
        if len(tr_r.tokens) != T:
            raise ValueError(f"pair tokenizes to different lengths: {clean!r} / {corrupt!r}")
        tau = T - 1
        (a_c,) = m.encode(ans_c_str)  # designated answers must be single tokens
        (a_r,) = m.encode(ans_r_str)
        if a_c == a_r:
            raise ValueError(f"pair has identical answers ({ans_c_str!r}): not a contrast")

        def ld(logits_row: np.ndarray) -> float:
            return float(logits_row[a_c]) - float(logits_row[a_r])

        ld_clean = ld(tr_c.logits[tau])
        ld_corrupt = ld(tr_r.logits[tau])
        denom = ld_clean - ld_corrupt

        grid_ld = np.zeros((nL + 1, T))
        for i in range(nL + 1):
            for p_pos in range(T):
                x = tr_r.resid[i].copy()
                x[p_pos] = tr_c.resid[i, p_pos]
                grid_ld[i, p_pos] = ld(_forward_from(m, x, i))

        pr_c = np.exp(tr_c.logits[tau] - tr_c.logits[tau].max())
        pr_c /= pr_c.sum()
        pr_r = np.exp(tr_r.logits[tau] - tr_r.logits[tau].max())
        pr_r /= pr_r.sum()
        rank_c = int(np.sum(tr_c.logits[tau] > tr_c.logits[tau, a_c])) + 1
        rank_r = int(np.sum(tr_r.logits[tau] > tr_r.logits[tau, a_r])) + 1
        grid_r = (grid_ld - ld_corrupt) / denom
        out_pairs.append({
            "slug": _slug(clean),
            "clean": clean,
            "corrupt": corrupt,
            "clean_strs": tr_c.token_strs,
            "corrupt_strs": tr_r.token_strs,
            "T": T,
            "diff_pos": [p for p in range(T) if tr_c.tokens[p] != tr_r.tokens[p]],
            # [str, logit in own run, p in own run, rank in own run]
            "ans_clean": [ans_c_str, round(float(tr_c.logits[tau, a_c]), 4),
                          round(float(pr_c[a_c]), 4), rank_c],
            "ans_corrupt": [ans_r_str, round(float(tr_r.logits[tau, a_r]), 4),
                            round(float(pr_r[a_r]), 4), rank_r],
            "ld_clean": round(ld_clean, 4),
            "ld_corrupt": round(ld_corrupt, 4),
            "ld": [round(float(v), 4) for v in grid_ld.reshape(-1)],  # row-major (layer, pos)
            "r": [round(float(v), 4) for v in grid_r.reshape(-1)],
        })

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "residual-stream activation patching: normalized recovery "
            "of the clean-vs-corrupt answer logit difference when one clean "
            "residual row is patched into the corrupt forward",
            "formula": "r[i,p] = (LD(patch resid_i[p]←clean, resume corrupt fwd) "
            "− LD_corrupt) / (LD_clean − LD_corrupt); LD = logit(ans_clean) − "
            "logit(ans_corrupt) at the final position",
            "note": "a true intervention (causal), not attribution: each cell is "
            "one full patched forward resume in the model's own float32. Row i "
            "is the residual entering block i (row 0 = embeddings; the last row "
            "is the final residual, where only the last position can matter). "
            "Answers are designated single-token contrasts (published "
            "convention), not necessarily the run's argmax — each answer's "
            "rank in its own run is exported. Single-site patches can "
            "understate redundant circuits; r can exceed [0,1] and is shown "
            "as computed. Raw patched LDs exported alongside. Rounded to 4 dp.",
            "n_layer": nL,
        },
        "pairs": out_pairs,
    }


def compute_induction(
    m: GPT2Numpy, period: int = 48, seed: int = 0, n_patterns: int = 8
) -> dict:
    """The induction-head diagnostic (Olsson et al. 2022) on a repeated
    random-token sequence — measured behavior, one real forward per seed.

    Sequence: <|endoftext|> (BOS/attention-sink, published convention) followed
    by `period` uniform-random tokens repeated twice. On the second repeat every
    token has occurred exactly once before, so a head implementing induction
    ("[A][B] … [A] → attend to [B]") must attend from position t to t−period+1.
    Three structural targets are scored per head, averaged over the second
    repeat only (first repeat has no previous occurrence to find):

        induction[l,h] = mean_t attn[l,h][t, t−period+1]
        duplicate[l,h] = mean_t attn[l,h][t, t−period]   (previous occurrence)
        prev[l,h]      = mean_t attn[l,h][t, t−1]        (previous token)

    Chance floor = what a uniform-attention head would score on any single
    target: mean_t 1/(t+1). A second seed is run in full and both score sets
    are exported, so score stability is data, not a promise. Full T×T patterns
    (seed A) ship only for the top-scoring heads — 144 patterns would be ~8 MB.

    Random tokens are drawn uniformly from the vocab excluding <|endoftext|>
    (mostly rare tokens — that is the published convention; the test isolates
    the copying mechanism from semantics)."""
    eot = m.V - 1  # 50256 = <|endoftext|>, GPT-2's only special token
    assert period * 2 + 1 <= m.n_ctx, "sequence must fit the context window"

    def run(sd: int) -> Trace:
        rng = np.random.default_rng(sd)
        ids = rng.integers(0, eot, size=period).tolist()  # high excl. → no eot
        return m.forward([eot] + ids + ids)

    tr_a, tr_b = run(seed), run(seed + 1)
    P = period
    T = len(tr_a.tokens)
    ts = np.arange(P + 1, 2 * P + 1)  # positions of the second repeat
    for tr in (tr_a, tr_b):  # the whole test rides on the sequence repeating
        assert all(tr.tokens[t] == tr.tokens[t - P] for t in ts)
    rowsum_drift = float(np.abs(tr_a.attn.sum(axis=-1) - 1.0).max())
    assert rowsum_drift < 1e-4, "attention rows must be a softmax"
    floor = float((1.0 / (ts + 1.0)).mean())

    def scores(tr: Trace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = tr.attn  # (nL, H, T, T); paired index arrays pick (t, target) cells
        ind = a[:, :, ts, ts - P + 1].mean(axis=-1)
        dup = a[:, :, ts, ts - P].mean(axis=-1)
        prev = a[:, :, ts, ts - 1].mean(axis=-1)
        return ind, dup, prev

    ind_a, dup_a, prev_a = scores(tr_a)
    ind_b, dup_b, prev_b = scores(tr_b)

    # patterns for the strongest heads only: top-4 induction ∪ top-2 duplicate
    # ∪ top-2 prev (dedup, order kept) — the heads the grid will invite you to
    # inspect. Anything else states "pattern not exported" on click.
    picks: list[tuple[int, int]] = []
    for arr, k in ((ind_a, 4), (dup_a, 2), (prev_a, 2)):
        for flat in np.argsort(arr.reshape(-1))[::-1][:k]:
            lh = (int(flat) // m.n_head, int(flat) % m.n_head)
            if lh not in picks:
                picks.append(lh)
    picks = picks[:n_patterns]

    def flat4(a: np.ndarray) -> list[float]:
        return [round(float(v), 4) for v in a.reshape(-1)]

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "per-head induction / duplicate-token / previous-token "
            "attention scores on a repeated random-token sequence, plus full "
            "attention patterns for the top-scoring heads",
            "formula": "induction[l,h] = mean over second-repeat positions t of "
            "attn[l,h][t, t−period+1]; duplicate: target t−period; prev: target "
            "t−1. Chance floor = mean_t 1/(t+1) (uniform attention).",
            "note": "measured behavior — one real forward per seed over "
            "<|endoftext|> + 48 uniform-random tokens repeated twice (published "
            "convention: random tokens isolate copying from semantics). Scores "
            "averaged over the second repeat only. Both seeds exported in full; "
            "patterns (seed A only) exported for the top heads by score — 144 "
            "full patterns would be ~8 MB. Rounded to 4 dp.",
            "n_layer": m.n_layer,
            "n_head": m.n_head,
            "period": P,
            "T": T,
            "seed_a": seed,
            "seed_b": seed + 1,
            "floor": round(floor, 4),
            "attn_rowsum_drift": rowsum_drift,
        },
        "token_strs": tr_a.token_strs,
        "ind": flat4(ind_a),
        "dup": flat4(dup_a),
        "prev": flat4(prev_a),
        "ind_b": flat4(ind_b),
        "dup_b": flat4(dup_b),
        "prev_b": flat4(prev_b),
        "patterns": [
            {
                "layer": layer,
                "head": head,
                "attn": flat4(tr_a.attn[layer, head]),  # row-major (from, to)
            }
            for layer, head in picks
        ],
    }


def _forward_ablate(
    m: GPT2Numpy, ids: list[int], sites: list[tuple[int, int]], mode: str
) -> np.ndarray:
    """Full forward pass with the given attention heads ablated; returns logits (T, V).

    Replicates GPT2Numpy.forward's float32 op order exactly, so an empty `sites`
    list reproduces Trace.logits bit-for-bit (the caller asserts this). At each
    ablated (layer, head) the head's context slice — its rows of a@v, i.e. the
    head's contribution entering c_proj — is zeroed ("zero") or replaced by its
    mean over this run's positions ("mean"). c_proj's bias belongs to no head
    and is never touched."""
    if mode not in ("zero", "mean"):
        raise ValueError(f"unknown ablation mode {mode!r}")
    T, d, H, dh = len(ids), m.d, m.n_head, m.d_head
    by_layer: dict[int, list[int]] = {}
    for lyr, hd in sites:
        by_layer.setdefault(lyr, []).append(hd)
    x = m.wte[ids] + m.wpe[:T]
    cmask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
    for L in range(m.n_layer):
        p = f"h.{L}."
        xn, _ = _layernorm(x, m._g(p + "ln_1.weight"), m._g(p + "ln_1.bias"))
        qkv = xn @ m._g(p + "attn.c_attn.weight") + m._g(p + "attn.c_attn.bias")
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(T, H, dh).transpose(1, 0, 2)
        k = k.reshape(T, H, dh).transpose(1, 0, 2)
        v = v.reshape(T, H, dh).transpose(1, 0, 2)
        a = _softmax((q @ k.transpose(0, 2, 1)) / np.sqrt(dh) + cmask, axis=-1)
        ctx = (a @ v).transpose(1, 0, 2).reshape(T, d)
        for hd in by_layer.get(L, ()):
            sl = slice(hd * dh, (hd + 1) * dh)
            if mode == "zero":
                ctx[:, sl] = 0.0
            else:
                ctx[:, sl] = ctx[:, sl].mean(axis=0, keepdims=True)
        x = x + (ctx @ m._g(p + "attn.c_proj.weight") + m._g(p + "attn.c_proj.bias"))
        xn2, _ = _layernorm(x, m._g(p + "ln_2.weight"), m._g(p + "ln_2.bias"))
        h = _gelu_new(xn2 @ m._g(p + "mlp.c_fc.weight") + m._g(p + "mlp.c_fc.bias"))
        x = x + (h @ m._g(p + "mlp.c_proj.weight") + m._g(p + "mlp.c_proj.bias"))
    xf, _ = _layernorm(x, m._g("ln_f.weight"), m._g("ln_f.bias"))
    # cast exactly as Trace.logits does, so the no-ablation path is bit-for-bit
    return (xf @ m.wte.T).astype(np.float32)


def _nll_curve(logits: np.ndarray, ids: list[int]) -> np.ndarray:
    """Per-position next-token NLL in nats: out[j−1] = −log p(ids[j] | ids[:j]),
    j = 1..T−1, via a numerically stable log-softmax."""
    rows = logits[:-1].astype(np.float64)  # (T−1, V)
    mx = rows.max(axis=-1, keepdims=True)
    lse = (mx[:, 0] + np.log(np.exp(rows - mx).sum(axis=-1)))
    tgt = np.asarray(ids[1:])
    return lse - rows[np.arange(len(tgt)), tgt]


def compute_ablation(m: GPT2Numpy, period: int = 48, seed: int = 0) -> dict:
    """Per-head causal ablation on the SAME repeated random-token sequence as
    compute_induction (same seed → identical tokens): does knocking a head out
    actually hurt in-context copying?

    Metric: mean next-token NLL (nats) over the induction window — predicted
    token indices j in [period+2, 2·period], i.e. the second-repeat tokens whose
    current context token has occurred before, so the induction rule
    "[A][B] … [A] → predict [B]" is available. (j = period+1's context token is
    the unique last token of the first repeat — no earlier occurrence, so it is
    excluded.) Δ[l,h] = window mean after ablating (l,h) − baseline.

    Two ablation modes, both exported — they disagree and that disagreement is
    data: "zero" deletes the head's output (off-distribution but standard);
    "mean" replaces it with its per-position mean over this run (keeps the
    head's average signal, removes its position-dependence). Full per-position
    NLL curves ship for every head and mode (~27k floats — small, unlike #2d's
    T×T patterns), so any head's "ghost" curve can be drawn against baseline.

    Combos (top-2 / top-4 induction-scoring heads ablated together, heads
    picked from THIS run's measured scores) expose redundancy: single-head
    ablation understates circuits with backup heads."""
    eot = m.V - 1
    rng = np.random.default_rng(seed)
    ids = [eot] + (lst := rng.integers(0, eot, size=period).tolist()) + lst
    P, T = period, len(ids)
    nL, H = m.n_layer, m.n_head

    tr = m.forward(ids)
    ident = float(np.abs(_forward_ablate(m, ids, [], "zero") - tr.logits).max())
    assert ident < 1e-5, "no-ablation forward must reproduce the baseline logits"

    base = _nll_curve(tr.logits, ids)  # (T−1,) indexed by predicted j−1
    win = slice(P + 1, 2 * P)  # curve indices for predicted j in [P+2, 2P]
    base_win = float(base[win].mean())
    first = float(base[1:P].mean())  # predicted j in [2, P]: inside first repeat
    assert base_win < first, "second-repeat loss must drop (in-context learning)"

    # this run's induction scores (same formula as compute_induction) — hover
    # context tying Δ-loss back to the behavioral score, from the same forward
    ts = np.arange(P + 1, 2 * P + 1)
    ind = tr.attn[:, :, ts, ts - P + 1].mean(axis=-1)  # (nL, H)

    def run(sites: list[tuple[int, int]], mode: str) -> tuple[float, np.ndarray]:
        nll = _nll_curve(_forward_ablate(m, ids, sites, mode), ids)
        return float(nll[win].mean() - base_win), nll

    d = {"zero": np.zeros((nL, H)), "mean": np.zeros((nL, H))}
    curves = {"zero": np.zeros((nL * H, T - 1)), "mean": np.zeros((nL * H, T - 1))}
    for L in range(nL):
        for hd in range(H):
            for mode in ("zero", "mean"):
                d[mode][L, hd], curves[mode][L * H + hd] = run([(L, hd)], mode)

    top = np.argsort(ind.reshape(-1))[::-1]
    combo_sites = [
        [(int(f) // H, int(f) % H) for f in top[:2]],
        [(int(f) // H, int(f) % H) for f in top[:4]],
    ]
    combos = []
    for sites in combo_sites:
        entry: dict = {
            "label": "+".join(f"L{lyr}H{hd}" for lyr, hd in sites),
            "sites": [[lyr, hd] for lyr, hd in sites],
        }
        for mode in ("zero", "mean"):
            dv, nll = run(sites, mode)
            entry[f"d_{mode}"] = round(dv, 4)
            entry[f"nll_{mode}"] = [round(float(v), 4) for v in nll]
        combos.append(entry)

    def flat4(a: np.ndarray) -> list[float]:
        return [round(float(v), 4) for v in a.reshape(-1)]

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "per-head causal ablation on the repeated random-token "
            "sequence: change in mean next-token NLL over the induction window, "
            "plus full per-position loss curves for every head and mode",
            "formula": "Δ[l,h] = mean_{j=P+2..2P} NLL_abl(j) − NLL_base(j); "
            "NLL(j) = −log p(s_j | s_<j) in nats; ablation replaces head (l,h)'s "
            "a@v slice before c_proj with 0 (zero) or its per-run positional "
            "mean (mean)",
            "note": "a true intervention: one full ablated forward per (head, "
            "mode) in the model's own float32 (the unablated path reproduces "
            "the baseline logits — drift exported). Window = second-repeat "
            "predictions whose context token has occurred before, so induction "
            "is available; j=P+1 excluded (its context token is unique). Zero-"
            "ablation is off-distribution; mean-ablation keeps the head's "
            "average signal — both shown, their disagreement is data. Single-"
            "head ablation understates redundant circuits (see combos). "
            "Rounded to 4 dp.",
            "n_layer": nL,
            "n_head": H,
            "period": P,
            "T": T,
            "seed": seed,
            "window": [P + 2, 2 * P],
            "base_window": round(base_win, 4),
            "base_first": round(first, 4),
            "ident_drift": ident,
            "n_forward": 1 + 2 * nL * H + 2 * len(combo_sites),
        },
        "token_strs": tr.token_strs,
        "ind": flat4(ind),
        "d_zero": flat4(d["zero"]),
        "d_mean": flat4(d["mean"]),
        "nll_base": [round(float(v), 4) for v in base],
        "nll_zero": flat4(curves["zero"]),  # row-major (l·H+h, predicted j−1)
        "nll_mean": flat4(curves["mean"]),
        "combos": combos,
    }


def _logsoftmax(row: np.ndarray) -> np.ndarray:
    r = row.astype(np.float64)
    mx = r.max()
    return r - (mx + np.log(np.exp(r - mx).sum()))


def compute_occlusion(m: GPT2Numpy, prompts: list[str] | None = None) -> dict:
    """Leave-one-token-out occlusion importance — exact input-level causal
    attribution, no gradients, one real forward per occlusion.

    For each prompt: the baseline forward fixes the model's own top-1 next
    token c at the final position. Then every position p is occluded two ways
    — both real, and they answer different questions:

      sub — token p replaced by <|endoftext|> (positions preserved; the model
            sees a sink token where the word was)
      del — token p deleted (every later token shifts one position left, so
            positional embeddings move too; disclosed)

    drop[p] = log p_base(c) − log p_occluded(c), in nats: positive = the token
    was supporting the prediction; negative = it was suppressing it. The new
    top-1 under each occlusion is exported, so a *flipped* prediction is
    visible exactly (e.g. IOI: delete the second name and watch the flip).

    Causality check baked in: deleting the FINAL token must reproduce the
    baseline's second-to-last-position logits (causal attention cannot look
    ahead); the max drift across prompts is asserted small and exported."""
    prompts = prompts or DEFAULT_PROMPTS
    eot = m.V - 1
    out = []
    causal_drift = 0.0
    for prompt in prompts:
        ids = m.encode(prompt)
        tr = m.forward(ids)
        T = len(ids)
        tau = T - 1
        base_row = tr.logits[tau]
        c = int(np.argmax(base_row))
        base_lp = _logsoftmax(base_row)
        pr = float(np.exp(base_lp[c]))

        # no-op check: "occluding" a position with its own token is the
        # baseline forward — must be bit-identical (deterministic numpy)
        same = list(ids)
        same[0] = ids[0]
        assert np.array_equal(m.forward(same).logits, tr.logits)

        modes: dict[str, dict[str, list]] = {}
        for mode in ("sub", "del"):
            drop_lp: list[float] = []
            drop_logit: list[float] = []
            new_top: list[list] = []
            for p in range(T):
                if mode == "sub":
                    occ = list(ids)
                    occ[p] = eot
                else:
                    occ = ids[:p] + ids[p + 1 :]
                    if not occ:  # single-token prompt fully deleted
                        drop_lp.append(0.0)
                        drop_logit.append(0.0)
                        new_top.append(["", 0.0])
                        continue
                row = m.forward(occ).logits[-1]
                if mode == "del" and p == tau:
                    # causal attention: dropping the last token must leave the
                    # earlier positions' logits (numerically) unchanged
                    causal_drift = max(causal_drift, float(np.abs(row - tr.logits[tau - 1]).max()))
                lp = _logsoftmax(row)
                drop_lp.append(round(float(base_lp[c] - lp[c]), 4))
                drop_logit.append(round(float(base_row[c]) - float(row[c]), 4))
                nt = int(np.argmax(row))
                new_top.append([m.decode1(nt), round(float(np.exp(lp[nt])), 4)])
            modes[mode] = {"drop_lp": drop_lp, "drop_logit": drop_logit, "new_top": new_top}

        out.append({
            "slug": _slug(prompt),
            "prompt": prompt,
            "token_strs": tr.token_strs,
            "T": T,
            "top1": [m.decode1(c), round(pr, 4), round(float(base_row[c]), 4)],
            "sub": modes["sub"],
            "del": modes["del"],
        })
    assert causal_drift < 1e-3, "deleting the final token must not change earlier logits"

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "leave-one-token-out occlusion importance: change in the "
            "baseline top-1 next token's log-probability when each prompt "
            "position is substituted with <|endoftext|> or deleted",
            "formula": "drop[p] = log p_base(c) − log p_occluded(c) at the final "
            "position, c = the BASELINE forward's own argmax next token; "
            "drop_logit is the same difference in raw logits",
            "note": "exact intervention — one real forward per (position, mode), "
            "no gradients, no approximation. sub keeps every position in place; "
            "del shifts later tokens left so positional embeddings move (both "
            "shown — they answer different questions). Occluding the final "
            "position changes the very token the model predicts from — kept, "
            "honestly showing exactly that. drop > 0: the token supported the "
            "prediction; drop < 0: it was suppressing it. new_top = the "
            "occluded run's own top-1 [str, p]. Rounded to 4 dp.",
            "n_forward": sum(1 + 2 * len(m.encode(p)) + 1 for p in prompts),
            "causal_drift": causal_drift,
        },
        "prompts": out,
    }


def compute_sae_web(
    repo: str = SAE_REPO,
    hook: str = SAE_HOOK,
    top_pairs: int = 60,
    baseline_pairs: int = 200_000,
    seed: int = 0,
) -> dict:
    """#12 Decoder Cosine Web — nearest-neighbor cosine structure of W_dec.

    For every SAE feature i, the maximum cosine similarity between its decoder
    direction and any OTHER feature's direction (exact, all 24576² pairs
    scanned in blocks), plus the global top pairs. High nearest-neighbor
    cosine is direct evidence of FEATURE SPLITTING — the SAE learning several
    near-duplicate directions for one underlying direction — a real
    superposition phenomenon, measured, not illustrated.

    Honesty: the scale for "high" is a MEASURED random-pair baseline (seeded
    sample of pairs, mean/std/p99/p99.9/max exported) — not an eyeballed
    threshold. "Mutual" is exact: i and j are each other's nearest neighbor.
    Rows are normalized to exact unit length before the scan, so these are
    true cosines even where ‖W_dec[i]‖ deviates from 1 (release range
    0.9998–1.0013)."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    cfg = json.loads(open(hf_hub_download(repo, f"{hook}/cfg.json")).read())
    t = load_file(hf_hub_download(repo, f"{hook}/sae_weights.safetensors"))
    W = t["W_dec"].astype(np.float64)
    d_sae, d_in = int(cfg["d_sae"]), int(cfg["d_in"])
    assert W.shape == (d_sae, d_in), f"W_dec shape {W.shape} != ({d_sae},{d_in})"
    U = (W / np.linalg.norm(W, axis=1, keepdims=True)).astype(np.float32)

    nn_idx = np.zeros(d_sae, dtype=np.int64)
    nn_cos = np.zeros(d_sae)
    cand: list[tuple[float, int, int]] = []  # global top-pair candidates (i<j)
    B = 2048
    for r0 in range(0, d_sae, B):
        r1 = min(r0 + B, d_sae)
        G = U[r0:r1] @ U.T  # (block, d_sae) float32
        rows = np.arange(r0, r1)
        G[np.arange(r1 - r0), rows] = -2.0  # mask self
        idx = np.argmax(G, axis=1)
        nn_idx[r0:r1] = idx
        nn_cos[r0:r1] = G[np.arange(r1 - r0), idx].astype(np.float64)
        # global-pair candidates from the strict upper triangle (dedupe i<j)
        G[np.arange(d_sae)[None, :] <= rows[:, None]] = -2.0
        flat = np.argpartition(G.reshape(-1), -4 * top_pairs)[-4 * top_pairs :]
        for f in flat:
            i_loc, j = divmod(int(f), d_sae)
            v = float(G[i_loc, j])
            if v > -2.0:
                cand.append((v, r0 + i_loc, j))
    cand.sort(reverse=True)

    # float32-matmul precision check: recompute a seeded sample of nn cosines
    # as exact float64 dots — the export is only honest if they agree
    rng = np.random.default_rng(seed)
    Wu64 = W / np.linalg.norm(W, axis=1, keepdims=True)
    sample = rng.integers(0, d_sae, size=64)
    worst = max(
        abs(float(Wu64[i] @ Wu64[nn_idx[i]]) - nn_cos[i]) for i in map(int, sample)
    )
    assert worst < 1e-5, f"float32 cosine drift {worst:.2e} ≥ 1e-5"

    # measured random-pair baseline — the yardstick for "unusually close"
    a = rng.integers(0, d_sae, size=baseline_pairs)
    b = rng.integers(0, d_sae, size=baseline_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    base = np.einsum("ij,ij->i", U[a].astype(np.float64), U[b].astype(np.float64))
    mutual = nn_idx[nn_idx] == np.arange(d_sae)

    pairs = [
        {
            "i": i,
            "j": j,
            "cos": round(v, 4),
            "mutual": bool(nn_idx[i] == j and nn_idx[j] == i),
        }
        for v, i, j in cand[:top_pairs]
    ]
    return {
        "meta": {
            "model": "gpt2",
            "created": _now(),
            "sae_repo": repo,
            "hook_point": hook,
            "quantity": "nearest-neighbor cosine between SAE decoder directions "
            "(rows of W_dec, unit-normalized) — every one of the "
            f"{d_sae}² ordered pairs scanned; evidence of feature splitting",
            "formula": "nn_cos[i] = max_{j≠i} cos(W_dec[i], W_dec[j]); mutual "
            "⇔ nn(nn(i)) = i. baseline = cosine over a seeded random sample "
            "of distinct pairs.",
            "note": "float32 matmul, verified against exact float64 dots on a "
            "seeded sample (worst |Δ| asserted < 1e-5). nn_cos rounded to 4 dp. "
            "No layout, no projection: both plot axes are computed quantities "
            "(release-measured firing sparsity × nn cosine).",
            "d_sae": d_sae,
            "d_in": d_in,
            "mutual_count": int(mutual.sum()),
            "baseline": {
                "n_pairs": int(a.size),
                "seed": seed,
                "mean": round(float(base.mean()), 4),
                "std": round(float(base.std()), 4),
                "p99": round(float(np.percentile(base, 99)), 4),
                "p999": round(float(np.percentile(base, 99.9)), 4),
                "max": round(float(base.max()), 4),
            },
        },
        "nn_idx": [int(v) for v in nn_idx],
        "nn_cos": [round(float(v), 4) for v in nn_cos],
        "mutual": [int(v) for v in mutual],
        "pairs": pairs,
    }


def compute_compass(
    m: GPT2Numpy,
    repo: str = SAE_REPO,
    hook: str = SAE_HOOK,
    baseline_n: int = 2000,
    top_exemplars: int = 8,
    seed: int = 0,
) -> dict:
    """#22 Direction Compass — where SAE feature directions point, measured
    against the model's own two big families of residual-stream directions.

    For every SAE decoder direction (unit-normalized row of W_dec, the same
    rows #5/#12 use), the exact maximum cosine to
      (a) every MLP neuron write direction — all n_layer·d_mlp rows of
          mlp.c_proj, the same rows #6 uses — and
      (b) every token embedding (row of W_E, the same rows #15 uses).
    Both scans are exhaustive (24576×36864 and 24576×50257 cosines), float32
    blocked matmuls verified against exact float64 dots on a seeded sample.

    Honesty:
    - The yardstick is a MEASURED baseline: `baseline_n` seeded random unit
      directions in the same d-space are scanned against each family the same
      way; the distribution of THEIR max-cos (mean/p99/max) is exported. In
      768-d a random direction still finds ~0.16 max-cos in a 36k family —
      raw cosines without this baseline would overstate alignment.
    - Cosines are SIGNED maxima: anti-aligned directions are not surfaced
      (stated in meta) — this asks "which direction writes most similarly",
      not "which is most correlated in magnitude".
    - Causal structure disclosed: the SAE reads blocks.8.hook_resid_pre = the
      residual BEFORE block 8, the sum of writes from layers 0–7 (+ emb). A best
      match in layers 8–11 is geometric similarity only — that neuron writes
      AFTER the hook and cannot be the feature's upstream source. Per-feature
      best-match layer is exported so the viewer can show the split.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    cfg = json.loads(open(hf_hub_download(repo, f"{hook}/cfg.json")).read())
    t = load_file(hf_hub_download(repo, f"{hook}/sae_weights.safetensors"))
    W = t["W_dec"].astype(np.float64)
    d_sae, d_in = int(cfg["d_sae"]), int(cfg["d_in"])
    assert W.shape == (d_sae, d_in), f"W_dec shape {W.shape} != ({d_sae},{d_in})"
    assert d_in == m.d, f"SAE d_in {d_in} != model d {m.d}"
    U = (W / np.linalg.norm(W, axis=1, keepdims=True)).astype(np.float32)

    d, d_mlp = m.d, 4 * m.d
    blocks = []
    for L in range(m.n_layer):
        Wp = m._g(f"h.{L}.mlp.c_proj.weight")
        assert Wp.shape == (d_mlp, d), f"c_proj L{L} shape {Wp.shape}"
        blocks.append(Wp)
    Nrows = np.concatenate(blocks, axis=0).astype(np.float64)  # (n_layer·d_mlp, d)
    n_neur = Nrows.shape[0]
    nn_norm = np.linalg.norm(Nrows, axis=1, keepdims=True)
    assert float(nn_norm.min()) > 1e-6, "degenerate zero-norm c_proj row"
    Nu = (Nrows / nn_norm).astype(np.float32)

    E = m.wte.astype(np.float64)  # (V, d)
    n_tok = E.shape[0]
    e_norm = np.linalg.norm(E, axis=1, keepdims=True)
    assert float(e_norm.min()) > 1e-6, "degenerate zero-norm embedding row"
    Eu = (E / e_norm).astype(np.float32)

    def max_cos(Q: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """exhaustive signed max cosine of each unit row of Q against family F"""
        n = Q.shape[0]
        best = np.zeros(n)
        idx = np.zeros(n, dtype=np.int64)
        B = 2048
        for r0 in range(0, n, B):
            r1 = min(r0 + B, n)
            G = Q[r0:r1] @ F.T
            j = np.argmax(G, axis=1)
            idx[r0:r1] = j
            best[r0:r1] = G[np.arange(r1 - r0), j].astype(np.float64)
        return best, idx

    nc, ni = max_cos(U, Nu)  # vs neuron write directions
    tc, ti = max_cos(U, Eu)  # vs token embeddings

    # float32-matmul precision check against exact float64 dots
    rng = np.random.default_rng(seed)
    Wu64 = W / np.linalg.norm(W, axis=1, keepdims=True)
    Nu64 = Nrows / nn_norm
    Eu64 = E / e_norm
    sample = rng.integers(0, d_sae, size=64)
    worst = max(
        max(abs(float(Wu64[i] @ Nu64[ni[i]]) - nc[i]) for i in map(int, sample)),
        max(abs(float(Wu64[i] @ Eu64[ti[i]]) - tc[i]) for i in map(int, sample)),
    )
    assert worst < 1e-5, f"float32 cosine drift {worst:.2e} ≥ 1e-5"

    # measured baseline: what max-cos does a RANDOM direction get per family?
    R = rng.standard_normal((baseline_n, d))
    Ru = (R / np.linalg.norm(R, axis=1, keepdims=True)).astype(np.float32)
    bn, _ = max_cos(Ru, Nu)
    bt, _ = max_cos(Ru, Eu)

    def base_stats(b: np.ndarray) -> dict:
        return {
            "mean": round(float(b.mean()), 4),
            "p99": round(float(np.percentile(b, 99)), 4),
            "max": round(float(b.max()), 4),
        }

    layer = (ni // d_mlp).astype(np.int64)
    layer_counts = np.bincount(layer, minlength=m.n_layer)
    upstream = int(layer_counts[:8].sum())  # layers 0–7 write before the hook

    # unique-string table for best-match tokens (hover text without a
    # client-side tokenizer); indices into it per feature
    tok_strs: list[str] = []
    tok_map: dict[int, int] = {}
    ti_u = np.zeros(d_sae, dtype=np.int64)
    for i in range(d_sae):
        tid = int(ti[i])
        if tid not in tok_map:
            tok_map[tid] = len(tok_strs)
            tok_strs.append(m.decode1(tid))
        ti_u[i] = tok_map[tid]

    # chip exemplars: strongest alignments, deduped by the MATCHED PARTNER —
    # eight top features can share one neuron (they do: several features tie
    # at ~0.97 on a single L2 neuron), and repeating it teaches nothing
    ex: list[dict] = []
    part_n: set[int] = set()
    for i in map(int, np.argsort(nc)[::-1]):
        if int(ni[i]) not in part_n:
            part_n.add(int(ni[i]))
            ex.append({"f": i, "kind": "neuron", "cos": round(float(nc[i]), 4)})
            if len(part_n) >= top_exemplars:
                break
    part_t: set[int] = set()
    for i in map(int, np.argsort(tc)[::-1]):
        if int(ti[i]) not in part_t:
            part_t.add(int(ti[i]))
            ex.append({"f": i, "kind": "token", "cos": round(float(tc[i]), 4)})
            if len(part_t) >= top_exemplars:
                break

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "sae_repo": repo,
            "hook_point": hook,
            "quantity": "per SAE feature: exact max cosine of its decoder "
            "direction against ALL MLP-neuron write directions and ALL token "
            "embeddings (both families unit-normalized, exhaustive scan)",
            "formula": "nc[i] = max_j cos(W_dec[i], c_proj_row[j]); "
            "tc[i] = max_v cos(W_dec[i], W_E[v]). baseline = the same scan "
            "for seeded random unit directions in the same d-space.",
            "note": "signed maxima — anti-aligned partners not surfaced. "
            "float32 matmul verified vs float64 on a seeded sample "
            "(worst |Δ| asserted < 1e-5). cosines rounded to 4 dp. Neuron "
            f"layers 8–11 write AFTER {hook}: a best match there is "
            "geometric similarity only, not an upstream source.",
            "d_sae": d_sae,
            "d": d,
            "n_neurons": int(n_neur),
            "n_tokens": int(n_tok),
            "d_mlp": d_mlp,
            "upstream_frac": round(upstream / d_sae, 4),
            "baseline": {
                "n_dirs": baseline_n,
                "seed": seed,
                "neuron": base_stats(bn),
                "token": base_stats(bt),
            },
        },
        "nc": [round(float(v), 4) for v in nc],
        "ni": [int(v) for v in ni],  # flat neuron idx: layer = i // d_mlp
        "tc": [round(float(v), 4) for v in tc],
        "ti": [int(v) for v in ti],  # token id
        "ti_u": [int(v) for v in ti_u],  # index into tok_strs
        "tok_strs": tok_strs,
        "layer_counts": [int(v) for v in layer_counts],
        "exemplars": ex,
    }


# Corpus for co-firing statistics: one complete public-domain book, stored in
# the repo and sha256-stamped into the bundle so every count is reproducible.
COFIRE_CORPUS = Path(__file__).parent / "corpus_alice.txt"


def compute_cofire(
    m: GPT2Numpy,
    repo: str = SAE_REPO,
    hook: str = SAE_HOOK,
    window: int = 128,
    min_count: int = 20,
    max_pairs: int = 20000,
    top_chips: int = 10,
    shuffle_sample: int = 200,
    recon_every: int = 4,
    seed: int = 0,
) -> dict:
    """#24 Co-Firing Venn — which SAE features fire TOGETHER, counted on a corpus.

    #5/#12/#22 are about where decoder directions POINT; this is about how the
    features BEHAVE: the res-jb encoder is run over every token of a real,
    disclosed corpus (Alice's Adventures in Wonderland, Project Gutenberg #11,
    public domain, sha256 in meta), and for every pair of features we count the
    exact number of positions where both fire (acts > 0). Joined with each
    pair's decoder cosine, this asks the polysemanticity question directly: do
    geometrically-similar features fire on the same tokens (feature splitting)
    or do they exclude each other (winner-take-all)?

    Exact quantities, no estimation:
      n_i  = #positions feature i fires        (integer)
      c_ij = #positions BOTH fire               (integer, sparse XᵀX)
      e_ij = n_i·n_j/N   expected under independence with these marginals
      lift = c_ij·N/(n_i·n_j);  PMI = log2 lift  (computed client-side from the
             exact integers — nothing pre-rounded enters the axes)
      cos  = cosine(W_dec[i], W_dec[j])          (float32, float64-verified)

    Honesty:
    - window = the SAE's own training context_size (128, asserted from cfg):
      feeding longer contexts is out-of-distribution for the SAE — a 512-token
      first attempt measured L0 ≈ 159 and recon cos 0.86 vs the published
      ~60-70 / 0.93+ regime.
    - Position 0 of every window is dropped from ALL statistics: a chunk
      boundary is not a real document start and carries GPT-2's massive-
      activation outlier (see compute_sae_acts). Stated in meta.
    - Selection is Dunning's G² (the standard 2×2 log-likelihood collocation
      statistic, exact from the same integers): the top max_pairs pairs by G²
      among all pairs with c_ij ≥ min_count. A raw count threshold was tried
      first and kept only always-on pairs (max lift 10); G² spans both tails —
      strong attraction AND strong avoidance above the support floor. The
      global cos↔PMI Pearson over ALL support pairs ships in meta so the
      truncation cannot fake a correlation. PMI has a hard ceiling
      log2(N/max(n_i,n_j)) — rare pairs dominate the top; disclosed.
    - c = 0 avoidance can't clear a support floor by definition, so the
      strongest below-independence pairs are additionally surfaced among the
      300 most active features where e ≥ 20 ("avoid" chips).
    - Measured yardstick: a seeded permutation of one feature's firing rows
      destroys pairing while keeping both marginals; shuffled counts are
      exported per chip and as an aggregate ratio vs e (expect ≈ 1).
    - Two independent count paths must agree: c_ij from the sparse matmul is
      asserted equal to the sorted-row intersection size for EVERY exported
      pair (which also yields the top co-firing token, so token-driven pairs
      are visible: "co-fires mostly on ' the'").
    - Chunking restarts the positional embedding every `window` tokens;
      position-sensitive features see that sawtooth (real, disclosed).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    from scipy import sparse as sp

    text = COFIRE_CORPUS.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

    cfg = json.loads(open(hf_hub_download(repo, f"{hook}/cfg.json")).read())
    t = load_file(hf_hub_download(repo, f"{hook}/sae_weights.safetensors"))
    W_enc, b_enc = t["W_enc"], t["b_enc"]
    W_dec, b_dec = t["W_dec"], t["b_dec"]
    d_sae, d_in = int(cfg["d_sae"]), int(cfg["d_in"])
    layer = int(cfg["hook_point_layer"])
    assert W_enc.shape == (d_in, d_sae), f"W_enc shape {W_enc.shape}"
    assert W_dec.shape == (d_sae, d_in), f"W_dec shape {W_dec.shape}"
    assert d_in == m.d, f"SAE d_in {d_in} != model d {m.d}"
    assert window == int(cfg["context_size"]), (
        f"window {window} != SAE training context_size {cfg['context_size']} — "
        "longer contexts are out-of-distribution for the SAE (measured: L0 159 "
        "vs the published ~65 at 512 tokens)"
    )

    ids = m.encode(text)
    n_tok = len(ids)

    pos_feats: list[np.ndarray] = []  # active feature ids per counted position
    tok_of_pos: list[int] = []  # token id at each counted position
    nj = np.zeros(d_sae, dtype=np.int64)
    l0_parts: list[np.ndarray] = []
    cos_sum, cos_n = 0.0, 0
    n_windows = 0
    for w0 in range(0, n_tok, window):
        chunk = ids[w0 : w0 + window]
        if len(chunk) < 2:
            break
        tr: Trace = m.forward(chunk)
        x = tr.resid[layer].astype(np.float32)
        xc = x - x.mean(axis=1, keepdims=True)  # TL center_writing_weights basis
        acts = np.maximum((xc - b_dec) @ W_enc + b_enc, 0.0)
        if n_windows % recon_every == 0:  # recon cosine sampled (cost = 2nd big matmul)
            recon = acts @ W_dec + b_dec
            rc = (recon[1:] * xc[1:]).sum(1) / (
                np.linalg.norm(recon[1:], axis=1) * np.linalg.norm(xc[1:], axis=1) + 1e-12
            )
            cos_sum += float(rc.sum())
            cos_n += rc.size
        n_windows += 1
        A = acts[1:] > 0.0  # (T-1, d_sae) — position 0 dropped from all stats
        nj += A.sum(0)
        l0_parts.append(A.sum(1))
        for r in range(A.shape[0]):
            pos_feats.append(np.flatnonzero(A[r]).astype(np.int32))
            tok_of_pos.append(int(chunk[r + 1]))

    N = len(pos_feats)
    l0 = np.concatenate(l0_parts)
    assert l0.size == N
    n_active = int((nj > 0).sum())

    # Features eligible for pair statistics. c_ij ≤ min(n_i, n_j), so no
    # dropped feature could have formed a qualifying pair — the census over
    # c ≥ min_count is exhaustive despite the restriction.
    kept = np.flatnonzero(nj >= min_count)
    remap = np.full(d_sae, -1, dtype=np.int32)
    remap[kept] = np.arange(kept.size, dtype=np.int32)

    indptr = np.zeros(N + 1, dtype=np.int64)
    idx_parts: list[np.ndarray] = []
    for r, f in enumerate(pos_feats):
        kf = remap[f]
        kf = kf[kf >= 0]
        idx_parts.append(kf)
        indptr[r + 1] = indptr[r] + kf.size
    indices = np.concatenate(idx_parts)
    X = sp.csr_matrix(
        (np.ones(indices.size, dtype=np.int32), indices, indptr), shape=(N, kept.size)
    )
    C = sp.triu(X.T @ X, k=1).tocoo()

    c_all = C.data.astype(np.int64)
    qual = c_all >= min_count
    pi_s = kept[C.row[qual]]  # ALL support pairs (before G² truncation)
    pj_s = kept[C.col[qual]]
    c_s = c_all[qual]
    n_support = int(c_s.size)
    na_s = nj[pi_s].astype(np.float64)
    nb_s = nj[pj_s].astype(np.float64)

    # Dunning's G² over the full 2×2 table — exact from the integer counts.
    def g2_of(cv: np.ndarray, na: np.ndarray, nb: np.ndarray) -> np.ndarray:
        tot = float(N)
        k11 = cv.astype(np.float64)
        k12, k21 = na - k11, nb - k11
        k22 = tot - na - nb + k11

        def term(k: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
            with np.errstate(divide="ignore", invalid="ignore"):
                v = k * np.log(k * tot / (row * col))
            return np.where(k > 0, v, 0.0)

        return 2.0 * (
            term(k11, na, nb)
            + term(k12, na, tot - nb)
            + term(k21, tot - na, nb)
            + term(k22, tot - na, tot - nb)
        )

    g2_s = g2_of(c_s, na_s, nb_s)

    # Decoder cosines for ALL support pairs (blocked) — needed for the global
    # cos↔PMI correlation, so the exported truncation cannot fake a trend.
    Wd64 = W_dec.astype(np.float64)
    dn = np.linalg.norm(Wd64, axis=1)
    assert float(dn.min()) > 1e-6, "zero-norm decoder row"
    U = (Wd64 / dn[:, None]).astype(np.float32)
    cos_s = np.empty(n_support, dtype=np.float64)
    B = 262144
    for r0 in range(0, n_support, B):
        r1 = min(r0 + B, n_support)
        cos_s[r0:r1] = np.einsum("ij,ij->i", U[pi_s[r0:r1]], U[pj_s[r0:r1]])
    pmi_s = np.log2(c_s * float(N) / (na_s * nb_s))
    pearson = float(np.corrcoef(cos_s, pmi_s)[0, 1])

    # Export: top max_pairs by G² (both tails), then sort by c desc for
    # transport determinism (the viewer reads c[0] as the color-scale max).
    sel_g2 = np.argsort(-g2_s, kind="stable")[:max_pairs]
    pi, pj, c = pi_s[sel_g2], pj_s[sel_g2], c_s[sel_g2]
    cosd, g2v = cos_s[sel_g2], g2_s[sel_g2]
    g2_min = float(g2v.min())
    order = np.lexsort((pj, pi, -c))  # deterministic: c desc, then ids
    pi, pj, c, cosd = pi[order], pj[order], c[order], cosd[order]
    n_pairs = int(c.size)

    # float64 verification of the float32 cosine scan on a seeded sample.
    rngv = np.random.default_rng(seed)
    samp = rngv.choice(n_pairs, size=min(64, n_pairs), replace=False)
    U64 = Wd64 / dn[:, None]
    worst = max(
        abs(float(U64[pi[k]] @ U64[pj[k]]) - float(cosd[k])) for k in map(int, samp)
    )
    assert worst < 1e-5, f"cosine verification failed: {worst}"

    # Sorted firing-row list per involved feature (from CSC — rows are sorted).
    Xc = X.tocsc()
    tok_arr = np.array(tok_of_pos, dtype=np.int32)

    def rows_of(f: int) -> np.ndarray:
        k = int(remap[f])
        assert k >= 0
        return Xc.indices[Xc.indptr[k] : Xc.indptr[k + 1]]

    # Per-pair: independent recount (intersection ≡ matmul count) + top co-token.
    tok_table: dict[int, int] = {}
    ctok_strs: list[str] = []
    tt = np.empty(n_pairs, dtype=np.int32)
    tshare = np.empty(n_pairs, dtype=np.float64)
    for k in range(n_pairs):
        co = np.intersect1d(rows_of(int(pi[k])), rows_of(int(pj[k])), assume_unique=True)
        assert co.size == int(c[k]), f"count mismatch pair {k}: {co.size} != {c[k]}"
        vals, cnts = np.unique(tok_arr[co], return_counts=True)
        b = int(cnts.argmax())  # ties → lowest token id (np.unique is sorted)
        tid = int(vals[b])
        if tid not in tok_table:
            tok_table[tid] = len(ctok_strs)
            ctok_strs.append(m.decode1(tid))
        tt[k] = tok_table[tid]
        tshare[k] = cnts[b] / co.size

    # Measured shuffle yardstick: permute one feature's rows (marginals kept,
    # pairing destroyed) — aggregate count over a seeded pair sample vs e.
    perm = np.random.default_rng(seed).permutation(N)
    rngs = np.random.default_rng(seed + 1)
    sel = rngs.choice(n_pairs, size=min(shuffle_sample, n_pairs), replace=False)

    def shuf_count(a: int, b: int) -> int:
        za = np.zeros(N, dtype=bool)
        za[rows_of(a)] = True
        zb = np.zeros(N, dtype=bool)
        zb[rows_of(b)] = True
        return int((za[perm] & zb).sum())

    sh_sum = sum(shuf_count(int(pi[k]), int(pj[k])) for k in map(int, sel))
    e_sum = float(sum(nj[pi[k]] * nj[pj[k]] / N for k in map(int, sel)))

    lift = c.astype(np.float64) * N / (nj[pi].astype(np.float64) * nj[pj])
    g2e = g2_of(c, nj[pi].astype(np.float64), nj[pj].astype(np.float64))

    def chip(k: int) -> dict:
        a, b = int(pi[k]), int(pj[k])
        return {
            "i": a,
            "j": b,
            "c": int(c[k]),
            "e": round(float(nj[a] * nj[b] / N), 2),
            "lift": round(float(lift[k]), 2),
            "cos": round(float(cosd[k]), 4),
            "tok": ctok_strs[int(tt[k])],
            "share": round(float(tshare[k]), 3),
            "shuf": shuf_count(a, b),
        }

    def chip_list(idx_order: np.ndarray) -> list[dict]:
        out: list[dict] = []
        used: set[int] = set()
        for k in map(int, idx_order):
            if int(pi[k]) in used or int(pj[k]) in used:
                continue  # chips deduped by feature (top pairs cluster in families)
            used.add(int(pi[k]))
            used.add(int(pj[k]))
            out.append(chip(k))
            if len(out) >= top_chips:
                break
        return out

    # assoc chips: strongest ABOVE-independence pairs by G² (both G² tails are
    # in the export; the below-independence tail is covered by avoid chips)
    assoc_order = np.argsort(-np.where(lift > 1.0, g2e, -1.0), kind="stable")
    chips_assoc = chip_list(assoc_order)
    chips_count = chip_list(np.arange(n_pairs))  # already sorted by c desc

    # Mutual exclusion: among the 300 most active features, the pairs whose
    # expected co-count is largest while the observed count sits far below it.
    ta = kept[np.argsort(-nj[kept], kind="stable")[:300]]
    Xd = np.zeros((N, ta.size), dtype=np.int32)
    for q, f in enumerate(map(int, ta)):
        Xd[rows_of(f), q] = 1
    C300 = Xd.T @ Xd
    E300 = np.outer(nj[ta], nj[ta]).astype(np.float64) / N
    iu, ju = np.triu_indices(ta.size, k=1)
    e_u, c_u = E300[iu, ju], C300[iu, ju].astype(np.float64)
    okm = e_u >= 20.0
    lift_u = c_u[okm] / e_u[okm]
    av_order = np.lexsort((-e_u[okm], lift_u))  # lift asc, then e desc
    ai, aj = ta[iu[okm]], ta[ju[okm]]
    chips_avoid: list[dict] = []
    used_a: set[int] = set()
    for k in map(int, av_order):
        a, b = int(ai[k]), int(aj[k])
        if a in used_a or b in used_a:
            continue
        used_a.add(a)
        used_a.add(b)
        cv = int(c_u[okm][k])
        co = np.intersect1d(rows_of(a), rows_of(b), assume_unique=True)
        assert co.size == cv
        if cv > 0:
            vals, cnts = np.unique(tok_arr[co], return_counts=True)
            bb = int(cnts.argmax())
            tok_s, share = m.decode1(int(vals[bb])), round(float(cnts[bb] / cv), 3)
        else:
            tok_s, share = None, 0.0
        chips_avoid.append(
            {
                "i": a,
                "j": b,
                "c": cv,
                "e": round(float(e_u[okm][k]), 2),
                "lift": round(float(lift_u[k]), 4),
                "cos": round(float(U64[a] @ U64[b]), 4),
                "tok": tok_s,
                "share": share,
                "shuf": shuf_count(a, b),
            }
        )
        if len(chips_avoid) >= top_chips:
            break

    # Marginals for every feature appearing anywhere in the export.
    f_ids = np.unique(
        np.concatenate(
            [pi, pj, [ch["i"] for ch in chips_avoid], [ch["j"] for ch in chips_avoid]]
        )
    )

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "sae_repo": repo,
            "hook_point": hook,
            "corpus": {
                "title": "Alice's Adventures in Wonderland — Lewis Carroll (1865)",
                "source": "Project Gutenberg eBook #11 (public domain); header/"
                "footer stripped, CRLF→LF normalized",
                "sha256": sha,
                "n_chars": len(text),
                "n_tokens": n_tok,
                "window": window,
                "n_windows": n_windows,
                "n_pos": N,
            },
            "quantity": "exact SAE co-firing counts over every corpus position: "
            "n_i (marginal), c_ij (joint), vs the independence expectation "
            "n_i·n_j/N — joined with each pair's decoder-direction cosine",
            "formula": "x̄ = x − mean(x); acts = ReLU((x̄ − b_dec)·W_enc + b_enc); "
            "fires = acts > 0 at blocks.8.hook_resid_pre. c_ij = |{p: i,j both "
            "fire}| (sparse XᵀX ≡ sorted-row intersection, asserted equal). "
            "e_ij = n_i·n_j/N; lift = c_ij·N/(n_i·n_j); PMI = log2 lift — "
            "computed in the viewer from the exact integers. selection = "
            "Dunning's G² over the full 2×2 table. "
            "cos = cosine(W_dec[i], W_dec[j]).",
            "note": f"window = SAE training context_size ({window}, asserted). "
            "position 0 of each window dropped from all statistics (chunk "
            f"boundary + massive-activation outlier). export = top {max_pairs} "
            f"pairs by G² among all {n_support} pairs with c ≥ {min_count} "
            "(both tails; smallest exported G² in meta). PMI ceiling = "
            "log2(N/max(n_i,n_j)) — rare pairs dominate the top. c = 0 "
            "avoidance can't clear a support floor, so below-independence "
            "exemplars are additionally drawn from the 300 most active "
            "features (e ≥ 20). shuffle = seeded permutation of one feature's "
            "rows (marginals kept). chips deduped by feature. recon cosine "
            f"sampled every {recon_every}th window. positional embedding "
            "restarts each window (sawtooth for position-sensitive features). "
            "counts exact; cos 4dp, shares 3dp.",
            "d_sae": d_sae,
            "layer": layer,
            "min_count": int(min_count),
            "selection": "top_g2",
            "g2_min": round(g2_min, 2),
            "n_support": n_support,
            "pearson_cos_pmi": round(pearson, 4),
            "n_active": n_active,
            "n_eligible": int(kept.size),
            "n_pairs": n_pairs,
            "l0_mean": round(float(l0.mean()), 2),
            "l0_median": int(np.median(l0)),
            "recon_cos_mean": round(cos_sum / max(cos_n, 1), 4),
            "recon_cos_n": cos_n,
            "shuffle": {
                "seed": seed,
                "n_sampled": int(sel.size),
                "shuf_sum": int(sh_sum),
                "e_sum": round(e_sum, 2),
                "agg_ratio": round(sh_sum / e_sum, 4),
            },
        },
        "N": N,
        "pi": [int(v) for v in pi],
        "pj": [int(v) for v in pj],
        "c": [int(v) for v in c],
        "cos": [round(float(v), 4) for v in cosd],
        "tt": [int(v) for v in tt],
        "tshare": [round(float(v), 3) for v in tshare],
        "ctok_strs": ctok_strs,
        "f_ids": [int(v) for v in f_ids],
        "f_n": [int(nj[v]) for v in f_ids],
        "chips": {"assoc": chips_assoc, "count": chips_count, "avoid": chips_avoid},
    }


def compute_tuned(
    m: GPT2Numpy,
    prompts: list[str] | None = None,
    window: int = 128,
    test_every: int = 4,
    eval_windows: int = 24,
    seed: int = 0,
) -> dict:
    """#20 Tuned-Lens Delta — a fitted affine translator per layer vs the raw
    logit lens, on a real disclosed corpus.

    The logit lens (#3) reads every layer's residual through the FINAL
    layernorm + unembedding — it asks "what if the model stopped here?", and
    at early layers that question is unfair: the residual basis drifts across
    depth, so early readouts look worse than what the layer actually knows.
    The tuned lens (Belrose et al. 2023) fixes this with a learned per-layer
    translator before the shared readout.

    CAVEAT (by design, stated everywhere): this is the LEAST-SQUARES affine
    translator — per layer L, (A_L, b_L) minimizing ‖A_L·h_L + b_L − h_final‖²
    over corpus positions, solved exactly from float64 normal equations. It is
    NOT the KL-trained translator of the paper; the fit objective is residual
    MSE, only the EVALUATION is distributional. Both lenses then share the
    model's own ln_f + tied unembedding.

    Fit/eval data: every token of the disclosed Alice corpus (sha256 in meta),
    non-overlapping windows of `window` tokens, position 0 of each window
    dropped (chunk boundary + GPT-2 massive-activation outlier — same rule as
    #24). Deterministic split: windows with idx % test_every == 0 are held out;
    translators are fit ONLY on the rest. Distributional metrics are computed
    on `eval_windows` seeded held-out windows (full 50257-way softmax — exact,
    no truncation):

      kl[L]    = KL(p_final ‖ p_lens_L) in bits, mean/p25/p50/p75
      agree[L] = fraction of positions where the lens top-1 == final top-1

    Honesty checks baked in:
    - layer 12 IS the final residual: the logit-lens KL there is asserted 0
      and agreement 1 (identity of the computation path, not a rounding claim).
    - the normal-equation solve residual ‖G·W − B‖/‖B‖ is asserted < 1e-8 and
      train R² per layer is exported (computed exactly from the accumulators).
    - the per-prompt grids (the 5 shared trace prompts) export BOTH lenses'
      top-1 token and probability and the exact per-position KL, so the
      "prediction emerges earlier under the tuned lens" claim is inspectable
      token by token — including where it does NOT hold.
    """
    prompts = prompts or DEFAULT_PROMPTS
    text = COFIRE_CORPUS.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    ids = m.encode(text)
    n_layer, d = m.n_layer, m.d

    # ---- pass over the corpus: accumulate normal equations on train windows,
    # stash residuals of the seeded eval subset of test windows ----
    starts = [s for s in range(0, len(ids), window) if len(ids) - s >= 2]
    test_idx = [w for w in range(len(starts)) if w % test_every == 0]
    rng = np.random.default_rng(seed)
    eval_pick = rng.choice(len(test_idx), size=min(eval_windows, len(test_idx)), replace=False)
    eval_set = {test_idx[i] for i in sorted(eval_pick.tolist())}

    G = np.zeros((n_layer, d + 1, d + 1), dtype=np.float64)
    B = np.zeros((n_layer, d + 1, d), dtype=np.float64)
    y_sumsq = 0.0
    n_train = 0
    eval_resid: list[np.ndarray] = []  # each (n_layer+1, P, d) float32
    for w, s in enumerate(starts):
        tr = m.forward(ids[s : s + window])
        r = tr.resid[:, 1:, :]  # drop position 0 (chunk boundary / outlier)
        if w in eval_set:
            eval_resid.append(r.copy())
        if w % test_every == 0:
            continue  # held out — never enters the fit
        P = r.shape[1]
        y = r[n_layer].astype(np.float64)
        y_sumsq += float((y * y).sum())
        n_train += P
        ones = np.ones((P, 1), dtype=np.float64)
        for L in range(n_layer):
            x = np.concatenate([r[L].astype(np.float64), ones], axis=1)
            G[L] += x.T @ x
            B[L] += x.T @ y
    assert n_train > 8 * (d + 1), f"train positions {n_train} too few for {d + 1}-dim fit"

    # ---- exact normal-equation solve (float64) + train R² from accumulators ----
    W = np.empty((n_layer, d + 1, d), dtype=np.float64)
    r2 = np.empty(n_layer)
    y_sum = B[0][d]  # bias row of X̃ᵀY = Σy (identical for every L)
    y_ss_centered = y_sumsq - float((y_sum * y_sum).sum()) / n_train
    solve_resid = 0.0
    for L in range(n_layer):
        W[L] = np.linalg.solve(G[L], B[L])
        solve_resid = max(solve_resid, float(np.linalg.norm(G[L] @ W[L] - B[L]) / np.linalg.norm(B[L])))
        # ‖Y − X̃W‖² = tr(YᵀY) − 2·tr(WᵀB) + tr(Wᵀ G W)
        sse = y_sumsq - 2.0 * float((W[L] * B[L]).sum()) + float((W[L] * (G[L] @ W[L])).sum())
        r2[L] = 1.0 - sse / y_ss_centered
    assert solve_resid < 1e-8, f"normal-equation solve residual {solve_resid:.2e}"
    assert r2[n_layer - 1] > r2[0], "translator fit must improve with depth"

    # ---- distributional eval on the seeded held-out windows ----
    g_f, b_f = m._g("ln_f.weight"), m._g("ln_f.bias")

    def lens_logits(h: np.ndarray) -> np.ndarray:
        xf, _ = _layernorm(h.astype(np.float32), g_f, b_f)
        return xf @ m.wte.T  # (P, V) float32

    def logprobs(rows: np.ndarray) -> np.ndarray:
        lp = rows.astype(np.float64)
        lp -= lp.max(axis=1, keepdims=True)
        lp -= np.log(np.exp(lp).sum(axis=1, keepdims=True))
        return lp

    def dist_stats(rows: np.ndarray, fin_lp: np.ndarray, fin_top: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lp = logprobs(rows)
        kl_nats = (np.exp(fin_lp) * (fin_lp - lp)).sum(axis=1)
        return kl_nats / np.log(2.0), rows.argmax(axis=1) == fin_top

    kl_all: list[list[np.ndarray]] = [[] for _ in range(2 * n_layer + 1)]
    ag_all: list[list[np.ndarray]] = [[] for _ in range(2 * n_layer + 1)]
    n_eval = 0
    for r in eval_resid:
        P = r.shape[1]
        n_eval += P
        fin_rows = lens_logits(r[n_layer])
        fin_lp = logprobs(fin_rows)
        fin_top = fin_rows.argmax(axis=1)
        ones = np.ones((P, 1))
        for L in range(n_layer):
            kb, ok = dist_stats(lens_logits(r[L]), fin_lp, fin_top)
            kl_all[L].append(kb)
            ag_all[L].append(ok)
            h_t = np.concatenate([r[L].astype(np.float64), ones], axis=1) @ W[L]
            kb, ok = dist_stats(lens_logits(h_t), fin_lp, fin_top)
            kl_all[n_layer + L].append(kb)
            ag_all[n_layer + L].append(ok)
        kb, ok = dist_stats(fin_rows, fin_lp, fin_top)
        kl_all[2 * n_layer].append(kb)
        ag_all[2 * n_layer].append(ok)

    def curve(idx: int) -> dict:
        kb = np.concatenate(kl_all[idx])
        ok = np.concatenate(ag_all[idx])
        q = np.percentile(kb, [25, 50, 75])
        return {
            "mean": round(float(kb.mean()), 4),
            "p25": round(float(q[0]), 4),
            "p50": round(float(q[1]), 4),
            "p75": round(float(q[2]), 4),
            "agree": round(float(ok.mean()), 4),
        }

    logit_curve = [curve(L) for L in range(n_layer)]
    tuned_curve = [curve(n_layer + L) for L in range(n_layer)]
    final_check = curve(2 * n_layer)
    # layer 12 through the identical readout path: KL exactly 0, agreement 1
    assert final_check["mean"] == 0.0 and final_check["agree"] == 1.0, final_check
    logit_curve.append(final_check)
    tuned_curve.append(final_check)  # translator at 12 is the identity by definition
    assert tuned_curve[0]["mean"] < logit_curve[0]["mean"], "tuned lens must beat logit lens at layer 0"

    # ---- per-prompt grids on the 5 shared trace prompts ----
    tok_strs: list[str] = []
    tok_of: dict[int, int] = {}

    def sid(t: int) -> int:
        if t not in tok_of:
            tok_of[t] = len(tok_strs)
            tok_strs.append(m.decode1(t))
        return tok_of[t]

    grids = []
    for prompt in prompts:
        tr = m.forward(prompt)
        T = len(tr.tokens)
        fin_rows = lens_logits(tr.resid[n_layer])
        assert np.abs(fin_rows - tr.logits).max() < 1e-3, "lens path must reproduce the forward's logits"
        fin_lp = logprobs(fin_rows)
        fin_top = fin_rows.argmax(axis=1)
        ones = np.ones((T, 1))
        cells: dict[str, list] = {"logit": [], "tuned": []}
        for L in range(n_layer + 1):
            if L < n_layer:
                rows_l = lens_logits(tr.resid[L])
                rows_t = lens_logits(np.concatenate([tr.resid[L].astype(np.float64), ones], axis=1) @ W[L])
            else:
                rows_l = rows_t = fin_rows
            for name, rows in (("logit", rows_l), ("tuned", rows_t)):
                lp = logprobs(rows)
                kl = (np.exp(fin_lp) * (fin_lp - lp)).sum(axis=1) / np.log(2.0)
                top = rows.argmax(axis=1)
                cells[name].append([
                    [sid(int(top[t])), round(float(np.exp(lp[t, top[t]])), 3), round(float(kl[t]), 3)]
                    for t in range(T)
                ])
        grids.append({
            "slug": _slug(prompt),
            "prompt": prompt,
            "token_strs": tr.token_strs,
            "T": T,
            "final_top": [[sid(int(fin_top[t])), round(float(np.exp(fin_lp[t, fin_top[t]])), 3)] for t in range(T)],
            "logit": cells["logit"],
            "tuned": cells["tuned"],
        })

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "quantity": "per-layer lens fidelity: KL(p_final ‖ p_lens) in bits and "
            "top-1 agreement, for the raw logit lens vs a least-squares affine "
            "translator, on held-out corpus windows",
            "formula": "tuned_L(h) = ln_f(A_L·h + b_L)·W_Eᵀ with (A_L,b_L) = "
            "argmin Σ‖A·h_L + b − h_final‖² over train positions (exact float64 "
            "normal equations); logit_L(h) = ln_f(h)·W_Eᵀ; KL in bits over the "
            "full 50,257-way softmax",
            "note": "NOT the KL-trained tuned lens of Belrose et al. 2023 — the "
            "translator is fit by residual least squares; only the evaluation is "
            "distributional. Both lenses share the model's own ln_f + tied "
            "unembedding. Position 0 of every window dropped (chunk boundary + "
            "massive-activation outlier). resid[L] is the stream ENTERING block "
            "L; layer 12 is the final residual, where both lenses are the "
            "identity (asserted KL 0).",
            "corpus": {
                "title": "Alice's Adventures in Wonderland",
                "source": "Project Gutenberg #11 (public domain)",
                "sha256": sha,
                "n_tokens": len(ids),
                "window": window,
                "n_windows": len(starts),
            },
            "split": f"window idx % {test_every} == 0 held out; fit on the rest",
            "n_train_pos": int(n_train),
            "n_eval_pos": int(n_eval),
            "eval_windows": len(eval_resid),
            "eval_seed": seed,
            "solve_resid": float(solve_resid),
            "r2_train": [round(float(v), 4) for v in r2],
            "kl_direction": "KL(p_final ‖ p_lens), bits",
        },
        "n_layer": n_layer,
        "logit": logit_curve,
        "tuned": tuned_curve,
        "tok_strs": tok_strs,
        "grids": grids,
    }


def compute_grok(
    p: int = 97,
    n_hidden: int = 128,
    frac: float = 0.22,
    lr: float = 1e-3,
    wd: float = 0.3,
    kappa: float = 16.0,
    steps: int = 40_000,
    seed: int = 0,
    ckpt_every: int = 100,
) -> dict:
    """Train a toy modular-addition MLP from scratch and record the grokking run.

    Gromov (2023) setup: y = ((onehot(a) ++ onehot(b)) @ W1)**2 @ W2, MSE loss
    against one-hot targets, full-batch AdamW. Weight decay drives the
    memorize->generalize transition; the generalizing solution is periodic in
    the token index, so the DFT of W1's rows over `a` concentrates onto a few
    key frequencies exactly when test accuracy jumps. This bundle is NOT
    derived from GPT-2 — it is a separately trained toy model.
    """
    rng = np.random.default_rng(seed)
    A, B = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = A.ravel(), B.ravel()
    c = (a + b) % p
    n = p * p
    perm = rng.permutation(n)
    ntr = int(frac * n)
    tr, te = perm[:ntr], perm[ntr:]

    X = np.zeros((n, 2 * p), dtype=np.float32)
    X[np.arange(n), a] = 1
    X[np.arange(n), p + b] = 1
    Y = np.zeros((n, p), dtype=np.float32)
    Y[np.arange(n), c] = 1
    Xtr, Ytr, ctr = X[tr], Y[tr], c[tr]
    Xte, Yte, cte = X[te], Y[te], c[te]

    W1 = (kappa * rng.standard_normal((2 * p, n_hidden)) / np.sqrt(2 * p)).astype(np.float32)
    W2 = (kappa * rng.standard_normal((n_hidden, p)) / np.sqrt(n_hidden)).astype(np.float32)

    n_freq = p // 2 + 1
    # rfft multiplicity: freq 0 appears once, freqs 1..(p-1)/2 fold conjugates
    mult = np.ones(n_freq)
    mult[1:] = 2.0

    def spectrum(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-freq power fraction + per-unit purity of the a-half rows.

        purity[u] = the largest single non-DC frequency's share of unit u's
        total spectral power — the quantity that jumps at the grok.
        Parseval-checked against p * ||W_a||^2.
        """
        Wa = W[:p].astype(np.float64)
        F = np.fft.rfft(Wa, axis=0)
        spec = mult[:, None] * np.abs(F) ** 2  # (n_freq, n_hidden)
        pw = spec.sum(1)
        total = pw.sum()
        parseval = p * float((Wa**2).sum())
        assert abs(total - parseval) < 1e-6 * max(parseval, 1.0), (total, parseval)
        dom = spec[1:].argmax(0) + 1
        purity = spec[dom, np.arange(spec.shape[1])] / spec.sum(0)
        return pw / total, purity

    def evaluate(Xs: np.ndarray, Ys: np.ndarray, cs: np.ndarray) -> tuple[float, float]:
        out = ((Xs @ W1) ** 2) @ W2
        acc = float((out.argmax(1) == cs).mean())
        loss = float(((out - Ys) ** 2).mean())
        return acc, loss

    m1 = np.zeros_like(W1)
    v1 = np.zeros_like(W1)
    m2 = np.zeros_like(W2)
    v2 = np.zeros_like(W2)
    b1m, b2m, eps = 0.9, 0.98, 1e-8

    ck_steps: list[int] = []
    ck_tracc: list[float] = []
    ck_teacc: list[float] = []
    ck_trloss: list[float] = []
    ck_teloss: list[float] = []
    ck_fpower: list[np.ndarray] = []
    ck_pur_med: list[float] = []
    ck_pur_q1: list[float] = []
    ck_pur_q3: list[float] = []

    def checkpoint(t: int) -> tuple[float, float]:
        atr, ltr = evaluate(Xtr, Ytr, ctr)
        ate, lte = evaluate(Xte, Yte, cte)
        pw, purity = spectrum(W1)
        ck_steps.append(t)
        ck_tracc.append(atr)
        ck_teacc.append(ate)
        ck_trloss.append(ltr)
        ck_teloss.append(lte)
        ck_fpower.append(pw)
        ck_pur_med.append(float(np.median(purity)))
        ck_pur_q1.append(float(np.percentile(purity, 25)))
        ck_pur_q3.append(float(np.percentile(purity, 75)))
        return atr, ate

    checkpoint(0)
    tr100_step: int | None = None
    grok_step: int | None = None
    for t in range(1, steps + 1):
        Z = Xtr @ W1
        H = Z * Z
        D = (H @ W2 - Ytr) * (2.0 / ntr)
        gW2 = H.T @ D
        dZ = (D @ W2.T) * 2 * Z
        gW1 = Xtr.T @ dZ
        for W, g, mm, vv in ((W1, gW1, m1, v1), (W2, gW2, m2, v2)):
            mm *= b1m
            mm += (1 - b1m) * g
            vv *= b2m
            vv += (1 - b2m) * g * g
            W -= lr * ((mm / (1 - b1m**t)) / (np.sqrt(vv / (1 - b2m**t)) + eps) + wd * W)
        if t % ckpt_every == 0:
            atr, ate = checkpoint(t)
            if atr > 0.999 and tr100_step is None:
                tr100_step = t
            if ate > 0.999 and grok_step is None:
                grok_step = t
            if grok_step is not None and t >= grok_step + 4000:
                break

    assert tr100_step is not None, "toy model never fit the training set"
    assert grok_step is not None, "toy model never generalized — no grokking run to export"
    assert grok_step - tr100_step >= 1000, (
        f"train->test gap only {grok_step - tr100_step} steps — not a grokking run"
    )

    # final re-verification from scratch: fresh forward equals stored last checkpoint
    ate_final, _ = evaluate(Xte, Yte, cte)
    assert abs(ate_final - ck_teacc[-1]) < 1e-12

    # the spectral signature of THIS architecture is per-unit, not aggregate:
    # each hidden unit becomes a near-pure single-frequency oscillator, but
    # different units pick different frequencies, so the aggregate spectrum
    # stays spread (unlike the sparse key-freq story of grokking transformers).
    init_pw = ck_fpower[0]
    assert float(init_pw[1:].max()) < 10.0 / n_freq, "init spectrum unexpectedly peaked"
    assert ck_pur_med[-1] > 0.8, f"final purity only {ck_pur_med[-1]:.2f} — units not single-freq"
    # while the net is memorized-but-not-generalized, purity must still be near
    # its chance level: the purity jump and the test-accuracy jump coincide
    memorized = [i for i, ta in enumerate(ck_teacc) if ta < 0.05]
    pur_at_mem = ck_pur_med[memorized[-1]] if memorized else float("nan")
    assert memorized and pur_at_mem < 0.3, f"purity already {pur_at_mem:.2f} before generalization"

    # per-hidden-unit dominant frequency + purity of the final network
    Wa64 = W1[:p].astype(np.float64)
    F = np.fft.rfft(Wa64, axis=0)
    spec_u = mult[:, None] * np.abs(F) ** 2  # (n_freq, n_hidden)
    unit_freq = spec_u[1:].argmax(0) + 1
    unit_frac = spec_u[unit_freq, np.arange(n_hidden)] / spec_u.sum(0)

    # the clock: project token rows onto the orthonormalized (cos, sin) Fourier
    # pair of a frequency k; the grokked embedding lays the p tokens on a circle
    # traversed k times, i.e. angle(token a) ~ +/- 2*pi*k*a/p + const. Clocks are
    # ranked by MEASURED circularity |mean exp(i(theta_a -+ 2*pi*k*a/p))| — the
    # selection criterion is itself a computed, exported quantity.
    final_pw = ck_fpower[-1]
    cands = []
    for k in range(1, n_freq):
        cosv = np.cos(2 * np.pi * k * np.arange(p) / p)
        sinv = np.sin(2 * np.pi * k * np.arange(p) / p)
        u_c = cosv @ Wa64
        u_s = sinv @ Wa64
        u_c /= np.linalg.norm(u_c)
        u_s -= (u_s @ u_c) * u_c
        u_s /= np.linalg.norm(u_s)
        xs = Wa64 @ u_c
        ys = Wa64 @ u_s
        theta = np.arctan2(ys, xs)
        best_r = max(
            float(abs(np.exp(1j * (theta - sign * 2 * np.pi * k * np.arange(p) / p)).mean()))
            for sign in (1, -1)
        )
        radii = np.hypot(xs, ys)
        cands.append((best_r, k, xs, ys, radii))
    cands.sort(key=lambda c: -c[0])
    clocks = [
        {
            "k": int(k),
            "circ": round(best_r, 4),
            "radius_cv": round(float(radii.std() / radii.mean()), 4),
            "n_units": int((unit_freq == k).sum()),
            "power_frac": round(float(final_pw[k]), 4),
            "xy": [round(float(v), 4) for pair in zip(xs, ys) for v in pair],
        }
        for best_r, k, xs, ys, radii in cands[:5]
    ]
    # the grokked clock must actually be a clock: near-perfect phase alignment
    assert clocks[0]["circ"] > 0.9, f"best clock circularity only {clocks[0]['circ']}"

    n_ck = len(ck_steps)
    return {
        "meta": {
            "model": f"toy MLP (2*{p} -> {n_hidden} -> {p}), phi(z)=z^2, no biases",
            "task": f"c = (a + b) mod {p}",
            "created": _now(),
            "note": "Trained from scratch in numpy — NOT GPT-2. Full-batch AdamW "
            f"(lr {lr}, wd {wd}, beta 0.9/0.98), init scale kappa={kappa}, "
            f"MSE loss on one-hot targets, seed {seed}. Accuracies/losses are "
            "exact full-split evaluations at every checkpoint (no smoothing).",
            "quantity": "train/test accuracy + MSE loss per checkpoint; per-unit "
            "single-frequency purity (median/quartiles) per checkpoint; DFT "
            "power fraction of W1's a-rows per frequency per checkpoint; final "
            "per-unit dominant frequency; token projections onto Fourier pairs "
            "of the most circular frequencies",
            "formula": "purity[u] = max_k mult_k|rfft_a W1[a,u]|_k^2 / sum_k (k>0); "
            "power_k = mult_k * sum_u |rfft_a W1[a,u]|_k^2, normalized per "
            "checkpoint (Parseval-checked vs p*||W1_a||^2); clock coords = "
            "W1_a rows projected on the orthonormalized (cos 2pi k a/p, "
            "sin 2pi k a/p) span; circ = |mean_a exp(i(theta_a -+ 2pi k a/p))|",
            "p": p,
            "n_hidden": n_hidden,
            "train_frac": frac,
            "n_train": int(ntr),
            "n_test": int(n - ntr),
            "steps_run": int(ck_steps[-1]),
            "ckpt_every": ckpt_every,
            "tr100_step": int(tr100_step),
            "grok_step": int(grok_step),
            "acc_threshold": 0.999,
            "purity_init": round(ck_pur_med[0], 4),
            "purity_at_memorized": round(pur_at_mem, 4),
            "purity_final": round(ck_pur_med[-1], 4),
            "top5_mass_final": round(float(np.sort(final_pw[1:])[::-1][:5].sum()), 4),
            "init_max_frac": round(float(init_pw[1:].max()), 4),
            "n_freq": n_freq,
            "spread_note": "aggregate spectrum stays SPREAD (top-5 freqs hold "
            "~15% of power) — the structure is per-unit: each unit becomes a "
            "near-pure single-frequency oscillator, units share the freqs out. "
            "This differs from the sparse key-frequency story of grokking "
            "transformers and is reported as measured.",
        },
        "steps": ck_steps,
        "train_acc": [round(v, 4) for v in ck_tracc],
        "test_acc": [round(v, 4) for v in ck_teacc],
        "train_loss": [round(v, 6) for v in ck_trloss],
        "test_loss": [round(v, 6) for v in ck_teloss],
        "purity_med": [round(v, 4) for v in ck_pur_med],
        "purity_q1": [round(v, 4) for v in ck_pur_q1],
        "purity_q3": [round(v, 4) for v in ck_pur_q3],
        "fpower": [round(float(v), 4) for row in ck_fpower for v in row],
        "unit_freq": [int(v) for v in unit_freq],
        "unit_frac": [round(float(v), 4) for v in unit_frac],
        "clocks": clocks,
        "n_ckpt": n_ck,
    }


def _logit_lens_topk(m: GPT2Numpy, resid_row: np.ndarray, k: int = 6) -> list:
    lg = m.logit_lens(resid_row)
    lg = lg - lg.max()
    pr = np.exp(lg)
    pr /= pr.sum()
    idx = np.argsort(pr)[::-1][:k]
    return [[m.decode1(int(i)), round(float(pr[i]), 4)] for i in idx]


def compute_trace(m: GPT2Numpy, prompt: str, attn_round: int = 4) -> dict:
    """One real forward pass, serialized to what the drivers render.

    Stores: full attention (rounded), per-layer residual L2 norms, logit-lens
    top-k at every layer for the LAST position (the prediction-sharpening
    'tunnel'), and per-position top-1 at the final layer. No full logits."""
    tr: Trace = m.forward(prompt)
    T = len(tr.tokens)

    # attention: (n_layer, n_head, T, T) rounded for transport
    attn = np.round(tr.attn, attn_round)
    # residual L2 norm per (layer, position) — real magnitude of the stream
    resid_norm = np.linalg.norm(tr.resid, axis=2)  # (n_layer+1, T)

    # logit-lens tunnel: top-k at the last position across all layers (0=embed)
    lens_last = [
        {"layer": L, "topk": _logit_lens_topk(m, tr.resid[L, -1])}
        for L in range(m.n_layer + 1)
    ]
    # per-position final prediction (top-1) — for simplex / attribution overlays
    final_lg = tr.logits  # (T, V)
    final_pred = []
    for pos in range(T):
        lg = final_lg[pos] - final_lg[pos].max()
        pr = np.exp(lg)
        pr /= pr.sum()
        j = int(np.argmax(pr))
        final_pred.append([m.decode1(j), round(float(pr[j]), 4)])
    # final-position next-token distribution top-k (Probability Simplex #18)
    final_topk = _logit_lens_topk(m, tr.resid[-1, -1], k=12)

    return {
        "meta": {
            "model": m.model_id,
            "created": _now(),
            "prompt": prompt,
            "n_layer": m.n_layer,
            "n_head": m.n_head,
            "d": m.d,
            "T": T,
            "quantity": "real forward pass: post-softmax attention, residual "
            "norms, logit-lens top-k, next-token distribution",
            "attn_rounding": attn_round,
        },
        "tokens": tr.tokens,
        "token_strs": tr.token_strs,
        "attn": attn.tolist(),  # (n_layer, n_head, T, T)
        "resid_norm": [[round(float(v), 3) for v in row] for row in resid_norm],
        "logit_lens_last": lens_last,
        "final_pred_per_pos": final_pred,
        "final_topk": final_topk,
    }


def write_bundles(
    model_id: str,
    out_root: Path,
    prompts: list[str] | None = None,
) -> list[Path]:
    """Compute + write all interp bundles for one model. Returns paths written."""
    m = GPT2Numpy(model_id)
    out_dir = out_root / model_id.replace("/", "__") / "interp"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def dump(name: str, obj: dict) -> None:
        p = out_dir / name
        p.write_text(json.dumps(obj, ensure_ascii=False))
        written.append(p)

    dump("weights.json", compute_weights(m))
    dump("fourier.json", compute_fourier(m))
    dump("embed.json", compute_embed(m))
    dump("neurons.json", compute_neurons(m))
    dump("heads.json", compute_heads(m, prompts=prompts or DEFAULT_PROMPTS))
    dump("ov_eigs.json", compute_ov_eigs(m))
    dump("comp.json", compute_comp(m))
    dump("attrib.json", compute_logit_attrib(m, prompts=prompts))
    dump("patch.json", compute_patching(m))
    dump("induction.json", compute_induction(m))
    dump("ablation.json", compute_ablation(m))
    dump("occlusion.json", compute_occlusion(m, prompts=prompts or DEFAULT_PROMPTS))
    dump("tuned.json", compute_tuned(m, prompts=prompts or DEFAULT_PROMPTS))
    # SAE decoder constellation — only where an open SAE release exists for the
    # model (res-jb covers gpt2-small). External download; skip loudly if absent.
    if model_id == "gpt2":
        try:
            dump("sae.json", compute_sae(m))
        except Exception as e:  # noqa: BLE001 — report and continue, never fake
            print(f"[interp] sae.json skipped: {e}")
        try:
            dump("sae_acts.json", compute_sae_acts(m, prompts=prompts))
        except Exception as e:  # noqa: BLE001
            print(f"[interp] sae_acts.json skipped: {e}")
        try:
            dump("sae_web.json", compute_sae_web())
        except Exception as e:  # noqa: BLE001
            print(f"[interp] sae_web.json skipped: {e}")
        try:
            dump("compass.json", compute_compass(m))
        except Exception as e:  # noqa: BLE001
            print(f"[interp] compass.json skipped: {e}")
        try:
            dump("cofire.json", compute_cofire(m))
        except Exception as e:  # noqa: BLE001
            print(f"[interp] cofire.json skipped: {e}")
        # grokking toy model — not derived from gpt2, but the Internals gallery
        # (and its index.json) lives in the gpt2 bundle dir
        dump("grok.json", compute_grok())

    traces_index = []
    for prompt in prompts or DEFAULT_PROMPTS:
        slug = _slug(prompt)
        dump(f"trace_{slug}.json", compute_trace(m, prompt))
        traces_index.append({"slug": slug, "prompt": prompt})

    # a tiny manifest so the viewer can discover what's available per model
    dump(
        "index.json",
        {
            "meta": {"model": model_id, "created": _now()},
            "bundles": [p.name for p in written],
            "traces": traces_index,
        },
    )
    return written
