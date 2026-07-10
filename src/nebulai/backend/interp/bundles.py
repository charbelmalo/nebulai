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
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    prompts = prompts or DEFAULT_PROMPTS
    cfg = json.loads(open(hf_hub_download(repo, f"{hook}/cfg.json")).read())
    t = load_file(hf_hub_download(repo, f"{hook}/sae_weights.safetensors"))
    sp = load_file(hf_hub_download(repo, f"{hook}/sparsity.safetensors"))
    W_enc, b_enc = t["W_enc"], t["b_enc"]
    W_dec, b_dec = t["W_dec"], t["b_dec"]
    d_sae, d_in = int(cfg["d_sae"]), int(cfg["d_in"])
    layer = int(cfg["hook_point_layer"])
    assert W_enc.shape == (d_in, d_sae), f"W_enc shape {W_enc.shape}"
    assert d_in == m.d, f"SAE d_in {d_in} != model d {m.d}"
    sparsity = sp["sparsity"].astype(np.float64)

    traces = []
    for prompt in prompts:
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

        traces.append(
            {
                "slug": _slug(prompt),
                "prompt": prompt,
                "token_strs": tr.token_strs,
                "T": len(tr.tokens),
                "l0": [int(v) for v in l0],
                "cos": [round(float(v), 4) for v in cos],
                "features": rows_for(ids),
                "sink_features": rows_for(sink_ids),
            }
        )

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
