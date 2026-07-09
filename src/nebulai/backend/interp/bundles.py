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

from .gpt2_numpy import GPT2Numpy, Trace

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
    # SAE decoder constellation — only where an open SAE release exists for the
    # model (res-jb covers gpt2-small). External download; skip loudly if absent.
    if model_id == "gpt2":
        try:
            dump("sae.json", compute_sae(m))
        except Exception as e:  # noqa: BLE001 — report and continue, never fake
            print(f"[interp] sae.json skipped: {e}")

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
