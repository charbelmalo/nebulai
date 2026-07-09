"""Interp bundle producers — real quantities → static JSON for the viewer.

Each producer returns a JSON-safe dict with a `meta` block that stamps the exact
provenance (model, formula, params) so a viewer feature can display where its
numbers came from. Nothing here fabricates or smooths data; arrays are rounded
only for transport (documented per bundle) and the rounding precision is stated.

Bundles written by `write_bundles`:
  weights.json    — SVD spectra of every weight matrix           (#21 Weight Spectrum)
  fourier.json    — DFT of positional embeddings W_pe            (#1  Fourier Atlas)
  embed.json      — PCA projection of the token embedding W_E    (#15 Embedding Constellation)
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
    We = m.wte.astype(np.float64)  # (V, d)
    mu = We.mean(axis=0)
    Wc = We - mu  # mean-centered rows
    cov = Wc.T @ Wc  # (d, d) — symmetric PSD, tiny
    evals, evecs = np.linalg.eigh(cov)  # ascending eigenpairs
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    axes = evecs[:, :dims]  # (d, dims) top principal axes
    # deterministic sign per axis: make its largest-magnitude loading positive,
    # so re-exports don't flip the constellation left/right between runs.
    for j in range(dims):
        k = int(np.argmax(np.abs(axes[:, j])))
        if axes[k, j] < 0:
            axes[:, j] = -axes[:, j]
    coords = Wc @ axes  # (V, dims) principal-component scores
    evr = evals[:dims] / evals.sum()  # explained-variance ratio of each PC
    norms = np.linalg.norm(We, axis=1)  # exact per-token embedding magnitude
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
        "total_variance": round(float(evals.sum()), 3),
        "coords": [round(float(v), 3) for v in xy],  # flat 2N (PC1, PC2)
        "z": [round(float(v), 3) for v in z],  # PC3 (hover only)
        "norm": [round(float(v), 3) for v in norms],
        "lead_space": lead,  # 1 if the token string starts with a space
        "strs": strs,
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
