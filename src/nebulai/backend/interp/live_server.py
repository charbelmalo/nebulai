"""Live probe server for Internals #25 (Live Prompt Nebula).

A tiny stdlib-only HTTP server that runs the SAME numpy GPT-2 forward pass as
every offline trace bundle (gpt2_numpy.GPT2Numpy) on text the user types in
the viewer, and returns real per-(layer, position) logit-lens readouts:

  - top-1 token + its probability under the lens distribution
  - Shannon entropy of the FULL |V|-way lens distribution, in bits
    (absolute scale: 0 .. log2(V) = 15.617 bits for GPT-2's 50257 vocab)
  - KL(p_final(t) || p_lens(L,t)) in bits, same position, full softmax
  - the final next-token candidates at the last position

Nothing is precomputed and nothing is smoothed; every number is computed on
request from the resident float32 weights, with log-softmax/entropy/KL in
float64. Layer 12 is the model's own output head, so its row is a standing
identity check: top-1 matches the final prediction and KL is ~0 (bounded by
the float32 resid snapshot of the float64 stream; measured < 1e-4 bits).

Run:  python -m nebulai.backend.interp.live_server [--port 8123] [--model gpt2]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from .bundles import SAE_HOOK, SAE_REPO, compute_trace, load_sae_weights, sae_trace_for_prompt
from .gpt2_numpy import GPT2Numpy, _layernorm

# A typed prompt is short; cap the forward so one request can't queue seconds
# of matmuls behind it. The cap is disclosed in the response (`truncated`).
MAX_TOKENS = 96
# /live/trace returns the full (n_layer, n_head, T, T) attention tensor —
# payload grows as T², so its cap is tighter (T=64 ≈ 4–5 MB of JSON).
MAX_TRACE_TOKENS = 64


def live_forward(m: GPT2Numpy, text: str, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    """One real forward pass + per-layer logit-lens readout. Pure function so
    it can be verified without HTTP (see scratchpad verify script)."""
    ids = m.encode(text)
    if not ids:
        raise ValueError("empty prompt (tokenizes to zero tokens)")
    truncated = len(ids) > max_tokens
    ids = ids[:max_tokens]
    t0 = time.perf_counter()
    tr = m.forward(ids)
    T, nL, V = len(ids), m.n_layer, m.V
    g_f, b_f = m._g("ln_f.weight"), m._g("ln_f.bias")

    # final log-probs per position (float64 log-softmax over the full vocab)
    flg = tr.logits.astype(np.float64)  # (T, V)
    flp = flg - flg.max(axis=1, keepdims=True)
    flp -= np.log(np.exp(flp).sum(axis=1, keepdims=True))
    fp = np.exp(flp)

    ln2 = np.log(2.0)
    cells: list[list[list[Any]]] = []
    for L in range(nL + 1):
        # the model's own readout applied to the stream entering block L
        # (L == nL is the final residual — identical to tr.logits' path)
        lg = (_layernorm(tr.resid[L], g_f, b_f)[0] @ m.wte.T).astype(np.float64)
        lp = lg - lg.max(axis=1, keepdims=True)
        lp -= np.log(np.exp(lp).sum(axis=1, keepdims=True))
        p = np.exp(lp)
        ent = -(p * lp).sum(axis=1) / ln2  # (T,) bits
        kl = (fp * (flp - lp)).sum(axis=1) / ln2  # (T,) bits
        top = lg.argmax(axis=1)  # (T,)
        row = [
            [
                m.decode1(int(top[t])),
                round(float(p[t, top[t]]), 4),
                round(float(ent[t]), 3),
                round(float(kl[t]), 3),
            ]
            for t in range(T)
        ]
        cells.append(row)

    order = np.argsort(flp[T - 1])[::-1][:8]
    final_top = [[m.decode1(int(i)), round(float(np.exp(flp[T - 1, i])), 4)] for i in order]
    ms = (time.perf_counter() - t0) * 1000.0

    return {
        "model": m.model_id,
        "T": T,
        "n_layer": nL,
        "truncated": truncated,
        "max_tokens": max_tokens,
        "ms": round(ms, 1),
        "tokens": list(tr.token_strs),
        "final_top": final_top,
        "cells": cells,  # [layer 0..nL][pos] = [top1_str, p, entropy_bits, kl_bits]
        "meta": {
            "formula": (
                "lens_L(t) = ln_f(resid[L,t])·W_E^T (the model's own final LN + tied "
                "unembedding — the raw logit lens, no trained translator); "
                "H = −Σ p·log2 p over all |V| tokens; KL(p_final(t) ‖ p_lens(L,t)) bits, "
                "same position, full softmax (float64)"
            ),
            "entropy_max": round(float(np.log2(V)), 4),
            "vocab": V,
            "resid_note": (
                "resid[L] = residual stream ENTERING block L; row 12 = the final "
                "residual readout — its KL vs the model output is ~0 (float32 "
                "snapshot of a float64 stream; measured < 1e-4 bits, rounds to 0.000)"
            ),
            "pos0_note": (
                "position 0 rides GPT-2's massive-activation outlier — its early-layer "
                "lens rows read near-uniform. Real, not a bug."
            ),
        },
    }


def live_trace(m: GPT2Numpy, text: str, max_tokens: int = MAX_TRACE_TOKENS) -> dict[str, Any]:
    """One real forward serialized EXACTLY like an offline trace_<slug>.json
    (same producer: bundles.compute_trace), so every trace-driven viewer
    feature renders a typed prompt with zero driver changes. Pure function —
    verifiable without HTTP. Extra top-level keys (`ms`, `truncated`,
    `max_tokens`) ride alongside the bundle shape, never inside it."""
    ids = m.encode(text)
    if not ids:
        raise ValueError("empty prompt (tokenizes to zero tokens)")
    truncated = len(ids) > max_tokens
    if truncated:
        # GPT-2 BPE round-trips exactly: decode of the kept ids re-encodes to
        # the same ids, so compute_trace sees precisely the truncated tokens
        text = "".join(m.decode1(int(i)) for i in ids[:max_tokens])
    t0 = time.perf_counter()
    out = compute_trace(m, text)
    out["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    out["truncated"] = truncated
    out["max_tokens"] = max_tokens
    return out


# res-jb SAE weights, loaded lazily on the first /live/sae and kept resident
# (~150 MB of float32; the download is HF-cached from the offline bundle run).
_sae_cache: tuple[dict, dict, np.ndarray] | None = None


def _sae() -> tuple[dict, dict, np.ndarray]:
    global _sae_cache
    if _sae_cache is None:
        t0 = time.perf_counter()
        print(f"[live] loading SAE {SAE_REPO} @ {SAE_HOOK}…")
        _sae_cache = load_sae_weights()
        print(f"[live] SAE resident in {time.perf_counter() - t0:.1f}s")
    return _sae_cache


def live_sae(m: GPT2Numpy, text: str, max_tokens: int = MAX_TRACE_TOKENS) -> dict[str, Any]:
    """The res-jb SAE encoder run on a typed prompt's real residual stream,
    serialized EXACTLY like one trace of the offline sae_acts.json (same
    producer: bundles.sae_trace_for_prompt), so the Piano-Roll renders it with
    the offline code path. Pure function — verifiable without HTTP."""
    ids = m.encode(text)
    if not ids:
        raise ValueError("empty prompt (tokenizes to zero tokens)")
    truncated = len(ids) > max_tokens
    if truncated:
        # GPT-2 BPE round-trips exactly (see live_trace)
        text = "".join(m.decode1(int(i)) for i in ids[:max_tokens])
    cfg, t, sparsity = _sae()
    d_in = int(cfg["d_in"])
    if d_in != m.d:
        raise ValueError(f"SAE d_in {d_in} != model d {m.d} — the res-jb release is gpt2-only")
    t0 = time.perf_counter()
    trace = sae_trace_for_prompt(m, text, t, sparsity, int(cfg["hook_point_layer"]))
    return {
        "model": m.model_id,
        "sae_repo": SAE_REPO,
        "hook_point": SAE_HOOK,
        "hook_layer": int(cfg["hook_point_layer"]),
        "d_sae": int(cfg["d_sae"]),
        "ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "truncated": truncated,
        "max_tokens": max_tokens,
        "trace": trace,
    }


class _Handler(BaseHTTPRequestHandler):
    m: GPT2Numpy  # set by serve()
    lock: threading.Lock

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # CORS preflight for POST + JSON
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/live/health":
            self._send(
                200,
                {
                    "ok": True,
                    "model": self.m.model_id,
                    "n_layer": self.m.n_layer,
                    "d_model": self.m.d,
                    "vocab": self.m.V,
                    "max_tokens": MAX_TOKENS,
                    "max_trace_tokens": MAX_TRACE_TOKENS,
                    "sae_repo": SAE_REPO,
                    "sae_hook": SAE_HOOK,
                    "sae_loaded": _sae_cache is not None,
                },
            )
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:
        if self.path not in ("/live/forward", "/live/trace", "/live/sae"):
            self._send(404, {"error": f"unknown path {self.path}"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            text = req.get("text", "")
            if not isinstance(text, str) or not text:
                self._send(400, {"error": "body must be {\"text\": \"<non-empty prompt>\"}"})
                return
            # serialize forwards — concurrent numpy matmuls only fight for RAM
            with self.lock:
                if self.path == "/live/trace":
                    out = live_trace(self.m, text)
                elif self.path == "/live/sae":
                    out = live_sae(self.m, text)
                else:
                    out = live_forward(self.m, text)
            self._send(200, out)
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # keep the server alive; report honestly
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[live] {self.address_string()} {fmt % args}")


def serve(model_id: str = "gpt2", host: str = "127.0.0.1", port: int = 8123) -> None:
    t0 = time.perf_counter()
    print(f"[live] loading {model_id} weights (float32, resident)…")
    m = GPT2Numpy(model_id)
    print(f"[live] loaded in {time.perf_counter() - t0:.1f}s — {m.n_layer} layers, vocab {m.V}")
    _Handler.m = m
    _Handler.lock = threading.Lock()
    srv = ThreadingHTTPServer((host, port), _Handler)
    print(
        f"[live] serving on http://{host}:{port}  "
        "(health: /live/health, forward: POST /live/forward, trace: POST /live/trace, "
        "sae: POST /live/sae)"
    )
    srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    a = ap.parse_args()
    serve(a.model, a.host, a.port)
