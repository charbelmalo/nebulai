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

from . import intervene as _iv
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


# An intervened generation is a forward pass per token with no KV cache, run
# twice (baseline + intervened). The cap keeps one typed request from holding
# the server's single lock for a minute; it is disclosed in the response.
MAX_GEN_TOKENS = 24


def live_intervene(
    m: GPT2Numpy,
    req: dict[str, Any],
    *,
    directions: dict[str, Any] | None = None,
    max_tokens: int = MAX_TRACE_TOKENS,
) -> dict[str, Any]:
    """`POST /live/intervene` — run one prompt with and without one verb.

    The allow-list is `intervene.build()` and nothing else: an unknown verb, an
    unknown key or a missing field comes back as a 400 with the reason, never
    as a default that turns a typo into a different experiment. This is the
    D6 boundary in the network layer — there is no endpoint here that emits a
    modified checkpoint, and adding one would have to delete
    `tests/test_intervene.py::test_no_weight_export`.

    `ablate` and `add` need a direction, and the server resolves them ONLY from
    the directions the caller's dataset actually shipped. A client cannot post
    a raw 768-number vector: that would let the browser run an arbitrary
    steering experiment whose provenance is a text box, and every figure this
    project exports carries a protocol string that has to mean something.
    """
    prompt = req.get("prompt", "")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError('body must be {"prompt": "…", "intervention": {…}}')
    ids = m.encode(prompt)
    if not ids:
        raise ValueError("empty prompt (tokenizes to zero tokens)")
    truncated = len(ids) > max_tokens
    if truncated:
        prompt = "".join(m.decode1(int(i)) for i in ids[:max_tokens])

    gen = int(req.get("max_tokens", 16))
    gen_capped = gen > MAX_GEN_TOKENS
    gen = max(1, min(gen, MAX_GEN_TOKENS))

    spec = req.get("intervention")
    if not isinstance(spec, dict):
        raise ValueError("intervention must be an object naming one of " + str(list(_iv.VERBS)))

    def resolve(did: str):
        if not directions:
            raise ValueError(
                f"this server has no directions loaded, so it cannot run {did!r}. "
                f"Start it with --directions <directions.json>; refusing rather "
                f"than accepting a vector from the wire."
            )
        d = directions.get(did)
        if d is None:
            raise ValueError(f"unknown direction {did!r}; have {sorted(directions)}")
        return d

    iv = _iv.build(spec, resolve_direction=resolve, n_layer=m.n_layer)

    sae_tensors = None
    if iv.verb == "clamp":
        cfg, tensors, _sp = _sae()
        sae_tensors = tensors
        if iv.sae_repo not in (None, SAE_REPO) or iv.sae_hook not in (None, SAE_HOOK):
            raise ValueError(
                f"this server serves exactly one SAE ({SAE_REPO} @ {SAE_HOOK}); "
                f"it will not pretend a clamp on another dictionary is the same "
                f"experiment"
            )

    t0 = time.perf_counter()
    out = _iv.run(m, prompt, iv, max_tokens=gen, sae=sae_tensors, targets=req.get("targets"))
    out["model"] = m.model_id
    out["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    out["truncated"] = truncated
    out["max_tokens"] = gen
    out["max_tokens_capped"] = gen_capped
    out["verbs"] = list(_iv.VERBS)
    out["d6"] = (
        "measured, never exported: this endpoint runs inference-time hooks and "
        "there is no path here that writes a modified checkpoint"
    )
    return out

# ── BEGIN /live/place (Attractors P2, §3.4) ─────────────────────────────────
# Everything this endpoint needs lives between these two markers, including its
# own lazy model cache, so that merging a sibling endpoint into this file is a
# mechanical insertion rather than a three-way reconciliation.
#
# `POST /live/place {space_id, texts}` -> `{space_id, layer, coords, fidelity}`.
# The persona model is a *different* model from the GPT-2 the rest of this
# server runs, so it gets its own cache: loaded on the first place request and
# kept resident, exactly like the SAE weights above.

#: The out-root persona spaces are read from, and an optional local weights
#: directory for a space whose revision is "local". Both set by `serve()`.
_place_root: str = "out"
_place_local_dir: str | None = None

#: `{(model_id, revision): LlamaNumpy}` — one entry in practice, but keyed so a
#: server asked for two spaces at different model sizes does not silently serve
#: the second from the first's weights.
_place_models: dict[tuple[str, str], Any] = {}
#: `{space_id: PersonaSpace}`
_place_spaces: dict[str, Any] = {}

#: A place request is one prompt per point; cap the batch so a single request
#: cannot occupy the server for minutes. Disclosed in the response.
MAX_PLACE_TEXTS = 64
MAX_PLACE_TOKENS = 512


def _place_space(space_id: str):
    from ..persona import read_space

    if space_id not in _place_spaces:
        from pathlib import Path

        _place_spaces[space_id] = read_space(space_id, Path(_place_root) / "persona")
    return _place_spaces[space_id]


def _place_model(space):
    from .llama_numpy import LlamaNumpy

    key = (space.model, space.revision)
    if key not in _place_models:
        t0 = time.perf_counter()
        print(f"[live] loading persona model {space.model} @ {space.revision}…")
        if space.revision == "local":
            if not _place_local_dir:
                raise ValueError(
                    f"space {space.space_id} was built from local weights and this "
                    f"server was not told where they are; start it with "
                    f"--persona-local-dir. Substituting the hub's copy of "
                    f"{space.model!r} would be a different set of bytes under the "
                    f"same name."
                )
            m = LlamaNumpy(space.model, local_dir=_place_local_dir)
        else:
            m = LlamaNumpy(space.model, revision=space.revision)
            if m.revision != space.revision:
                raise ValueError(
                    f"asked for {space.model} @ {space.revision}, got {m.revision}"
                )
        print(f"[live] persona model resident in {time.perf_counter() - t0:.1f}s")
        _place_models[key] = m
    return _place_models[key]


def live_place(space_id: Any, texts: Any) -> dict[str, Any]:
    """Place free text into a built persona space. Pure function, HTTP-free.

    Each text is rendered through the model's own chat template as a *user*
    turn with no system prompt, run to its last token, read at the space's
    pinned layer, and projected through the space's fixed basis. That is
    deterministic end to end — no sampling, no fitted reducer — which is why
    the fidelity is `deterministic` and not an estimate.
    """
    if not isinstance(space_id, str) or not space_id:
        raise ValueError('body must be {"space_id": "<id>", "texts": ["…"]}')
    if not isinstance(texts, list) or not texts or not all(
        isinstance(t, str) and t for t in texts
    ):
        raise ValueError("`texts` must be a non-empty list of non-empty strings")
    if len(texts) > MAX_PLACE_TEXTS:
        raise ValueError(f"{len(texts)} texts exceeds the per-request cap of {MAX_PLACE_TEXTS}")

    space = _place_space(space_id)
    model = _place_model(space)
    rendered = [
        model.apply_chat_template([{"role": "user", "content": t}]) for t in texts
    ]
    ids = [model.encode(r)[-MAX_PLACE_TOKENS:] for r in rendered]
    t0 = time.perf_counter()
    acts = model.capture_resid(ids, [space.layer], batch_size=8)[space.layer]
    coords = space.project(acts)[:, :2]
    return {
        "space_id": space.space_id,
        "model": space.model,
        "revision": space.revision,
        "layer": space.layer,
        "coords": [[round(float(x), 5), round(float(y), 5)] for x, y in coords],
        "fidelity": "deterministic",
        # The verdict travels with every placement: a point drawn in a space
        # whose PC1 did not clear its null is still a real projection, but the
        # card that shows it has to say so, and it cannot be the default frame.
        "verdict": space.control.verdict,
        "pc1_evr": round(space.control.pc1_evr, 6),
        "pc1_evr_null_p95": round(space.control.pc1_evr_null_p95, 6),
        "ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "max_texts": MAX_PLACE_TEXTS,
    }


# ── END /live/place ─────────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    m: GPT2Numpy  # set by serve()
    lock: threading.Lock
    directions: dict[str, Any] = {}  # id -> Direction, empty unless --directions

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
                    "intervene_verbs": list(_iv.VERBS),
                    "directions": sorted(self.directions),
                    "max_gen_tokens": MAX_GEN_TOKENS,
                },
            )
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:
        if self.path not in ("/live/forward", "/live/trace", "/live/sae", "/live/intervene", "/live/place"):
            self._send(404, {"error": f"unknown path {self.path}"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            if self.path == "/live/intervene":
                # serialize: an intervened generation is 2 x gen forward passes
                with self.lock:
                    out = live_intervene(self.m, req, directions=self.directions)
                self._send(200, out)
                return
            # ── BEGIN /live/place dispatch ───────────────────────────────
            # Its body is {space_id, texts}, not {text}, so it branches before
            # the shared prompt validation below.
            if self.path == "/live/place":
                with self.lock:
                    out = live_place(req.get("space_id"), req.get("texts"))
                self._send(200, out)
                return
            # ── END /live/place dispatch ────────────────────────────────
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


def _load_directions(path: str | None, d_model: int) -> dict[str, Any]:
    """Load the directions this server will let a client name.

    Width is checked here rather than at request time so the refusal happens
    once, at boot, with the whole table printed — the same refusal
    `check_dimensionality` makes on import, in the place a user can see it.
    """
    if not path:
        return {}
    from ...backend.directions import Direction, read_directions

    doc = read_directions(path)
    if doc is None:
        print(f"[live] no directions at {path} — /live/intervene will refuse add/ablate")
        return {}
    out: dict[str, Any] = {}
    for obj in doc.get("directions", []):
        dd = Direction.from_json(obj)
        if dd.d != d_model:
            print(f"[live] skipping {dd.id}: {dd.d} wide, this model's stream is {d_model}")
            continue
        out[dd.id] = dd
    print(f"[live] {len(out)} direction(s) available to /live/intervene: {sorted(out)}")
    return out


def serve(
    model_id: str = "gpt2",
    host: str = "127.0.0.1",
    port: int = 8123,
    directions_path: str | None = None,
    *,
    persona_root: str = "out",
    persona_local_dir: str | None = None,
) -> None:
    # ── BEGIN /live/place wiring ───────────────────────────────────────
    global _place_root, _place_local_dir
    _place_root, _place_local_dir = persona_root, persona_local_dir
    # ── END /live/place wiring ─────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"[live] loading {model_id} weights (float32, resident)…")
    m = GPT2Numpy(model_id)
    print(f"[live] loaded in {time.perf_counter() - t0:.1f}s — {m.n_layer} layers, vocab {m.V}")
    _Handler.m = m
    _Handler.lock = threading.Lock()
    _Handler.directions = _load_directions(directions_path, m.d)
    srv = ThreadingHTTPServer((host, port), _Handler)
    print(
        f"[live] serving on http://{host}:{port}  "
        "(health: /live/health, forward: POST /live/forward, trace: POST /live/trace, "
        "sae: POST /live/sae, intervene: POST /live/intervene, place: POST /live/place)"
    )
    srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument(
        "--directions",
        default=None,
        help="path to a directions.json; /live/intervene's add and ablate "
        "verbs resolve ids against it and refuse anything not in it",
    )
    # ── BEGIN /live/place flags ────────────────────────────────────────
    ap.add_argument(
        "--persona-root",
        default="out",
        help="output root holding persona/<space_id>/space.json (default: out)",
    )
    ap.add_argument(
        "--persona-local-dir",
        default=None,
        help="weights directory for a persona space whose revision is 'local'; "
        "without it such a space is refused rather than served from a "
        "same-named hub checkout",
    )
    # ── END /live/place flags ──────────────────────────────────────────
    a = ap.parse_args()
    serve(
        a.model,
        a.host,
        a.port,
        a.directions,
        persona_root=a.persona_root,
        persona_local_dir=a.persona_local_dir,
    )
