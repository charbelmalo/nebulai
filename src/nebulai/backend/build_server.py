"""Build server for the viewer's "Map builder" (Settings → Model Probing).

A tiny stdlib-only HTTP server that runs the REAL token-map pipeline — the
same `nebulai tokens` CLI that produced every artifact in out/ — as a
subprocess, and turns its stage prints (`[k/5] … [12.3s]`) into a status
object the viewer polls. Nothing is simulated: progress is the pipeline's own
stage output, and the finished artifact is whatever the CLI wrote to
out/<model>/nebulai.json (plus the refreshed out/index.json).

One build at a time (a second /build/start gets 409). The subprocess runs
with -u / PYTHONUNBUFFERED so stage lines arrive as they happen — without it
the pipe is block-buffered and status would stall until exit.

Stage model: seeing `[k/5]` means stage k just FINISHED, so pct = k/5 and the
current stage is the NEXT one. Stages are wildly uneven (UMAP is 15–30 min on
the full vocab; everything else is seconds), so within a stage the bar stays
put and `elapsed_s` — recomputed on every poll — is the liveness signal.

Run:  python -m nebulai.backend.build_server [--port 8124] [--out out]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# Small models with a known embedding key (tokens.py picks wte.weight /
# embed_in.weight / embed_tokens.weight). `interp` = GPT-2 family, the only
# architecture gpt2_numpy implements, so only those get the Internals page.
CURATED_MODELS: list[dict[str, Any]] = [
    {"id": "gpt2", "label": "GPT-2 (124M)", "interp": True},
    {"id": "distilgpt2", "label": "DistilGPT-2 (82M, distilled GPT-2)", "interp": True},
    {"id": "gpt2-medium", "label": "GPT-2 Medium (355M)", "interp": True},
    {"id": "EleutherAI/pythia-70m", "label": "Pythia-70M", "interp": False},
    {"id": "EleutherAI/pythia-160m", "label": "Pythia-160M", "interp": False},
    {"id": "HuggingFaceTB/SmolLM2-135M", "label": "SmolLM2-135M", "interp": False},
    {"id": "Qwen/Qwen2.5-0.5B", "label": "Qwen2.5-0.5B", "interp": False},
]

NAMERS = ["auto", "ollama", "openrouter", "anthropic", "none"]
SOURCES = [
    {"id": "hf", "label": "model weights (W_E — the model's own geometry)"},
    {"id": "api", "label": "API text embeddings (external embedder — labeled on the map)"},
]

# `[3/5] HDBSCAN: 208 clusters, 55% noise [4.2s]` -> (3, "HDBSCAN: … [4.2s]")
_STAGE_LINE = re.compile(r"^\[([1-5])/5\]\s+(.*\S)\s*$")

# stage k finished -> the pipeline is now in stage k+1 (viewer ProbeStage names)
_STAGE_AFTER = {0: "loading", 1: "reducing", 2: "clustering", 3: "naming", 4: "exporting", 5: "exporting"}

_MODEL_ID = re.compile(r"^[\w.\-]+(/[\w.\-]+)?$")


def parse_stage_line(line: str) -> tuple[int, str] | None:
    """Extract (k, message) from a `[k/5] …` pipeline stage line, else None.

    Pure function — the whole progress model hangs off it, so it's unit-tested
    against the exact prints in cli._run_tokens.
    """
    m = _STAGE_LINE.match(line.strip())
    return (int(m.group(1)), m.group(2)) if m else None


def dataset_id_for(model: str, source: str, embed_model: str | None = None) -> str:
    """out/ directory name for a build — mirrors cli.py's model.replace("/","__");
    api-embedding builds get their own dir so the W_E artifact is never clobbered."""
    base = model.replace("/", "__")
    if source == "api":
        return f"{base}__api-{(embed_model or 'embed').replace('/', '__')}"
    return base


def build_cmd(model: str, source: str, params: dict[str, Any]) -> list[str]:
    """argv for the real pipeline run. Pure function (unit-tested); list argv,
    never a shell string."""
    if not _MODEL_ID.match(model):
        raise ValueError(f"invalid model id {model!r}")
    if source not in ("hf", "api"):
        raise ValueError(f"invalid source {source!r} (hf|api)")
    p = params or {}
    cmd = [sys.executable, "-u", "-m", "nebulai", "tokens", "--model", model]
    if p.get("max_tokens"):
        cmd += ["--max-tokens", str(int(p["max_tokens"]))]
    if p.get("n_neighbors"):
        cmd += ["--n-neighbors", str(int(p["n_neighbors"]))]
    if p.get("seed") is not None:
        cmd += ["--seed", str(int(p["seed"]))]
    if p.get("min_cluster_size"):
        cmd += ["--min-cluster-size", str(int(p["min_cluster_size"]))]
    if p.get("min_samples"):
        cmd += ["--min-samples", str(int(p["min_samples"]))]
    if p.get("cluster_method"):
        if p["cluster_method"] not in ("leaf", "eom"):
            raise ValueError(f"invalid cluster_method {p['cluster_method']!r}")
        cmd += ["--cluster-method", p["cluster_method"]]
    if p.get("namer"):
        if p["namer"] not in NAMERS:
            raise ValueError(f"invalid namer {p['namer']!r}")
        cmd += ["--namer", p["namer"]]
    if p.get("edges"):
        if p["edges"] not in ("knn", "cluster", "none"):
            raise ValueError(f"invalid edges {p['edges']!r}")
        cmd += ["--edges", p["edges"]]
    if p.get("force"):
        cmd += ["--force"]
    if source == "api":
        cmd += ["--source", "api"]
        if p.get("embed_host"):
            cmd += ["--embed-host", str(p["embed_host"])]
        if p.get("embed_model"):
            cmd += ["--embed-model", str(p["embed_model"])]
        if p.get("embed_api"):
            if p["embed_api"] not in ("ollama", "openai"):
                raise ValueError(f"invalid embed_api {p['embed_api']!r}")
            cmd += ["--embed-api", p["embed_api"]]
    return cmd


def _idle_status() -> dict[str, Any]:
    return {
        "running": False,
        "model": None,
        "source": None,
        "stage": "idle",
        "stage_index": 0,
        "pct": 0.0,
        "message": "",
        "elapsed_s": 0.0,
        "log": [],
        "done": False,
        "error": None,
        "artifact": None,
        "dataset_id": None,
    }


class _BuildState:
    """The one running build. All mutation under the lock; GET /build/status
    reads a snapshot with elapsed_s recomputed so every poll shows liveness."""

    def __init__(self, out_root: Path) -> None:
        self.lock = threading.Lock()
        self.out_root = out_root
        self.proc: subprocess.Popen[str] | None = None
        self.cancelled = False
        self.t0 = 0.0
        self.status = _idle_status()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            s = dict(self.status)
            s["log"] = list(self.status["log"][-40:])  # poll payload: tail only
            if s["running"]:
                s["elapsed_s"] = round(time.time() - self.t0, 1)
            return s

    def start(self, model: str, source: str, params: dict[str, Any]) -> None:
        cmd = build_cmd(model, source, params)  # validates before we lock in
        cmd += ["--out", str(self.out_root)]  # pipeline writes where we serve from
        dsid = dataset_id_for(model, source, params.get("embed_model"))
        with self.lock:
            if self.status["running"]:
                raise RuntimeError(f"a build of {self.status['model']} is already running")
            self.cancelled = False
            self.t0 = time.time()
            self.status = _idle_status()
            self.status.update(
                running=True, model=model, source=source, stage="loading",
                message=f"starting {model}…", dataset_id=dsid,
            )
            self.proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        threading.Thread(target=self._pump, args=(self.proc, dsid), daemon=True).start()

    def _pump(self, proc: subprocess.Popen[str], dsid: str) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            with self.lock:
                log = self.status["log"]
                log.append(line)
                if len(log) > 200:
                    del log[: len(log) - 200]
                st = parse_stage_line(line)
                if st:
                    k, msg = st
                    self.status["stage_index"] = k
                    self.status["pct"] = round(k / 5, 2)
                    self.status["stage"] = _STAGE_AFTER[k]
                    self.status["message"] = msg
        rc = proc.wait()
        with self.lock:
            self.status["running"] = False
            self.status["elapsed_s"] = round(time.time() - self.t0, 1)
            if self.cancelled:
                self.status["stage"] = "idle"
                self.status["message"] = "cancelled"
            elif rc == 0 and self.status["stage_index"] == 5:
                self.status.update(
                    stage="done", pct=1.0, done=True,
                    artifact=f"{dsid}/nebulai.json",
                    message=f"build complete — out/{dsid}/nebulai.json",
                )
            else:
                tail = " | ".join(self.status["log"][-3:]) or f"exit code {rc}"
                self.status["stage"] = "error"
                self.status["error"] = f"pipeline exited {rc}: {tail}"

    def cancel(self) -> bool:
        with self.lock:
            proc = self.proc
            if not (proc and self.status["running"]):
                return False
            self.cancelled = True
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return True

    def models(self) -> dict[str, Any]:
        out = []
        for m in CURATED_MODELS:
            d = self.out_root / m["id"].replace("/", "__")
            cached = d / "reduced.params.json"
            out.append(
                {
                    **m,
                    "built": (d / "nebulai.json").exists(),
                    "cached_reduce": json.loads(cached.read_text()) if cached.exists() else None,
                }
            )
        return {"models": out, "namers": NAMERS, "sources": SOURCES}


class _Handler(BaseHTTPRequestHandler):
    state: _BuildState  # set by serve()

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
        if self.path == "/build/health":
            self._send(200, {"ok": True, "running": self.state.snapshot()["running"], "out": str(self.state.out_root)})
        elif self.path == "/build/status":
            self._send(200, self.state.snapshot())
        elif self.path == "/build/models":
            self._send(200, self.state.models())
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:
        if self.path == "/build/cancel":
            self._send(200, {"cancelled": self.state.cancel()})
            return
        if self.path != "/build/start":
            self._send(404, {"error": f"unknown path {self.path}"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            model = req.get("model", "")
            if not isinstance(model, str) or not model:
                self._send(400, {"error": 'body must be {"model": "<hf id>", "source"?, "params"?}'})
                return
            self.state.start(model, req.get("source", "hf"), req.get("params") or {})
            self._send(200, self.state.snapshot())
        except RuntimeError as e:  # already running
            self._send(409, {"error": str(e)})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # keep the server alive; report honestly
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[build] {self.address_string()} {fmt % args}")


def serve(host: str = "127.0.0.1", port: int = 8124, out: str = "out") -> None:
    out_root = (REPO_ROOT / out).resolve() if not Path(out).is_absolute() else Path(out)
    _Handler.state = _BuildState(out_root)
    srv = ThreadingHTTPServer((host, port), _Handler)
    print(
        f"[build] serving on http://{host}:{port}  (health: /build/health, "
        "models: /build/models, start: POST /build/start, status: /build/status, "
        f"cancel: POST /build/cancel) — artifacts under {out_root}"
    )
    srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    serve(a.host, a.port, a.out)
