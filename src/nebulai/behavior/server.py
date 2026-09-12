"""Loopback-only runner server for the Behavior page (plan §8.4).

The page is a static artifact by default: it reads `behavior.json` and works
with no server at all. This server is the *optional* half — it lets a local user
start a run from the browser and watch progress. Three properties make that
safe enough to ship:

* **Loopback only.** The socket binds `127.0.0.1` and the server refuses any
  `Host` that is not loopback. It is a local tool, not a service.
* **Paid runs require two separate acts.** `--allow-paid` is a launch-time
  decision made in a terminal, and each paid run still needs an explicit
  `approve: true` in its request body after the estimate has been fetched.
  A browser cannot spend money by itself.
* **Every response says what it is.** `/health` reports whether paid runs are
  permitted and what the store already holds, so the page never has to guess
  whether the absence of a number means "zero" or "not measured".
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..llm import BudgetError, RunBudget
from .contract import load_manifest
from .runner import Runner, estimate
from .store import TrialStore

#: Progress of the run currently in flight, if any. One at a time on purpose:
#: two concurrent runs against one SQLite store would interleave their spend
#: against a single ceiling, and neither would know it.
_STATE: dict[str, Any] = {"running": False, "arm": "", "done": 0, "total": 0, "spent_usd": 0.0}
_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    out_dir: str = "out/behavior"
    allow_paid: bool = False

    # -- plumbing ---------------------------------------------------------
    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # No CORS header. The page and the server are the same local origin in
        # the supported configuration; opening this up would turn a loopback
        # tool into something any page in the browser could drive.
        self.end_headers()
        self.wfile.write(body)

    def _guard(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost", "[::1]", "::1"):
            self._json(403, {"error": f"refused: Host {host!r} is not loopback"})
            return False
        return True

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default
        pass

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        if not self._guard():
            return
        u = urlparse(self.path)
        if u.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "time": datetime.now(UTC).isoformat(timespec="seconds"),
                    "allow_paid": self.allow_paid,
                    "out_dir": self.out_dir,
                    "studies": sorted(p.name for p in Path(self.out_dir).glob("*") if p.is_dir()),
                    # Who holds each store's writer lock. A run started from a
                    # terminal is invisible to this server's own `_STATE`, so
                    # without this the page would offer to start a run that is
                    # already in flight elsewhere and then report a refusal it
                    # could have predicted.
                    "writers": _writers(self.out_dir),
                },
            )
            return
        if u.path == "/progress":
            with _LOCK:
                self._json(200, dict(_STATE))
            return
        if u.path == "/estimate":
            q = parse_qs(u.query)
            try:
                m = load_manifest(q["manifest"][0])
                est = estimate(m, [q.get("arm", ["discovery"])[0]])
            except Exception as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(
                200,
                {
                    "n_trials": est.n_trials,
                    "n_paid_trials": est.n_paid_trials,
                    # null, not 0, when the price is unknown.
                    "est_cost_usd": est.est_cost_usd,
                    "est_seconds": est.est_seconds,
                    "est_bytes": est.est_bytes,
                    "unpriced_models": est.unpriced_models,
                    "text": est.render(),
                },
            )
            return
        self._json(404, {"error": f"no route {u.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._guard():
            return
        u = urlparse(self.path)
        if u.path != "/run":
            self._json(404, {"error": f"no route {u.path}"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            self._json(400, {"error": f"bad JSON: {exc}"})
            return

        with _LOCK:
            if _STATE["running"]:
                self._json(409, {"error": "a run is already in flight", "state": dict(_STATE)})
                return

        try:
            m = load_manifest(body["manifest"])
        except Exception as exc:
            self._json(400, {"error": str(exc)})
            return
        arm = body.get("arm", "discovery")
        est = estimate(m, [arm] if arm != "canary" else [])
        if est.n_paid_trials:
            if not self.allow_paid:
                self._json(
                    403,
                    {
                        "error": (
                            "this run includes paid trials and the server was "
                            "started without --allow-paid. That is a terminal "
                            "decision, not a browser one."
                        ),
                        "n_paid_trials": est.n_paid_trials,
                        "est_cost_usd": est.est_cost_usd,
                    },
                )
                return
            if not body.get("approve"):
                self._json(
                    402,
                    {
                        "error": "paid run not approved",
                        "estimate": est.render(),
                        "n_paid_trials": est.n_paid_trials,
                        "est_cost_usd": est.est_cost_usd,
                        "how": "resend with approve: true after showing this estimate",
                    },
                )
                return

        t = threading.Thread(
            target=_execute, args=(m, arm, self.out_dir, int(body.get("limit") or 0) or None),
            daemon=True,
        )
        with _LOCK:
            _STATE.update({"running": True, "arm": arm, "done": 0, "total": est.n_trials,
                           "spent_usd": 0.0, "error": "", "halted": ""})
        t.start()
        self._json(202, {"started": True, "arm": arm, "estimate": est.render()})


def _execute(m: Any, arm: str, out_dir: str, limit: int | None) -> None:
    store: TrialStore | None = None

    def on_progress(p: dict[str, Any]) -> None:
        with _LOCK:
            _STATE.update({"done": p["done"], "total": p["total"], "spent_usd": p["spent_usd"]})

    # Everything, setup included, inside the try. This runs on a daemon thread
    # whose only channel back to the page is `_STATE`: an exception raised
    # before the try — opening the store on a read-only directory, a preflight
    # that trips the ceiling — would kill the thread with `running` still True,
    # and the page would show a run in flight forever with no reason given. A
    # failure that cannot be reported is worse than the failure.
    try:
        store = TrialStore(Path(out_dir) / m.study_id / "trials.sqlite")
        budget = RunBudget(m.max_cost_usd, label=f"behavior:{m.study_id}")
        est = estimate(m, [arm] if arm != "canary" else [])
        if est.n_paid_trials:
            budget.preflight(est.est_cost_usd or 0.0, est.n_paid_trials, "paid arms")
            budget.approve()  # the HTTP layer already required an explicit approval

        res = Runner(m, store, budget=budget, progress=on_progress).run(arm, limit=limit)
        with _LOCK:
            _STATE.update(
                {
                    "running": False,
                    "done": res.completed,
                    "errors": res.errors,
                    "halted": res.halted,
                    "not_run": res.not_run,
                    "spent_usd": res.spent_usd,
                }
            )
    except BudgetError as exc:
        with _LOCK:
            _STATE.update({"running": False, "halted": str(exc)})
    except Exception as exc:  # surfaced, never swallowed
        with _LOCK:
            _STATE.update({"running": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if store is not None:
            store.close()


def _writers(out_dir: str) -> dict[str, dict]:
    """Per-study writer-lock state, read-only and never fatal.

    A store that cannot be opened (mid-creation, permissions, a directory with
    no sqlite in it yet) is simply absent from the map rather than an error:
    /health exists to be reachable.
    """
    out: dict[str, dict] = {}
    for d in sorted(Path(out_dir).glob("*")):
        f = d / "trials.sqlite"
        if not f.exists():
            continue
        try:
            with TrialStore(f) as s:
                lock = s.writer_lock()
        except Exception:
            continue
        if lock:
            out[d.name] = lock
    return out


def serve(out_dir: str, *, host: str = "127.0.0.1", port: int = 8765, allow_paid: bool = False) -> None:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(
            f"refusing to bind {host!r}: the behavior runner server is loopback-only."
        )
    Handler.out_dir = out_dir
    Handler.allow_paid = allow_paid
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"behavior runner on http://{host}:{port}  (paid runs "
          f"{'ALLOWED' if allow_paid else 'refused'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
