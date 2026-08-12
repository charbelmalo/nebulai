"""SessionSeer's HTTP + SSE server. Port 8125, beside the build server's 8124.

Stdlib only, same shape as `backend/build_server.py`: a `ThreadingHTTPServer`, a
`_Handler` with `do_GET`/`do_POST`/`do_OPTIONS`, permissive CORS so the viewer
can reach it from the dev server or from a `file://` build.

SSE rather than polling because the thing being watched is a live agent: a
2-second poll on a run that changes state four times in a second shows a
researcher a slideshow of a trajectory instead of the trajectory. Each client
gets its own bounded queue; a client that stops reading is dropped rather than
allowed to back-pressure the collector, because the collector falling behind
would distort the very timings we are recording.

Read paths never block on a write lock: they go through `EventStore`, whose
SQLite handle is in WAL mode. A live run and a viewer refresh are readers and
writers of the same file at the same time by design.

Run:  python -m nebulai.seer.server [--port 8125] [--root ~/.nebulai/seer]
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .analysis import analyze
from .attach import CodexAttachment, ProtocolMismatch
from .collector import SpoolCollector
from .compare import compare, summarize_refusals
from .contract import (
    SCHEMA_VERSION,
    CaptureMode,
    Event,
    EventType,
    Fidelity,
    Source,
)
from .export import export
from .reconcile import reconcile_codex
from .recover import recover_orphans
from .redaction import ContentLevel, parse_level
from .reducer import reduce_run
from .runner import Runner
from .store import DEFAULT_ROOT, EventStore

DEFAULT_PORT = 8125
#: Per-client SSE backlog. Small on purpose: a viewer that cannot keep up should
#: be disconnected and reconnect fresh, not silently receive a delayed stream it
#: will render as if it were live.
CLIENT_QUEUE_MAX = 512


class Broadcaster:
    """Fan events out to SSE subscribers. Drops slow clients, never blocks."""

    def __init__(self) -> None:
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self.dropped = 0

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=CLIENT_QUEUE_MAX)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, payload: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                # The client is not reading. Cut it loose and count it — a
                # silently-lagging viewer is worse than a disconnected one,
                # because the researcher cannot tell it is lagging.
                self.dropped += 1
                self.unsubscribe(q)


class SeerState:
    def __init__(self, root: Path, *, watch: bool = False) -> None:
        self.store = EventStore(root)
        self.bus = Broadcaster()
        self.runners: dict[str, Runner] = {}
        #: attached captures, keyed the same way. Kept apart from
        #: `runners` because cancelling one means interrupting a turn,
        #: not killing a process we own.
        self.attachments: dict[str, CodexAttachment] = {}
        #: one import pass at a time. Two concurrent passes would each read the
        #: store before the other wrote, and the dedup key would come back
        #: clean for both — the exact double this pass exists to prevent.
        self._reconciling = False
        self._lock = threading.Lock()
        #: Runs a previous process was capturing when it died. Swept here,
        #: before anything can read them: a run left in `running` renders as
        #: live forever, and the first thing a researcher does with a live run
        #: is wait for it to finish.
        self.recovered = recover_orphans(self.store)
        for r in self.recovered:
            print(
                f"[seer] recovered {r['run_id']}: was {r['was']}, "
                f"{r['n_events']} events — capture process is gone, "
                "recorded as interrupted"
            )
        self.collector: SpoolCollector | None = None
        if watch:
            self.collector = SpoolCollector(
                self.store, self.store.root, on_events=self._on_events
            )
            self.collector.start()

    def _on_events(self, events: list[Event]) -> None:
        # Observed runs reach the viewer through the same SSE frames as driven
        # ones, so the page needs no second code path for "runs I did not start".
        for e in events:
            self._on_event(e)

    # ── launching ────────────────────────────────────────────────────────

    def start(self, req: dict[str, Any]) -> dict[str, Any]:
        agent = str(req.get("agent") or "").lower()
        prompt = str(req.get("prompt") or "")
        if not agent or not prompt:
            raise ValueError('body must be {"agent": "codex|claude|hermes", "prompt": "…"}')

        runner = Runner(
            agent,
            prompt,
            store=self.store,
            cwd=req.get("cwd"),
            model=req.get("model"),
            extra_args=req.get("extra_args") or [],
            keep_reasoning=bool(req.get("keep_reasoning")),
            label=req.get("label"),
            on_event=self._on_event,
        )
        with self._lock:
            self.runners[runner.run_id] = runner
        # Addressable before the subprocess has said anything: we are about to
        # return this id to a client that will ask for it on the next line.
        self.store.register_run(
            runner.run_id,
            agent=agent,
            agent_version=runner.adapter.agent_version,
            capture_mode=runner.adapter.capture_mode.value,
            label=runner.label,
            repo=runner.adapter.repo,
            started_at=time.time(),
        )

        def _go() -> None:
            try:
                runner.run(timeout_s=req.get("timeout_s"))
            finally:
                # Out of `runners` the moment it stops running, the way an
                # attachment leaves `attachments`. `/seer/health` was reading
                # `proc.poll()` to decide what was active, so the leak never
                # showed there — it only surfaced once something *else* asked
                # the same dict whether a run was live, and every finished run
                # said yes.
                with self._lock:
                    self.runners.pop(runner.run_id, None)
                self.bus.publish(
                    {"kind": "run_finished", "run_id": runner.run_id,
                     "view": runner.reducer.finalize().to_dict()}
                )

        threading.Thread(target=_go, daemon=True, name=f"seer-{runner.run_id}").start()
        return {"run_id": runner.run_id, "agent": agent, "state": "starting"}

    def attach(self, req: dict[str, Any]) -> dict[str, Any]:
        """Capture Codex through its app-server. With a prompt we drive one
        turn; without one we watch a daemon, if there is one to watch."""
        prompt = req.get("prompt")
        att = CodexAttachment(
            store=self.store,
            sock=req.get("sock"),
            cwd=req.get("cwd"),
            keep_reasoning=bool(req.get("keep_reasoning")),
            label=req.get("label"),
            on_event=self._on_event,
        )
        try:
            att.open(prefer_daemon=not req.get("no_daemon"))
        except ProtocolMismatch as exc:
            # `open` already wrote the refusal into the log under this id; the
            # HTTP error carries the pointer so the viewer can show the record
            # rather than only a toast that disappears.
            exc.run_id = att.run_id  # type: ignore[attr-defined]
            raise
        with self._lock:
            self.attachments[att.run_id] = att

        timeout_s = float(req.get("timeout_s") or 900.0)

        def _go() -> None:
            try:
                if prompt is None:
                    att.watch(timeout_s)
                else:
                    att.drive(str(prompt), model=req.get("model"), timeout_s=timeout_s)
            finally:
                with self._lock:
                    self.attachments.pop(att.run_id, None)
                self.bus.publish(
                    {"kind": "run_finished", "run_id": att.run_id,
                     "view": att.reducer.finalize().to_dict()}
                )

        threading.Thread(target=_go, daemon=True, name=f"seer-attach-{att.run_id}").start()
        return {
            "run_id": att.run_id, "agent": "codex", "state": "starting",
            "transport": att.transport, "protocol": att.protocol,
            "driving": prompt is not None,
        }

    def reconcile(self, req: dict[str, Any]) -> dict[str, Any]:
        """Import persisted Codex threads in the background.

        Not synchronous: a pass over twenty-five threads takes tens of seconds
        of `thread/read`, and an HTTP call that holds the page for that long
        looks exactly like one that has hung. Each imported run is announced on
        the bus as it lands, in the same frame a finished capture uses, so the
        viewer needs no second code path to show them arriving.
        """
        with self._lock:
            if self._reconciling:
                raise ValueError("a reconcile pass is already running")
            self._reconciling = True

        limit = int(req.get("limit") or 25)
        since_days = req.get("since_days")
        since = (
            time.time() - float(since_days) * 86400.0
            if since_days is not None else None
        )

        def _go() -> None:
            try:
                report = reconcile_codex(
                    store=self.store, limit=limit,
                    only_cwd=req.get("only_cwd"), since=since,
                    keep_reasoning=bool(req.get("keep_reasoning")),
                )
                for imp in report.imported:
                    self.bus.publish({"kind": "run_finished", "run_id": imp.run_id,
                                      "view": imp.view.to_dict()})
                self.bus.publish({"kind": "reconcile_done",
                                  "n_seen": report.n_seen,
                                  "n_imported": len(report.imported),
                                  "n_skipped": len(report.skipped),
                                  "failed": report.failed})
            except Exception as exc:  # report, never swallow
                self.bus.publish({"kind": "reconcile_failed",
                                  "error": f"{type(exc).__name__}: {exc}"})
            finally:
                with self._lock:
                    self._reconciling = False

        threading.Thread(target=_go, daemon=True, name="seer-reconcile").start()
        return {"started": True, "limit": limit, "agent": "codex"}

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            att = self.attachments.get(run_id)
            if att is not None:
                # `stop`, not `interrupt`: a watcher has no turn to interrupt,
                # and in proxy mode the turn belongs to whoever is driving the
                # thread — cancelling our observation must not cancel their work.
                att.stop()
                return True
            runner = self.runners.get(run_id)
        return runner.cancel() if runner else False

    def delete(self, run_id: str) -> dict[str, Any]:
        """Remove a run entirely, unless we are still capturing it.

        Refusing a live run is not squeamishness: the runner holds an open
        append handle and will keep writing, so a delete now produces a run
        that comes back a few lines shorter and missing its beginning. Cancel
        it first, then delete what it left.
        """
        with self._lock:
            live = run_id in self.runners or run_id in self.attachments
        if live:
            raise ValueError(
                f"{run_id} is still being captured — cancel it first, "
                "then delete what it recorded"
            )
        gone = self.store.delete_run(run_id)
        self.bus.publish({"kind": "run_deleted", **gone})
        return gone

    def _on_event(self, e: Event) -> None:
        self.bus.publish({"kind": "event", "event": e.to_dict()})

    # ── reading ──────────────────────────────────────────────────────────

    def view(self, run_id: str, now: float | None = None) -> dict[str, Any] | None:
        summary = self.store.get_run(run_id)
        if summary is None:
            return None
        v = reduce_run(run_id, self.store.read(run_id), now=now)
        d = v.to_dict()
        d["summary"] = summary.to_dict()
        return d

    def analysis(self, run_id: str) -> dict[str, Any] | None:
        if self.store.get_run(run_id) is None:
            return None
        events = list(self.store.read(run_id))
        return analyze(reduce_run(run_id, events), events)

    def annotate(self, req: dict[str, Any]) -> dict[str, Any]:
        """Append a human note to the run's own log.

        Not a side table: the note has to survive export, replay and deletion
        with the events it is about, and the append-only log is the only place
        in this system where that is already true.
        """
        run_id = str(req.get("run_id") or "")
        text = str(req.get("text") or "").strip()
        if not text:
            raise ValueError("an annotation needs `text`")
        summary = self.store.get_run(run_id)
        if summary is None:
            raise ValueError(f"unknown run {run_id!r}")
        e = Event(
            event_type=EventType.ANNOTATION_ADDED,
            source=Source(
                agent=summary.agent,
                agent_version=summary.agent_version or "unknown",
                adapter="human",
                adapter_version="1",
                capture_mode=CaptureMode(summary.capture_mode or "driven"),
                # native to its author: a person reported this, and no part of
                # it was derived by us
                fidelity=Fidelity.NATIVE,
            ),
            run_id=run_id,
            session_id=run_id,
            span_id=req.get("span_id"),
            ts=float(req["ts"]) if req.get("ts") is not None else time.time(),
            native_type="human.annotation",
            payload={
                "text": text,
                "tags": [str(t) for t in (req.get("tags") or [])],
                "author": req.get("author"),
            },
            # `content` because it is free text, `author_supplied` because it
            # is the one kind of text in the log that a person typed knowing it
            # was going into the log.
            privacy={"content_level": ContentLevel.CONTENT.value,
                     "author_supplied": True},
        )
        self.store.append(e)
        self._on_event(e)
        return {"ok": True, "event_id": e.event_id, "ts": e.ts}

    def comparison(self, run_ids: list[str]) -> dict[str, Any]:
        views = []
        for rid in run_ids:
            if self.store.get_run(rid) is None:
                raise ValueError(f"unknown run {rid!r}")
            views.append(reduce_run(rid, self.store.read(rid)))
        c = compare(views)
        d = c.to_dict()
        d["summary"] = summarize_refusals(c)
        return d


class _Handler(BaseHTTPRequestHandler):
    state: SeerState

    protocol_version = "HTTP/1.1"

    # ── plumbing ─────────────────────────────────────────────────────────

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = _dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, code: int, body: str, ctype: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(self, code: int, raw: bytes, ctype: str,
                    filename: str | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[seer] {self.address_string()} {fmt % args}")

    # ── routes ───────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path.rstrip("/") or "/"
        st = self.state

        if path == "/seer/health":
            self._send(200, {
                "ok": True,
                "schema_version": SCHEMA_VERSION,
                "root": str(st.store.root),
                "runs": len(st.store.list_runs(limit=1000)),
                "active": [r for r, run in st.runners.items()
                           if run.proc and run.proc.poll() is None],
                "sse_clients_dropped": st.bus.dropped,
                # `null` means nothing is watching the spool — which looks
                # exactly like "you have had no sessions" from the outside, so
                # the page has to be able to tell the two apart.
                "observing": st.collector.status() if st.collector else None,
            })
            return

        if path == "/seer/runs":
            limit = int((q.get("limit") or ["100"])[0])
            agent = (q.get("agent") or [None])[0]
            self._send(200, {
                "runs": [r.to_dict() for r in st.store.list_runs(limit, agent)]
            })
            return

        if path.startswith("/seer/run/"):
            rest = path[len("/seer/run/"):]
            run_id, _, tail = rest.partition("/")
            if tail == "events":
                # An unknown run is not a run with no events. Returning [] would
                # let a typo render as a captured-but-silent session.
                if st.store.get_run(run_id) is None:
                    self._send(404, {"error": f"unknown run {run_id!r}"})
                    return
                since = int((q.get("since") or ["0"])[0])
                events = [e.to_dict() for e in st.store.read(run_id, since_line=since)]
                self._send(200, {"run_id": run_id, "since": since, "events": events})
                return
            if tail == "analysis":
                doc = st.analysis(run_id)
                if doc is None:
                    self._send(404, {"error": f"unknown run {run_id!r}"})
                    return
                self._send(200, doc)
                return
            view = st.view(run_id)
            if view is None:
                self._send(404, {"error": f"unknown run {run_id!r}"})
                return
            self._send(200, view)
            return

        if path == "/seer/compare":
            ids = [r for r in (q.get("runs") or [""])[0].split(",") if r]
            try:
                self._send(200, st.comparison(ids))
            except ValueError as e:
                self._send(400, {"error": str(e)})
            return

        if path == "/seer/export":
            run_id = (q.get("run_id") or [""])[0]
            fmt = (q.get("format") or ["jsonl"])[0]
            if st.store.get_run(run_id) is None:
                self._send(404, {"error": f"unknown run {run_id!r}"})
                return
            events = list(st.store.read(run_id))
            try:
                keep = parse_level(r) if (r := (q.get("redact") or [""])[0]) else None
                body, ctype, filename = export(
                    fmt, reduce_run(run_id, events), events, keep
                )
            except ValueError as e:
                self._send(400, {"error": str(e)})
                return
            except RuntimeError as e:
                # a format we support in principle and cannot produce here.
                # 501, not 500: the request was fine, this install is not.
                self._send(501, {"error": str(e)})
                return
            self._send_bytes(200, body, ctype, filename)
            return

        if path == "/seer/live":
            self._sse((q.get("run_id") or [None])[0])
            return

        self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, {"error": f"bad JSON body: {e}"})
            return

        if path == "/seer/start":
            try:
                self._send(200, self.state.start(req))
            except ValueError as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # keep serving; report honestly
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return

        if path == "/seer/attach":
            try:
                self._send(200, self.state.attach(req))
            except ProtocolMismatch as e:
                # 409, not 500: the request was fine and the server is fine.
                # This install's Codex and this adapter disagree, and the run
                # id in the body points at the record that says how.
                self._send(409, {"error": str(e), "run_id": getattr(e, "run_id", None)})
            except ValueError as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # keep serving; report honestly
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return

        if path == "/seer/reconcile":
            try:
                self._send(200, self.state.reconcile(req))
            except ProtocolMismatch as e:
                self._send(409, {"error": str(e)})
            except ValueError as e:
                self._send(409, {"error": str(e)})
            except Exception as e:  # keep serving; report honestly
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return

        if path == "/seer/cancel":
            run_id = str(req.get("run_id") or "")
            self._send(200, {"cancelled": self.state.cancel(run_id)})
            return

        if path == "/seer/annotate":
            try:
                self._send(200, self.state.annotate(req))
            except ValueError as e:
                self._send(400, {"error": str(e)})
            return

        if path == "/seer/reindex":
            n = self.state.store.reindex(req.get("run_id"))
            self._send(200, {"reindexed": n})
            return

        self._send(404, {"error": f"unknown path {self.path}"})

    def do_DELETE(self) -> None:
        """`DELETE /seer/run/<id>` — the whole run, log and index.

        Its own verb rather than a POST: a client that has to choose DELETE
        cannot reach it by resubmitting a form, and a proxy or a browser that
        replays GETs will never replay this.
        """
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        if not path.startswith("/seer/run/"):
            self._send(404, {"error": f"unknown path {self.path}"})
            return
        run_id = path[len("/seer/run/"):]
        try:
            self._send(200, self.state.delete(run_id))
        except KeyError:
            self._send(404, {"error": f"unknown run {run_id!r}"})
        except ValueError as e:
            # a run we are still capturing; cancel it first
            self._send(409, {"error": str(e)})

    # ── SSE ──────────────────────────────────────────────────────────────

    def _sse(self, run_id: str | None) -> None:
        q = self.state.bus.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        # Chunked is the default for HTTP/1.1 with no Content-Length; SSE needs
        # the connection held open, so no length is sent.
        self.end_headers()
        try:
            self._sse_write("hello", {"schema_version": SCHEMA_VERSION, "run_id": run_id})
            last_ping = time.time()
            while True:
                try:
                    msg = q.get(timeout=1.0)
                except queue.Empty:
                    # A comment line every 15s so proxies and sleeping laptops
                    # do not silently close a stream that is merely quiet. A
                    # quiet agent and a dead connection look identical without
                    # it, and telling those apart is half the product.
                    if time.time() - last_ping > 15:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_ping = time.time()
                    continue
                if run_id and msg.get("run_id", _run_of(msg)) != run_id:
                    continue
                self._sse_write(msg.get("kind", "event"), msg)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.state.bus.unsubscribe(q)

    def _sse_write(self, event: str, data: dict[str, Any]) -> None:
        payload = _dumps(data)
        self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()


def _finite(o: Any) -> Any:
    """Replace `inf`/`nan` with `null`, recursively.

    Python writes them as bare `Infinity` and `NaN`, which are not JSON —
    `JSON.parse` rejects the *whole document*, so one unreachable field would
    blank the entire page rather than one number. `null` is also the honest
    reading: an infinite clock resolution means we have no clock.
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def _dumps(payload: Any) -> str:
    return json.dumps(_finite(payload), ensure_ascii=False, default=str)


def _run_of(msg: dict[str, Any]) -> str | None:
    ev = msg.get("event")
    return ev.get("run_id") if isinstance(ev, dict) else None


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT,
          root: Path | str | None = None, *, watch: bool = False) -> None:
    st = _Handler.state = SeerState(Path(root) if root else DEFAULT_ROOT, watch=watch)
    srv = ThreadingHTTPServer((host, port), _Handler)
    srv.daemon_threads = True
    print(
        f"[seer] serving on http://{host}:{port}  "
        "(health: /seer/health, runs: /seer/runs, run: /seer/run/<id>, "
        "live: /seer/live, compare: /seer/compare?runs=a,b, "
        "start: POST /seer/start, attach: POST /seer/attach, "
        "reconcile: POST /seer/reconcile) — log under "
        f"{st.store.root}"
    )
    if st.collector:
        d = st.collector.reader.dir
        print(
            f"[seer] watching {d}"
            if d.is_dir()
            else f"[seer] no spool at {d} — run `seer install` to capture "
                 "your own sessions"
        )
    try:
        srv.serve_forever()
    finally:
        if st.collector:
            # Leave the runs open: they did not end because the server did.
            st.collector.stop(close_runs=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--root", default=None)
    ap.add_argument("--watch", action="store_true",
                    help="also collect your own sessions from the hook spool")
    a = ap.parse_args()
    serve(a.host, a.port, a.root, watch=a.watch)
