"""ATTACHED capture: speak the agent's own protocol instead of reading its UI.

For Codex that protocol is `codex app-server`, a JSON-RPC service over stdio.
There are two ways to reach one, and they are not the same measurement:

* **proxy** — `codex app-server proxy --sock <path>` joins a *running* daemon,
  so we observe threads someone else is driving. Nothing we do changes what the
  agent does. This is `CaptureMode.ATTACHED`.
* **spawn** — with no daemon we start our own `codex app-server` and drive a
  thread through it. We own the process, so honestly this is
  `CaptureMode.DRIVEN` — but through the app-server adapter, which sees 68
  notification kinds where `codex exec --json` shows 7. Capture *mode* answers
  "do we own the process"; the *adapter* answers "how much can we see". They
  are recorded separately because they vary independently.

**We never start a daemon.** `codex app-server daemon start` leaves a service
running after we exit; installing durable state as a side effect of "show me
this session" is not a decision a measurement tool gets to make. When no daemon
is running we say so and spawn a private server instead.

**We never approve anything.** A spawned thread is started with
`approvalPolicy: "never"` so the server does not ask; if it asks anyway we
answer `decline` and record — as an adapter warning, not as a user action — that
the answer came from SessionSeer and not from a person. The alternative is a
deadlock, and the alternative to *that* is a tool that silently authorises file
writes and shell commands on someone's behalf.

The version gate runs before the first event is stored. Its rule is asymmetric
on purpose: a mapped method that has **disappeared** is fatal, because whatever
depended on it would quietly read zero and the run would look like one where
nothing happened. A method we have **never heard of** is recorded and ignored,
because ignoring it cannot corrupt anything.
"""

from __future__ import annotations

import itertools
import json
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .adapters.codex_app_server import (
    CodexAppServerAdapter,
    ProtocolMismatch,
    check_protocol,
)
from .contract import (
    CaptureMode,
    Event,
    EventType,
    Fidelity,
    Outcome,
    new_run_id,
    new_session_id,
)
from .reducer import Reducer, RunView
from .runner import _repo_context, agent_version
from .store import EventStore

#: Where `codex app-server daemon` puts its control socket. Only ever read —
#: its absence is a fact we report, never one we fix.
DEFAULT_SOCK = Path.home() / ".codex" / "app-server-control" / "app-server-control.sock"

#: The recorded method surface, shipped with the tests so the gate still has
#: something to compare against on a machine where `codex` is not installed.
GOLDEN_PROTOCOL = (
    Path(__file__).resolve().parents[3]
    / "tests" / "fixtures" / "seer" / "codex-appserver-protocol.json"
)

#: The only answer we will give to an approval request. See the module docstring.
DECLINE = "decline"

OBSERVER_ONLY = (
    "SessionSeer is observing this session and cannot answer for the user"
)

#: How long a plain request may take before we stop waiting. Turn *completion*
#: is not a request — it arrives as a notification — so this covers only the
#: handshake-shaped calls.
REQUEST_TIMEOUT_S = 30.0


def load_golden(path: Path | None = None) -> dict[str, Any] | None:
    try:
        return json.loads((path or GOLDEN_PROTOCOL).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _methods(doc: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for variant in doc.get("oneOf", []):
        enum = variant.get("properties", {}).get("method", {}).get("enum")
        if enum:
            out.add(enum[0])
    return out


def live_protocol(codex_bin: str = "codex") -> tuple[set[str], set[str]]:
    """The method surface of the *installed* binary, from its own generator.

    Asking the binary beats parsing its version string. Two builds can carry the
    same version and differ, and what breaks a capture is a method going away,
    not a number changing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [codex_bin, "app-server", "generate-json-schema", "--out", tmp],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            raise ProtocolMismatch(
                f"{codex_bin} cannot describe its own protocol "
                f"({proc.stderr.strip()[:200] or 'no output'}) — this build is "
                "older than attached mode supports"
            )
        d = Path(tmp)
        return (
            _methods(json.loads((d / "ServerNotification.json").read_text())),
            _methods(json.loads((d / "ServerRequest.json").read_text())),
        )


def gate(codex_bin: str = "codex", *, golden: dict[str, Any] | None = None
         ) -> dict[str, Any]:
    """Run the version gate. Raises `ProtocolMismatch` when it fails closed."""
    notes, reqs = live_protocol(codex_bin)
    report = check_protocol(notes, reqs, load_golden() if golden is None else golden)
    if not report["compatible"]:
        gone = report["missing_notifications"] + report["missing_requests"]
        raise ProtocolMismatch(
            "this codex build no longer sends " + ", ".join(gone)
            + " — attached capture would report what those carried as absent "
              "rather than as unmeasured, so it is refused until the adapter "
              "is updated"
        )
    return report


def daemon_running(sock: Path | None = None) -> bool:
    return (sock or DEFAULT_SOCK).exists()


def protocol_note(report: dict[str, Any]) -> str:
    new = report.get("new_since_golden") or []
    if not new:
        return "protocol matches the recorded surface"
    shown = ", ".join(new[:5]) + (" …" if len(new) > 5 else "")
    return (
        f"this build sends {len(new)} notification(s) the adapter does not "
        f"read: {shown}"
    )


#: How a closing outcome becomes a terminal event. Written out rather than
#: derived so that adding an `Outcome` forces a decision here instead of
#: defaulting a failure into `session.completed`.
_CLOSING_EVENT: dict[Outcome, EventType] = {
    Outcome.AGENT_CLAIMED_COMPLETE: EventType.SESSION_COMPLETED,
    Outcome.UNVERIFIED_COMPLETE: EventType.SESSION_COMPLETED,
    Outcome.VERIFIED_PASS: EventType.SESSION_COMPLETED,
    Outcome.VERIFIED_PARTIAL: EventType.SESSION_COMPLETED,
    Outcome.VERIFIED_FAIL: EventType.SESSION_FAILED,
    Outcome.INFRASTRUCTURE_FAILURE: EventType.SESSION_FAILED,
    Outcome.INTERRUPTED: EventType.SESSION_INTERRUPTED,
    Outcome.UNKNOWN: EventType.SESSION_STATE_CHANGED,
}


@dataclass(slots=True)
class AttachResult:
    run_id: str
    thread_id: str | None
    view: RunView
    n_events: int
    protocol: dict[str, Any]
    transport: str


class CodexAttachment:
    """One connection to a `codex app-server`, and the run it fills in.

    Not reusable: an attachment is a run, the same way a `Runner` is.
    """

    agent = "codex"

    def __init__(
        self,
        *,
        store: EventStore,
        codex_bin: str = "codex",
        sock: Path | str | None = None,
        cwd: Path | str | None = None,
        keep_reasoning: bool = False,
        label: str | None = None,
        on_event: Callable[[Event], None] | None = None,
        golden: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.codex_bin = codex_bin
        self.sock = Path(sock) if sock else None
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.keep_reasoning = keep_reasoning
        self.label = label
        self.on_event = on_event
        self.golden = golden

        self.run_id = new_run_id()
        self.session_id = new_session_id()
        self.reducer = Reducer(self.run_id)
        self.protocol: dict[str, Any] = {}
        self.transport = "unopened"
        self.thread_id: str | None = None

        self.proc: subprocess.Popen | None = None
        self.adapter: CodexAppServerAdapter | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._turn_done = threading.Event()
        #: set when the server's stdout closes or `stop()` is called — the
        #: thing a watcher waits on, as opposed to one turn ending
        self._finished = threading.Event()
        self._stopping = False
        self._closed = False
        self._indexed_state: str | None = None
        self._stderr_tail: list[str] = []

    # ── opening ──────────────────────────────────────────────────────────

    def open(self, *, prefer_daemon: bool = True) -> CodexAttachment:
        if shutil.which(self.codex_bin) is None:
            raise ProtocolMismatch(f"{self.codex_bin!r} is not on PATH")

        use_daemon = prefer_daemon and daemon_running(self.sock)
        self.transport = "daemon-proxy" if use_daemon else "own-app-server"
        mode = CaptureMode.ATTACHED if use_daemon else CaptureMode.DRIVEN

        self.adapter = CodexAppServerAdapter(
            run_id=self.run_id,
            session_id=self.session_id,
            agent_version=agent_version(self.codex_bin),
            capture_mode=mode,
            keep_reasoning=self.keep_reasoning,
            repo=_repo_context(self.cwd),
        )
        self.store.register_run(
            self.run_id, agent=self.agent,
            agent_version=self.adapter.agent_version,
            capture_mode=mode.value, label=self.label,
            repo=self.adapter.repo, started_at=time.time(),
        )
        self._emit([
            self.adapter.event(
                EventType.RUN_STARTED,
                fidelity=Fidelity.DETERMINISTIC,
                payload={
                    "agent": self.agent,
                    "command": " ".join(self._command(use_daemon)),
                    "cwd": str(self.cwd),
                    "label": self.label,
                    "transport": self.transport,
                },
            )
        ])

        try:
            self.protocol = gate(self.codex_bin, golden=self.golden)
        except ProtocolMismatch as exc:
            # A refusal is a result: the run exists, says why it is empty, and
            # ends. An exception alone would leave a run that merely looks quiet.
            self._emit([
                self.adapter.event(
                    EventType.ADAPTER_INCOMPATIBLE,
                    fidelity=Fidelity.DETERMINISTIC,
                    payload={"message": str(exc), "codex_bin": self.codex_bin},
                )
            ])
            self._close_record(Outcome.INFRASTRUCTURE_FAILURE, str(exc))
            raise

        self._emit([
            self.adapter.event(
                EventType.ADAPTER_WARNING if self.protocol.get("new_since_golden")
                else EventType.SESSION_STATE_CHANGED,
                fidelity=Fidelity.DETERMINISTIC,
                native_type="protocol.checked",
                payload={
                    "state": "connected",
                    "note": protocol_note(self.protocol),
                    "transport": self.transport,
                    **self.protocol,
                },
            )
        ])

        self.proc = subprocess.Popen(
            self._command(use_daemon), cwd=str(self.cwd),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._pump, daemon=True,
                         name=f"seer-attach-{self.run_id}").start()
        threading.Thread(target=self._pump_stderr, daemon=True,
                         name=f"seer-attach-err-{self.run_id}").start()

        self.request("initialize", {
            "clientInfo": {"name": "sessionseer", "title": "SessionSeer",
                           "version": "0.1.0"}
        })
        self.notify("initialized", {})
        return self

    def _command(self, use_daemon: bool) -> list[str]:
        if use_daemon:
            return [self.codex_bin, "app-server", "proxy",
                    "--sock", str(self.sock or DEFAULT_SOCK)]
        return [self.codex_bin, "app-server"]

    # ── driving a thread (spawned server only) ───────────────────────────

    def start_thread(self, *, model: str | None = None,
                     sandbox: str = "workspace-write") -> str:
        """Open a thread on our own server. `approvalPolicy: never` because we
        are not a person and must not answer for one."""
        params: dict[str, Any] = {
            "cwd": str(self.cwd),
            "approvalPolicy": "never",
            "sandbox": sandbox,
        }
        if model:
            params["model"] = model
        res = self.request("thread/start", params)
        thread = res.get("thread") or {}
        self.thread_id = thread.get("id") or res.get("threadId")
        return self.thread_id or ""

    def send_turn(self, prompt: str, *, model: str | None = None,
                  effort: str | None = None) -> dict[str, Any]:
        if not self.thread_id:
            self.start_thread(model=model)
        self._turn_done.clear()
        params: dict[str, Any] = {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": prompt}],
            "cwd": str(self.cwd),
        }
        if model:
            params["model"] = model
        if effort:
            params["effort"] = effort
        return self.request("turn/start", params)

    def wait_for_turn(self, timeout_s: float | None = None) -> bool:
        """True if a turn completed, False on timeout. The wait is on the
        notification, not on the request — `turn/start` returns as soon as the
        turn exists, which is long before it is over."""
        return self._turn_done.wait(timeout_s)

    def interrupt(self) -> None:
        if self.thread_id:
            self.notify("turn/interrupt", {"threadId": self.thread_id})

    def stop(self) -> None:
        """Detach now, from another thread.

        Interrupts a turn if we started one — but only that. In proxy mode the
        thread belongs to whoever is driving it, and 'stop watching' must not
        mean 'stop their agent'.
        """
        self._stopping = True
        if self.thread_id and self.transport != "daemon-proxy":
            self.interrupt()
        self._turn_done.set()
        self._finished.set()

    # ── the two shapes of an attached capture ────────────────────────────

    def drive(self, prompt: str, *, model: str | None = None,
              timeout_s: float | None = 900.0) -> AttachResult:
        """Run one turn through our own app-server and close."""
        self.send_turn(prompt, model=model)
        done = self.wait_for_turn(timeout_s)
        if self._stopping or not done:
            self.interrupt()
            return self.close(outcome=Outcome.INTERRUPTED)
        return self.close(outcome=Outcome.AGENT_CLAIMED_COMPLETE)

    def watch(self, timeout_s: float | None = 900.0) -> AttachResult:
        """Observe until the connection drops, `stop()`, or the timeout.

        Deliberately *not* until the first `turn/completed`: a session someone
        else is driving has many turns, and a watcher that quits after one
        would silently truncate the run it was asked to observe.
        """
        self._finished.wait(timeout_s)
        return self.close()

    # ── JSON-RPC plumbing ────────────────────────────────────────────────

    def request(self, method: str, params: dict[str, Any] | None = None,
                *, timeout_s: float = REQUEST_TIMEOUT_S) -> dict[str, Any]:
        rid = next(self._ids)
        box: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            self._pending[rid] = box
        self._write({"jsonrpc": "2.0", "id": rid, "method": method,
                     "params": params or {}})
        try:
            msg = box.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError(
                f"{method} did not answer in {timeout_s:g}s"
            ) from None
        finally:
            with self._lock:
                self._pending.pop(rid, None)
        if "error" in msg:
            err = msg["error"] or {}
            raise RuntimeError(f"{method}: {err.get('message', err)}")
        return msg.get("result") or {}

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _write(self, obj: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("attachment is not open")
        with self._lock:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                self._line(line)
            if self.adapter is not None:
                self._emit(self.adapter.finish())
        finally:
            # A dropped connection has to wake a watcher, or `watch()` sits on
            # its timeout observing a server that is no longer there.
            self._turn_done.set()
            self._finished.set()

    def _pump_stderr(self) -> None:
        """Diagnostics, not data — kept for the failure message, never folded
        into a count. Same rule as the runner's."""
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            if line.strip():
                self._stderr_tail.append(line.rstrip())
                del self._stderr_tail[:-50]

    def _line(self, line: str) -> None:
        raw = line.strip()
        if not raw.startswith("{"):
            return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            msg = None

        # Every line reaches the adapter, including replies to our own calls —
        # withholding one would make the stored log something other than what
        # the server said, and replay would no longer be replay.
        if self.adapter is not None:
            self._emit(self.adapter.feed(raw))

        if not isinstance(msg, dict):
            return
        if msg.get("method") is None and msg.get("id") is not None:
            with self._lock:
                box = self._pending.get(_as_int(msg["id"]))
            if box is not None:
                try:
                    box.put_nowait(msg)
                except queue.Full:  # pragma: no cover - one reply per id
                    pass
        elif msg.get("method") is not None and msg.get("id") is not None:
            self._answer_server_request(msg)

    def _answer_server_request(self, msg: dict[str, Any]) -> None:
        """The server asked a human a question. We are not one.

        Approvals get `decline`, which the protocol defines as "the agent will
        continue the turn" — the only answer that authorises nothing while
        leaving the run alive. A free-text prompt (`requestUserInput`) has no
        such answer, so it gets a JSON-RPC error saying who we are; inventing a
        reply would put words into the transcript that no one said.

        Either way the fact that a machine answered is recorded. A run where
        SessionSeer declined an approval is not the same run as one where a
        person did, and nothing downstream should be able to confuse them.
        """
        method = str(msg.get("method"))
        low = method.lower()
        if "approval" in low:
            reply: dict[str, Any] = {"result": {"decision": DECLINE}}
            what = f"declined {method}"
        elif "requestuserinput" in low:
            reply = {"error": {"code": -32001, "message": OBSERVER_ONLY}}
            what = f"refused {method}"
        else:
            return
        answer = {"jsonrpc": "2.0", "id": msg["id"], **reply}
        self._write(answer)
        if self.adapter is not None:
            # Our reply goes out on stdin and never comes back on stdout, so
            # the adapter would leave the approval open forever and the time
            # spent waiting on it would be unmeasurable. Feeding it the exact
            # bytes we sent is not fabrication — it is the missing half of a
            # conversation we are one side of.
            self._emit(self.adapter.feed(json.dumps(answer)))
            self._emit([self.adapter.warn(
                f"{what} automatically — SessionSeer never answers for a user. "
                "Drive this thread in Codex itself to make the decision."
            )])

    # ── closing ──────────────────────────────────────────────────────────

    def close(self, *, outcome: Outcome | None = None) -> AttachResult:
        if self.thread_id and self.transport == "daemon-proxy":
            # Leave the daemon as we found it. Unsubscribing is the difference
            # between detaching and quietly staying attached forever.
            try:
                self.notify("thread/unsubscribe", {"threadId": self.thread_id})
            except RuntimeError:  # pragma: no cover - already gone
                pass
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - slow exit
                self.proc.kill()
        return self._close_record(outcome, None)

    def _close_record(self, outcome: Outcome | None, reason: str | None
                      ) -> AttachResult:
        if self._closed:
            return self._result()
        self._closed = True
        assert self.adapter is not None
        if outcome is None:
            # An attachment that never saw a thread has no terminal event of its
            # own, and a run stuck at `starting` forever is indistinguishable in
            # the list from one that is still live. `detached` says the true
            # thing: we stopped looking, and nothing is claimed about the agent.
            self._emit([
                self.adapter.event(
                    EventType.SESSION_STATE_CHANGED,
                    fidelity=Fidelity.DETERMINISTIC,
                    native_type="attach.closed",
                    payload={"state": "detached",
                             "reason": "SessionSeer closed the connection"},
                )
            ])
        if outcome is not None:
            self._emit([
                self.adapter.event(
                    _CLOSING_EVENT[outcome],
                    fidelity=Fidelity.DETERMINISTIC,
                    native_type="attach.closed",
                    payload={"outcome": outcome.value, "reason": reason,
                             "stderr_tail": self._stderr_tail[-5:]},
                )
            ])
        self._emit([
            self.adapter.event(
                EventType.RUN_COMPLETED,
                fidelity=Fidelity.DETERMINISTIC,
                payload={
                    "outcome": self.reducer.view.outcome.value,
                    "n_events": self.reducer.view.n_events,
                    "transport": self.transport,
                },
            )
        ])
        view = self.reducer.finalize()
        self.store.set_state(self.run_id, view.state.value, self.label)
        return self._result()

    def _result(self) -> AttachResult:
        return AttachResult(
            run_id=self.run_id, thread_id=self.thread_id,
            view=self.reducer.finalize(), n_events=self.reducer.view.n_events,
            protocol=self.protocol, transport=self.transport,
        )

    # ── shared with the runner ───────────────────────────────────────────

    def _emit(self, events: list[Event]) -> None:
        if not events:
            return
        self.store.append_many(events)
        for e in events:
            self.reducer.push(e)
            if e.event_type is EventType.TURN_COMPLETED:
                self._turn_done.set()
            if self.on_event:
                self.on_event(e)
        state = self.reducer.view.state.value
        if state != self._indexed_state:
            self._indexed_state = state
            self.store.set_state(self.run_id, state, self.label)


def _as_int(v: Any) -> Any:
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def attach_codex(prompt: str | None = None, *, store: EventStore,
                 model: str | None = None, timeout_s: float | None = 900.0,
                 prefer_daemon: bool = True, **kw: Any) -> AttachResult:
    """Open an attachment, drive one turn through it or watch, close it.

    With no `prompt` this is pure observation: it connects, records what the
    server says, and returns when the connection drops or `timeout_s` elapses.
    """
    att = CodexAttachment(store=store, **kw).open(prefer_daemon=prefer_daemon)
    try:
        return (att.watch(timeout_s) if prompt is None
                else att.drive(prompt, model=model, timeout_s=timeout_s))
    finally:
        if att.proc is not None and att.proc.poll() is None:  # pragma: no cover
            att.close()


__all__ = [
    "DECLINE",
    "DEFAULT_SOCK",
    "GOLDEN_PROTOCOL",
    "AttachResult",
    "CodexAttachment",
    "ProtocolMismatch",
    "attach_codex",
    "daemon_running",
    "gate",
    "live_protocol",
    "load_golden",
    "protocol_note",
]
