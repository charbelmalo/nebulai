"""Launch an agent, capture it, store it. The DRIVEN capture mode, concretely.

One subprocess, one adapter, one append-only log. The runner's whole job is to
not lose anything and to not invent anything:

* stdout is read line by line on a reader thread so a long-running agent
  streams instead of arriving all at once at exit;
* stderr is captured separately and only enters the record as an adapter
  warning — agents write progress chatter there and treating it as data would
  put an agent's stderr formatting into research metrics;
* a non-zero exit with no terminal event in the stream becomes
  `INFRASTRUCTURE_FAILURE`, never a completed run;
* the agent's own environment is passed through unchanged. Building a clean env
  would change the run being measured, and a measurement that alters what it
  measures is not one a researcher can use.

Cancellation sends SIGTERM then SIGKILL after a grace period, and records
`INTERRUPTED` — a killed run is a real outcome, not a gap.
"""

from __future__ import annotations

import os
import queue
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .adapters import adapter_for
from .adapters.hermes import HermesOneshotAdapter, HermesStateDbReconciler
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
from .store import EventStore

#: `codex exec` blocks reading stdin when it is not a TTY, which hangs the whole
#: capture with no output and no error. Every agent is therefore launched with
#: stdin closed. This cost an afternoon to find and is exactly the kind of thing
#: that looks like "the agent is slow" from the outside.
_STDIN = subprocess.DEVNULL

TERMINAL_EVENTS = frozenset(
    {
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    }
)


def build_command(agent: str, prompt: str, *, model: str | None = None,
                  extra: list[str] | None = None) -> list[str]:
    """The headless invocation per agent, with the flags that make it
    machine-readable. Kept in one place so a flag change is one edit and so the
    exact command can be written into the run record."""
    extra = list(extra or [])
    a = agent.lower()
    if a == "codex":
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check"]
        if model:
            cmd += ["-m", model]
        return cmd + extra + [prompt]
    if a == "claude":
        cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose"]
        if model:
            cmd += ["--model", model]
        return cmd + extra + [prompt]
    if a == "hermes":
        # `-z` prints only the final text; the structure comes from the
        # state.db reconciliation pass. See adapters/hermes.py.
        cmd = ["hermes", "-z", prompt]
        if model:
            cmd += ["-m", model]
        return cmd + extra
    raise ValueError(f"no launcher for agent {agent!r}")


#: `<agent> --version` output, cached per process. Claude reports its version in
#: the stream itself and the adapter overwrites this; Codex and Hermes do not,
#: and a run whose agent version is `unknown` cannot be pooled with other runs
#: of the same agent — which makes every longitudinal comparison guesswork.
_VERSION_CACHE: dict[str, str] = {}


def parse_version(out: str) -> str:
    """Pull the version out of a `--version` line.

    Three agents, three shapes, and one of them is a sentence:

        codex-cli 0.144.6
        2.1.222 (Claude Code)
        Hermes Agent v0.16.0 (2026.6.5) · upstream a41d280f · local 53147743

    The first *version-shaped* token wins, not the first token starting with a
    digit — Hermes puts a build date in parentheses right after the version,
    and taking that gave every Hermes run an `agent_version` of `2026.6.5)`,
    which is neither a version nor, with the bracket still attached, anything
    a later comparison could match.
    """
    line = (out.strip().splitlines() or [""])[0]
    for tok in line.replace("(", " ").replace(")", " ").split():
        core = tok[1:] if tok[:1] in "vV" else tok
        # a dot is what separates a version from a bare number in the sentence;
        # letters after it are fine — `0.146.0-alpha.9.2` is a real codex build
        if core[:1].isdigit() and "." in core:
            return core
    return line or "unknown"


def agent_version(agent: str) -> str:
    if agent in _VERSION_CACHE:
        return _VERSION_CACHE[agent]
    try:
        out = subprocess.run(
            [agent, "--version"], capture_output=True, text=True, timeout=10
        ).stdout
        v = parse_version(out)
    except (OSError, subprocess.SubprocessError):
        v = "unknown"
    _VERSION_CACHE[agent] = v
    return v


@dataclass(slots=True)
class RunResult:
    run_id: str
    agent: str
    exit_code: int | None
    view: RunView
    n_events: int
    stderr_tail: list[str] = field(default_factory=list)


class Runner:
    """One agent launch. Not reusable — a Runner is a run."""

    def __init__(
        self,
        agent: str,
        prompt: str,
        *,
        store: EventStore,
        cwd: Path | str | None = None,
        model: str | None = None,
        extra_args: list[str] | None = None,
        keep_reasoning: bool = False,
        label: str | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        self.agent = agent.lower()
        self.prompt = prompt
        self.store = store
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.model = model
        self.extra_args = extra_args or []
        self.label = label
        self.on_event = on_event

        self.run_id = new_run_id()
        self.session_id = new_session_id()
        self.adapter = adapter_for(
            self.agent,
            run_id=self.run_id,
            session_id=self.session_id,
            agent_version=agent_version(self.agent),
            capture_mode=CaptureMode.DRIVEN,
            keep_reasoning=keep_reasoning,
            repo=_repo_context(self.cwd),
        )
        self.reducer = Reducer(self.run_id)
        self.proc: subprocess.Popen | None = None
        self.stderr_tail: list[str] = []
        self._cancelled = False
        self._launched_at: float | None = None
        self._indexed_state: str | None = None

    # ── the run ──────────────────────────────────────────────────────────

    def run(self, timeout_s: float | None = None) -> RunResult:
        cmd = build_command(
            self.agent, self.prompt, model=self.model, extra=self.extra_args
        )
        self._launched_at = time.time()
        self._emit(
            [
                Event(
                    event_type=EventType.RUN_STARTED,
                    source=self.adapter._source(Fidelity.DETERMINISTIC),
                    run_id=self.run_id,
                    session_id=self.session_id,
                    repo=_repo_context(self.cwd),
                    payload={
                        "agent": self.agent,
                        # the exact command, so a run can be reproduced and so
                        # a flag change shows up as a change in the record
                        "command": " ".join(shlex.quote(c) for c in cmd),
                        "cwd": str(self.cwd),
                        "label": self.label,
                        "model_requested": self.model,
                    },
                )
            ]
        )

        # Hermes emits no start marker of its own.
        if isinstance(self.adapter, HermesOneshotAdapter):
            self._emit(self.adapter.open())

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.cwd),
                stdin=_STDIN,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line buffered; without it a long run looks hung
                env=os.environ.copy(),
            )
        except FileNotFoundError:
            self._emit([self.adapter.warn(f"{self.agent} is not on PATH")])
            return self._finish(exit_code=127, reason="agent not installed")

        lines: queue.Queue[tuple[str, str] | None] = queue.Queue()
        threads = [
            threading.Thread(target=_pump, args=(self.proc.stdout, "out", lines), daemon=True),
            threading.Thread(target=_pump, args=(self.proc.stderr, "err", lines), daemon=True),
        ]
        for t in threads:
            t.start()

        deadline = None if timeout_s is None else time.time() + timeout_s
        open_streams = 2
        while open_streams:
            try:
                item = lines.get(timeout=0.25)
            except queue.Empty:
                if deadline and time.time() > deadline:
                    self.cancel(reason="timeout")
                    deadline = None
                continue
            if item is None:
                open_streams -= 1
                continue
            stream, line = item
            if stream == "err":
                self._stderr(line)
            else:
                self._emit(self.adapter.feed(line))

        exit_code = self.proc.wait()

        if isinstance(self.adapter, HermesOneshotAdapter):
            self._emit(self.adapter.close(exit_code=exit_code))
        else:
            self._emit(self.adapter.finish())

        if self.agent == "hermes":
            self._reconcile_hermes()

        return self._finish(exit_code=exit_code)

    def cancel(self, reason: str = "cancelled") -> bool:
        if self.proc is None or self.proc.poll() is not None:
            return False
        self._cancelled = True
        try:
            self.proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return False
        # Grace, then force. A stuck agent must not stop the collector.
        def _kill_later() -> None:
            time.sleep(5.0)
            if self.proc and self.proc.poll() is None:
                self.proc.kill()

        threading.Thread(target=_kill_later, daemon=True).start()
        self._emit(
            [
                self.adapter.event(
                    EventType.SESSION_INTERRUPTED,
                    fidelity=Fidelity.DETERMINISTIC,
                    payload={"outcome": Outcome.INTERRUPTED.value, "reason": reason},
                )
            ]
        )
        return True

    # ── pieces ───────────────────────────────────────────────────────────

    def _stderr(self, line: str) -> None:
        """stderr is diagnostics, not data.

        Agents write progress bars, deprecation notices and provider warnings
        here. It reaches the record only as an adapter warning, never as an
        event with an action or a count — otherwise a chatty release of one
        agent would show up as that agent doing more work.
        """
        line = line.rstrip()
        if not line:
            return
        self.stderr_tail.append(line)
        del self.stderr_tail[:-50]

    def _reconcile_hermes(self) -> None:
        rec = HermesStateDbReconciler(
            run_id=self.run_id,
            session_id=self.session_id,
            agent_version=self.adapter.agent_version,
        )
        self._emit(
            rec.reconcile(
                started_after=self._launched_at or 0.0,
                cwd=str(self.cwd),
            )
        )

    def _emit(self, events: list[Event]) -> None:
        if not events:
            return
        self.store.append_many(events)
        for e in events:
            self.reducer.push(e)
            if self.on_event:
                self.on_event(e)
        # Keep the index's state column current *during* the run, not only at
        # the end. `seer list` and the viewer's run list read the index; without
        # this a live agent reads as `starting` for its whole life, which is the
        # opposite of what a real-time overview is for. Transitions are rare
        # (single digits per run), so this is a handful of UPDATEs, not one per
        # event.
        state = self.reducer.view.state.value
        if state != self._indexed_state:
            self._indexed_state = state
            self.store.set_state(self.run_id, state, self.label)

    def _finish(self, *, exit_code: int | None, reason: str | None = None) -> RunResult:
        view = self.reducer.finalize()
        saw_terminal = view.state.value in ("completed", "failed", "interrupted")

        if not saw_terminal or (exit_code not in (0, None) and not self._cancelled):
            # A stream that ended without saying how, or a non-zero exit the
            # agent never reported. Either way the honest label is that our
            # side failed to observe an ending — not that the task succeeded.
            outcome = (
                Outcome.INTERRUPTED if self._cancelled
                else Outcome.INFRASTRUCTURE_FAILURE
            )
            closing = self.adapter.event(
                EventType.SESSION_FAILED,
                fidelity=Fidelity.DETERMINISTIC,
                native_type="process.exit",
                payload={
                    "outcome": outcome.value,
                    "exit_code": exit_code,
                    "reason": reason or f"process exited {exit_code}",
                    "stderr_tail": self.stderr_tail[-5:],
                },
            )
            self._emit([closing])
            view = self.reducer.finalize()

        self._emit(
            [
                self.adapter.event(
                    EventType.RUN_COMPLETED,
                    fidelity=Fidelity.DETERMINISTIC,
                    payload={
                        "exit_code": exit_code,
                        "outcome": view.outcome.value,
                        "n_events": view.n_events,
                    },
                )
            ]
        )
        view = self.reducer.finalize()
        self.store.set_state(self.run_id, view.state.value, self.label)
        return RunResult(
            run_id=self.run_id,
            agent=self.agent,
            exit_code=exit_code,
            view=view,
            n_events=view.n_events,
            stderr_tail=self.stderr_tail[-10:],
        )


def _pump(stream, name: str, out: queue.Queue) -> None:
    try:
        for line in stream:
            out.put((name, line))
    finally:
        out.put(None)


def _repo_context(cwd: Path) -> dict[str, Any] | None:
    """Branch and HEAD, if this is a git worktree.

    Recorded per run because "which commit was this measured against" is the
    first question anyone asks of a result, and reconstructing it later from
    timestamps is guesswork.
    """
    try:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=3
            ).stdout.strip()

        root = git("rev-parse", "--show-toplevel")
        if not root:
            return None
        return {
            "root_id": root,
            "branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "head": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return None
