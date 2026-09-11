"""The cost gate, extended to Seer (Attractors P3).

`llm.py`'s `RunBudget` prices a run *we* pay for: we pick the endpoint, we hold
the key, and the provider hands back a usage block we can multiply by a corpus
rate. None of that is true here. `seer run` launches somebody else's agent
binary, which talks to somebody else's account over somebody else's plan. The
money leaves through a door this process cannot see.

So the honest thing to say about the first run of a protocol is **nothing**:

    run 1 of a new protocol has NO estimate. Its fidelity is `MISSING`.

`MISSING` is not `0`. A `$0.0000` printed before a twenty-run repeat would be a
fabricated number wearing a currency symbol, and the whole point of `--repeat`
is that it multiplies whatever run 1 costs by twenty. An unpriced repeat is
therefore refused unless the human says, in the command, that they accept
spending an unknown amount: `acknowledge_unpriced=True`
(`--acknowledge-unpriced` on the command line). That acknowledgement is the
substitute for an estimate, and it is recorded in the ensemble so a reader can
see that nobody knew.

Runs 2…N are different. By then run 1 has *happened*, and if the agent reported
what it cost — Claude Code does on its terminal line, Hermes on its usage line;
`codex exec --json` does not — that is a MEASURED number about this exact
protocol on this exact machine. Multiplying it is an `ESTIMATED` figure with a
real basis, and it goes through the same `preflight → approve` step as any other
spend, against the same `DEFAULT_MAX_COST_USD` ceiling `corpus.py` sets for the
rest of the project.

**What a "protocol" is.** Two runs belong to the same protocol when they would
cost the same because they ask the same thing of the same agent in the same
place. That is decided by a `sha256` over exactly five fields —

    agent · model · prompt · cwd · extra_args

— and the first 16 hex digits become the id (`proto_…`). Three notes on the
choice, because "new protocol" being decidable is what makes run 1 identifiable
at all:

* the **prompt is hashed, never stored**. A hash leaks nothing and the ledger
  lives on disk next to the event log;
* `model=None` and `model="gpt-5.1-codex"` are **different protocols**. An
  unpinned run does not get to inherit the measured cost of a pinned one — that
  would be a pin substitution arriving from the money side;
* the model string is hashed **verbatim**, not case-folded. `same_model` in
  `llm.py` tolerates case because those are the same model; a protocol id is
  about the same *command*, and normalising it here would quietly merge two
  ledger rows that a reader wrote deliberately.

A protocol that has never been measured stays unpriced forever if its agent
never reports a cost — which is the true state of affairs for `codex exec`, and
the ensemble document says so rather than filling the hole in.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..corpus import DEFAULT_MAX_COST_USD
from ..llm import BudgetError
from .contract import Fidelity
from .store import EventStore

#: How a protocol id is made. Both halves ship in every document that carries
#: an id, so "how was this hashed" never has to be recovered from source.
PROTOCOL_HASH_ALGORITHM = "sha256"
PROTOCOL_HASH_FIELDS = ("agent", "model", "prompt", "cwd", "extra_args")
PROTOCOL_ID_PREFIX = "proto_"
PROTOCOL_ID_HEX = 16

#: The ledger of what protocols have actually cost, under the store root.
LEDGER_FILENAME = "budget.json"
LEDGER_VERSION = 1


class SeerBudgetError(BudgetError):
    """A repeat was refused before anything was spent.

    Subclasses `llm.BudgetError` so existing `except BudgetError` sites catch
    it, and terminal for the same reason theirs is: continuing past a refused
    ceiling is the behaviour the ceiling exists to prevent.
    """


# ── protocol identity ───────────────────────────────────────────────────────


def protocol_id(
    agent: str,
    prompt: str,
    *,
    model: str | None = None,
    cwd: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """The stable id of "this command, run again". See the module docstring."""
    canonical = json.dumps(
        {
            "agent": (agent or "").strip().lower(),
            "model": (model or "").strip() or None,
            "prompt": prompt or "",
            "cwd": str(cwd) if cwd else None,
            "extra_args": [str(a) for a in (extra_args or [])],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return PROTOCOL_ID_PREFIX + digest[:PROTOCOL_ID_HEX]


def protocol_fingerprint(
    agent: str,
    prompt: str,
    *,
    model: str | None = None,
    cwd: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """The id plus how it was made. The prompt and the cwd are hashed into the
    id and deliberately not reproduced here — a document that travels to a
    viewer should not carry the prompt text as a side effect of carrying a
    cost estimate."""
    return {
        "id": protocol_id(
            agent, prompt, model=model, cwd=cwd, extra_args=extra_args
        ),
        "hash_algorithm": PROTOCOL_HASH_ALGORITHM,
        "hash_fields": list(PROTOCOL_HASH_FIELDS),
        "agent": (agent or "").strip().lower(),
        # `null` is a real value here: it means "no model was pinned", which is
        # a different protocol from any pinned one.
        "model": (model or "").strip() or None,
    }


# ── what a protocol has actually cost ───────────────────────────────────────


@dataclass(frozen=True)
class Measurement:
    """What the ledger knows about one protocol. Only ever written from a run
    that really reported a cost — an unpriced run leaves no row, because a row
    saying `$0.00 over 1 run` would be read as a measurement of zero."""

    protocol_id: str
    n_runs: int
    total_usd: float
    per_run_usd: float
    #: the fidelity of the *agent's* cost report the row was built from
    source_fidelity: str
    last_run_id: str | None = None
    updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "n_runs": self.n_runs,
            "total_usd": self.total_usd,
            "per_run_usd": self.per_run_usd,
            "source_fidelity": self.source_fidelity,
            "last_run_id": self.last_run_id,
            "updated": self.updated,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Measurement":
        return Measurement(
            protocol_id=str(d["protocol_id"]),
            n_runs=int(d.get("n_runs", 0)),
            total_usd=float(d.get("total_usd", 0.0)),
            per_run_usd=float(d.get("per_run_usd", 0.0)),
            source_fidelity=str(d.get("source_fidelity", Fidelity.NATIVE.value)),
            last_run_id=d.get("last_run_id"),
            updated=str(d.get("updated", "")),
        )


@dataclass(frozen=True)
class Estimate:
    """What a repeat is expected to cost, and how well that is known.

    `usd is None` and `fidelity is MISSING` travel together and are the normal
    state for the first run of a protocol. Nothing in this module ever returns
    `usd=0.0` for an unknown cost.
    """

    protocol_id: str
    n_runs: int
    usd: float | None
    fidelity: Fidelity
    reason: str
    ceiling_usd: float
    per_run_usd: float | None = None
    measured_runs: int = 0
    measured_from_run_id: str | None = None
    acknowledged_unpriced: bool = False

    @property
    def priced(self) -> bool:
        return self.usd is not None and self.fidelity is not Fidelity.MISSING

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "n_runs": self.n_runs,
            "usd": self.usd,
            "fidelity": self.fidelity.value,
            "reason": self.reason,
            "ceiling_usd": self.ceiling_usd,
            "per_run_usd": self.per_run_usd,
            "measured_runs": self.measured_runs,
            "measured_from_run_id": self.measured_from_run_id,
            "acknowledged_unpriced": self.acknowledged_unpriced,
        }

    def text(self) -> str:
        """The line a human approves against."""
        if self.priced:
            return (
                f"  cost: ~${self.usd:.4f} for {self.n_runs} run(s) of "
                f"{self.protocol_id} (ceiling ${self.ceiling_usd:.2f})\n"
                f"        estimated — {self.reason}"
            )
        return (
            f"  cost: MISSING for {self.n_runs} run(s) of {self.protocol_id} "
            f"(ceiling ${self.ceiling_usd:.2f})\n"
            f"        {self.reason}\n"
            f"        MISSING is not $0.00 — this repeat is unpriced, not free."
        )


class SeerBudget:
    """A spend ceiling for one `seer run --repeat N`, with an approval step.

    Same shape as `llm.RunBudget` on purpose — `preflight()` prints an estimate,
    `approve()` records that a human read it, `charge_run()` records what really
    happened — and the same two properties:

    **Approval is a step, not a flag.** `charge_run()` raises until `approve()`
    has been called, and `approve()` is unreachable before `preflight()`.

    **It charges what was reported, never the estimate.** A run whose agent said
    nothing about money is counted as an *unpriced run*, not as $0.00: the
    counter `n_unpriced_runs` is what the ensemble document reports so that
    `spent_usd` is never read as "the total", only as "the part we could see".
    """

    def __init__(
        self,
        store: EventStore | None = None,
        *,
        root: Path | str | None = None,
        ceiling_usd: float = DEFAULT_MAX_COST_USD,
        label: str = "seer repeat",
    ) -> None:
        if store is None and root is None:
            raise ValueError("SeerBudget needs a store or a root")
        if ceiling_usd < 0:
            raise ValueError(f"ceiling must be >= 0, got {ceiling_usd}")
        self.root = Path(root) if root is not None else Path(store.root)  # type: ignore[union-attr]
        self.ceiling_usd = float(ceiling_usd)
        self.label = label
        self.calls = 0
        self.n_unpriced_runs = 0
        self._spent_usd = 0.0
        self._approved = False
        self._preflighted = False
        self._acknowledged_unpriced = False
        self._last_estimate: Estimate | None = None

    # ── the ledger ───────────────────────────────────────────────────────

    @property
    def ledger_path(self) -> Path:
        return self.root / LEDGER_FILENAME

    def _read_ledger(self) -> dict[str, Any]:
        path = self.ledger_path
        if not path.exists():
            return {"version": LEDGER_VERSION, "protocols": {}}
        try:
            doc = json.loads(path.read_text())
        except ValueError:
            # A truncated ledger is a lost measurement, not a licence to
            # invent one: we fall back to "never measured", which routes
            # straight back through the unpriced refusal.
            return {"version": LEDGER_VERSION, "protocols": {}}
        doc.setdefault("protocols", {})
        return doc

    def _write_ledger(self, doc: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    def measurement(self, pid: str) -> Measurement | None:
        """What this protocol has cost before, or None if nobody knows."""
        row = self._read_ledger()["protocols"].get(pid)
        return Measurement.from_dict(row) if row else None

    def record_measurement(
        self,
        pid: str,
        run_id: str,
        usd: float,
        *,
        source_fidelity: str = Fidelity.NATIVE.value,
    ) -> Measurement:
        """Fold one real, reported cost into the protocol's row.

        A running mean over every measured run of the protocol rather than
        "the last one wins": run-to-run spend varies with how much work the
        agent chose to do, and one cheap run should not reprice a fan.
        """
        doc = self._read_ledger()
        prev = doc["protocols"].get(pid)
        n = int(prev["n_runs"]) + 1 if prev else 1
        total = float(prev["total_usd"]) + float(usd) if prev else float(usd)
        m = Measurement(
            protocol_id=pid,
            n_runs=n,
            total_usd=total,
            per_run_usd=total / n,
            source_fidelity=source_fidelity,
            last_run_id=run_id,
            updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        doc["protocols"][pid] = m.to_dict()
        doc["version"] = LEDGER_VERSION
        self._write_ledger(doc)
        return m

    # ── spend so far ─────────────────────────────────────────────────────

    @property
    def spent_usd(self) -> float:
        """What we could *see* being spent. Never "the total" when
        `n_unpriced_runs` is non-zero."""
        return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.ceiling_usd - self._spent_usd)

    @property
    def acknowledged_unpriced(self) -> bool:
        return self._acknowledged_unpriced

    # ── the gate ─────────────────────────────────────────────────────────

    def estimate(self, pid: str, n_runs: int) -> Estimate:
        """Price `n_runs` of this protocol, or say honestly that we cannot."""
        if n_runs < 0:
            raise ValueError(f"n_runs must be >= 0, got {n_runs}")
        m = self.measurement(pid)
        if m is None or m.n_runs <= 0:
            return Estimate(
                protocol_id=pid,
                n_runs=n_runs,
                usd=None,
                fidelity=Fidelity.MISSING,
                reason=(
                    "no run of this protocol has ever reported a cost. Seer "
                    "launches the agent's own binary, which bills through an "
                    "account this process cannot read, so the first run of a "
                    "protocol has no estimate at all"
                ),
                ceiling_usd=self.ceiling_usd,
            )
        return Estimate(
            protocol_id=pid,
            n_runs=n_runs,
            usd=m.per_run_usd * n_runs,
            fidelity=Fidelity.ESTIMATED,
            reason=(
                f"{n_runs} x ${m.per_run_usd:.4f}/run, measured over "
                f"{m.n_runs} run(s) of this protocol that reported a cost"
            ),
            ceiling_usd=self.ceiling_usd,
            per_run_usd=m.per_run_usd,
            measured_runs=m.n_runs,
            measured_from_run_id=m.last_run_id,
        )

    def preflight(
        self,
        pid: str,
        n_runs: int,
        *,
        acknowledge_unpriced: bool = False,
        quiet: bool = False,
    ) -> Estimate:
        """The step before anything is spent. Returns the estimate it printed.

        Raises `SeerBudgetError` in exactly two situations, both of them before
        a single agent has been launched:

        * the protocol is unpriced and the caller did not acknowledge that;
        * the estimate *alone*, plus what this budget has already spent, is
          over the ceiling.
        """
        est = self.estimate(pid, n_runs)
        if not est.priced:
            if not acknowledge_unpriced:
                raise SeerBudgetError(
                    f"{self.label}: {n_runs} run(s) of {pid} cannot be priced, "
                    f"so nothing was launched.\n"
                    f"  {est.reason}.\n"
                    f"  This is MISSING, not $0.00: the runs will cost "
                    f"whatever your agent's account is charged, multiplied by "
                    f"{n_runs}.\n"
                    f"  Re-run with --acknowledge-unpriced to proceed anyway. "
                    f"The acknowledgement is recorded in the ensemble, so a "
                    f"reader can see the spend was never known.\n"
                    f"  After one run of this protocol reports a cost, later "
                    f"repeats are estimated from it and gated by the "
                    f"${self.ceiling_usd:.2f} ceiling."
                )
            est = Estimate(
                protocol_id=est.protocol_id,
                n_runs=est.n_runs,
                usd=None,
                fidelity=Fidelity.MISSING,
                reason=est.reason,
                ceiling_usd=est.ceiling_usd,
                acknowledged_unpriced=True,
            )
            self._acknowledged_unpriced = True
            self._preflighted = True
            self._last_estimate = est
            if not quiet:
                print(est.text())
            return est

        if not quiet:
            print(est.text())
        projected = self._spent_usd + float(est.usd or 0.0)
        if projected > self.ceiling_usd:
            raise SeerBudgetError(
                f"{self.label}: {n_runs} run(s) of {pid} would cost about "
                f"${est.usd:.4f}"
                + (
                    f" on top of ${self._spent_usd:.4f} already spent"
                    if self._spent_usd
                    else ""
                )
                + f", over the ${self.ceiling_usd:.2f} ceiling, so nothing "
                f"further was launched.\n"
                f"  {est.reason}.\n"
                f"  Lower --repeat, or raise the ceiling deliberately with "
                f"--max-cost-usd {projected:.4f}."
            )
        self._preflighted = True
        self._last_estimate = est
        return est

    def approve(self) -> None:
        """Record that the estimate was read and accepted."""
        if not self._preflighted:
            raise RuntimeError(
                "approve() before preflight() — approval must follow an "
                "estimate the human could actually read"
            )
        self._approved = True

    def _require_approval(self) -> None:
        if not self._approved:
            raise SeerBudgetError(
                f"{self.label}: a run was charged before its cost estimate was "
                f"approved. Call preflight() then approve()."
            )

    def charge_run(
        self,
        pid: str,
        run_id: str,
        cost_usd: float | None,
        *,
        source_fidelity: str = Fidelity.NATIVE.value,
    ) -> float:
        """Record one finished run. `None` means the agent reported no cost.

        A `None` is counted under `n_unpriced_runs` and adds nothing to the
        running total — it does NOT add zero, and it does not write a ledger
        row, because a protocol whose agent never reports a cost must stay
        unpriced rather than acquire a measurement of $0.00.
        """
        self._require_approval()
        self.calls += 1
        if cost_usd is None:
            self.n_unpriced_runs += 1
            return self._spent_usd
        self._spent_usd += float(cost_usd)
        self.record_measurement(
            pid, run_id, float(cost_usd), source_fidelity=source_fidelity
        )
        if self._spent_usd > self.ceiling_usd:
            raise SeerBudgetError(
                f"{self.label}: cumulative reported spend "
                f"${self._spent_usd:.4f} over {self.calls} run(s) has passed "
                f"the ${self.ceiling_usd:.2f} ceiling, so the repeat stopped "
                f"here.\n"
                f"  The runs that completed were really run, and the ensemble "
                f"reports its true n — it is a smaller fan, not a broken one.\n"
                f"  Re-run with a higher ceiling to continue."
            )
        return self._spent_usd

    # ── reporting ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """What the ensemble document carries about the money."""
        priced = self.calls - self.n_unpriced_runs
        return {
            "ceiling_usd": self.ceiling_usd,
            # `spent_usd` is what was *reported*. When n_unpriced_runs > 0 it
            # is a lower bound on the true spend and the fidelity says so.
            "spent_usd": self._spent_usd if priced else None,
            "spent_fidelity": (
                Fidelity.NATIVE.value if priced else Fidelity.MISSING.value
            ),
            "n_runs_charged": self.calls,
            "n_runs_priced": priced,
            "n_runs_unpriced": self.n_unpriced_runs,
            "acknowledged_unpriced": self._acknowledged_unpriced,
            "estimate": self._last_estimate.to_dict() if self._last_estimate else None,
        }

    def summary(self) -> str:
        if self.n_unpriced_runs and self.calls == self.n_unpriced_runs:
            return (
                f"{self.label}: {self.calls} run(s), cost MISSING — the agent "
                f"reported none, and the repeat was run on an explicit "
                f"acknowledgement rather than an estimate"
            )
        line = (
            f"{self.label}: ${self._spent_usd:.4f} reported of "
            f"${self.ceiling_usd:.2f} over {self.calls} run(s)"
        )
        if self.n_unpriced_runs:
            line += (
                f"; {self.n_unpriced_runs} of them reported no cost, so this "
                f"is a lower bound"
            )
        return line


__all__ = [
    "LEDGER_FILENAME",
    "LEDGER_VERSION",
    "PROTOCOL_HASH_ALGORITHM",
    "PROTOCOL_HASH_FIELDS",
    "PROTOCOL_ID_PREFIX",
    "Estimate",
    "Measurement",
    "SeerBudget",
    "SeerBudgetError",
    "protocol_fingerprint",
    "protocol_id",
]
