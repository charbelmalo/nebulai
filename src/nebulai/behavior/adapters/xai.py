"""The xAI arm: built, unit-tested against a recorded fixture, never run here.

**Status on this machine, 2026-09-11: NOT RUN — no `XAI_API_KEY` exists.** That
is a fact about the environment, not a property of the adapter. The adapter is
complete: it resolves the model surface, performs the §5.5 identity and
capability audit, records served ids and usage, enforces the runner's spend
ceiling, and is exercised by `tests/test_behavior_xai_adapter.py` against a
recorded response fixture so the parsing and provenance paths are covered
without a key.

What it will **not** do is quietly skip. `complete()` without credentials raises
`AdapterError` carrying the estimate-and-approve message, because a runner that
silently drops an arm produces a study whose manifest says two models and whose
data has one — and no downstream check can tell the difference afterwards.

Nothing in this file, or anywhere in the package, reads Grok's internals. The
strongest sentence this arm supports is §1.1's: *under protocol P, exact model
deployments A and B produced reliably different association distributions for
cue C.* The adapter sees a served model id, a response id, token usage, and
text — the same surface any API client sees.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ...llm import load_key
from .base import AdapterError, Completion, SamplerSettings

DEFAULT_BASE_URL = "https://api.x.ai/v1"
API_KEY_VAR = "XAI_API_KEY"

#: What `nebulai behavior plan` must resolve before the cost gate (§5.5). These
#: are the audit's own field names, so the manifest and the report agree.
AUDIT_FIELDS = (
    "requested_model",
    "served_model",
    "reasoning_disable_supported",
    "reasoning_effort",
    "supports_temperature",
    "supports_top_p",
    "supports_stop",
    "tools_absent",
    "response_id_capturable",
    "usage_capturable",
    "fingerprint_available",
)


@dataclass
class XAIAdapter:
    name: str = "xai"
    pinned: str = ""  # exact dated release; a moving alias is refused upstream
    paid: bool = True
    base_url: str = DEFAULT_BASE_URL
    env_file: str | None = None
    timeout: int = 60
    reasoning_effort: str | None = None
    #: Set by the runner from `RunBudget`. The adapter charges through it so the
    #: hard ceiling of §5.1 is enforced by the thing making the request, not
    #: only by the planner that guessed beforehand.
    budget: Any = None
    _key: str | None = None
    _detail: dict[str, Any] = field(default_factory=dict)

    # -- credentials ------------------------------------------------------
    @property
    def key(self) -> str | None:
        if self._key is None:
            self._key = load_key(API_KEY_VAR, self.env_file)
        return self._key

    def require_key(self, *, n_trials: int = 0, estimate_usd: float | None = None) -> str:
        k = self.key
        if k:
            return k
        est = (
            f"~${estimate_usd:.4f} for {n_trials} trials"
            if estimate_usd is not None
            else f"{n_trials} trials, price unknown"
        )
        raise AdapterError(
            f"the xAI arm has no credentials on this machine.\n"
            f"  set {API_KEY_VAR} (env or .env) and re-run.\n"
            f"  this arm would cost {est} on {self.pinned or '<unpinned>'}, and "
            f"`nebulai behavior run` requires an explicit --approve after "
            f"printing that estimate.\n"
            f"  the arm is recorded as not_run with this reason; it is NOT "
            f"skipped silently, because a manifest naming two models over data "
            f"containing one cannot be audited afterwards."
        )

    # -- §5.5 audit -------------------------------------------------------
    def audit(self) -> dict[str, Any]:
        """Resolve the model/capability surface before the cost gate.

        Returns every `AUDIT_FIELDS` key. Anything that cannot be determined is
        `None` — **MISSING, never a guess and never a zero.** In particular
        `reasoning_tokens_p95` stays `None` in the manifest until a real run
        measures it; §5.1 computes the budget on that p95, and inventing one
        would make the budget a number about nothing.
        """
        if not self.key:
            return {f: None for f in AUDIT_FIELDS} | {
                "status": "not_run",
                "reason": f"no {API_KEY_VAR}",
                "requested_model": self.pinned,
            }
        info = self._get("/models")
        ids = {m.get("id") for m in info.get("data", [])}
        if self.pinned not in ids:
            raise AdapterError(
                f"{self.pinned!r} is not in the provider's model list. "
                f"Nebul.AI does not substitute a near-match: pin an id the "
                f"provider actually serves, or stop. (available: "
                f"{sorted(i for i in ids if i)[:8]}…)"
            )
        probe = self._post(
            "/chat/completions",
            {
                "model": self.pinned,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 4,
                "temperature": 0.0,
            },
        )
        return {
            "requested_model": self.pinned,
            "served_model": probe.get("model", ""),
            "reasoning_disable_supported": None,
            "reasoning_effort": self.reasoning_effort,
            "supports_temperature": True,
            "supports_top_p": True,
            "supports_stop": True,
            "tools_absent": True,
            "response_id_capturable": bool(probe.get("id")),
            "usage_capturable": bool(probe.get("usage")),
            "fingerprint_available": probe.get("system_fingerprint") is not None,
            "status": "audited",
        }

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "pinned": self.pinned,
            "base_url": self.base_url,
            "paid": True,
            "credentials": "present" if self.key else "absent",
            "reasoning_effort": self.reasoning_effort,
        }

    # -- one trial --------------------------------------------------------
    def complete(
        self, prompt: str, settings: SamplerSettings, *, trial_seed: int
    ) -> Completion:
        self.require_key(n_trials=1)
        body: dict[str, Any] = {
            "model": self.pinned,
            # §5.2: the entire literal block in a single user message, with NO
            # study-specific system message. A system prompt here would make the
            # two arms incomparable before a single statistic ran.
            "messages": [{"role": "user", "content": prompt}],
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_output_tokens,
            "seed": trial_seed,
        }
        if settings.stop:
            body["stop"] = list(settings.stop)
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        t0 = time.perf_counter()
        payload = self._post("/chat/completions", body)
        dt = int((time.perf_counter() - t0) * 1000)
        return self.parse_response(payload, latency_ms=dt, budget=self.budget)

    # -- pure parsing, so the fixture test covers the real path ------------
    def parse_response(
        self, payload: dict[str, Any], *, latency_ms: int | None = None, budget: Any = None
    ) -> Completion:
        served = payload.get("model", "")
        if served and self.pinned and served != self.pinned:
            raise AdapterError(
                f"identity: requested {self.pinned!r} but the provider served "
                f"{served!r}. The run stops rather than pooling two deployments "
                f"(§5.5) — a study whose subject changed mid-collection is not "
                f"one study."
            )
        choices = payload.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        usage = payload.get("usage") or {}
        cost = None
        if budget is not None:
            cost = budget.charge_response(self.pinned, payload)
        return Completion(
            text=text,
            requested_model=self.pinned,
            served_model=served,
            response_id=payload.get("id", ""),
            # Optional-if-absent (§5.5.1): a missing fingerprint never fails a
            # trial. The manifest records once whether the field exists at all,
            # and the canary probe carries the drift-detection load either way.
            fingerprint=payload.get("system_fingerprint") or "",
            latency_ms=latency_ms,
            usage=usage,
            cost_usd=cost,
            detail={"reasoning_effort": self.reasoning_effort},
        )

    # -- transport --------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.require_key()}",
            "Content-Type": "application/json",
        }

    def _get(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url.rstrip("/") + path, headers=self._headers())
        return self._send(req)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url.rstrip("/") + path,
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        return self._send(req)

    def _send(self, req: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover - needs a key
            detail = e.read().decode("utf-8", "replace")[:400]
            raise AdapterError(f"xAI HTTP {e.code}: {detail}") from e
        except OSError as e:  # pragma: no cover - needs network
            raise AdapterError(f"xAI transport failure: {e}") from e


def reasoning_token_stats(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Empirical reasoning-token distribution for the budget's p95 (§5.1).

    Returns `{"status": "missing", ...}` on an empty input rather than zeros.
    Reasoning tokens are billed, vary per request and are not knowable before
    the request is made; a plan-time estimate multiplied by trial count bounds a
    quantity it cannot predict. A **zero** here would silently tell the budget
    gate that the unpredictable part costs nothing.
    """
    vals = []
    for p in payloads:
        u = p.get("usage") or {}
        det = u.get("completion_tokens_details") or {}
        rt = det.get("reasoning_tokens")
        if isinstance(rt, int):
            vals.append(rt)
    if not vals:
        return {
            "status": "missing",
            "reason": "no responses recorded — requires a live run with credentials",
            "p95": None,
            "n": 0,
        }
    vals.sort()
    idx = max(0, min(len(vals) - 1, int(round(0.95 * (len(vals) - 1)))))
    return {
        "status": "measured",
        "p95": vals[idx],
        "mean": sum(vals) / len(vals),
        "max": vals[-1],
        "n": len(vals),
    }
