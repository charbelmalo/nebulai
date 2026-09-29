"""Versioned study, model, cue, trial and manifest schemas — plan §9.1/§9.4.

The load-bearing object here is `Manifest`. Everything the study is allowed to
claim depends on parameters that must be chosen *before* data is seen: the
primary statistic's form, the effect floor, the compliance-parity threshold, the
informativeness floor, RBO's `p`, the JSD correction, the embedder SHA. A free
parameter that is not in the manifest is a researcher degree of freedom (plan
§9.4), so the manifest carries all of them and hashes them.

Immutability is enforced the same way `backend/instrument.py` enforces the
question-set freeze, and for the same reason: the failure — quietly changing a
threshold after seeing discovery results — has no migration path, so the
mechanism has to make it loud rather than merely discouraged. `Manifest.freeze()`
stamps a content hash over exactly the fields that determine what a number
means; `load_manifest()` re-derives it and refuses a post-freeze edit.

Three small rules that appear trivial and are not:

* `missing` is not `0`. A parse that produced nothing is `None`, never a zero
  count, everywhere in this module (plan §7 of the sibling plan says the same
  thing for scores; it is the same mistake).
* A pinned model id is never substituted. `ModelRef.pinned` is what was asked
  for; `served` is what came back; a mismatch is an error, not a note.
* Cue text is stored verbatim. Normalization is a derived view (`normalize.py`),
  never a replacement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import BEHAVIOR_SCHEMA_VERSION

MANIFEST_FORMAT_VERSION = 1

#: Evidence states, plan §6.5. `incomparable` is a real state here (unlike in
#: the sibling generative-variance plan, which refuses the word because Seer's
#: refusal engine owns it) because the behavioral study genuinely has a
#: protocol-integrity failure mode distinct from "not enough evidence".
EVIDENCE_STATES = (
    "confirmed",
    "suggestive",
    "frame-specific",
    "unstable",
    "capability-attributable",
    "no-detected-deviation",
    "insufficient-evidence",
    "incomparable",
)

#: Which arm a trial belongs to. `discovery` ranks candidates; `R` re-tests them
#: on the SAME frame with fresh trials (was the hit real?); `G` re-tests them on
#: held-out frames (does it survive a different way of asking?). Splitting the
#: two is what makes `suggestive` and `frame-specific` decidable — see §5.4.
ARMS = ("discovery", "R", "G")

#: A canary trial is excluded from every semantic metric (§5.5.1); it exists to
#: detect deployment drift behaviourally when `system_fingerprint` is absent.
CANARY_CUE = "__canary__"


class ManifestError(RuntimeError):
    """The manifest is malformed, unfrozen, or was edited after freezing.

    Not a `ValueError` subclass, for the same reason `InstrumentError` is not:
    parsing code all over this tree catches `ValueError`, and a manifest
    integrity failure must not be swallowed by a handler written for a bad int.
    """


@dataclass(frozen=True)
class ModelRef:
    """One arm of the study — what was asked for, and how to reach it.

    `pinned` is an EXACT id. Strict mode refuses a moving alias (`latest`,
    a bare family name) because a study whose subject can change underneath it
    is not reproducible; exploratory mode allows one but can never produce a
    `confirmed` cue (§5.5).
    """

    key: str  # arm label used in artifacts: "A", "B", "cap_small", …
    adapter: str  # "gpt2-local" | "xai" | "fake"
    pinned: str  # exact model id, never a family or alias
    revision: str | None = None  # HF commit sha for local models
    label: str = ""  # display name; never implies quality
    reasoning_effort: str | None = None  # recorded when it cannot be disabled
    notes: str = ""

    def validate(self, strict: bool) -> None:
        if not self.key or not self.pinned:
            raise ManifestError("a model arm needs both a key and a pinned id")
        if strict and _is_moving_alias(self.pinned):
            raise ManifestError(
                f"{self.pinned!r} is a moving alias, and a strict study may not "
                f"use one: the served weights can change mid-study with no "
                f"signal, which makes 'confirmed' unsupportable. Pin an exact "
                f"dated release, or run with strict=False and accept that no "
                f"cue can be confirmed (§5.5)."
            )


_MOVING = ("latest", "newest", "preview", "beta")


def _is_moving_alias(model_id: str) -> bool:
    tail = model_id.rsplit("/", 1)[-1].lower()
    if tail in _MOVING:
        return True
    return any(tail.endswith(f"-{m}") or tail.endswith(f":{m}") for m in _MOVING)


@dataclass(frozen=True)
class Cue:
    """A cue word and its preregistered stratum.

    `stratum` is the ONLY thing the landscape may be coloured by (§7.1.2) — it
    is declared before collection, so unlike a discovered cluster it cannot be
    seed-dependent. `pack` groups matched controls (the "daddy" pack) without
    granting them a different statistical threshold.
    """

    text: str
    stratum: str
    pack: str = ""
    notes: str = ""


@dataclass(frozen=True)
class PromptFrame:
    """One literal prompt form. `role` decides which partition may use it.

    The primary frame is what discovery and arm R use; `heldout` frames are
    hidden during discovery and used only by arm G. Keeping them in one file
    with an explicit role — rather than in two files — is what lets the manifest
    hash cover the held-out text too, so it cannot be authored after seeing
    discovery results.
    """

    id: str
    role: str  # "primary" | "heldout" | "canary"
    template: str  # must contain "{cue}"
    stop: tuple[str, ...] = ()

    def render(self, cue: str) -> str:
        if "{cue}" not in self.template:
            raise ManifestError(f"frame {self.id!r} has no {{cue}} placeholder")
        return self.template.replace("{cue}", cue)


@dataclass
class Manifest:
    """Every parameter that decides what a number means, frozen before data.

    The field list is long on purpose — it is §9.4's list. A shorter manifest is
    not a simpler design, it is an undeclared degree of freedom.
    """

    study_id: str
    created: str
    models: list[ModelRef]
    cues: list[Cue]
    frames: list[PromptFrame]

    # --- sampling ---------------------------------------------------------
    trials_per_cue: int = 40
    min_valid_trials: int = 20
    min_within_block_trials: int = 5
    n_time_blocks: int = 4
    temperature: float = 0.8
    top_p: float = 0.95
    max_output_tokens: int = 48
    seed: int = 42
    strict: bool = True

    # --- representation ---------------------------------------------------
    embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_sha: str = ""
    embedder_dtype: str = "float32"
    embedder_pooling: str = "mean"
    embedder_normalize: bool = True
    secondary_embedder_id: str = ""
    secondary_embedder_sha: str = ""
    #: Which resolution of §6.5.4 is in force: "strengthen" (a genuinely
    #: different representation kind) or "downgrade" (two similar encoders, and
    #: the copy says "not specific to one of two similar encoders").
    second_embedder_resolution: str = "downgrade"
    rank_weights: tuple[float, ...] = (1.0, 0.5, 1.0 / 3.0)

    # --- statistics (all frozen before discovery) -------------------------
    delta_hat_form: str = "difference"  # "difference" | "standardized"
    split_half_draws: int = 25
    mmd_bandwidth: float | None = None  # None until calibration freezes it
    n_permutations: int = 2000
    n_bootstrap: int = 1000
    effect_floor: float = 0.02
    reliability_floor: float = 0.0
    q_threshold: float = 0.05
    multiple_testing: str = "BY"
    compliance_parity_max: float = 0.25
    min_distinct_types: int = 3
    min_associate_entropy: float = 0.8
    oov_differential_max: float = 0.25
    rbo_p: float = 0.9
    jsd_correction: str = "miller-madow"

    # --- budget / provenance ----------------------------------------------
    max_cost_usd: float = 1.0
    reasoning_tokens_p95: int | None = None  # None = MISSING, never 0
    fingerprint_available: bool | None = None  # None = not yet audited
    canary_frame_id: str = "canary"
    positive_control_rate: dict[str, float] = field(default_factory=dict)
    p_floor: float | None = None
    git_commit: str = ""
    analysis_version: int = BEHAVIOR_SCHEMA_VERSION
    notes: str = ""

    frozen_at: str | None = None
    frozen_hash: str | None = None

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        keys = [m.key for m in self.models]
        if len(set(keys)) != len(keys):
            raise ManifestError(f"duplicate model arm keys: {keys}")
        for m in self.models:
            m.validate(self.strict)
        if not any(f.role == "primary" for f in self.frames):
            raise ManifestError("no primary prompt frame declared")
        texts = [c.text for c in self.cues]
        if len(set(texts)) != len(texts):
            raise ManifestError("duplicate cue text — cues must be unique")
        if self.delta_hat_form not in ("difference", "standardized"):
            raise ManifestError(
                f"delta_hat_form must be 'difference' or 'standardized', "
                f"not {self.delta_hat_form!r} (§6.4.1 — which one is primary is "
                f"declared before discovery and never changed)"
            )
        if self.second_embedder_resolution not in ("strengthen", "downgrade"):
            raise ManifestError(
                "second_embedder_resolution must be 'strengthen' or "
                "'downgrade' (§6.5.4) — the manifest declares which is in force"
            )

    # ------------------------------------------------------------------
    @property
    def is_frozen(self) -> bool:
        return self.frozen_hash is not None

    @property
    def primary_frame(self) -> PromptFrame:
        return next(f for f in self.frames if f.role == "primary")

    @property
    def heldout_frames(self) -> list[PromptFrame]:
        return [f for f in self.frames if f.role == "heldout"]

    def frame(self, frame_id: str) -> PromptFrame:
        for f in self.frames:
            if f.id == frame_id:
                return f
        raise ManifestError(f"no frame {frame_id!r} in this manifest")

    def model(self, key: str) -> ModelRef:
        for m in self.models:
            if m.key == key:
                return m
        raise ManifestError(f"no model arm {key!r} in this manifest")

    def protocol_hash(self) -> str:
        """Hash of the PROTOCOL only — frames + sampler + task shape.

        Separate from the manifest hash because two studies may legitimately
        share a protocol while differing in cue set or thresholds, and the
        comparison "were these collected the same way?" is exactly that
        question.
        """
        payload = json.dumps(
            {
                "frames": [asdict(f) for f in self.frames],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_output_tokens": self.max_output_tokens,
                "trials_per_cue": self.trials_per_cue,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def compute_hash(self) -> str:
        """Content hash over everything that decides what a number means.

        Excluded: `notes`, `frozen_at`, `frozen_hash`, and `created` — editorial
        or self-referential fields. Everything else is in, including held-out
        frame text, so a frame cannot be authored after discovery.
        """
        d = self.to_dict()
        for k in ("notes", "frozen_at", "frozen_hash", "created"):
            d.pop(k, None)
        payload = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def freeze(self, at: str) -> str:
        if self.is_frozen:
            raise ManifestError(
                f"study {self.study_id} was already frozen at {self.frozen_at} "
                f"({self.frozen_hash}). Re-freezing would redefine the study "
                f"every trial so far was collected under. Changing a threshold "
                f"after seeing results creates a NEW study revision (§5.4); it "
                f"cannot rewrite this one."
            )
        if not self.cues:
            raise ManifestError("a study with no cues has nothing to measure")
        if len(self.models) < 2:
            raise ManifestError(
                "a divergence study needs at least two model arms; got "
                f"{len(self.models)}"
            )
        self.frozen_at = at
        self.frozen_hash = self.compute_hash()
        return self.frozen_hash

    def require_frozen(self) -> str:
        if not self.is_frozen:
            raise ManifestError(
                f"study {self.study_id} is a draft, so no trial may be "
                f"collected against it. Freezing first is what makes the "
                f"analysis preregistered rather than chosen after the fact."
            )
        return self.frozen_hash  # type: ignore[return-value]

    def verify_integrity(self) -> None:
        if not self.is_frozen:
            return
        actual = self.compute_hash()
        if actual != self.frozen_hash:
            raise ManifestError(
                f"study {self.study_id} was edited after it was frozen.\n"
                f"  frozen: {self.frozen_hash}\n"
                f"  now:    {actual}\n"
                f"Every trial already collected was collected under the frozen "
                f"manifest. Restore it, or start a new study revision."
            )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["models"] = [asdict(m) for m in self.models]
        d["cues"] = [asdict(c) for c in self.cues]
        d["frames"] = [{**asdict(f), "stop": list(f.stop)} for f in self.frames]
        d["rank_weights"] = list(self.rank_weights)
        d["manifest_format_version"] = MANIFEST_FORMAT_VERSION
        return d

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return p


def manifest_from_dict(d: dict[str, Any]) -> Manifest:
    d = dict(d)
    d.pop("manifest_format_version", None)
    d["models"] = [ModelRef(**m) for m in d.get("models", [])]
    d["cues"] = [Cue(**c) for c in d.get("cues", [])]
    d["frames"] = [
        PromptFrame(**{**f, "stop": tuple(f.get("stop", ()))}) for f in d.get("frames", [])
    ]
    if "rank_weights" in d:
        d["rank_weights"] = tuple(d["rank_weights"])
    return Manifest(**d)


def load_manifest(path: str | Path) -> Manifest:
    """Load and immediately re-derive the hash — a post-freeze edit fails here,
    not at analysis time when the numbers are already on screen."""
    m = manifest_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    m.verify_integrity()
    return m


@dataclass
class TrialRecord:
    """One request/response pair and everything needed to audit it (§6.1).

    `raw_output` is append-only. A correction is a derived normalization row,
    never an overwrite: what the model actually returned is the evidence.
    """

    study_id: str
    arm: str
    cue: str
    frame_id: str
    model_key: str
    repeat: int
    block: int
    prompt: str
    prompt_sha: str
    raw_output: str | None = None
    associates: list[str] = field(default_factory=list)
    valid: bool = False
    invalid_reason: str = ""
    requested_model: str = ""
    served_model: str = ""
    response_id: str = ""
    fingerprint: str = ""
    latency_ms: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    parser_version: int = 1
    created: str = ""
    error: str = ""


def prompt_sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
