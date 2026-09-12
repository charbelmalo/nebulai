"""Frozen prompt sets.

A prompt set in here is a *pinned artifact*, not a configuration file: its
sha256 is stamped into every space or direction computed from it, and changing
one makes a new id rather than an edit in place. The same discipline
`backend/instrument.py` applies to its question set, for the same reason — a
coordinate system whose basis silently moved is worse than no coordinate
system, because figures made a month apart still look comparable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent


class PromptSetError(ValueError):
    """A prompt set is missing, malformed, or not the one that was asked for."""


def prompt_set_path(prompt_set_id: str) -> Path:
    """The file for `prompt_set_id`, e.g. `personas.v1` -> personas.v1.json."""
    if "/" in prompt_set_id or "\\" in prompt_set_id or prompt_set_id.startswith("."):
        raise PromptSetError(f"{prompt_set_id!r} is not a prompt-set id")
    return PROMPTS_DIR / f"{prompt_set_id}.json"


def load_prompt_set(prompt_set_id: str) -> tuple[dict[str, Any], str]:
    """Return `(document, sha256-of-the-file-bytes)`.

    The digest is over the raw bytes rather than over a re-serialised dict so
    that reformatting the file — which changes nothing semantically — still
    registers as a change. A prompt set is frozen; "semantically equivalent" is
    not the standard.
    """
    path = prompt_set_path(prompt_set_id)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        have = sorted(p.stem for p in PROMPTS_DIR.glob("*.json"))
        raise PromptSetError(
            f"no prompt set {prompt_set_id!r} in {PROMPTS_DIR}; have {have}"
        ) from None
    doc = json.loads(raw)
    if doc.get("id") != prompt_set_id:
        raise PromptSetError(
            f"{path.name} declares id {doc.get('id')!r}, not {prompt_set_id!r}"
        )
    return doc, hashlib.sha256(raw).hexdigest()


__all__ = ["PROMPTS_DIR", "PromptSetError", "load_prompt_set", "prompt_set_path"]
