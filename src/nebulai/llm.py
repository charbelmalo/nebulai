"""The shared LLM layer: credentials, model identity, the cost gate, transport.

Two product surfaces in this repo talk to language models — `backend/name.py`
titles clusters, `frontends/probe.py` grows a concept cloud from a seed — and
before either of them moves a single token they both need the same four things:
a credential resolved from wherever this box happens to keep it, a model id that
is *the model that was asked for*, a spend estimate that is allowed to refuse,
and a parser that survives a server which ignores `response_format`. That is
what lives here.

It lives here rather than inside the namer because probe was reaching into
`backend/name.py` for seven underscore-private symbols to get at it. A second
product depending on the first product's privates is the wrong shape twice over:
it makes the namer un-editable without breaking probe, and it makes these rules
read as naming rules when they are house rules that every endpoint call in the
project is held to.

The rules are unchanged by the move, and they are the whole reason this layer
exists at all:

**Identity.** `same_model` is deliberately narrow — case and an ollama `:tag`
are tolerated because those are the same model, and nothing else is. A pin means
THIS model, so a reachable *different* model is not a fallback, it is a
fabrication, and `IdentityError` says which backends were tried and why each
declined instead of quietly substituting one.

**Cost.** `cost_gate` prices the job from the corpus's measured per-token rates
before anything reaches a paid endpoint. Over the ceiling it raises
`BudgetError` with the arithmetic and a list of cheaper models it deliberately
did NOT select — auto-downgrading to fit a budget is the same substitution bug
arriving from the money side. `actual_cost` then prices what was really spent,
because the estimate is an upper bound by construction and stamping it would
overstate every run.

Callers reach all of this through the module object (`from .. import llm`, then
`llm.chat_openai(...)`) rather than by importing the names one by one. The
functions here read `HF_TOKEN_FILE`, `DEFAULT_ENV_FILE` and the rest as module
globals at CALL time, so a test can point them at a tmp_path; a `from` import at
the call site would freeze a copy of the old value and the patch would silently
do nothing while the call went to the real network.

`NamerIdentityError` and `NamerBudgetError` remain below as aliases of these
same two classes: that is the spelling the namer raised under before this split,
and `except NamerIdentityError` in code that predates it must keep catching.
"""

import json
import os
import urllib.request
from pathlib import Path

from .corpus import CORPUS, DEFAULT_MAX_COST_USD, ModelSpec, estimate_naming_cost


class IdentityError(RuntimeError):
    """A pinned model could not be served, and substituting is not allowed.

    Raised only when the caller pinned an identity (`model=` / `--namer-model`).
    In `auto` there is no identity to violate, so the chain still falls through
    — it just has to stamp whatever answered.
    """


class BudgetError(RuntimeError):
    """The estimated spend exceeds the ceiling, so nothing was sent.

    Deliberately terminal: it is NOT caught by the fall-through chain. Quietly
    continuing to a cheaper model after refusing would be the auto-downgrade
    this exists to prevent, and quietly continuing to centroid would hide a
    refusal the human asked to be told about.
    """


# the Namer* spelling is the historical one these two were raised under while
# they lived in backend/name.py; same objects, so existing `except` sites catch.
NamerIdentityError = IdentityError
NamerBudgetError = BudgetError


DEFAULT_ENV_FILE = "~/.config/nebulai/.env"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# HF Inference Providers speak the OpenAI chat protocol behind one router, so
# the same request/parse code serves both remotes — see
# backend.name._name_with_remote.
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN_FILE = "~/.cache/huggingface/token"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"  # local ollama server
DEFAULT_LLM_HOST = "http://localhost:8050"  # OpenAI-compatible chat server


# --- model identity -------------------------------------------------------
#
# The whole point of the corpus is that `endpoint` is a pinned model id. These
# helpers are what stop a pin from drifting into a family or a neighbour.


def corpus_entry(model_id: str) -> ModelSpec | None:
    """The corpus spec this id names, by key / repo / endpoint / hf_endpoint.

    `corpus.spec()` raises and does not know about `hf_endpoint`; naming needs a
    lookup that can also say "not in the corpus" without an exception, because
    most OpenRouter slugs legitimately aren't.
    """
    want = (model_id or "").strip().lower()
    if not want:
        return None
    for s in CORPUS.values():
        for alias in (s.key, s.repo, s.endpoint, s.hf_endpoint):
            if alias and alias.lower() == want:
                return s
    return None


def _pin_aliases(pinned: str) -> list[str]:
    """Every id that names the SAME model as `pinned` (its corpus row, if any).

    A corpus model is reachable under four spellings — `muse-glimmer-30b`, the
    HF repo, the OpenRouter slug, the HF router id — and they are one model, so
    a pin written in any of them must match a backend serving any other.
    """
    s = corpus_entry(pinned)
    if s is None:
        return [pinned.strip()]
    return [a for a in (s.key, s.repo, s.endpoint, s.hf_endpoint) if a]


def same_model(served: str, wanted: str) -> bool:
    """True only when `served` IS `wanted` — never a family or fragment match.

    Two tolerances, both measured rather than assumed:

    * case. The HF router serves `google/gemma-4-26B-A4B-it` while the corpus
      writes `google/gemma-4-26b-a4b-it` (checked against /v1/models on
      2026-08-12). That is one repo, and a case-exact compare would refuse a
      model that is in fact being served.
    * an ollama `:tag`. `mymodel:q4_K_M` is a BUILD of `mymodel`, not another
      model — so it matches, and the *tag* is what gets stamped into
      `namer_model`, because the quantisation is part of what answered.

    Nothing else matches. `openai_pick_model`'s substring fallback is exactly
    the substitution a pin exists to forbid, so the pinned path never uses it.
    """
    a, b = (served or "").strip().lower(), (wanted or "").strip().lower()
    if not a or not b:
        return False
    return a == b or a.split(":", 1)[0] == b or a == b.split(":", 1)[0]


def serves_pin(served: str, pinned: str) -> bool:
    return any(same_model(served, alias) for alias in _pin_aliases(pinned))


# --- the cost gate --------------------------------------------------------


def is_free(model_id: str, s: ModelSpec | None) -> bool:
    """A genuinely $0 endpoint. Corpus price wins; otherwise the `:free` suffix
    OpenRouter uses to mark its no-charge variants."""
    if s is not None:
        return s.usd_in == 0.0 and s.usd_out == 0.0
    return model_id.strip().endswith(":free")


def _alternatives(exclude: str, n_clusters: int, batch_size: int) -> list[tuple[str, str, float]]:
    """Cheaper corpus models for this job, cheapest first. Listed for a human
    to choose from — deliberately never auto-selected."""
    rows = []
    for s in CORPUS.values():
        if s.key == exclude:
            continue
        rows.append((s.key, s.endpoint, estimate_naming_cost(n_clusters, s.key, batch_size)))
    return sorted(rows, key=lambda r: r[2])


def cost_gate(
    model_id: str,
    n_clusters: int,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
    batch_size: int = 15,
) -> float | None:
    """Estimate the spend before sending. Returns USD, or None if unpriceable.

    Raises `BudgetError` over budget. It never returns a *different* model
    to fit the ceiling: the alternatives go in the message for a human to pick
    from, because swapping the model silently changes what the map is evidence
    of — the exact failure `IdentityError` exists to prevent, arrived at
    from the money side instead of the reachability side.

    A model with no corpus row cannot be priced (most OpenRouter slugs), so the
    ceiling is unenforceable for it and this says so rather than pretending.
    """
    s = corpus_entry(model_id)
    if is_free(model_id, s):
        return 0.0
    if s is None:
        print(
            f"  cost: {model_id} has no corpus price — the "
            f"${max_cost_usd:.2f} ceiling cannot be enforced for it"
        )
        return None

    est = estimate_naming_cost(n_clusters, s.key, batch_size)
    if est <= max_cost_usd:
        print(
            f"  cost: ~${est:.4f} to name {n_clusters} clusters on "
            f"{s.endpoint} (ceiling ${max_cost_usd:.2f}) — proceeding"
        )
        return est

    batches = -(-n_clusters // batch_size)
    lines = [
        f"naming {n_clusters} clusters on {s.endpoint} would cost about "
        f"${est:.4f}, over the ${max_cost_usd:.2f} ceiling (--max-cost-usd).",
        f"  {batches} batches x {batch_size} clusters, ~1500 prompt + ~400 "
        f"completion tokens each, at ${s.usd_in}/M in and ${s.usd_out}/M out.",
        "cheaper corpus models for this job — NOT selected, because a silent "
        "downgrade would change which model the map is evidence of:",
    ]
    for key, endpoint, alt in _alternatives(s.key, n_clusters, batch_size):
        if alt < est:
            lines.append(f"    {key:<16} {endpoint:<34} ~${alt:.4f}")
    lines.append(
        f"re-run with --namer-model <one of the above>, or raise the ceiling "
        f"with --max-cost-usd {est:.4f}"
    )
    raise BudgetError("\n".join(lines))


def accumulate_usage(payload: dict, usage: dict) -> None:
    """Add one response's reported token counts to the running total.

    Actual usage, not the estimate, is what gets stamped — the estimate is an
    upper bound by construction (see corpus.estimate_naming_cost) and reporting
    it as the cost would overstate every run.
    """
    got = payload.get("usage") or {}
    usage["prompt_tokens"] += int(got.get("prompt_tokens") or 0)
    usage["completion_tokens"] += int(got.get("completion_tokens") or 0)
    if got.get("cost") is not None:  # OpenRouter's own authoritative number
        usage["cost"] = (usage.get("cost") or 0.0) + float(got["cost"])


def actual_cost(model_id: str, usage: dict) -> float | None:
    """USD actually spent, from reported usage. None when unpriceable.

    The provider's own `cost` wins when it reports one; otherwise the corpus's
    measured per-token prices are applied to the reported token counts.
    """
    if usage.get("cost") is not None:
        return round(float(usage["cost"]), 8)
    s = corpus_entry(model_id)
    if s is None:
        return None
    if not (usage.get("prompt_tokens") or usage.get("completion_tokens")):
        return None
    # 8 places, not 6: one naming batch on Ling costs $0.000027, and rounding a
    # real spend to $0.00 is the kind of "free" that stops being true at scale
    return round(
        (usage["prompt_tokens"] * s.usd_in + usage["completion_tokens"] * s.usd_out)
        / 1e6,
        8,
    )


# --- credentials ----------------------------------------------------------


def load_openrouter_key(env_file: str | None) -> str | None:
    """os.environ first, then the last uncommented OPENROUTER_API_KEY= in the
    .env file."""
    return load_key("OPENROUTER_API_KEY", env_file)


def load_key(var: str, env_file: str | None) -> str | None:
    """os.environ first, then the last uncommented `<var>=` in the .env file."""
    key = os.environ.get(var)
    if key:
        return key.strip()
    path = Path(env_file or DEFAULT_ENV_FILE).expanduser()
    if not path.exists():
        return None
    found = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith(f"{var}="):
            found = s.split("=", 1)[1].strip().strip("'\"")
    return found or None


def load_hf_token(env_file: str | None) -> str | None:
    """HF_TOKEN / HUGGINGFACE_HUB_TOKEN, the .env file, then the CLI's own
    token file — the same order `huggingface_hub` itself resolves in, so a box
    already logged in with `hf auth login` needs no extra configuration."""
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        tok = load_key(var, env_file)
        if tok:
            return tok
    path = Path(HF_TOKEN_FILE).expanduser()
    if path.exists():
        tok = path.read_text().strip()
        if tok:
            return tok
    return None


# --- ollama ---------------------------------------------------

def ollama_tags(host: str) -> list[str]:
    """Text-capable model tags on an ollama server ([] when unreachable)."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as r:
            tags = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []
    # never pick an embedding model for text generation
    return [t for t in tags if "embed" not in t.lower()]


def ollama_pick_model(host: str, preferred: str) -> str | None:
    tags = ollama_tags(host)
    if not tags:
        return None
    for t in tags:
        if t == preferred or t.startswith(preferred + ":"):
            return t
    return tags[0]


# --- OpenAI-compatible chat server ----------------------------------------

# /v1/models on a multi-purpose server (the M4 lists rerankers, whisper, TTS,
# embedders and a diffusion checkpoint alongside its chat model) — anything
# matching these is not a text generator and must never be auto-picked.
# A reasoning model's scratchpad alone ran 300-700 tokens here on trivial
# prompts and blew past 2048 on a 12-concept expansion. An unused ceiling is
# free — the model stops when it stops — so this floor is set high enough that
# truncation is the exception, not the thing every caller has to tune around.
MIN_CHAT_TOKENS = 4096


class ChatTruncated(RuntimeError):
    """The model kept generating until it hit the token ceiling.

    Distinct from every other chat failure because it means the opposite thing:
    an unreachable host or a non-chat model produces no output at all, whereas
    this one produced *too much*. Callers probing whether a backend works at all
    should treat it as a pass — see frontends.probe._make_expander, where
    treating it as a failure retired a perfectly good generator.
    """

NON_CHAT_HINTS = (
    "embed",
    "rerank",
    "whisper",
    "tts",
    "flux",
    "nsfw",
    "clip",
    "vae",
    # sentence-transformer encoders whose ids do NOT contain "embed" —
    # all-MiniLM-L6-v2 is served by the M4 on the chat port too, and picking it
    # as a generator fails deep in the run with a shape error, not a clear one
    "minilm",
    "sentence-transformers",
    "bge-",
    "gte-",
    "e5-",
)


def json_object(text: str) -> dict:
    """Parse one JSON object out of an LLM reply.

    Servers that honour `response_format` return bare JSON, but the same model
    behind a server that ignores it wraps the object in ```json fences and
    sometimes prefaces it with a sentence. Falling back to the first balanced
    {...} keeps one backend working against both instead of failing on syntax.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        s = s.rsplit("```", 1)[0].strip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start : i + 1])
    raise ValueError(f"unbalanced JSON in reply: {text[:200]!r}")


def openai_list_models(host: str, api_key: str | None = None) -> list[str]:
    """Model ids advertised by an OpenAI-compatible server ([] if unreachable).

    Split out from openai_pick_model because the PINNED path must compare
    against this raw list itself: the picker's substring fallback below is
    exactly the "close enough" match that a pinned identity forbids.
    """
    req = urllib.request.Request(f"{host.rstrip('/')}/v1/models")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return [m["id"] for m in json.load(r).get("data", [])]
    except Exception:
        return []


def openai_pick_model(host: str, preferred: str, api_key: str | None = None) -> str | None:
    """Resolve a chat model id on an OpenAI-compatible server, or None."""
    ids = openai_list_models(host, api_key)
    if not ids:
        return None
    if preferred and preferred in ids:
        return preferred
    # a caller-supplied fragment ("qwen") should still resolve rather than
    # silently falling through to an unrelated model
    if preferred:
        for i in ids:
            if preferred.lower() in i.lower():
                return i
    chat = [i for i in ids if not any(h in i.lower() for h in NON_CHAT_HINTS)]
    return chat[0] if chat else None


def chat_openai(
    host: str,
    model: str,
    system: str,
    user: str,
    schema: dict,
    schema_name: str,
    api_key: str | None = None,
    max_tokens: int = 2000,
    timeout: float = 300.0,
) -> dict:
    """One structured chat completion against an OpenAI-compatible server.

    Reasoning models put their scratchpad in a separate `reasoning_content`
    field, so `content` is already clean — but json_object() still guards the
    servers that inline the fences.

    Those same models spend max_tokens on the scratchpad BEFORE emitting any
    content, and how much is unpredictable — 300-700 tokens for identical-shaped
    prompts here, but a distilled reasoner burned through 8192 on one ordinary
    12-concept expansion. A budget sized for the answer alone therefore truncates
    mid-fence and surfaces as a bogus JSON error, so a `length` finish doubles
    the budget and retries. Retries are only ever paid on an actual truncation,
    so the ceiling is set where a genuinely stuck model stops, not where a
    verbose one does.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    budget = max(MIN_CHAT_TOKENS, max_tokens)
    for attempt in (1, 2, 3):
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": budget,
            "temperature": 0.2,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
        req = urllib.request.Request(
            f"{host.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
        if "choices" not in payload:
            raise RuntimeError(f"{host} returned no choices: {str(payload)[:200]}")
        choice = payload["choices"][0]
        if choice.get("finish_reason") == "length" and attempt < 3:
            budget *= 2
            continue
        content = choice["message"].get("content") or ""
        if choice.get("finish_reason") == "length":
            raise ChatTruncated(
                f"{model} truncated its reply at {budget} tokens (reasoning "
                "scratchpad consumed the budget) — raise max_tokens or use a "
                "smaller batch"
            )
        return json_object(content)
    raise AssertionError("unreachable")
