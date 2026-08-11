"""Cluster naming with a pluggable backend chain.

auto mode tries: a local ollama server (free, private) -> an OpenAI-compatible
chat server (also local: LM Studio, vLLM, llama.cpp, an MLX box on the LAN) ->
OpenRouter (key from env or a .env file) -> centroid fallback (title = the
members nearest the cluster centroid). The pipeline therefore always
completes; the LLM namers simply activate when one is reachable or a key is
present. Anthropic stays available via `--namer anthropic`, HF Inference
Providers via `--namer hf`.

Two rules constrain that chain, and both exist because a map's titles are
evidence about a specific model.

**Identity.** `auto` means "any namer": the caller has expressed no identity
requirement, so falling through is honest — provided the map records what
actually answered. `model=` (CLI `--namer-model`) means "THIS model", and then
a reachable *different* model is not a fallback, it is a fabrication: the
export would claim Glimmer's semantics while carrying some other model's. So a
pin never substitutes. If no configured backend can serve that exact id,
`NamerIdentityError` says which backends were tried and why each declined, and
the centroid fallback is not offered either — centroid is not the pinned model.

**Cost.** Before anything reaches a paid endpoint the spend is estimated from
the real cluster count and the corpus's measured per-token prices. Over
`max_cost_usd` the run is REFUSED (`NamerBudgetError`) with the numbers, and the
cheaper corpus models are printed but never selected — auto-downgrading to fit a
budget is the same substitution bug wearing a different hat. A $0.00 endpoint
(Gemma-4) skips the gate entirely.

Whatever path runs, `units.meta` is stamped with `namer_backend`, `namer_model`
(the exact id that answered), `namer_identity` and — when the endpoint reports
usage — `namer_cost_usd`, so an exported map can always say which model titled
it. `export_json` splats `units.meta`, so those keys land in `nebulai.json`.
"""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from ..corpus import CORPUS, DEFAULT_MAX_COST_USD, ModelSpec, estimate_naming_cost
from ..units import Units


class NamerIdentityError(RuntimeError):
    """A pinned model could not be served, and substituting is not allowed.

    Raised only when the caller pinned an identity (`model=` / `--namer-model`).
    In `auto` there is no identity to violate, so the chain still falls through
    — it just has to stamp whatever answered.
    """


class NamerBudgetError(RuntimeError):
    """The estimated spend exceeds the ceiling, so nothing was sent.

    Deliberately terminal: it is NOT caught by the fall-through chain. Quietly
    continuing to a cheaper model after refusing would be the auto-downgrade
    this exists to prevent, and quietly continuing to centroid would hide a
    refusal the human asked to be told about.
    """

_SYSTEM = (
    "You name clusters of tokens drawn from a language model's vocabulary. "
    "For each cluster you are shown representative member tokens (quoted; a "
    "leading space inside the quotes is part of the token). Reply with a "
    "short, specific title of 2-5 words describing what unites the members, "
    "e.g. 'days & months', 'programming keywords', 'country names'. "
    "Prefer concrete categories over vague ones."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                },
                "required": ["id", "title"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["titles"],
    "additionalProperties": False,
}

_DEFAULT_ENV_FILE = "~/.config/nebulai/.env"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# HF Inference Providers speak the OpenAI chat protocol behind one router, so
# the same request/parse code serves both remotes — see _name_with_remote.
_HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
_HF_TOKEN_FILE = "~/.cache/huggingface/token"
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"  # local ollama server
_DEFAULT_LLM_HOST = "http://localhost:8050"  # OpenAI-compatible chat server


def _representatives(
    units: Units, member_idx: np.ndarray, k: int = 20
) -> list[str]:
    """Members nearest the cluster centroid (cosine), most central first."""
    V = units.vectors[member_idx]
    c = V.mean(axis=0)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    cn = c / (np.linalg.norm(c) + 1e-8)
    order = np.argsort(-(Vn @ cn))
    return [units.labels[member_idx[i]] for i in order[:k]]


def _batch_lines(reps: dict[int, list[str]], cids: list[int]) -> str:
    return "\n".join(
        f"cluster {cid}: " + ", ".join(repr(t) for t in reps[cid]) for cid in cids
    )


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

    Nothing else matches. `_openai_pick_model`'s substring fallback is exactly
    the substitution a pin exists to forbid, so the pinned path never uses it.
    """
    a, b = (served or "").strip().lower(), (wanted or "").strip().lower()
    if not a or not b:
        return False
    return a == b or a.split(":", 1)[0] == b or a == b.split(":", 1)[0]


def _serves_pin(served: str, pinned: str) -> bool:
    return any(same_model(served, alias) for alias in _pin_aliases(pinned))


# --- the cost gate --------------------------------------------------------


def _is_free(model_id: str, s: ModelSpec | None) -> bool:
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

    Raises `NamerBudgetError` over budget. It never returns a *different* model
    to fit the ceiling: the alternatives go in the message for a human to pick
    from, because swapping the model silently changes what the map is evidence
    of — the exact failure `NamerIdentityError` exists to prevent, arrived at
    from the money side instead of the reachability side.

    A model with no corpus row cannot be priced (most OpenRouter slugs), so the
    ceiling is unenforceable for it and this says so rather than pretending.
    """
    s = corpus_entry(model_id)
    if _is_free(model_id, s):
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
    raise NamerBudgetError("\n".join(lines))


# --- remote OpenAI-protocol endpoints (OpenRouter, HF Inference Providers) --


def _load_openrouter_key(env_file: str | None) -> str | None:
    """os.environ first, then the last uncommented OPENROUTER_API_KEY= in the
    .env file."""
    return _load_key("OPENROUTER_API_KEY", env_file)


def _load_key(var: str, env_file: str | None) -> str | None:
    """os.environ first, then the last uncommented `<var>=` in the .env file."""
    key = os.environ.get(var)
    if key:
        return key.strip()
    path = Path(env_file or _DEFAULT_ENV_FILE).expanduser()
    if not path.exists():
        return None
    found = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith(f"{var}="):
            found = s.split("=", 1)[1].strip().strip("'\"")
    return found or None


def _load_hf_token(env_file: str | None) -> str | None:
    """HF_TOKEN / HUGGINGFACE_HUB_TOKEN, the .env file, then the CLI's own
    token file — the same order `huggingface_hub` itself resolves in, so a box
    already logged in with `hf auth login` needs no extra configuration."""
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        tok = _load_key(var, env_file)
        if tok:
            return tok
    path = Path(_HF_TOKEN_FILE).expanduser()
    if path.exists():
        tok = path.read_text().strip()
        if tok:
            return tok
    return None


def _remote_body(reps: dict[int, list[str]], batch: list[int], model: str) -> dict:
    """The one request shape both remotes send, so they cannot drift apart."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": "Name each cluster.\n\n" + _batch_lines(reps, batch),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "cluster_titles",
                "strict": True,
                "schema": _SCHEMA,
            },
        },
        "max_tokens": 2000,
        "temperature": 0.2,
    }


def _accumulate_usage(payload: dict, usage: dict) -> None:
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


def _name_with_remote(
    reps: dict[int, list[str]],
    url: str,
    headers: dict[str, str],
    model: str,
    batch_size: int = 15,
    usage: dict | None = None,
    expect_model: str | None = None,
    timeout: float = 120.0,
) -> dict[int, str]:
    """Batched structured naming against an OpenAI-protocol remote.

    Shared by OpenRouter and the HF router: same batching, same json_schema
    response_format, same parse. The only differences between the two are the
    URL and the auth header, which is why they are parameters here rather than
    a second copy of this function.

    `expect_model` is the anti-substitution check on the wire. Both routers echo
    the model they actually ran in the completion, and a router that quietly
    served a neighbour (a `:free` variant, a provider's own repack) would
    otherwise be indistinguishable from one that honoured the request.
    """
    titles: dict[int, str] = {}
    cids = sorted(reps)
    for start in range(0, len(cids), batch_size):
        batch = cids[start : start + batch_size]
        body = json.dumps(_remote_body(reps, batch, model)).encode()
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
        if "choices" not in payload:
            raise RuntimeError(f"{url} returned no choices: {str(payload)[:200]}")
        served = str(payload.get("model") or "")
        if expect_model and served and not same_model(served, expect_model):
            raise NamerIdentityError(
                f"asked {url} for {expect_model!r} and it answered as "
                f"{served!r} — a different model's titles are not this "
                "model's semantics, so the reply is discarded"
            )
        if usage is not None:
            _accumulate_usage(payload, usage)
        content = payload["choices"][0]["message"].get("content") or ""
        for item in json_object(content).get("titles", []):
            cid = int(item["id"])
            if cid in reps:  # a hallucinated id must not invent a cluster
                titles[cid] = str(item["title"]).strip() or "unnamed"
    return titles


def _name_with_openrouter(
    reps: dict[int, list[str]],
    model: str,
    env_file: str | None,
    batch_size: int = 15,
    usage: dict | None = None,
    expect_model: str | None = None,
) -> dict[int, str]:
    key = _load_openrouter_key(env_file)
    if not key:
        raise RuntimeError("no OPENROUTER_API_KEY in env or .env file")
    body_headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Title": "Nebul.AI",
    }
    return _name_with_remote(
        reps,
        _OPENROUTER_URL,
        body_headers,
        model,
        batch_size=batch_size,
        usage=usage,
        expect_model=expect_model,
    )


def _name_with_hf(
    reps: dict[int, list[str]],
    model: str,
    env_file: str | None,
    batch_size: int = 15,
    usage: dict | None = None,
    expect_model: str | None = None,
) -> dict[int, str]:
    """Name via HF Inference Providers' OpenAI-compatible router.

    The model id here is the HF repo (`meta-models/Muse-Glimmer-30B`), not an
    OpenRouter slug; the router picks the provider serving it. Corpus rows carry
    `hf_endpoint=None` where no provider serves the model at all (Ling), and
    that is a refusal, never a reroute to a different model.
    """
    token = _load_hf_token(env_file)
    if not token:
        raise RuntimeError(
            "no HF token in HF_TOKEN / HUGGINGFACE_HUB_TOKEN, the .env file, or "
            f"{_HF_TOKEN_FILE}"
        )
    return _name_with_remote(
        reps,
        _HF_ROUTER_URL,
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        model,
        batch_size=batch_size,
        usage=usage,
        expect_model=expect_model,
    )


# --- ollama ---------------------------------------------------

def _ollama_tags(host: str) -> list[str]:
    """Text-capable model tags on an ollama server ([] when unreachable)."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as r:
            tags = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []
    # never pick an embedding model for text generation
    return [t for t in tags if "embed" not in t.lower()]


def _ollama_pick_model(host: str, preferred: str) -> str | None:
    tags = _ollama_tags(host)
    if not tags:
        return None
    for t in tags:
        if t == preferred or t.startswith(preferred + ":"):
            return t
    return tags[0]


def _name_with_ollama(
    reps: dict[int, list[str]], host: str, preferred: str
) -> dict[int, str]:
    model = _ollama_pick_model(host, preferred)
    if model is None:
        raise RuntimeError(f"ollama at {host} unreachable or no text models")
    titles: dict[int, str] = {}
    for cid, tokens in reps.items():
        prompt = (
            _SYSTEM
            + "\n\nTokens: "
            + ", ".join(repr(t) for t in tokens)
            + '\n\nReply as JSON: {"title": "<2-5 word cluster name>"}'
        )
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"num_predict": 60},
            }
        ).encode()
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)["response"]
        titles[cid] = str(json.loads(out).get("title", "")).strip() or "unnamed"
    return titles


# --- OpenAI-compatible chat server ----------------------------------------

# /v1/models on a multi-purpose server (the M4 lists rerankers, whisper, TTS,
# embedders and a diffusion checkpoint alongside its chat model) — anything
# matching these is not a text generator and must never be auto-picked.
# A reasoning model's scratchpad alone ran 300-700 tokens here on trivial
# prompts and blew past 2048 on a 12-concept expansion. An unused ceiling is
# free — the model stops when it stops — so this floor is set high enough that
# truncation is the exception, not the thing every caller has to tune around.
_MIN_CHAT_TOKENS = 4096


class ChatTruncated(RuntimeError):
    """The model kept generating until it hit the token ceiling.

    Distinct from every other chat failure because it means the opposite thing:
    an unreachable host or a non-chat model produces no output at all, whereas
    this one produced *too much*. Callers probing whether a backend works at all
    should treat it as a pass — see frontends.probe._make_expander, where
    treating it as a failure retired a perfectly good generator.
    """

_NON_CHAT_HINTS = (
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


def _openai_list_models(host: str, api_key: str | None = None) -> list[str]:
    """Model ids advertised by an OpenAI-compatible server ([] if unreachable).

    Split out from _openai_pick_model because the PINNED path must compare
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


def _openai_pick_model(host: str, preferred: str, api_key: str | None = None) -> str | None:
    """Resolve a chat model id on an OpenAI-compatible server, or None."""
    ids = _openai_list_models(host, api_key)
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
    chat = [i for i in ids if not any(h in i.lower() for h in _NON_CHAT_HINTS)]
    return chat[0] if chat else None


def _chat_openai(
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

    budget = max(_MIN_CHAT_TOKENS, max_tokens)
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


def _name_with_openai(
    reps: dict[int, list[str]],
    host: str,
    model: str,
    api_key: str | None = None,
    batch_size: int = 20,
) -> dict[int, str]:
    """Name every cluster via an OpenAI-compatible chat server, in batches.

    Batching matters on a local box: one cluster per request costs a full
    prefill of the system prompt each time, so a 200-cluster map goes from
    ~200 round-trips to ~10.
    """
    titles: dict[int, str] = {}
    cids = sorted(reps)
    for start in range(0, len(cids), batch_size):
        batch = cids[start : start + batch_size]
        got = _chat_openai(
            host,
            model,
            _SYSTEM,
            "Name each cluster.\n\n" + _batch_lines(reps, batch),
            _SCHEMA,
            "cluster_titles",
            api_key=api_key,
            max_tokens=600 + 60 * len(batch),
        )
        for item in got.get("titles", []):
            cid = int(item["id"])
            if cid in reps:  # a hallucinated id must not invent a cluster
                titles[cid] = str(item["title"]).strip() or "unnamed"
    missing = [c for c in cids if c not in titles]
    if missing:
        raise RuntimeError(
            f"{host} named {len(titles)}/{len(cids)} clusters "
            f"(missing {missing[:5]}{'...' if len(missing) > 5 else ''})"
        )
    return titles


# --- Claude CLI -----------------------------------------------------------
#
# Names clusters by shelling out to the `claude` binary in headless mode, so the
# work runs against an existing Claude subscription instead of being billed per
# token through the Anthropic API. Same prompt and same parser as every other
# backend; only the transport differs.
#
# The flags are not optional decoration. A bare `claude -p` loads the full Claude
# Code harness — system prompt, tool schemas, MCP servers, project CLAUDE.md —
# which measured 41,454 tokens of context on a one-word prompt, dwarfing the
# ~1.2k the naming prompt itself needs. Stripping settings, MCP and tools cut
# that to 6,394 new tokens against a 19k cached prefix on the same probe. The
# harness is pure overhead here: this task wants one turn, no tools, no repo
# context, so anything the agent could reach for is a liability rather than a
# capability.
_CLAUDE_CLI_FLAGS = [
    "--output-format", "json",
    "--setting-sources", "",          # no CLAUDE.md / settings.json
    "--strict-mcp-config",
    "--mcp-config", '{"mcpServers":{}}',  # no MCP tool schemas
    "--allowed-tools", "",            # one turn, no tool use
]

# a chunk of 15 clusters x 20 representatives is ~1.2k tokens in and ~250 out;
# the ceiling is for a wedged process, not for normal work
_CLAUDE_CLI_TIMEOUT = 300


def _cli_prompt(reps: dict[int, list[str]], batch: list[int]) -> str:
    """The one prompt shared by every CLI-transport namer.

    Agent CLIs have no structured-output parameter the way the APIs do, so the
    JSON contract has to live in the prompt itself. Keeping it in one place is
    what makes a claude-vs-codex comparison a comparison of MODELS rather than
    of two prompts that drifted apart.
    """
    return (
        f"{_SYSTEM}\n\nName each cluster. Reply with ONLY a JSON object of the "
        'form {"titles": [{"id": <cluster id>, "title": "<2-5 words>"}]} — one '
        "entry per cluster, no prose, no code fence.\n\n"
        + _batch_lines(reps, batch)
    )


def _ingest_titles(text: str, titles: dict[int, str]) -> bool:
    """Merge a CLI reply's titles in place; True if it contributed any."""
    before = len(titles)
    try:
        for item in json_object(str(text)).get("titles", []):
            cid, title = item.get("id"), str(item.get("title", "")).strip()
            if cid is not None and title:
                titles[int(cid)] = title
    except Exception:
        return False
    return len(titles) > before


def _name_with_claude_cli(
    reps: dict[int, list[str]],
    model: str = "",
    batch_size: int = 15,
    binary: str = "claude",
) -> dict[int, str]:
    """Name every cluster through the `claude` CLI, one subprocess per chunk.

    Chunks run sequentially rather than in parallel: they share one subscription
    rate limit, and the prompt prefix is identical across chunks so sequential
    calls hit the prompt cache instead of racing to recreate it.
    """
    import subprocess

    cids = sorted(reps)
    titles: dict[int, str] = {}
    failed: list[str] = []

    for start in range(0, len(cids), batch_size):
        batch = cids[start : start + batch_size]
        cmd = [binary, "-p", _cli_prompt(reps, batch), *_CLAUDE_CLI_FLAGS]
        if model:
            cmd += ["--model", model]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_CLAUDE_CLI_TIMEOUT
            )
        except FileNotFoundError as e:
            # a missing binary is a setup problem for the WHOLE run, not one
            # chunk — fail loudly instead of silently degrading to centroid
            raise RuntimeError(
                f"{binary!r} not on PATH — install the Claude CLI or pick "
                "another --namer"
            ) from e
        except subprocess.TimeoutExpired:
            failed.append(f"chunk-{start}:timeout")
            continue

        if proc.returncode != 0:
            failed.append(f"chunk-{start}:exit{proc.returncode}")
            continue

        # the CLI wraps the reply in its own envelope; `result` is the text the
        # model actually produced, which is what the shared parser expects
        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError:
            failed.append(f"chunk-{start}:unparsable-envelope")
            continue
        if envelope.get("is_error"):
            failed.append(f"chunk-{start}:{envelope.get('subtype', 'error')}")
            continue

        if not _ingest_titles(envelope.get("result", ""), titles):
            failed.append(f"chunk-{start}:no-titles")

    if failed and not titles:
        raise RuntimeError(f"every claude-cli chunk failed: {', '.join(failed[:5])}")
    if failed:
        # partial is reported, never silently padded — the caller stamps the
        # shortfall into the namer string so the map discloses it
        print(f"  {len(failed)} claude-cli chunks failed: {failed[:5]}")
    return titles


# --- Codex CLI ------------------------------------------------------------
#
# Same transport idea as claude-cli, against the `codex` binary, so the two can
# be compared on identical prompts. Codex is an agent with a sandbox and a repo
# checkout, none of which naming needs, so the flags disable all of it:
#
#   --ephemeral            no session files left behind per chunk
#   --ignore-user-config   ignore ~/.codex/config.toml, so the run is not
#                          silently steered by whatever model/effort/profile the
#                          user happens to have configured — the point of this
#                          backend is that the model is the variable under test
#   --skip-git-repo-check  naming is not a repo operation
#   -s read-only           the model is asked for JSON; it must not run commands
#   --color never          keep the captured text clean
#
# `--ignore-user-config` drops the configured default model too, so an explicit
# `-m` is required for a meaningful comparison; the caller passes it.
_CODEX_CLI_FLAGS = [
    "--ephemeral",
    "--ignore-user-config",
    "--skip-git-repo-check",
    "-s", "read-only",
    "--color", "never",
]

_CODEX_CLI_TIMEOUT = 600  # reasoning models at high effort are slow per chunk


def _name_with_codex_cli(
    reps: dict[int, list[str]],
    model: str = "",
    batch_size: int = 15,
    binary: str = "codex",
) -> dict[int, str]:
    """Name every cluster through the `codex` CLI, one subprocess per chunk.

    Reads the reply from `--output-last-message` rather than parsing the JSONL
    event stream: the stream interleaves reasoning traces with the answer, and
    scraping the last assistant event out of it would break the first time the
    event schema changed.
    """
    import subprocess
    import tempfile

    cids = sorted(reps)
    titles: dict[int, str] = {}
    failed: list[str] = []

    for start in range(0, len(cids), batch_size):
        batch = cids[start : start + batch_size]
        with tempfile.TemporaryDirectory() as td:
            last = os.path.join(td, "last.txt")
            cmd = [binary, "exec", _cli_prompt(reps, batch), *_CODEX_CLI_FLAGS, "-o", last]
            if model:
                cmd += ["-m", model]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=_CODEX_CLI_TIMEOUT
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"{binary!r} not on PATH — install the Codex CLI or pick "
                    "another --namer"
                ) from e
            except subprocess.TimeoutExpired:
                failed.append(f"chunk-{start}:timeout")
                continue

            if proc.returncode != 0:
                failed.append(f"chunk-{start}:exit{proc.returncode}")
                continue
            try:
                with open(last, encoding="utf-8") as fh:
                    reply = fh.read()
            except OSError:
                failed.append(f"chunk-{start}:no-output-file")
                continue

        if not _ingest_titles(reply, titles):
            failed.append(f"chunk-{start}:no-titles")

    if failed and not titles:
        raise RuntimeError(f"every codex-cli chunk failed: {', '.join(failed[:5])}")
    if failed:
        print(f"  {len(failed)} codex-cli chunks failed: {failed[:5]}")
    return titles


# --- Anthropic (kept for --namer anthropic) -------------------------------

# naming is embarrassingly parallel and nobody is waiting on it interactively,
# which is exactly the Batch API's case: same model, HALF the token cost, and
# the chunks stop being 14 blocking round-trips. Below this many clusters the
# submit+poll overhead isn't worth it and the synchronous path wins on latency.
_BATCH_API_MIN_CLUSTERS = 60
_BATCH_POLL_SECONDS = 10
_BATCH_MAX_WAIT_SECONDS = 3600  # most batches finish well under an hour


def _chunk_params(reps: dict[int, list[str]], batch: list[int], model: str) -> dict:
    """The one request shape, shared by the sync and batch paths so they can
    never drift apart."""
    return {
        "model": model,
        "max_tokens": 2000,
        "system": _SYSTEM,
        "output_config": {"format": {"type": "json_schema", "schema": _SCHEMA}},
        "messages": [
            {
                "role": "user",
                "content": "Name each cluster.\n\n" + _batch_lines(reps, batch),
            }
        ],
    }


def _parse_titles(content, into: dict[int, str]) -> None:
    text = next(b.text for b in content if b.type == "text")
    for item in json.loads(text)["titles"]:
        into[int(item["id"])] = str(item["title"]).strip()


def _name_with_anthropic_batch(
    reps: dict[int, list[str]], model: str, batch_size: int = 15
) -> dict[int, str]:
    """Name every cluster through the Batch API — 50% cheaper, one submission.

    Results come back in ARBITRARY order, so each chunk is keyed by `custom_id`
    and mapped back through `by_id`. Indexing results positionally would
    silently attach the wrong titles to the wrong clusters, which is the exact
    failure this map cannot afford.
    """
    import time

    import anthropic

    client = anthropic.Anthropic()
    cids = sorted(reps)

    by_id: dict[str, list[int]] = {}
    requests = []
    for start in range(0, len(cids), batch_size):
        batch = cids[start : start + batch_size]
        custom_id = f"chunk-{start:06d}"
        by_id[custom_id] = batch
        requests.append({"custom_id": custom_id, "params": _chunk_params(reps, batch, model)})

    submitted = client.messages.batches.create(requests=requests)
    print(
        f"  anthropic batch {submitted.id}: {len(requests)} requests "
        f"({len(cids)} clusters) — polling every {_BATCH_POLL_SECONDS}s"
    )

    waited = 0
    status = submitted
    while status.processing_status != "ended":
        if waited >= _BATCH_MAX_WAIT_SECONDS:
            raise TimeoutError(
                f"batch {submitted.id} still {status.processing_status} after "
                f"{waited}s; it is not cancelled — retrieve it later with "
                f"client.messages.batches.results('{submitted.id}')"
            )
        time.sleep(_BATCH_POLL_SECONDS)
        waited += _BATCH_POLL_SECONDS
        status = client.messages.batches.retrieve(submitted.id)

    titles: dict[int, str] = {}
    failed: list[str] = []
    for entry in client.messages.batches.results(submitted.id):
        if entry.result.type != "succeeded":
            failed.append(f"{entry.custom_id}:{entry.result.type}")
            continue
        _parse_titles(entry.result.message.content, titles)

    if failed and not titles:
        raise RuntimeError(f"every batch request failed: {', '.join(failed[:5])}")
    if failed:
        # partial is reported, never silently padded — the caller stamps the
        # shortfall into the namer string so the map discloses it
        print(f"  {len(failed)} of {len(requests)} batch requests failed: {failed[:5]}")
    return titles


def _name_with_anthropic(
    reps: dict[int, list[str]],
    model: str,
    batch_size: int = 15,
    use_batch_api: bool | None = None,
) -> dict[int, str]:
    """Synchronous naming, chunked. `use_batch_api=None` picks the batch path
    for maps big enough to pay for the submit+poll overhead."""
    import anthropic

    if use_batch_api is None:
        use_batch_api = len(reps) >= _BATCH_API_MIN_CLUSTERS
    if use_batch_api:
        return _name_with_anthropic_batch(reps, model, batch_size)

    client = anthropic.Anthropic()
    titles: dict[int, str] = {}
    cids = sorted(reps)
    for start in range(0, len(cids), batch_size):
        batch = cids[start : start + batch_size]
        resp = client.messages.create(**_chunk_params(reps, batch, model))
        _parse_titles(resp.content, titles)
    return titles


def _name_with_centroid(reps: dict[int, list[str]]) -> dict[int, str]:
    return {
        cid: " · ".join(dict.fromkeys(t.strip() for t in tokens[:4] if t.strip()))
        for cid, tokens in reps.items()
    }


def placeholder_titles(
    cluster_ids: np.ndarray, unit_noun: str
) -> tuple[dict[int, str], str]:
    """Honest cluster titles when EVERY member label is a placeholder.

    An LLM namer given only placeholder labels ("neuron 3 (unlabeled)")
    invents semantics from zero information — observed producing "token
    clusters" on an all-placeholder neuron map. Callers must use this instead
    of name_clusters when units.meta["n_labeled"] == 0. `unit_noun` names the
    unit type ("neurons", "features"); the namer stamp records why no LLM ran.
    """
    titles = {
        int(cid): f"unlabeled {unit_noun} (cluster {int(cid)})"
        for cid in sorted({int(c) for c in cluster_ids if c >= 0})
    }
    return titles, "none(all-placeholder-labels)"


def stamp_identity(
    meta: dict,
    backend: str,
    model: str,
    identity: str,
    cost_usd: float | None = None,
    tokens: dict | None = None,
) -> None:
    """Record WHICH model titled this map, on every path including centroid.

    `export_json` splats `units.meta`, so these land in nebulai.json. Without
    them a map that silently fell through to a different model is
    indistinguishable from one that used the model the command named — which is
    the whole failure this module is built against.
    """
    meta["namer_backend"] = backend
    meta["namer_model"] = model
    meta["namer_identity"] = identity
    meta["namer_cost_usd"] = cost_usd
    if tokens and (tokens.get("prompt_tokens") or tokens.get("completion_tokens")):
        meta["namer_tokens"] = {
            "prompt": tokens["prompt_tokens"],
            "completion": tokens["completion_tokens"],
        }


# Chains for `auto` (no identity requirement) — unchanged behaviour, plus the
# new `hf` remote as an explicit choice.
_CHAINS: dict[str, list[str]] = {
    "auto": ["ollama", "openai", "openrouter", "centroid"],
    "openrouter": ["openrouter", "centroid"],
    "hf": ["hf", "centroid"],
    "ollama": ["ollama", "centroid"],
    "openai": ["openai", "centroid"],
    "anthropic": ["anthropic", "centroid"],
    "claude-cli": ["claude-cli", "centroid"],
    "codex-cli": ["codex-cli", "centroid"],
    "none": ["centroid"],
}

# Under a pin the chain is "every host that could serve THIS model", cheapest
# and most local first — and centroid is absent, because centroid is not the
# pinned model, it is four token strings joined by a dot.
_PINNED_AUTO_CHAIN = ["ollama", "openai", "hf", "openrouter"]


def name_clusters(
    units: Units,
    cluster_ids: np.ndarray,
    namer: str = "auto",
    openrouter_model: str = "openai/gpt-oss-120b:free",
    ollama_model: str = "liquidai/lfm2.5-1.2b-instruct",
    ollama_host: str = _DEFAULT_OLLAMA_HOST,
    anthropic_model: str = "claude-opus-5",
    env_file: str | None = None,
    llm_host: str = _DEFAULT_LLM_HOST,
    llm_model: str = "",
    llm_api_key: str | None = None,
    claude_cli_model: str = "",
    codex_cli_model: str = "",
    hf_model: str = "",
    model: str | None = None,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
) -> tuple[dict[int, str], str]:
    """Returns ({cluster_id: title}, backend_used), and stamps `units.meta`.

    `model=` pins an identity: only a backend that can serve THAT exact id may
    run, and if none can this raises `NamerIdentityError` instead of naming the
    map with whatever was reachable. Without it the chain behaves exactly as
    before — but either way `units.meta` records the model that answered.
    """
    reps = {
        int(cid): _representatives(units, np.where(cluster_ids == cid)[0])
        for cid in sorted(set(cluster_ids.tolist()))
        if cid >= 0
    }
    if not reps:
        stamp_identity(units.meta, "none", "", "auto")
        return {}, "none"

    pinned = (model or "").strip() or None
    identity = "pinned" if pinned else "auto"
    n = len(reps)

    if pinned:
        if namer == "none":
            raise NamerIdentityError(
                f"--namer none names nothing, so it cannot serve the pinned "
                f"model {pinned!r}; drop one of the two flags"
            )
        chain = _PINNED_AUTO_CHAIN if namer == "auto" else [namer]
    else:
        chain = _CHAINS[namer]

    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def run(backend: str) -> tuple[dict[int, str], str, str]:
        """(titles, label, exact model id that answered) for one backend."""
        if backend in ("openrouter", "hf"):
            if pinned:
                s = corpus_entry(pinned)
                if backend == "hf" and s is not None and s.hf_endpoint is None:
                    raise RuntimeError(
                        f"no HF inference provider serves {s.key} "
                        f"({s.repo}) — corpus hf_endpoint is None"
                    )
                target = (
                    (s.hf_endpoint if backend == "hf" else s.endpoint)
                    if s is not None
                    else pinned
                )
            else:
                target = hf_model if backend == "hf" else openrouter_model
                if backend == "hf" and not target:
                    raise RuntimeError(
                        "--namer hf needs a model: pass --namer-model (pinned) "
                        "or --hf-model"
                    )
            cost_gate(target, n, max_cost_usd)
            fn = _name_with_openrouter if backend == "openrouter" else _name_with_hf
            titles = fn(
                reps,
                target,
                env_file,
                usage=usage,
                expect_model=target if pinned else None,
            )
            return titles, f"{backend}:{target}", target

        if backend == "ollama":
            if pinned:
                served = next(
                    (t for t in _ollama_tags(ollama_host) if _serves_pin(t, pinned)),
                    None,
                )
                if served is None:
                    have = _ollama_tags(ollama_host)
                    raise RuntimeError(
                        f"ollama at {ollama_host} serves no build of {pinned!r} "
                        f"(has: {have[:6] or 'nothing reachable'})"
                    )
            else:
                served = _ollama_pick_model(ollama_host, ollama_model)
            # the *tag* is stamped, not the pin: a q4 build is what answered
            return (
                _name_with_ollama(reps, ollama_host, served or ollama_model),
                f"ollama:{served}",
                str(served),
            )

        if backend == "openai":
            if pinned:
                # exact only. _openai_pick_model's substring fallback would
                # happily return a neighbour, which is the substitution bug.
                ids = _openai_list_models(llm_host, llm_api_key)
                served = next((i for i in ids if _serves_pin(i, pinned)), None)
                if served is None:
                    raise RuntimeError(
                        f"{llm_host} does not serve {pinned!r} "
                        f"(it lists: {ids[:6] or 'nothing reachable'})"
                    )
            else:
                served = _openai_pick_model(llm_host, llm_model, llm_api_key)
                if served is None:
                    raise RuntimeError(
                        f"no chat model on {llm_host} (unreachable, or it serves "
                        "only embedding/rerank/audio models)"
                    )
            return (
                _name_with_openai(reps, llm_host, served, llm_api_key),
                f"openai:{served}",
                served,
            )

        if backend in ("claude-cli", "codex-cli"):
            fn = (
                _name_with_claude_cli
                if backend == "claude-cli"
                else _name_with_codex_cli
            )
            cli_model = pinned or (
                claude_cli_model if backend == "claude-cli" else codex_cli_model
            )
            titles = fn(reps, cli_model)
            # the model IS the variable under test for these backends, so it
            # is stamped even when it fell back to the CLI's own default —
            # 'default' is a question the reader can answer, '' is not
            label = f"{backend}:{cli_model or 'default'}"
            if len(titles) < len(reps):
                label += f"(partial:{len(titles)}/{len(reps)})"
            return titles, label, cli_model or "default"

        if backend == "anthropic":
            target = pinned or anthropic_model
            cost_gate(target, n, max_cost_usd)
            titles = _name_with_anthropic(reps, target)
            # the batch path can come back short if individual requests
            # errored; say so in the namer rather than letting a map look
            # fully named when some clusters export an empty title
            label = f"anthropic:{target}"
            if len(titles) < len(reps):
                label += f"(partial:{len(titles)}/{len(reps)})"
            return titles, label, target

        return _name_with_centroid(reps), "centroid", "centroid"

    declined: list[str] = []
    last_err: Exception | None = None
    for backend in chain:
        try:
            titles, label, used = run(backend)
        except NamerBudgetError:
            # terminal on purpose: refusing over budget is the ANSWER, and
            # falling through to a cheaper model would be the auto-downgrade
            # the gate exists to prevent
            raise
        except Exception as e:
            if pinned:
                declined.append(f"{backend}: {type(e).__name__}: {e}")
                continue
            last_err = e
            print(
                f"  namer '{backend}' unavailable ({type(e).__name__}: {e}); falling back"
            )
            continue
        stamp_identity(
            units.meta, backend, used, identity, actual_cost(used, usage), usage
        )
        return titles, label

    if pinned:
        reasons = "\n".join(f"  - {d}" for d in declined)
        raise NamerIdentityError(
            f"no configured backend can serve the pinned model {pinned!r}, and "
            "substituting a different model would make this map claim semantics "
            "it does not have.\n"
            f"tried {len(declined)} backend(s):\n{reasons}\n"
            "fix one of those backends, or drop --namer-model to let the chain "
            "use whatever is reachable (it will be stamped in meta.namer_model)."
        )
    raise RuntimeError(f"all namers failed: {last_err}")
