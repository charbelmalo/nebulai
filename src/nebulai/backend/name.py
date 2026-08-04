"""Cluster naming with a pluggable backend chain.

auto mode tries: a local ollama server (free, private) -> an OpenAI-compatible
chat server (also local: LM Studio, vLLM, llama.cpp, an MLX box on the LAN) ->
OpenRouter (key from env or a .env file) -> centroid fallback (title = the
members nearest the cluster centroid). The pipeline therefore always
completes; the LLM namers simply activate when one is reachable or a key is
present. Anthropic stays available via `--namer anthropic`.
"""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from ..units import Units

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


# --- OpenRouter -----------------------------------------------------------

def _load_openrouter_key(env_file: str | None) -> str | None:
    """os.environ first, then the last uncommented OPENROUTER_API_KEY= in the
    .env file."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    path = Path(env_file or _DEFAULT_ENV_FILE).expanduser()
    if not path.exists():
        return None
    found = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("OPENROUTER_API_KEY="):
            found = s.split("=", 1)[1].strip().strip("'\"")
    return found or None


def _name_with_openrouter(
    reps: dict[int, list[str]],
    model: str,
    env_file: str | None,
    batch_size: int = 15,
) -> dict[int, str]:
    key = _load_openrouter_key(env_file)
    if not key:
        raise RuntimeError("no OPENROUTER_API_KEY in env or .env file")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Title": "Nebul.AI",
    }
    titles: dict[int, str] = {}
    cids = sorted(reps)
    for start in range(0, len(cids), batch_size):
        batch = cids[start : start + batch_size]
        body = json.dumps(
            {
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
        ).encode()
        req = urllib.request.Request(_OPENROUTER_URL, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            content = json.load(r)["choices"][0]["message"]["content"]
        for item in json.loads(content)["titles"]:
            titles[int(item["id"])] = str(item["title"]).strip()
    return titles


# --- ollama ---------------------------------------------------

def _ollama_pick_model(host: str, preferred: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as r:
            tags = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return None
    # never pick an embedding model for text generation
    tags = [t for t in tags if "embed" not in t.lower()]
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


def _openai_pick_model(host: str, preferred: str, api_key: str | None = None) -> str | None:
    """Resolve a chat model id on an OpenAI-compatible server, or None."""
    req = urllib.request.Request(f"{host.rstrip('/')}/v1/models")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            ids = [m["id"] for m in json.load(r).get("data", [])]
    except Exception:
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
) -> tuple[dict[int, str], str]:
    """Returns ({cluster_id: title}, backend_used)."""
    reps = {
        int(cid): _representatives(units, np.where(cluster_ids == cid)[0])
        for cid in sorted(set(cluster_ids.tolist()))
        if cid >= 0
    }
    if not reps:
        return {}, "none"

    chain = {
        "auto": ["ollama", "openai", "openrouter", "centroid"],
        "openrouter": ["openrouter", "centroid"],
        "ollama": ["ollama", "centroid"],
        "openai": ["openai", "centroid"],
        "anthropic": ["anthropic", "centroid"],
        "claude-cli": ["claude-cli", "centroid"],
        "codex-cli": ["codex-cli", "centroid"],
        "none": ["centroid"],
    }[namer]

    last_err: Exception | None = None
    for backend in chain:
        try:
            if backend == "openrouter":
                return (
                    _name_with_openrouter(reps, openrouter_model, env_file),
                    f"openrouter:{openrouter_model}",
                )
            if backend == "ollama":
                model = _ollama_pick_model(ollama_host, ollama_model)
                return (
                    _name_with_ollama(reps, ollama_host, ollama_model),
                    f"ollama:{model}",
                )
            if backend == "openai":
                model = _openai_pick_model(llm_host, llm_model, llm_api_key)
                if model is None:
                    raise RuntimeError(
                        f"no chat model on {llm_host} (unreachable, or it serves "
                        "only embedding/rerank/audio models)"
                    )
                return (
                    _name_with_openai(reps, llm_host, model, llm_api_key),
                    f"openai:{model}",
                )
            if backend in ("claude-cli", "codex-cli"):
                fn = (
                    _name_with_claude_cli
                    if backend == "claude-cli"
                    else _name_with_codex_cli
                )
                cli_model = (
                    claude_cli_model if backend == "claude-cli" else codex_cli_model
                )
                titles = fn(reps, cli_model)
                # the model IS the variable under test for these backends, so it
                # is stamped even when it fell back to the CLI's own default —
                # 'default' is a question the reader can answer, '' is not
                label = f"{backend}:{cli_model or 'default'}"
                if len(titles) < len(reps):
                    label += f"(partial:{len(titles)}/{len(reps)})"
                return titles, label
            if backend == "anthropic":
                titles = _name_with_anthropic(reps, anthropic_model)
                # the batch path can come back short if individual requests
                # errored; say so in the namer rather than letting a map look
                # fully named when some clusters export an empty title
                label = f"anthropic:{anthropic_model}"
                if len(titles) < len(reps):
                    label += f"(partial:{len(titles)}/{len(reps)})"
                return titles, label
            return _name_with_centroid(reps), "centroid"
        except Exception as e:  # fall through the chain, remember why
            last_err = e
            print(
                f"  namer '{backend}' unavailable ({type(e).__name__}: {e}); falling back"
            )
    raise RuntimeError(f"all namers failed: {last_err}")
