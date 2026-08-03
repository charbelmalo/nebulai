"""Cluster naming with a pluggable backend chain.

auto mode tries: a local ollama server (free, private) -> OpenRouter
(key from env or a .env file) -> centroid fallback (title = the members
nearest the cluster centroid). The pipeline therefore always completes; the
LLM namers simply activate when a reachable ollama or a key is present.
Anthropic stays available via `--namer anthropic`.
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
        "auto": ["ollama", "openrouter", "centroid"],
        "openrouter": ["openrouter", "centroid"],
        "ollama": ["ollama", "centroid"],
        "anthropic": ["anthropic", "centroid"],
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
