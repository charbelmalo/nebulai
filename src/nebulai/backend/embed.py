"""Text embeddings via an ollama or OpenAI-compatible embeddings endpoint
(default: mxbai-embed-large on a local ollama server).

Used by the cross-model comparison (compare.py): each model's named clusters
are embedded in this neutral, model-independent semantic space so clouds from
different models can be laid out and categorized together honestly — the
comparison happens in a third-party embedder, never by pretending two models'
raw geometries share a basis. Also the vector source for the api-embeddings
token front-end (frontends/api_tokens.py), under the same honesty rule.

Unlike the namer, this module has never had a fall-through chain: the caller
always names the embedder and no code here picks one, so there is no path by
which a different embedder is silently substituted. The one gap that DID exist
is closed by `EmbedIdentityError` below — a multi-model server that ignores the
`model` field and serves whatever it has loaded.

REACHABILITY — corrected 2026-08-13, and the correction is the whole point.

An earlier note here (2026-08-12) said "nothing serves it" and concluded that
`nebulai compare` was dead until someone built a new embedder. That was wrong,
and it was wrong for a boring reason: it probed **:11434**, the stock ollama
port, and the M4 worker has never used it. `docs/M4-OLLAMA-HANDOVER.md` has said
`OLLAMA_HOST=0.0.0.0:11435` since 2026-08-04. Verified working today:

    GET  http://<m4-host>:11435/api/tags   -> mxbai-embed-large:latest,
                                             334M params, F16 GGUF, 1024-dim
    GET  http://<m4-host>:11435/api/version-> 0.23.1
    GET  http://<m4-host>:8100/v1/status/ollama -> running:true, port:11435
    embed_texts(...) -> (n,1024) float32, L2-normalized, semantically sane
                        (cos 0.70 related / 0.31 unrelated)

So the default below is a *local* default, not a claim that the LAN box is down.
Point `--embed-host` (or NEBULAI_EMBED_HOST) at `http://<m4-host>:11435` and this
module works as designed. Two things are worth not re-deriving:

  - :8050 on the same box is a DIFFERENT server (OpenAI-compatible, `omlx`)
    carrying `all-MiniLM-L6-v2` and `nomic-embed-text-v1.5`. It does not serve
    mxbai. Use `--embed-api openai` for it, and remember it is a different
    neutral space — not interchangeable with mxbai vectors.
  - The HF router still has no /v1/embeddings route at all (GET returns 404),
    and the only live HF path for this embedder is `hf-inference`'s
    feature-extraction pipeline, whose response shape neither branch below
    parses. There is still no faithful remote drop-in.

The ollama `mxbai-embed-large` tag is an F16 GGUF build of
`mixedbread-ai/mxbai-embed-large-v1`, so its vectors are not bit-identical to
the fp32 repo's. That matters less than the earlier note implied — F16 is half
precision, not a 4-bit quant — but it is still a different artifact behind a
MUTABLE tag, so it cannot satisfy a "pinned to an exact revision" requirement.
Swapping `compare` to the fp32 repo would change the neutral space, which is
precisely the substitution this file must not make quietly.
"""

import ipaddress
import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import numpy as np

_DEFAULT_OLLAMA_HOST = "http://localhost:11434"  # local ollama server
_DEFAULT_EMBED_MODEL = "mxbai-embed-large"

#: Env override for the embeddings base URL. Exists because the working host on
#: this network is a LAN box on a non-stock port (see the module docstring), and
#: hardcoding a LAN IP as the library default would be wrong for everyone else
#: while retyping `--embed-host` every run is how the port drift went unnoticed
#: for a month. Setting it once fixes both.
EMBED_HOST_ENV = "NEBULAI_EMBED_HOST"

#: Values of EMBED_HOST_ENV (or `--embed-host`) that mean "don't hardcode the M4
#: worker's address — go find it". The box is company-managed: its DHCP IP and
#: Private-Wi-Fi MAC both rotate, so a literal `http://192.168.0.200:11435` goes
#: stale. Setting `NEBULAI_EMBED_HOST=auto` once resolves the host dynamically at
#: run time instead. Mirrors the sentinels the m4host resolver itself honors.
_DISCOVER_SENTINELS = frozenset({"auto", "discover", "dynamic", "m4"})

#: The M4 worker's ollama embed port — only the HOST is discovered; the port and
#: the `/api/embed` path (appended by _embed_batch) are unchanged. See the module
#: docstring and docs/M4-OLLAMA-HANDOVER.md (the box binds 11435, not stock 11434).
_M4_EMBED_PORT = 11435

#: Last-resort host when dynamic discovery is unavailable (the resolver cannot be
#: imported, or itself gives up). The historical literal, kept so a resolver
#: problem degrades to the old behavior rather than breaking a command.
_M4_FALLBACK_HOST = "192.168.0.200"


def _looks_like_discover(value: str) -> bool:
    return value.strip().lower() in _DISCOVER_SENTINELS


def _discover_embed_host() -> str:
    """Build the M4 embed base URL from a live host discovery. Best-effort.

    Uses the vendored `m4host` resolver, which finds the worker by the service it
    serves rather than a fixed address. ANY failure — the resolver missing, an
    import error, an unexpected bug — collapses to the historical literal so
    asking for discovery can never take down a command: worst case it behaves
    exactly as the old hardcoded default did. The resolver never scans when
    `MIND_M4_STRICT=1` (with a pin) or `MIND_M4_DISCOVERY=0` is set.
    """
    host = _M4_FALLBACK_HOST
    try:
        from . import m4host

        resolved = (m4host.resolve() or "").strip()
        if resolved:
            host = resolved
    except Exception:
        host = _M4_FALLBACK_HOST
    return f"http://{host}:{_M4_EMBED_PORT}"


def resolve_embed_host(host: str | None) -> str | None:
    """Interpret one embed-host value shared by every entry point.

    A discovery sentinel (`auto` / `m4` / `discover` / `dynamic`) becomes the
    dynamically resolved M4 URL; anything else — a real URL, or None/blank — is
    returned unchanged so an explicit endpoint still wins verbatim.
    """
    if host and _looks_like_discover(host):
        return _discover_embed_host()
    return host


def default_embed_host() -> str:
    """The embeddings base URL: `NEBULAI_EMBED_HOST` if set, else local ollama.

    The caller still names the endpoint — an env var is the caller naming it once
    instead of per command, and a concrete URL is used verbatim with nothing here
    silently substituting a different one. The single exception is an explicit
    discovery sentinel (`NEBULAI_EMBED_HOST=auto`), which is the caller asking, in
    so many words, to have the (address-rotating) M4 worker located at run time;
    only then does this consult the resolver. Unset/blank stays the local ollama
    default — no probe, no network.
    """
    raw = (os.environ.get(EMBED_HOST_ENV) or "").strip()
    if not raw:
        return _DEFAULT_OLLAMA_HOST
    if _looks_like_discover(raw):
        return _discover_embed_host()
    return raw


#: What `embed_host` becomes in an exported artifact when the endpoint was not
#: loopback. A marker, not a redaction of the fact: provenance still records
#: that an external service placed these points.
PUBLISHED_REMOTE_HOST = "remote"


def public_embed_host(host: str) -> str:
    """The `embed_host` value that is safe to stamp into an exported artifact.

    Loopback endpoints pass through verbatim — they name no machine but the
    reader's own. Everything else collapses to `"remote"`.

    The reason is that `nebulai.json` is served publicly while `--embed-host`
    is an operator's private network detail, and the host is not evidence
    about the map anyway: what makes these vectors what they are is the
    *model*, and `embed_model`/`embed_api` are stamped separately and
    untouched. Five shipped artifacts published a LAN address this way before
    this function existed (docs/ONBOARDING.md blocker 1).

    Deliberately not a general-purpose "is this address private" classifier.
    That call fails OPEN — one wrong verdict publishes the address, and the
    zoo of RFC1918 / CGNAT / link-local / mDNS / bare-LAN-hostname cases is
    exactly where such a classifier gets it wrong. Only loopback, which needs
    no judgement, survives; anything unparseable is treated as remote.
    """
    host = (host or "").strip()
    if not host:
        return ""
    try:
        # a bare "localhost:11434" has no scheme, so urlsplit reads it as a
        # path and yields no hostname; the "//" prefix forces netloc parsing.
        name = (urlsplit(host).hostname or urlsplit(f"//{host}").hostname or "").lower()
    except ValueError:  # malformed IPv6 literal, etc.
        return PUBLISHED_REMOTE_HOST
    if not name:
        return PUBLISHED_REMOTE_HOST
    if name == "localhost" or name.endswith(".localhost"):
        return host
    try:
        # covers 127.0.0.0/8 and ::1 without hardcoding either
        if ipaddress.ip_address(name).is_loopback:
            return host
    except ValueError:
        pass
    return PUBLISHED_REMOTE_HOST


class EmbedIdentityError(RuntimeError):
    """The endpoint served a different embedding model than the one requested.

    `compare` exists to put several models' clusters into ONE neutral space, so
    which embedder produced the vectors is not a detail — it IS the space. A
    host that ignores the `model` field and serves whatever it has loaded (the
    multi-model OpenAI-compatible servers do exactly this) would yield a map
    stamped `mxbai-embed-large` and positioned by something else. Same rule as
    llm.IdentityError, applied to the embedder.
    """


def parse_embed_response(payload: dict, api: str, host: str, model: str) -> list[list[float]]:
    """Extract embeddings from one response, in input order.

    ollama's /api/embed returns {"embeddings": [...]}; OpenAI-compatible
    /v1/embeddings returns {"data": [{"index", "embedding"}, ...]} whose order
    is not guaranteed — sort by index.

    When the response names the model it ran, that name is checked against the
    one asked for. Servers that omit the field are taken at their word — there
    is nothing to check against, and refusing them would break every endpoint
    that simply does not report it.
    """
    served = str(payload.get("model") or "")
    if served and model and served.split(":", 1)[0].lower() != model.split(":", 1)[0].lower():
        raise EmbedIdentityError(
            f"asked {host} to embed with {model!r} and it answered as "
            f"{served!r} — a different embedder is a different semantic space, "
            "so these vectors are not comparable with the rest of the map"
        )
    if api == "openai":
        data = payload.get("data")
        if not data:
            raise RuntimeError(f"no embeddings returned from {host} ({model})")
        return [d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))]
    embs = payload.get("embeddings")
    if not embs:
        raise RuntimeError(f"no embeddings returned from {host} ({model})")
    return embs


def _embed_batch(
    batch: list[str],
    host: str,
    model: str,
    timeout: float,
    retries: int,
    api: str = "ollama",
    api_key: str | None = None,
) -> list[list[float]]:
    """One embed call, retried on transient network/timeout errors.

    The M4 is shared: a concurrent naming run can queue this request behind
    dozens of generate calls, so a single timeout is not a hard failure —
    back off and retry before giving up.
    """
    url = f"{host.rstrip('/')}/v1/embeddings" if api == "openai" else f"{host}/api/embed"
    body = json.dumps({"model": model, "input": batch}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.load(r)
            return parse_embed_response(payload, api, host, model)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))  # linear backoff
    # Name the fix, not just the failure. The last time this raised, the message
    # was a bare URLError and the conclusion drawn from it was "no embedder
    # exists anywhere" — when the real cause was a wrong port on a host that was
    # up the whole time. Anything that costs 4 retries should say where to look.
    raise RuntimeError(
        f"embed request to {host} ({model}) failed after {retries} attempts: {last}\n"
        f"  - check the endpoint is up:  curl {host.rstrip('/')}/api/tags\n"
        f"  - ollama's stock port is 11434, but a host may bind elsewhere "
        f"(this project's LAN box uses 11435 — see docs/M4-OLLAMA-HANDOVER.md)\n"
        f"  - set {EMBED_HOST_ENV} or pass --embed-host to point somewhere else\n"
        f"  - for an OpenAI-compatible server use --embed-api openai"
    )


def embed_texts(
    texts: list[str],
    host: str = _DEFAULT_OLLAMA_HOST,
    model: str = _DEFAULT_EMBED_MODEL,
    batch_size: int = 32,
    timeout: float = 300.0,
    retries: int = 4,
    api: str = "ollama",
    api_key: str | None = None,
) -> np.ndarray:
    """Return (n, d) L2-normalized float32 embeddings for texts.

    api="ollama" posts to {host}/api/embed; api="openai" posts to any
    OpenAI-compatible {host}/v1/embeddings (with optional bearer api_key).
    """
    if api not in ("ollama", "openai"):
        raise ValueError(f"unknown embed api {api!r} (expected 'ollama' or 'openai')")
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        out.extend(_embed_batch(batch, host, model, timeout, retries, api, api_key))
    arr = np.asarray(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr
