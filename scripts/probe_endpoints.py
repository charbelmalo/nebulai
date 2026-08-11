#!/usr/bin/env python3
"""Reachability and cost matrix for the endpoint-era corpus — run this BEFORE
planning a map, and paste its output into any claim about what a run will cost.

Two questions, one command:

1. **Can we read the rows without downloading the checkpoint?** For each corpus
   model it resolves the pinned revision, range-reads the safetensors header,
   locates the W_E (and W_U) tensor, and reports what a curated token map would
   actually stream against the size of the checkpoint we deliberately do not
   download. With `--rows N` it also range-reads the first N embedding rows and
   decodes them, which is the only part of that chain that can prove the byte
   offsets are right rather than merely plausible.

2. **Who can serve the pinned model id, and what would naming cost?** It looks
   for an OpenRouter key, an HF token and a local ollama server, checks each
   corpus model against OpenRouter's (public) catalogue and HF's per-model
   provider mapping, and prices a naming run against the live rates.

    scripts/probe_endpoints.py
    scripts/probe_endpoints.py --models muse-glimmer-30b mistral-nemo
    scripts/probe_endpoints.py --max-tokens 20000 --rows 8
    scripts/probe_endpoints.py --weights-only --json

**No credentials are required and none are read from the network.** "no key
configured" is a result, not an error: every repo below is public and every
catalogue it queries is public, so the weights half runs identically with an
empty environment and the endpoint half tells you exactly what you would need
to configure. Nothing here sends a chat request, so running it costs $0.

Prices are fetched live from OpenRouter and may differ from the `usd_in` /
`usd_out` stamped in `corpus.py`; the script flags any drift rather than
silently trusting either side. Cost figures are estimates from the measured
shape of a naming request (see `corpus.estimate_naming_cost`), not receipts.

It takes the corpus from `nebulai.corpus` and reimplements everything else —
the range reads, the header parse, the key lookup — deliberately. This is the
probe you reach for when the loader or the namer is the thing under suspicion,
so it must not fail with them.
"""

import argparse
import json
import array
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# stdlib only, on purpose — not even numpy. `nebulai.corpus` is pure dataclasses,
# so this runs in a bare python3.11+ with no venv, which is exactly the state the
# machine is in when you most need to know whether the repos are reachable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nebulai.corpus import (  # noqa: E402
    CORPUS,
    DEFAULT_MAX_COST_USD,
    EMBED_KEY_SUFFIXES,
    UNEMBED_KEY_SUFFIXES,
    estimate_naming_cost,
)

HF = "https://huggingface.co"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"  # matches backend/name.py
ENV_FILE = "~/.config/nebulai/.env"  # matches backend/name.py

# Shown only when a model breaches the cost gate, and shown as a *menu*: each of
# these is a different model with different semantics, so swapping one in changes
# what the map is evidence of. The choice is a human's, never the tool's.
ALTERNATIVES = (
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-3.1-8b-instruct",
)

# safetensors dtype tag -> bytes per element. Only what a checkpoint's embedding
# table is plausibly stored as; anything else is reported verbatim and its row
# size is left unknown rather than guessed.
ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1}


# --- transport ------------------------------------------------------------
# urllib rather than requests: the repo has no HTTP dependency and this script
# has to run in a bare checkout. Redirects matter twice here — HF answers a
# case-mismatched repo id with 307, and every resolve/ URL 302s to a CDN host —
# and urllib follows both while preserving the Range header.


def _req(url: str, headers: dict[str, str] | None, timeout: float):
    r = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(r, timeout=timeout)


def http_json(url: str, timeout: float = 30.0, token: str | None = None):
    """GET and parse JSON. Returns (data, status, error_string)."""
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with _req(url, headers, timeout) as r:
            return json.load(r), r.status, None
    except urllib.error.HTTPError as e:
        return None, e.code, f"HTTP {e.code}"
    except Exception as e:  # DNS, TLS, timeout — all "unreachable" to a caller
        return None, 0, type(e).__name__


def http_range(url: str, start: int, length: int, timeout: float = 30.0):
    """Range-read `length` bytes. Returns (bytes, status, error_string).

    A 200 here is a *failure* of the premise, not a success: it means the host
    ignored the Range header and is about to hand us the whole shard. The
    caller checks the status, so that case is reported rather than swallowed.
    """
    end = start + length - 1
    try:
        with _req(url, {"Range": f"bytes={start}-{end}"}, timeout) as r:
            return r.read(length), r.status, None
    except urllib.error.HTTPError as e:
        return b"", e.code, f"HTTP {e.code}"
    except Exception as e:
        return b"", 0, type(e).__name__


def resolve_url(repo: str, rev: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/{rev}/{path}"


# --- credentials ----------------------------------------------------------
# Detected, never required. Each returns (value_or_None, where_it_came_from).


def find_openrouter_key() -> tuple[str | None, str]:
    """os.environ first, then the last uncommented OPENROUTER_API_KEY= in the
    .env file — the same rule backend/name.py applies. Deliberately duplicated
    rather than imported: this probe has to keep working while the namer it
    reports on is being rewritten."""
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return key.strip(), "$OPENROUTER_API_KEY"
    path = Path(ENV_FILE).expanduser()
    if path.exists():
        found = None
        for line in path.read_text().splitlines():
            s = line.strip()
            if s.startswith("OPENROUTER_API_KEY="):
                found = s.split("=", 1)[1].strip().strip("'\"")
        if found:
            return found, str(path)
    return None, f"not in env or {ENV_FILE}"


def find_hf_token() -> tuple[str | None, str]:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if val := os.environ.get(var):
            return val.strip(), f"${var}"
    cached = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "token"
    if cached.exists() and (t := cached.read_text().strip()):
        return t, str(cached)
    return None, "not in env or ~/.cache/huggingface/token"


# --- weights probe --------------------------------------------------------


def read_header(url: str, timeout: float) -> tuple[dict | None, int, str | None]:
    """The safetensors header of a remote shard, in two range reads: 8 bytes of
    length, then that many bytes of JSON. Returns (header, data_start, error).

    This is the whole trick the endpoint architecture rests on — a few hundred
    KB tells you every tensor's dtype, shape and byte range in a 200 GB
    checkpoint.
    """
    raw, status, err = http_range(url, 0, 8, timeout)
    if err or len(raw) != 8:
        return None, 0, err or f"short read ({len(raw)}B)"
    if status != 206:
        return None, 0, f"no partial content (HTTP {status})"
    n = struct.unpack("<Q", raw)[0]
    if n > 200 * 1024 * 1024:  # a sane ceiling; 25k tensors is ~7 MB of JSON
        return None, 0, f"implausible header length {n}"
    body, status, err = http_range(url, 8, n, timeout)
    if err or len(body) != n:
        return None, 0, err or f"short header read ({len(body)}/{n}B)"
    try:
        return json.loads(body), 8 + n, None
    except json.JSONDecodeError as e:
        return None, 0, f"bad header JSON: {e}"


def decode_f32(raw: bytes, dtype: str) -> array.array:
    """Raw tensor bytes -> f32 values, without numpy.

    BF16 is exactly the high 16 bits of an IEEE-754 float32, so widening is a
    byte move: drop each pair into the top half of a 4-byte word and read the
    buffer as floats. That is the same lossless move `weights.py` makes with a
    shift, restated here so the probe carries no dependencies.
    """
    if dtype == "BF16":
        wide = bytearray(len(raw) * 2)
        wide[2::4] = raw[0::2]
        wide[3::4] = raw[1::2]
        buf, fmt = bytes(wide), "f"
    elif dtype == "F16":
        return array.array("f", struct.unpack(f"<{len(raw) // 2}e", raw))
    else:
        buf, fmt = raw, "f"
    a = array.array(fmt)
    a.frombytes(buf)
    if sys.byteorder != "little":  # the safetensors buffer is little-endian
        a.byteswap()
    return a


def match_suffix(keys, suffixes) -> str | None:
    """Exact-key lookup is what fails on multimodal wrappers (Glimmer and
    Gemma-4 nest their text weights under `model.language_model.`) and on
    Bailing's `word_embeddings`, so resolve by suffix. Shortest match wins:
    on a checkpoint carrying both a wrapper key and a plain one, the plain one
    is the tensor the text model actually reads."""
    hits = [k for k in keys if any(k.endswith(s) for s in suffixes)]
    return min(hits, key=len) if hits else None


def probe_weights(spec, max_tokens: int, n_rows: int, timeout: float) -> dict:
    """Everything the loader needs to know about one repo, without downloading
    it: resolved revision, shard layout, W_E/W_U location and shape, the bytes a
    map would stream, and (optionally) proof that those bytes decode."""
    out = {
        "key": spec.key,
        "repo": spec.repo,
        "revision_pinned": spec.revision,
        "ok": False,
        "notes": [],
    }

    info, status, err = http_json(f"{HF}/api/models/{spec.repo}", timeout)
    if info is None:
        out["error"] = f"repo info unreachable: {err or status}"
        return out
    canonical = info.get("id", spec.repo)
    out["canonical_repo"] = canonical
    out["gated"] = bool(info.get("gated"))
    if canonical != spec.repo:
        out["notes"].append(f"repo id redirects to {canonical} (HF 307, case)")
    if out["gated"]:
        out["notes"].append("repo is GATED — range reads will need a token")

    # `main` floats. Resolve it now so a run can be replayed against the exact
    # commit it read, which is the whole point of pinning.
    out["revision_resolved"] = str(info.get("sha", ""))[:12] or "?"

    files = [s.get("rfilename", "") for s in info.get("siblings", [])]
    shards = sorted(f for f in files if f.endswith(".safetensors"))
    if not shards:
        out["error"] = "no .safetensors files in repo"
        return out

    # The index gives shard routing and the exact byte total of the checkpoint.
    index, _, _ = http_json(
        resolve_url(spec.repo, spec.revision, "model.safetensors.index.json"), timeout
    )
    weight_map = (index or {}).get("weight_map", {})
    # Count the shards the *index* routes to, not the .safetensors files in the
    # repo: Mistral ships an extra un-sharded `consolidated.safetensors`, and
    # counting files would report a checkpoint one shard larger than it is.
    indexed = sorted(set(weight_map.values()))
    out["n_shards"] = len(indexed) or len(shards)
    out["n_tensors"] = len(weight_map) or None
    if indexed and (extra := sorted(set(shards) - set(indexed))):
        out["notes"].append(f"repo also carries {', '.join(extra)} (not in the index)")
    total = ((index or {}).get("metadata") or {}).get("total_size")
    out["total_gb"] = (total / 1e9) if total else None
    if total is None:
        out["notes"].append("no index.json — single-shard repo, size from spec")
        out["total_gb"] = spec.total_gb

    # Read only the headers of the shards that actually hold W_E / W_U.
    headers: dict[str, tuple[dict, int]] = {}

    def header_for(shard: str):
        if shard not in headers:
            h, start, e = read_header(resolve_url(spec.repo, spec.revision, shard), timeout)
            if h is None:
                out["notes"].append(f"{shard}: {e}")
                return None
            headers[shard] = (h, start)
        return headers[shard]

    if weight_map:
        embed_key = match_suffix(weight_map, EMBED_KEY_SUFFIXES)
        unembed_key = match_suffix(weight_map, UNEMBED_KEY_SUFFIXES)
        embed_shard = weight_map.get(embed_key)
        unembed_shard = weight_map.get(unembed_key)
    else:
        hdr = header_for(shards[0])
        if hdr is None:
            out["error"] = "could not read shard header"
            return out
        keys = [k for k in hdr[0] if k != "__metadata__"]
        embed_key = match_suffix(keys, EMBED_KEY_SUFFIXES)
        unembed_key = match_suffix(keys, UNEMBED_KEY_SUFFIXES)
        embed_shard = unembed_shard = shards[0]

    out["range_ok"] = None
    for role, key, shard in (
        ("embed", embed_key, embed_shard),
        ("unembed", unembed_key, unembed_shard),
    ):
        if not key or not shard:
            out[role] = None
            continue
        hdr = header_for(shard)
        out["range_ok"] = hdr is not None if out["range_ok"] is None else out["range_ok"]
        if hdr is None:
            out[role] = {"key": key, "shard": shard, "error": "header unreadable"}
            continue
        header, data_start = hdr
        entry = header.get(key)
        if entry is None:
            out[role] = {"key": key, "shard": shard, "error": "key absent from shard"}
            continue
        out[role] = {
            "key": key,
            "shard": shard,
            "dtype": entry["dtype"],
            "shape": list(entry["shape"]),
            "data_start": data_start,
            "offsets": entry["data_offsets"],
        }

    emb = out.get("embed")
    if not emb or "shape" not in emb:
        out["error"] = f"no W_E resolved (tried suffixes {EMBED_KEY_SUFFIXES})"
        return out

    vocab, hidden = emb["shape"][0], emb["shape"][-1]
    item = ITEMSIZE.get(emb["dtype"])
    rows = min(max_tokens, vocab)
    out["vocab_size"], out["hidden_size"] = vocab, hidden
    out["rows_mapped"] = rows
    if item:
        stream = rows * hidden * item
        out["stream_mb"] = stream / 1e6
        # An untied model's headline experiment reads both matrices.
        out["stream_both_mb"] = (
            2 * stream / 1e6 if out.get("unembed") and "shape" in out["unembed"] else None
        )
        if out["total_gb"]:
            out["share_pct"] = 100 * stream / (out["total_gb"] * 1e9)

    # Decode real rows. Everything above is arithmetic on a JSON header and
    # would look identical if the offsets were wrong by a shard; this is the
    # step that can fail.
    if n_rows and item:
        begin = emb["data_start"] + emb["offsets"][0]
        nbytes = n_rows * hidden * item
        t0 = time.time()
        raw, status, err = http_range(
            resolve_url(spec.repo, spec.revision, emb["shard"]), begin, nbytes, timeout
        )
        if err or len(raw) != nbytes or status != 206:
            out["decode"] = {"error": err or f"HTTP {status}, {len(raw)}/{nbytes}B"}
        else:
            vals = decode_f32(raw, emb["dtype"])
            norms = [
                math.sqrt(math.fsum(v * v for v in vals[r * hidden : (r + 1) * hidden]))
                for r in range(n_rows)
            ]
            out["decode"] = {
                "rows": n_rows,
                "bytes": nbytes,
                "seconds": time.time() - t0,
                "finite": all(math.isfinite(v) for v in vals),
                "norm_min": min(norms),
                "norm_max": max(norms),
            }

    # Cross-check the header against what corpus.py claims. A silent drift here
    # is the failure mode that would make every downstream map wrong.
    if vocab != spec.vocab_size or hidden != spec.hidden_size:
        out["notes"].append(
            f"shape {[vocab, hidden]} != corpus.py {[spec.vocab_size, spec.hidden_size]}"
        )
    if emb["key"] != spec.embed_key:
        out["notes"].append(f"W_E key {emb['key']} != corpus.py {spec.embed_key}")
    tied_here = out.get("unembed") is None
    tied_spec = spec.unembed_key is None
    if tied_here != tied_spec:
        out["notes"].append(
            f"tie mismatch: repo says {'tied' if tied_here else 'untied'}, "
            f"corpus.py says {'tied' if tied_spec else 'untied'}"
        )
    out["ok"] = True
    return out


# --- endpoint probe -------------------------------------------------------


def probe_openrouter(timeout: float) -> dict:
    """OpenRouter's catalogue is public, so the *identity* half of the check
    works with no key: is this exact model id served, and at what price? Only
    actually sending a request needs the key."""
    data, status, err = http_json(OPENROUTER_MODELS, timeout)
    if data is None:
        return {"reachable": False, "error": err or f"HTTP {status}", "models": {}}
    models = {}
    for m in data.get("data", []):
        p = m.get("pricing") or {}
        try:
            models[m["id"]] = (float(p.get("prompt", 0)) * 1e6, float(p.get("completion", 0)) * 1e6)
        except (TypeError, ValueError):
            models[m["id"]] = (None, None)
    return {"reachable": True, "n": len(models), "models": models}


def probe_hf_providers(repo: str, timeout: float, token: str | None) -> dict:
    """Which HF Inference Provider, if any, serves this exact repo. `None` is a
    legitimate answer and means the HF route does not exist for this model —
    not that a similar model should be used instead."""
    data, status, err = http_json(
        f"{HF}/api/models/{repo}?expand[]=inferenceProviderMapping", timeout, token
    )
    if data is None:
        return {"error": err or f"HTTP {status}"}
    mapping = data.get("inferenceProviderMapping") or {}
    if isinstance(mapping, list):  # older shape: list of {provider, status}
        mapping = {m.get("provider"): m.get("status") for m in mapping}
    else:
        mapping = {k: (v or {}).get("status") for k, v in mapping.items()}
    return {"providers": {k: v for k, v in mapping.items() if v == "live"} or {}}


def probe_ollama(host: str, timeout: float = 3.0) -> dict:
    data, status, err = http_json(f"{host}/api/tags", timeout)
    if data is None:
        return {"up": False, "error": err or f"HTTP {status}"}
    tags = [m["name"] for m in data.get("models", []) if "embed" not in m["name"].lower()]
    return {"up": True, "models": tags}


def corpus_clusters(root: Path) -> tuple[int, int]:
    """(maps, clusters) actually built in out/ — the real denominator for a
    "re-name the whole corpus" estimate, instead of a remembered one."""
    maps = clusters = 0
    for p in sorted(root.glob("*/nebulai.json")):
        try:
            meta = json.loads(p.read_text())["meta"]
        except Exception:
            continue
        if (n := meta.get("n_clusters")) is not None:
            maps += 1
            clusters += int(n)
    return maps, clusters


# --- rendering ------------------------------------------------------------


def table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "  (nothing to report)"
    w = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def line(cells: list[str]) -> str:
        return "  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(cells))

    return "\n".join(
        [line(headers), line(["-" * x for x in w])] + [line(r) for r in rows]
    ).rstrip()


def fmt_gb(x: float | None) -> str:
    return "?" if x is None else (f"{x:.2f} GB" if x >= 1 else f"{x * 1000:.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=sorted(CORPUS), help="corpus keys")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=50_000,
        help="curated vocab size to price the stream against (default 50000)",
    )
    ap.add_argument(
        "--rows",
        type=int,
        default=2,
        help="embedding rows to actually range-read and decode (0 = skip)",
    )
    ap.add_argument("--clusters", type=int, default=250, help="clusters per map for costing")
    ap.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD)
    ap.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--weights-only", action="store_true")
    ap.add_argument("--endpoints-only", action="store_true")
    ap.add_argument("--json", action="store_true", help="machine-readable dump instead")
    args = ap.parse_args()

    unknown = [m for m in args.models if m not in CORPUS]
    if unknown:
        sys.exit(f"not in corpus: {unknown}; have {sorted(CORPUS)}")
    specs = [CORPUS[k] for k in args.models]
    report: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    if not args.endpoints_only:
        report["weights"] = [
            probe_weights(s, args.max_tokens, max(0, args.rows), args.timeout) for s in specs
        ]

    if not args.weights_only:
        or_key, or_where = find_openrouter_key()
        hf_token, hf_where = find_hf_token()
        report["credentials"] = {
            "openrouter": {"present": bool(or_key), "source": or_where},
            "hf": {"present": bool(hf_token), "source": hf_where},
        }
        report["openrouter"] = probe_openrouter(args.timeout)
        report["ollama"] = probe_ollama(args.ollama_host)
        report["providers"] = {
            s.key: probe_hf_providers(s.repo, args.timeout, hf_token) for s in specs
        }
        report["built"] = dict(zip(("maps", "clusters"), corpus_clusters(args.out)))

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    print(f"nebulai endpoint probe   {report['generated']}   {len(specs)} model(s)")

    if w := report.get("weights"):
        print(
            f"\nWEIGHTS — range reads against public HF repos, no download, no auth"
            f"   (map = {args.max_tokens} tokens)"
        )
        rows = []
        for r in w:
            emb, un = r.get("embed") or {}, r.get("unembed") or {}
            rows.append(
                [
                    r["key"],
                    r.get("revision_resolved", "?"),
                    "206" if r.get("range_ok") else "FAIL",
                    f"{r.get('n_shards', '?')}",
                    fmt_gb(r.get("total_gb")),
                    f"{emb.get('dtype', '?')} {emb.get('shape', '?')}",
                    "tied" if not un else f"{un.get('dtype', '?')} {un.get('shape', '?')}",
                    f"{r['stream_mb']:.0f} MB" if r.get("stream_mb") else "?",
                    f"{r['share_pct']:.2f}%" if r.get("share_pct") else "?",
                ]
            )
        print(
            table(
                ["model", "rev", "range", "shards", "checkpoint", "W_E", "W_U", "streams", "share"],
                rows,
            )
        )

        if args.rows:
            print("\n  decoded rows (the falsifiable part — offsets, dtype and widening)")
            drows = []
            for r in w:
                d = r.get("decode") or {}
                if "error" in d:
                    drows.append([r["key"], "FAILED", d["error"], "", ""])
                elif d:
                    drows.append(
                        [
                            r["key"],
                            f"{d['rows']} rows / {d['bytes'] / 1000:.0f} kB",
                            f"{d['seconds']:.2f}s",
                            "finite" if d["finite"] else "NON-FINITE",
                            f"norm {d['norm_min']:.3f}–{d['norm_max']:.3f}",
                        ]
                    )
            print(table(["model", "read", "time", "values", "row norms"], drows))

        both = [r for r in w if r.get("stream_both_mb")]
        if both:
            print(
                "\n  'streams' is W_E alone. The W_E-vs-W_U experiment reads both matrices: "
                + ", ".join(f"{r['key']} {r['stream_both_mb']:.0f} MB" for r in both)
            )

        notes = [(r["key"], n) for r in w for n in r.get("notes", [])]
        notes += [(r["key"], r["error"]) for r in w if r.get("error")]
        if notes:
            print("\n  notes")
            for k, n in notes:
                print(f"    {k}: {n}")

    if not args.weights_only:
        creds, orr, oll = report["credentials"], report["openrouter"], report["ollama"]
        print("\nCHAT ROUTES — who can serve the pinned id (catalogues are public)")
        print(
            table(
                ["route", "credential", "status"],
                [
                    [
                        "openrouter",
                        "key present" if creds["openrouter"]["present"] else "NO KEY",
                        (
                            f"catalogue: {orr['n']} models"
                            if orr["reachable"]
                            else f"unreachable: {orr.get('error')}"
                        )
                        + ("" if creds["openrouter"]["present"] else f"  ({creds['openrouter']['source']})"),
                    ],
                    [
                        "hf router",
                        "token present" if creds["hf"]["present"] else "NO TOKEN",
                        "provider mapping is public; a token is needed to send"
                        + ("" if creds["hf"]["present"] else f"  ({creds['hf']['source']})"),
                    ],
                    [
                        "ollama",
                        args.ollama_host,
                        (
                            f"up, {len(oll['models'])} text model(s): "
                            + ", ".join(oll["models"][:3])
                            if oll["up"]
                            else f"down ({oll.get('error')})"
                        ),
                    ],
                ],
            )
        )

        maps, clusters = report["built"]["maps"], report["built"]["clusters"]
        print(
            f"\nNAMING COST — live OpenRouter prices; nothing below was sent."
            f"   (gate ${args.max_cost_usd:.2f})"
        )
        rows = []
        route_notes: list[str] = []
        for s in specs:
            live = orr["models"].get(s.endpoint)
            served = "yes" if live else ("NOT SERVED" if orr["reachable"] else "?")
            drift = ""
            if live and live[0] is not None:
                if abs(live[0] - s.usd_in) > 1e-9 or abs(live[1] - s.usd_out) > 1e-9:
                    drift = f"  (corpus.py says {s.usd_in}/{s.usd_out})"
            one = estimate_naming_cost(args.clusters, s.key)
            whole = estimate_naming_cost(clusters, s.key) if clusters else 0.0
            provs = (report["providers"].get(s.key) or {}).get("providers") or {}
            rows.append(
                [
                    s.key,
                    s.endpoint,
                    served,
                    f"{live[0]:.3f}" if live and live[0] is not None else "?",
                    f"{live[1]:.3f}" if live and live[1] is not None else "?",
                    f"${one:.4f}",
                    f"${whole:.4f}",
                    "over gate" if one > args.max_cost_usd else "ok",
                    (",".join(sorted(provs)) or "none") + drift,
                ]
            )
            # An `hf_endpoint` nobody serves is exactly the situation the no-
            # substitution rule exists for: the route must be reported dead, not
            # quietly satisfied by a neighbouring model on the same router.
            if s.hf_endpoint and not provs:
                route_notes.append(
                    f"{s.key}: corpus.py names hf_endpoint={s.hf_endpoint} but no HF "
                    "provider serves it today — that route is dead, use OpenRouter"
                )
            if not live and orr["reachable"]:
                route_notes.append(
                    f"{s.key}: {s.endpoint} is not in OpenRouter's catalogue — refuse "
                    "rather than pick a neighbour"
                )
        print(
            table(
                [
                    "model",
                    "openrouter id",
                    "served",
                    "$/M in",
                    "$/M out",
                    f"{args.clusters} cl.",
                    f"{maps} maps/{clusters} cl.",
                    "gate",
                    "hf providers",
                ],
                rows,
            )
        )
        if route_notes:
            print("\n  notes")
            for n in route_notes:
                print(f"    {n}")

        over = [
            s.key for s in specs if estimate_naming_cost(args.clusters, s.key) > args.max_cost_usd
        ]
        if over and orr["reachable"]:
            batches = -(-args.clusters // 15)  # same batch shape corpus.py prices
            alts = [
                [
                    alt,
                    f"{orr['models'][alt][0]:.3f}",
                    f"{orr['models'][alt][1]:.3f}",
                    f"${(batches * 1500 * orr['models'][alt][0] + batches * 400 * orr['models'][alt][1]) / 1e6:.4f}",
                ]
                for alt in ALTERNATIVES
                if orr["models"].get(alt) and orr["models"][alt][0] is not None
            ]
            print(
                f"\n  OVER GATE: {', '.join(over)}. Cheaper models exist, and each is a"
                "\n  DIFFERENT model — titles from one are not the other's semantics. Pick"
                "\n  one deliberately or raise --max-cost-usd; nothing is auto-substituted."
            )
            print(table(["alternative", "$/M in", "$/M out", f"{args.clusters} cl."], alts))

    print(
        "\n  What this does and does not establish. A 206 plus a decoded row proves the\n"
        "  rows are readable at that revision — it does not prove the endpoint in the\n"
        "  same line serves those weights. Nothing can prove that from outside, which is\n"
        "  why the model id is pinned and a missing route is refused rather than swapped\n"
        "  for a cheaper one. Costs are estimates from a measured request shape, and a\n"
        "  'NOT SERVED' or 'none' is a real result: that route does not exist today."
    )


if __name__ == "__main__":
    main()
