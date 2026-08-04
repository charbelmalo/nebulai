"""Probe front-end: a semantic cloud grown from a seed topic.

The other three front-ends decompose a MODEL — token embeddings, SAE decoder
directions, MLP write directions. This one has no model at all. You give it a
word, an LLM proposes related concepts breadth-first, a text embedder places
them, and a cosine gate against the seed decides what stays. The output is the
same `Units` contract, so reduce → cluster → name → export → render are
untouched.

WHAT THIS MEASURES, and it is not the same thing the other front-ends measure:
a probe cloud is the joint opinion of two models — the generator that proposed
the terms and the embedder that positioned them. It is NOT a fact about
language, and it is not the geometry of any model under study. A term missing
from the cloud means the generator did not propose it; a term sitting far from
the seed means the embedder put it there. Both are stamped into meta and the
`geometry` field says so in the same words the api-embedding front-end uses, so
the viewer labels it identically.

The gate is deliberately cosine-to-SEED rather than cosine-to-parent. Chaining
parent similarity lets a depth-3 term drift arbitrarily far from the topic
while every individual hop looks reasonable; anchoring every term to the seed
is what makes `--sensitivity` mean one legible thing at any depth.
"""

import json
import re
import urllib.request

import numpy as np

from ..units import Units

_SYSTEM = (
    "You expand a seed concept into related concepts for a semantic map. "
    "Given a term, reply with concepts that a person exploring that topic "
    "would expect to find nearby: sub-topics, adjacent fields, key entities, "
    "and common associations. Prefer specific, nameable concepts over vague "
    "abstractions, and never repeat the term you were given."
)

# Term used to check a generator is alive. Must be concrete and unambiguous:
# asking a reasoning model to expand the word "test" gives it nothing to work
# with, and it burns thousands of scratchpad tokens deciding what was meant.
_PROBE_TERM = "ocean"

_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["concepts"],
    "additionalProperties": False,
}


def probe_dataset_id(seed: str) -> str:
    """out/ directory name for a probe cloud.

    Shared by the CLI (which writes it) and build_server (which must predict it
    to report the artifact path before the run finishes) — two copies of this
    slug rule would drift and the server would poll for a directory the
    pipeline never created.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", seed.lower()).strip("-")[:48] or "probe"
    return f"probe__{slug}"


def _norm(term: str) -> str:
    """Dedup key — case and surrounding punctuation only. Distinct wordings of
    the same idea are NOT merged: deciding that 'car' and 'automobile' are one
    concept is exactly the judgement this tool exists to show, not to make."""
    return re.sub(r"\s+", " ", term.strip().strip(".,;:!?\"'()[]")).lower()


def _clean(raw: list, seen: set[str], limit: int) -> list[str]:
    out: list[str] = []
    for item in raw:
        term = re.sub(r"\s+", " ", str(item).strip().strip(".,;:!?\"'()[]"))
        if not term or len(term) > 80:
            continue
        k = _norm(term)
        if k in seen:
            continue
        seen.add(k)
        out.append(term)
        if len(out) >= limit:
            break
    return out


# --- expansion backends ---------------------------------------------------


def _expand_ollama(term: str, n: int, host: str, model: str) -> list:
    prompt = (
        f"{_SYSTEM}\n\nTerm: {term!r}\n\n"
        f'Reply as JSON: {{"concepts": [{n} short concept names]}}'
    )
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"num_predict": 40 * n + 80},
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.load(r)["response"]
    return json.loads(out).get("concepts", [])


def _expand_openrouter(term: str, n: int, model: str, env_file: str | None) -> list:
    from ..backend.name import _load_openrouter_key

    key = _load_openrouter_key(env_file)
    if not key:
        raise RuntimeError("no OPENROUTER_API_KEY")
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": f"Term: {term!r}\n\n"
                    f'Reply as JSON: {{"concepts": [{n} short concept names]}}',
                },
            ],
            "response_format": {"type": "json_object"},
        }
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        payload = json.load(r)
    return json.loads(payload["choices"][0]["message"]["content"]).get("concepts", [])


def _expand_openai(
    term: str, n: int, host: str, model: str, api_key: str | None
) -> list:
    from ..backend.name import _chat_openai

    got = _chat_openai(
        host,
        model,
        _SYSTEM,
        f"Term: {term!r}\n\n"
        f'Reply as JSON: {{"concepts": [{n} short concept names]}}',
        _SCHEMA,
        "concepts",
        api_key=api_key,
        max_tokens=120 * n + 400,
    )
    return got.get("concepts", [])


def _expand_anthropic(term: str, n: int, model: str) -> list:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1000,
        system=_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": f"Term: {term!r}. Give {n} concepts."}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text).get("concepts", [])


def _make_expander(
    generator: str,
    ollama_host: str,
    ollama_model: str,
    openrouter_model: str,
    anthropic_model: str,
    env_file: str | None,
    llm_host: str = "http://localhost:8050",
    llm_model: str = "",
    llm_api_key: str | None = None,
):
    """Resolve the generator once, up front — a cloud whose first half came
    from one model and second half from another after a mid-run fallback would
    be uninterpretable, so this fails loudly instead of degrading."""
    from ..backend.name import _ollama_pick_model, _openai_pick_model

    chain = {
        "auto": ["ollama", "openai", "openrouter", "anthropic"],
        "ollama": ["ollama"],
        "openai": ["openai"],
        "openrouter": ["openrouter"],
        "anthropic": ["anthropic"],
    }[generator]

    tried: list[str] = []
    for backend in chain:
        try:
            if backend == "ollama":
                picked = _ollama_pick_model(ollama_host, ollama_model)
                if picked is None:
                    raise RuntimeError(f"ollama at {ollama_host} unreachable")
                _expand_ollama(_PROBE_TERM, 2, ollama_host, picked)
                return (
                    lambda t, n: _expand_ollama(t, n, ollama_host, picked),
                    f"ollama:{picked}",
                )
            if backend == "openai":
                picked = _openai_pick_model(llm_host, llm_model, llm_api_key)
                if picked is None:
                    raise RuntimeError(
                        f"no chat model on {llm_host} (unreachable, or it "
                        "serves only embedding/rerank/audio models)"
                    )
                # a truncation here is not a failed backend: the model answered,
                # at length. Only a hard error means it can't generate.
                from ..backend.name import ChatTruncated

                try:
                    _expand_openai(_PROBE_TERM, 2, llm_host, picked, llm_api_key)
                except ChatTruncated as trunc:
                    print(f"  generator openai:{picked} rambles ({trunc}) — using it anyway")
                return (
                    lambda t, n: _expand_openai(t, n, llm_host, picked, llm_api_key),
                    f"openai:{picked}",
                )
            if backend == "openrouter":
                _expand_openrouter(_PROBE_TERM, 2, openrouter_model, env_file)
                return (
                    lambda t, n: _expand_openrouter(t, n, openrouter_model, env_file),
                    f"openrouter:{openrouter_model}",
                )
            _expand_anthropic(_PROBE_TERM, 2, anthropic_model)
            return (
                lambda t, n: _expand_anthropic(t, n, anthropic_model),
                f"anthropic:{anthropic_model}",
            )
        except Exception as e:
            tried.append(backend)
            print(f"  generator '{backend}' unavailable ({type(e).__name__}: {e})")
    # the per-backend reasons are already on stdout; summarising with only the
    # LAST exception reads as if that one backend were the whole problem
    raise RuntimeError(
        f"no expansion backend available (tried: {', '.join(tried)}) — see the "
        "reasons above. A probe needs a generator and an embedder; neither is "
        "bundled."
    )


# --- the cloud ------------------------------------------------------------


def load_probe_units(
    seed: str,
    depth: int = 2,
    breadth: int = 12,
    sensitivity: float = 0.35,
    generator: str = "auto",
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "liquidai/lfm2.5-1.2b-instruct",
    openrouter_model: str = "openai/gpt-oss-120b:free",
    anthropic_model: str = "claude-opus-5",
    env_file: str | None = None,
    embed_host: str = "http://localhost:11434",
    embed_model: str = "mxbai-embed-large",
    embed_api: str = "ollama",
    embed_api_key: str | None = None,
    llm_host: str = "http://localhost:8050",
    llm_model: str = "",
    llm_api_key: str | None = None,
    max_terms: int = 4000,
    reuse_terms: list[str] | None = None,
    reused_from: str | None = None,
) -> Units:
    """Grow a semantic cloud from `seed` and return it as Units.

    `sensitivity` is a cosine floor against the seed: 0.0 keeps everything the
    generator proposed, ~0.35 keeps a recognisable topic, ~0.6 keeps only close
    synonyms. Dropped terms are counted into meta rather than discarded
    silently — the drop rate is the most useful single diagnostic this produces,
    because a high rate means the generator wandered and the map you are looking
    at is narrower than the terms that were actually proposed.
    """
    from ..backend.embed import embed_texts

    # Re-embedding a FIXED concept set is the only way to change this map's
    # embedding space without also changing its points. The generator is
    # stochastic, so a plain rebuild against a new embedder moves both variables
    # at once and the two maps are not comparable. It is also the only way to
    # rebuild at all when the generator and the embedder cannot be resident
    # simultaneously — the real constraint on a 48GB box serving a 21.8GB
    # generator and a 23.5GB embedding host.
    if reuse_terms is not None:
        terms = list(reuse_terms)
        if len(terms) < 3:
            raise RuntimeError(
                f"reuse_terms had {len(terms)} terms; need at least 3"
            )
        if _norm(terms[0]) != _norm(seed):
            raise RuntimeError(
                f"reuse_terms[0] is {terms[0]!r} but the seed is {seed!r} — the "
                "seed anchors the map and every similarity is measured against "
                "it, so a mismatch would silently re-centre the cloud"
            )
        generator_used = f"reused:{reused_from}" if reused_from else "reused"
        # depth/parent structure is not recoverable from an exported label list,
        # and inventing one would put a false tree in meta
        depths = [0] * len(terms)
        print(f"  reusing {len(terms)} concepts from {reused_from or 'caller'} — no generator call")
    else:
        expand, generator_used = _make_expander(
            generator,
            ollama_host,
            ollama_model,
            openrouter_model,
            anthropic_model,
            env_file,
            llm_host,
            llm_model,
            llm_api_key,
        )

        seen: set[str] = {_norm(seed)}
        terms = [seed]
        depths = [0]
        parents: list[int] = [-1]

        frontier = [(seed, 0)]
        for d in range(1, depth + 1):
            next_frontier: list[tuple[str, int]] = []
            for i, (term, parent_idx) in enumerate(frontier, 1):
                if len(terms) >= max_terms:
                    break
                # one sequential LLM call per term, so a wide depth is minutes
                # long with nothing else to show for it — the build server turns
                # this into the live progress message
                print(f"  depth {d}: expanding {i}/{len(frontier)} {term!r}…", flush=True)
                try:
                    raw = expand(term, breadth)
                except Exception as e:
                    print(f"  expansion failed for {term!r} ({type(e).__name__}: {e})")
                    continue
                fresh = _clean(raw, seen, breadth)
                for f in fresh:
                    terms.append(f)
                    depths.append(d)
                    parents.append(parent_idx)
                    next_frontier.append((f, len(terms) - 1))
            print(f"  depth {d}: +{len(next_frontier)} terms (total {len(terms)})")
            frontier = next_frontier
            if not frontier:
                break

    if len(terms) < 3:
        raise RuntimeError(
            f"probe produced only {len(terms)} terms — the generator "
            f"({generator_used}) returned nothing usable for {seed!r}"
        )

    V = embed_texts(
        terms,
        host=embed_host,
        model=embed_model,
        api=embed_api,
        api_key=embed_api_key,
    )
    # embed_texts L2-normalizes, so cosine to the seed is a plain dot product
    sims = V @ V[0]

    keep = np.where(sims >= sensitivity)[0]
    if 0 not in keep:  # the seed anchors the map; it is never gated out
        keep = np.concatenate([[0], keep])
    keep = np.unique(keep)
    n_dropped = len(terms) - len(keep)

    if len(keep) < 3:
        raise RuntimeError(
            f"sensitivity {sensitivity} kept only {len(keep)} of {len(terms)} "
            "terms — lower it or raise --breadth"
        )

    kept_terms = [terms[i] for i in keep]
    print(
        f"  sensitivity {sensitivity}: kept {len(keep)}/{len(terms)} "
        f"(dropped {n_dropped}, {n_dropped / len(terms):.0%})"
    )

    return Units(
        ids=list(range(len(keep))),
        vectors=np.ascontiguousarray(V[keep], dtype=np.float32),
        labels=kept_terms,
        meta={
            "model": seed,
            "unit": f"probe_concept({embed_model})",
            "geometry": "third-party text-embedding space — NOT model-internal",
            "probe_seed": seed,
            "generator": generator_used,
            "embed_model": embed_model,
            "embed_host": embed_host,
            "embed_api": embed_api,
            "depth": depth,
            "breadth": breadth,
            "sensitivity": sensitivity,
            "n_proposed": len(terms),
            "kept": len(keep),
            "n_dropped": n_dropped,
            # a reused set was already sensitivity-filtered once, in a DIFFERENT
            # embedding space — so `n_proposed` here counts the terms this run
            # was handed, not terms any generator proposed. Say which it is.
            "terms_reused": reuse_terms is not None,
            "reused_from": reused_from,
            "max_depth_reached": (
                None if reuse_terms is not None else int(max(depths[i] for i in keep))
            ),
            "centered": False,
            "seed_similarity_min": round(float(sims[keep].min()), 4),
            "seed_similarity_mean": round(float(sims[keep].mean()), 4),
        },
    )
