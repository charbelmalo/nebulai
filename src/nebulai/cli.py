import argparse
import json
import time
from pathlib import Path

import numpy as np


def _timer():
    t0 = time.time()
    return lambda: f"{time.time() - t0:.1f}s"


def _default_embed_host() -> str:
    """`--embed-host` default, read at parser-build time so `--help` shows it.

    Imported lazily to keep this module's import cheap, matching the rest of the
    file's lazy-import style.
    """
    from .backend.embed import default_embed_host

    return default_embed_host()


def _update_index(out_root: Path) -> Path:
    """Rewrite out/index.json so the static viewer can discover datasets."""
    datasets = []
    for jp in sorted(out_root.glob("*/nebulai.json")):
        doc = json.loads(jp.read_text())
        meta = doc["meta"]
        datasets.append(
            {
                "id": jp.parent.name,
                "model": meta.get("model", jp.parent.name),
                # names the geometry's true origin — "token_embedding" for W_E
                # maps, "api_text_embedding(<embedder>)" for api-source maps
                "unit": meta.get("unit", "token_embedding"),
                "path": f"{jp.parent.name}/nebulai.json",
                "schema_version": meta.get("schema_version", 1),
                "n_points": meta["n_points"],
                "n_clusters": meta["n_clusters"],
                "noise_fraction": meta["noise_fraction"],
                "namer": meta.get("namer"),
                "has_edges": "edges" in doc,
                # the viewer's Internals model picker reads this instead of
                # probing each model's interp/index.json over the network
                "has_interp": (jp.parent / "interp" / "index.json").exists(),
            }
        )
    index = out_root / "index.json"
    index.write_text(json.dumps({"datasets": datasets}, ensure_ascii=False))
    return index


def _run_tokens(args: argparse.Namespace) -> None:
    import os

    from .backend.cluster import cluster_units, resolve_cluster_params
    from .backend.export import export_json
    from .backend.name import name_clusters
    from .backend.reduce import reduce_vectors
    from .backend.viz import render

    dataset_id = args.model.replace("/", "__")
    if args.source == "api":
        dataset_id += f"__api-{args.embed_model.replace('/', '__')}"
    elif getattr(args, "unembedding", False):
        # W_U is a different matrix from W_E — never share an output dir (or a
        # reduction cache) with the embedding map
        dataset_id += "__unembed"
    out_dir = Path(args.out) / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t = _timer()
    if args.source == "api":
        from .frontends.api_tokens import load_api_token_units

        units = load_api_token_units(
            args.model,
            embed_host=args.embed_host,
            embed_model=args.embed_model,
            api=args.embed_api,
            api_key=os.environ.get("EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            center=not args.no_center,
            max_tokens=args.max_tokens,
            out_root=Path(args.out),
        )
        print(
            f"[1/5] loaded {len(units)} token units from {args.model} via "
            f"{args.embed_model}@{args.embed_host} — api text embeddings, "
            f"NOT model-internal geometry (vocab {units.meta['vocab_size']}, "
            f"curated to {units.meta['kept']}) [{t()}]"
        )
    else:
        from .frontends.tokens import load_token_units

        units = load_token_units(
            args.model,
            center=not args.no_center,
            max_tokens=args.max_tokens,
            revision=args.revision,
            remote=args.remote,
            which="output" if args.unembedding else "input",
        )
        mb = units.meta["bytes_fetched"] / 1e6
        print(
            f"[1/5] loaded {len(units)} token units from {args.model} "
            f"@{units.meta['revision'][:8]} ({units.meta['which']} / "
            f"{units.meta['weight_key']}, vocab {units.meta['vocab_size']}, "
            f"curated to {units.meta['kept']}, {units.meta['source']}, "
            f"{mb:.0f} MB fetched) [{t()}]"
        )

    # UMAP is the expensive step — cache reductions keyed by their params
    reduce_params = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "center": not args.no_center,
        "cluster_dim": args.cluster_dim,
        "n_neighbors": args.n_neighbors,
        "seed": args.seed,
    }
    if args.source != "api":
        # geometry-identifying params: a cached reduction from a different
        # revision or matrix is a different cloud, not a cache hit
        reduce_params["revision"] = units.meta["revision"]
        reduce_params["which"] = units.meta["which"]
    if args.source == "api":
        # extra keys only for api builds, so existing hf caches stay valid
        reduce_params["source"] = "api"
        reduce_params["embed_model"] = args.embed_model
    cache = out_dir / "reduced.npz"
    cache_meta = out_dir / "reduced.params.json"
    t = _timer()
    if (
        not args.force
        and cache.exists()
        and cache_meta.exists()
        and json.loads(cache_meta.read_text()) == reduce_params
    ):
        z = np.load(cache)
        u_cluster, u3, u2 = z["u_cluster"], z["u3"], z["u2"]
        print(f"[2/5] reused cached reductions from {cache} [{t()}]")
    else:
        u_cluster, u3, u2 = reduce_vectors(
            units.vectors,
            cluster_dim=args.cluster_dim,
            n_neighbors=args.n_neighbors,
            seed=args.seed,
        )
        np.savez_compressed(cache, u_cluster=u_cluster, u3=u3, u2=u2)
        cache_meta.write_text(json.dumps(reduce_params))
        print(f"[2/5] UMAP -> {args.cluster_dim}d/3d/2d [{t()}]")

    t = _timer()
    cluster_ids, probs = cluster_units(
        u_cluster,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        method=args.cluster_method,
    )
    # record the params HDBSCAN actually saw so the map is reproducible and
    # the metrics table can report how each cloud was clustered (defaults are
    # n-dependent, so a re-sweep's chosen values must be stamped, not implied)
    units.meta["hdbscan"] = resolve_cluster_params(
        len(u_cluster),
        args.min_cluster_size,
        args.min_samples,
        args.cluster_method,
    )
    n_clusters = len({int(c) for c in cluster_ids if c >= 0})
    noise = float((cluster_ids < 0).mean())
    print(f"[3/5] HDBSCAN: {n_clusters} clusters, {noise:.0%} noise [{t()}]")

    t = _timer()
    titles, namer_used = name_clusters(
        units,
        cluster_ids,
        namer=args.namer,
        openrouter_model=args.openrouter_model,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        anthropic_model=args.anthropic_model,
        claude_cli_model=args.claude_cli_model,
        codex_cli_model=args.codex_cli_model,
        env_file=args.env_file,
        llm_host=args.llm_host,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        hf_model=args.hf_model,
        model=args.namer_model,
        max_cost_usd=args.max_cost_usd,
    )
    print(f"[4/5] named {len(titles)} clusters via '{namer_used}' [{t()}]")

    t = _timer()
    json_path = out_dir / "nebulai.json"
    meta = export_json(
        json_path,
        units,
        u2,
        u3,
        cluster_ids,
        probs,
        titles,
        namer_used,
        u_cluster=u_cluster,
        edges_mode=args.edges,
    )
    png = out_dir / "map_static.png"
    html = out_dir / "map_interactive.html"
    geometry_line = (
        f"api text embeddings ({args.embed_model}) — not model-internal"
        if args.source == "api"
        else "embedding rows"
    )
    render(
        u2,
        cluster_ids,
        titles,
        units.labels,
        png,
        html,
        title=f"Nebul.AI — {args.model} token map",
        sub_title=(
            f"{meta['n_points']} tokens · {meta['n_clusters']} clusters · "
            f"{geometry_line} -> UMAP -> HDBSCAN"
        ),
    )
    _update_index(Path(args.out))
    print(f"[5/5] exported [{t()}]")
    for p in (json_path, png, html):
        print(f"  {p}")


def _run_sae(args: argparse.Namespace) -> None:
    """Plan A: SAE decoder-direction map. Mirrors _run_tokens's 5-stage
    structure and its exact `[k/5] ...` prints (build_server parses them)."""
    from .backend.cluster import cluster_units, resolve_cluster_params
    from .backend.export import export_json
    from .backend.name import name_clusters, placeholder_titles
    from .backend.reduce import reduce_vectors
    from .backend.viz import render
    from .frontends.sae import load_sae_units, sae_dataset_id

    if args.source == "label":
        raise SystemExit(
            "label-space projection is not implemented — it would lay the map "
            "out by the label-embedder's semantics, not the model's geometry; "
            "the MVP ships model space (decoder) only"
        )

    t = _timer()
    units = load_sae_units(
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        max_features=args.max_features,
        center=args.center,
        labels_source=args.labels,
        out_root=Path(args.out),
    )
    dataset_id = sae_dataset_id(units.meta["model"], args.sae_id)
    out_dir = Path(args.out) / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[1/5] loaded {len(units)} SAE feature units from "
        f"{args.sae_release}/{args.sae_id} (d_sae {units.meta['d_sae']}, "
        f"curated to {units.meta['kept']}, {units.meta['n_labeled']} labeled) [{t()}]"
    )

    reduce_params = {
        "release": args.sae_release,
        "sae_id": args.sae_id,
        "max_features": args.max_features,
        "source": args.source,
        "center": args.center,
        "cluster_dim": args.cluster_dim,
        "n_neighbors": args.n_neighbors,
        "seed": args.seed,
    }
    cache = out_dir / "reduced.npz"
    cache_meta = out_dir / "reduced.params.json"
    t = _timer()
    if (
        not args.force
        and cache.exists()
        and cache_meta.exists()
        and json.loads(cache_meta.read_text()) == reduce_params
    ):
        z = np.load(cache)
        u_cluster, u3, u2 = z["u_cluster"], z["u3"], z["u2"]
        print(f"[2/5] reused cached reductions from {cache} [{t()}]")
    else:
        u_cluster, u3, u2 = reduce_vectors(
            units.vectors,
            cluster_dim=args.cluster_dim,
            n_neighbors=args.n_neighbors,
            seed=args.seed,
        )
        np.savez_compressed(cache, u_cluster=u_cluster, u3=u3, u2=u2)
        cache_meta.write_text(json.dumps(reduce_params))
        print(f"[2/5] UMAP -> {args.cluster_dim}d/3d/2d [{t()}]")

    t = _timer()
    cluster_ids, probs = cluster_units(
        u_cluster,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        method=args.cluster_method,
    )
    # record the params HDBSCAN actually saw so the map is reproducible and
    # the metrics table can report how each cloud was clustered (defaults are
    # n-dependent, so a re-sweep's chosen values must be stamped, not implied)
    units.meta["hdbscan"] = resolve_cluster_params(
        len(u_cluster),
        args.min_cluster_size,
        args.min_samples,
        args.cluster_method,
    )
    n_clusters = len({int(c) for c in cluster_ids if c >= 0})
    noise = float((cluster_ids < 0).mean())
    print(f"[3/5] HDBSCAN: {n_clusters} clusters, {noise:.0%} noise [{t()}]")

    t = _timer()
    if units.meta.get("n_labeled", 0) == 0:
        # every member label is a placeholder (--labels none) — an LLM namer
        # would invent semantics from zero information, so title honestly
        titles, namer_used = placeholder_titles(cluster_ids, "features")
    else:
        titles, namer_used = name_clusters(
            units,
            cluster_ids,
            namer=args.namer,
            openrouter_model=args.openrouter_model,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
            anthropic_model=args.anthropic_model,
            claude_cli_model=args.claude_cli_model,
            codex_cli_model=args.codex_cli_model,
            env_file=args.env_file,
            llm_host=args.llm_host,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            hf_model=args.hf_model,
            model=args.namer_model,
            max_cost_usd=args.max_cost_usd,
        )
    print(f"[4/5] named {len(titles)} clusters via '{namer_used}' [{t()}]")

    t = _timer()
    json_path = out_dir / "nebulai.json"
    meta = export_json(
        json_path,
        units,
        u2,
        u3,
        cluster_ids,
        probs,
        titles,
        namer_used,
        u_cluster=u_cluster,
        edges_mode=args.edges,
    )
    png = out_dir / "map_static.png"
    html = out_dir / "map_interactive.html"
    render(
        u2,
        cluster_ids,
        titles,
        units.labels,
        png,
        html,
        title=f"Nebul.AI — {units.meta['model']} SAE feature map",
        sub_title=(
            f"{meta['n_points']} SAE features · {meta['n_clusters']} clusters · "
            f"SAE decoder directions ({args.sae_id}, {meta.get('hook', 'resid')}-SAE)"
            f" -> UMAP -> HDBSCAN"
        ),
    )
    _update_index(Path(args.out))
    print(f"[5/5] exported [{t()}]")
    for p in (json_path, png, html):
        print(f"  {p}")


def _run_neurons(args: argparse.Namespace) -> None:
    """Plan B: MLP-neuron write-direction map. Mirrors _run_sae's 5-stage
    structure and its exact `[k/5] ...` prints (build_server parses them)."""
    from .backend.cluster import cluster_units, resolve_cluster_params
    from .backend.export import export_json
    from .backend.name import name_clusters
    from .backend.reduce import reduce_vectors
    from .backend.viz import render
    from .frontends.neurons import (
        load_neuron_units,
        neuron_dataset_id,
        placeholder_titles,
    )

    if args.source == "label":
        raise SystemExit(
            "label-space projection is not implemented — it would lay the map "
            "out by the label-embedder's semantics, not the model's geometry; "
            "the MVP ships model space (w_out rows) only"
        )

    t = _timer()
    expert = args.expert
    if expert is not None and expert not in ("shared", "dense"):
        try:
            expert = int(expert)
        except ValueError:
            raise SystemExit(
                f"--expert must be an integer, 'shared' or 'dense', not {expert!r}"
            ) from None
    units = load_neuron_units(
        model_repo=args.model,
        layer=args.layer,
        max_neurons=args.max_neurons,
        center=args.center,
        labels_source=args.labels,
        out_root=Path(args.out),
        revision=args.revision,
        expert=expert,
        remote=args.remote,
    )
    # arch-aware path (gpt2 c_proj vs llama down_proj) is stamped by the loader
    tensor_path = units.meta["tensor_path"]
    dataset_id = neuron_dataset_id(units.meta["model"], tensor_path)
    out_dir = Path(args.out) / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[1/5] loaded {len(units)} MLP neuron units from "
        f"{args.model}/{tensor_path} (d_mlp {units.meta['d_mlp']}, "
        f"curated to {units.meta['kept']}, {units.meta['n_labeled']} labeled) [{t()}]"
    )

    reduce_params = {
        "model_repo": args.model,
        "layer": args.layer,
        "max_neurons": args.max_neurons,
        "source": args.source,
        "revision": units.meta["revision"],
        "expert": units.meta["expert"],
        "center": args.center,
        "cluster_dim": args.cluster_dim,
        "n_neighbors": args.n_neighbors,
        "seed": args.seed,
    }
    cache = out_dir / "reduced.npz"
    cache_meta = out_dir / "reduced.params.json"
    t = _timer()
    if (
        not args.force
        and cache.exists()
        and cache_meta.exists()
        and json.loads(cache_meta.read_text()) == reduce_params
    ):
        z = np.load(cache)
        u_cluster, u3, u2 = z["u_cluster"], z["u3"], z["u2"]
        print(f"[2/5] reused cached reductions from {cache} [{t()}]")
    else:
        u_cluster, u3, u2 = reduce_vectors(
            units.vectors,
            cluster_dim=args.cluster_dim,
            n_neighbors=args.n_neighbors,
            seed=args.seed,
        )
        np.savez_compressed(cache, u_cluster=u_cluster, u3=u3, u2=u2)
        cache_meta.write_text(json.dumps(reduce_params))
        print(f"[2/5] UMAP -> {args.cluster_dim}d/3d/2d [{t()}]")

    t = _timer()
    cluster_ids, probs = cluster_units(
        u_cluster,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        method=args.cluster_method,
    )
    # record the params HDBSCAN actually saw so the map is reproducible and
    # the metrics table can report how each cloud was clustered (defaults are
    # n-dependent, so a re-sweep's chosen values must be stamped, not implied)
    units.meta["hdbscan"] = resolve_cluster_params(
        len(u_cluster),
        args.min_cluster_size,
        args.min_samples,
        args.cluster_method,
    )
    n_clusters = len({int(c) for c in cluster_ids if c >= 0})
    noise = float((cluster_ids < 0).mean())
    print(f"[3/5] HDBSCAN: {n_clusters} clusters, {noise:.0%} noise [{t()}]")

    t = _timer()
    if units.meta.get("n_labeled", 0) == 0:
        # every member label is a placeholder — an LLM namer would invent
        # semantics from zero information, so title clusters honestly instead
        titles, namer_used = placeholder_titles(cluster_ids)
    else:
        titles, namer_used = name_clusters(
            units,
            cluster_ids,
            namer=args.namer,
            openrouter_model=args.openrouter_model,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
            anthropic_model=args.anthropic_model,
            claude_cli_model=args.claude_cli_model,
            codex_cli_model=args.codex_cli_model,
            env_file=args.env_file,
            llm_host=args.llm_host,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            hf_model=args.hf_model,
            model=args.namer_model,
            max_cost_usd=args.max_cost_usd,
        )
    print(f"[4/5] named {len(titles)} clusters via '{namer_used}' [{t()}]")

    t = _timer()
    json_path = out_dir / "nebulai.json"
    meta = export_json(
        json_path,
        units,
        u2,
        u3,
        cluster_ids,
        probs,
        titles,
        namer_used,
        u_cluster=u_cluster,
        edges_mode=args.edges,
    )
    png = out_dir / "map_static.png"
    html = out_dir / "map_interactive.html"
    render(
        u2,
        cluster_ids,
        titles,
        units.labels,
        png,
        html,
        title=f"Nebul.AI — {units.meta['model']} MLP neuron map",
        sub_title=(
            f"{meta['n_points']} MLP neurons · {meta['n_clusters']} clusters · "
            f"MLP write directions ({tensor_path}) -> UMAP -> HDBSCAN"
        ),
    )
    _update_index(Path(args.out))
    print(f"[5/5] exported [{t()}]")
    for p in (json_path, png, html):
        print(f"  {p}")


def _run_edges(args: argparse.Namespace) -> None:
    """Backfill schema-v2 edges into existing nebulai.json artifacts.

    Uses the cached 10-D u_cluster from reduced.npz — no UMAP recompute.
    """
    from .backend.edges import compute_edges
    from .backend.export import SCHEMA_VERSION

    out_root = Path(args.out)
    for m in args.models:
        model_dir = out_root / m.replace("/", "__")
        jp = model_dir / "nebulai.json"
        npz = model_dir / "reduced.npz"
        params_path = model_dir / "reduced.params.json"
        for req in (jp, npz, params_path):
            if not req.exists():
                raise SystemExit(
                    f"missing {req} — run `nebulai tokens --model {m}` first"
                )
        params = json.loads(params_path.read_text())
        if params.get("model") != m:
            raise SystemExit(
                f"{params_path} was built for model "
                f"{params.get('model')!r}, not {m!r} — refusing to backfill"
            )

        t = _timer()
        doc = json.loads(jp.read_text())
        u_cluster = np.load(npz)["u_cluster"]
        if len(u_cluster) != len(doc["points"]):
            raise SystemExit(
                f"{npz} has {len(u_cluster)} rows but {jp} has "
                f"{len(doc['points'])} points — stale cache, rerun tokens"
            )
        cluster_ids = np.array(
            [p["cluster_id"] for p in doc["points"]], dtype=np.int64
        )
        doc["edges"] = compute_edges(
            u_cluster, cluster_ids, include_knn=(args.mode == "knn")
        )
        doc["meta"]["schema_version"] = SCHEMA_VERSION
        jp.write_text(json.dumps(doc, ensure_ascii=False))
        e = doc["edges"]
        knn = f", knn k={e['knn']['k']}" if "knn" in e else ""
        print(
            f"{m}: {len(e['cluster_edges'])} cluster edges{knn} "
            f"({e['metric']}@{e['space']}) [{t()}]"
        )
    index = _update_index(out_root)
    print(f"  {index}")


def _run_interp(args: argparse.Namespace) -> None:
    """Compute real interp bundles (weight spectra, positional DFT, forward
    traces) for the Phase-2 viewer's mechanistic-interpretability features.

    Pure numpy from the model's safetensors — no torch. See
    docs/INTERP_FEATURES.md for the feature → bundle map.
    """
    from .backend.interp.bundles import DEFAULT_PROMPTS, write_bundles

    prompts = None
    if args.prompts_file:
        prompts = [
            ln.strip()
            for ln in Path(args.prompts_file).read_text().splitlines()
            if ln.strip()
        ]

    t = _timer()
    written = write_bundles(args.model, Path(args.out), prompts)
    print(
        f"interp bundles for {args.model}: {len(written)} files "
        f"({len(prompts or DEFAULT_PROMPTS)} traces) [{t()}]"
    )
    for p in written:
        print(f"  {p}")
    # refresh the discovery index so the viewer's model picker sees the new
    # bundle without a re-run of `nebulai tokens`
    _update_index(Path(args.out))


def _run_metrics(args: argparse.Namespace) -> None:
    """Structural comparison table (silhouette / noise / cluster count) across
    several already-built maps — the quantitative artifact behind the A-vs-B-vs-C
    claim. Writes out/compare/metrics.json and prints an aligned table."""
    from .backend.metrics import add_verdict, compute_map_metrics, format_table

    out_root = Path(args.out)
    rows = []
    for m in args.datasets:
        dd = out_root / m.replace("/", "__")
        if not (dd / "nebulai.json").exists():
            raise SystemExit(
                f"missing {dd / 'nebulai.json'} — build it first "
                f"(nebulai tokens/sae/neurons)"
            )
        rows.append(compute_map_metrics(dd))

    print(format_table(rows))

    cmp_dir = out_root / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cmp_dir / "metrics.json"
    # the null-margin verdict is derived HERE rather than in the viewer: the
    # `!` / `?` rules in margin_flags() are the honesty logic, and reimplementing
    # them in TypeScript would give the browser a second opinion that drifts from
    # this table. The viewer renders what this file already decided.
    metrics_path.write_text(
        json.dumps(
            {"maps": [add_verdict(r) for r in rows]}, ensure_ascii=False, indent=2
        )
    )
    print(f"\n  {metrics_path}")


def _run_probe(args: argparse.Namespace) -> None:
    """Semantic cloud grown from a seed topic — the one front-end that needs no
    model weights. Mirrors _run_tokens's 5-stage structure and its exact
    `[k/5] ...` prints (build_server parses them)."""
    import os

    from .backend.cluster import cluster_units, resolve_cluster_params
    from .backend.export import export_json
    from .backend.name import name_clusters
    from .backend.reduce import reduce_vectors
    from .backend.viz import render
    from .frontends.probe import load_probe_units, probe_dataset_id

    dataset_id = probe_dataset_id(args.seed)
    out_dir = Path(args.out) / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --reuse-terms re-embeds an EXISTING map's concepts instead of generating
    # new ones. That holds the point set fixed so a change of embedder is the
    # only variable, and it drops the generator from the run entirely — which is
    # what makes a rebuild possible when the generator and the embedding host
    # cannot fit in RAM at the same time.
    reuse_terms = reused_from = None
    if args.reuse_terms:
        src = Path(args.out) / args.reuse_terms
        src_json = src / "nebulai.json"
        if not src_json.exists():
            raise SystemExit(f"probe: --reuse-terms {args.reuse_terms} has no {src_json}")
        src_doc = json.loads(src_json.read_text())
        src_meta = src_doc["meta"]
        if not str(src_meta.get("unit", "")).startswith("probe_concept"):
            raise SystemExit(
                f"probe: --reuse-terms {args.reuse_terms} is a "
                f"{src_meta.get('unit')!r} map, not a probe map — its labels are "
                "not concept strings"
            )
        if str(src_meta.get("probe_seed")) != str(args.seed):
            raise SystemExit(
                f"probe: --reuse-terms {args.reuse_terms} was grown from "
                f"{src_meta.get('probe_seed')!r} but --seed is {args.seed!r}; "
                "similarities are measured against the seed, so reusing another "
                "cloud's terms would re-centre the map"
            )
        reuse_terms = [str(p["label"]) for p in src_doc["points"]]
        reused_from = args.reuse_terms

    t = _timer()
    # unlike the weight-reading front-ends, this one cannot run offline: it needs
    # a generator AND an embedder. Both failure modes are ordinary setup problems
    # (no ollama, no key, threshold too high), so report them as such rather than
    # dumping a traceback — load_probe_units already names every backend it tried.
    try:
        units = load_probe_units(
            args.seed,
            depth=args.depth,
            breadth=args.breadth,
            sensitivity=args.sensitivity,
            generator=args.generator,
            ollama_host=args.ollama_host,
            ollama_model=args.ollama_model,
            openrouter_model=args.openrouter_model,
            anthropic_model=args.anthropic_model,
            # NB: no claude_cli/codex_cli here — those are NAMER backends. The
            # probe generator has its own backend set (_make_expander).
            env_file=args.env_file,
            embed_host=args.embed_host,
            embed_model=args.embed_model,
            embed_api=args.embed_api,
            embed_api_key=os.environ.get("EMBED_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            llm_host=args.llm_host,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            hf_model=args.hf_model,
            # the generator is pinned by the SAME flag as the namer: a probe
            # run names one model, and the cloud and its titles should not be
            # able to disagree about which one it was
            generator_model=args.namer_model,
            max_cost_usd=args.max_cost_usd,
            reuse_terms=reuse_terms,
            reused_from=reused_from,
        )
    except RuntimeError as exc:
        raise SystemExit(f"probe: {exc}") from exc
    m = units.meta
    print(
        f"[1/5] grew {len(units)} concepts from {args.seed!r} via {m['generator']} "
        f"-> {m['embed_model']} — third-party embedding space, NOT model-internal "
        f"(proposed {m['n_proposed']}, kept {m['kept']}) [{t()}]"
    )

    # the expansion is nondeterministic, so there is no param-keyed reduction
    # cache here: reusing one across runs would silently pair a new cloud's
    # labels with an old cloud's layout
    t = _timer()
    n_neighbors = args.n_neighbors or max(5, min(30, len(units) // 10))
    u_cluster, u3, u2 = reduce_vectors(
        units.vectors,
        cluster_dim=args.cluster_dim,
        n_neighbors=n_neighbors,
        seed=args.seed_rng,
    )
    np.savez_compressed(
        out_dir / "reduced.npz", u_cluster=u_cluster, u3=u3, u2=u2
    )
    print(f"[2/5] UMAP -> {args.cluster_dim}d/3d/2d (n_neighbors={n_neighbors}) [{t()}]")

    t = _timer()
    cluster_ids, probs = cluster_units(
        u_cluster,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        method=args.cluster_method,
    )
    units.meta["hdbscan"] = resolve_cluster_params(
        len(u_cluster), args.min_cluster_size, args.min_samples, args.cluster_method
    )
    n_clusters = len({int(c) for c in cluster_ids if c >= 0})
    noise = float((cluster_ids < 0).mean())
    print(f"[3/5] HDBSCAN: {n_clusters} clusters, {noise:.0%} noise [{t()}]")

    t = _timer()
    titles, namer_used = name_clusters(
        units,
        cluster_ids,
        namer=args.namer,
        openrouter_model=args.openrouter_model,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        anthropic_model=args.anthropic_model,
        claude_cli_model=args.claude_cli_model,
        codex_cli_model=args.codex_cli_model,
        env_file=args.env_file,
        llm_host=args.llm_host,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        hf_model=args.hf_model,
        model=args.namer_model,
        max_cost_usd=args.max_cost_usd,
    )
    print(f"[4/5] named {len(titles)} clusters via '{namer_used}' [{t()}]")

    t = _timer()
    json_path = out_dir / "nebulai.json"
    meta = export_json(
        json_path,
        units,
        u2,
        u3,
        cluster_ids,
        probs,
        titles,
        namer_used,
        u_cluster=u_cluster,
        edges_mode=args.edges,
    )
    png = out_dir / "map_static.png"
    html = out_dir / "map_interactive.html"
    render(
        u2,
        cluster_ids,
        titles,
        units.labels,
        png,
        html,
        title=f"Nebul.AI — semantic probe: {args.seed}",
        sub_title=(
            f"{meta['n_points']} concepts · {meta['n_clusters']} clusters · "
            f"{m['generator']} -> {m['embed_model']} (sensitivity "
            f"{args.sensitivity}) -> UMAP -> HDBSCAN"
        ),
    )
    _update_index(Path(args.out))
    print(f"[5/5] exported [{t()}]")
    for p in (json_path, png, html):
        print(f"  {p}")


def _run_validate(args: argparse.Namespace) -> None:
    """Independent validation for already-built maps — the checks that are NOT
    the construction procedure (see backend/validate.py).

    Writes `validation.json` next to each map's `nebulai.json`; `nebulai
    metrics` picks it up from there and adds the trust / seed.ARI / null.sil
    columns. Kept a separate command because it re-runs UMAP: this is minutes
    per map, not part of a build."""
    from .backend.validate import validate_map

    out_root = Path(args.out)
    for m in args.datasets:
        dd = out_root / m.replace("/", "__")
        if not (dd / "nebulai.json").exists():
            raise SystemExit(
                f"missing {dd / 'nebulai.json'} — build it first "
                f"(nebulai tokens/sae/neurons)"
            )
        t = _timer()
        print(f"validating {dd.name} — reloading front-end vectors ...")
        try:
            report = validate_map(
                dd,
                trust_neighbors=args.trust_neighbors,
                stability_seeds=tuple(args.seeds),
                trust_sample_cap=args.trust_sample,
                stability_sample_cap=args.stability_sample,
                skip_stability=args.skip_stability,
                skip_null=args.skip_null,
            )
        except (ValueError, FileNotFoundError, KeyError) as e:
            # an honest skip beats a number that describes a different point set
            print(f"  skipped: {e}")
            continue

        path = dd / "validation.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

        tw = report["trustworthiness"]
        print(
            f"  trustworthiness {tw['trustworthiness']} "
            f"(k={tw['n_neighbors']}, n={tw['n_scored']}"
            f"{', subsampled' if tw['subsampled'] else ''})"
        )
        if "stability" in report:
            s = report["stability"]
            print(
                f"  seed stability  mean ARI {s['mean_ari']} / min {s['min_ari']} "
                f"over seeds {s['seeds']} (n={s['n_scored']}, "
                f"method={report['cluster_method']})"
            )
        if "null_baseline" in report:
            n = report["null_baseline"]
            print(
                f"  null baseline   silhouette {n['silhouette']} "
                f"({n['n_clusters']} clusters, {n['noise_fraction']:.0%} noise, "
                f"n={n['n_scored']})"
            )
        print(f"  {path} [{t()}]")


def _run_rename(args: argparse.Namespace) -> None:
    """Re-title already-built maps without rebuilding them (see backend/rename.py).

    A separate command because titles and geometry have separate lifetimes: the
    points here were built once and are correct, while the namer that titled
    them has been replaced twice. Rebuilding to fix only the titles would
    re-run UMAP over 50k vectors to land back on the same coordinates."""
    from .backend.rename import rename_map, sync_index
    from .llm import BudgetError, IdentityError

    out_root = Path(args.out)
    if len(args.datasets) == 1 and args.datasets[0] == "all":
        args.datasets = sorted(
            d.name for d in out_root.iterdir() if (d / "nebulai.json").is_file()
        )
        if not args.datasets:
            raise SystemExit(f"no built maps under {out_root}/")

    done, skipped = [], []
    for m in args.datasets:
        dd = out_root / m.replace("/", "__")
        t = _timer()
        print(f"renaming {dd.name} ...")
        try:
            r = rename_map(
                dd,
                namer=args.namer,
                openrouter_model=args.openrouter_model,
                ollama_model=args.ollama_model,
                ollama_host=args.ollama_host,
                anthropic_model=args.anthropic_model,
                llm_host=args.llm_host,
                llm_model=args.llm_model,
                llm_api_key=args.llm_api_key,
                hf_model=args.hf_model,
                model=args.namer_model,
                max_cost_usd=args.max_cost_usd,
                claude_cli_model=args.claude_cli_model,
                codex_cli_model=args.codex_cli_model,
                env_file=args.env_file,
            )
        except (IdentityError, BudgetError) as e:
            # NOT a per-map skip: an unservable pin or a blown budget will hold
            # for every remaining map too, so skipping would print the same
            # refusal N times and still exit 0 with nothing renamed
            raise SystemExit(f"rename: {e}") from e
        except (ValueError, FileNotFoundError, KeyError) as e:
            # a map left with its old titles is a better outcome than a map
            # given invented ones, so this is a skip, not a failure
            print(f"  skipped: {e}")
            skipped.append(dd.name)
            continue
        done.append(r)
        print(
            f"  {r['renamed']}/{r['n_clusters']} titles from {r['namer']} "
            f"(was {r['was']}) [{t()}]"
        )

    sync_index(out_root)
    print(f"\n{len(done)} renamed, {len(skipped)} skipped")
    for r in done:
        print(f"  {r['id']:<52} {r['was']} -> {r['namer']}")


def _run_compare(args: argparse.Namespace) -> None:
    import os

    from .backend import embed as embed_mod
    from .backend.compare import build_comparison, export_comparison
    from .backend.viewer import write_viewer

    out_root = Path(args.out)

    # `all` is not a convenience — hand-listing eleven dataset ids is how a
    # comparison silently ends up missing the front-ends it exists to contrast
    if len(args.models) == 1 and args.models[0] == "all":
        names = sorted(
            d.name for d in out_root.iterdir() if (d / "nebulai.json").is_file()
        )
        if not names:
            raise SystemExit(f"no built maps under {out_root}/")
        print(f"  comparing all {len(names)} maps under {out_root}/")
        args.models = names

    json_paths = []
    for m in args.models:
        # a model id ("EleutherAI/pythia-70m") and a dataset id
        # ("gpt2__sae__blocks.8...") both land here — the SAE/neuron/api
        # front-ends only ever produce the latter
        jp = out_root / m.replace("/", "__") / "nebulai.json"
        if not jp.exists():
            avail = sorted(
                d.name for d in out_root.iterdir() if (d / "nebulai.json").is_file()
            )
            raise SystemExit(
                f"missing {jp}\n  build it first (`nebulai tokens --model {m}`), "
                f"or pass one of the {len(avail)} maps already in {out_root}/:\n    "
                + "\n    ".join(avail)
            )
        json_paths.append(jp)

    t = _timer()
    # explicit flag > NEBULAI_EMBED_HOST > --ollama-host. The env var sits in the
    # middle so a box whose embedder lives somewhere other than the namer (here:
    # ollama on :11435, not the stock :11434) can be configured once instead of
    # per invocation — see backend/embed.py's docstring for why that matters.
    # resolve_embed_host turns a discovery sentinel ("auto"/"m4"/...) — whether it
    # arrives via --embed-host or NEBULAI_EMBED_HOST — into the dynamically located
    # M4 URL, and passes any concrete URL (or None) through untouched.
    embed_host = (
        embed_mod.resolve_embed_host(args.embed_host)
        or embed_mod.resolve_embed_host(os.environ.get(embed_mod.EMBED_HOST_ENV))
        or args.ollama_host
    )
    try:
        comp = build_comparison(
            json_paths,
            embed_host=embed_host,
            embed_model=args.embed_model,
            seed=args.seed,
            embed_api=args.embed_api,
            embed_api_key=os.environ.get("EMBED_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
        )
    except RuntimeError as e:
        # `compare` is the one command that cannot run without a reachable
        # embedder, and it used to surface that as a raw traceback. The message
        # from _embed_batch already names the fix; just don't bury it in a stack.
        raise SystemExit(f"compare needs a reachable embedder.\n{e}") from e
    print(
        f"[1/2] built comparison: {comp['meta']['n_points']} clusters, "
        f"{comp['meta']['n_meta_clusters']} meta-clusters, "
        f"{comp['stats']['n_shared_concepts']} shared [{t()}]"
    )

    t = _timer()
    cmp_dir = out_root / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    export_comparison(cmp_dir / "compare.json", comp)
    write_viewer(cmp_dir / "index.html", comp)
    print(f"[2/2] exported [{t()}]")
    print(f"  {cmp_dir / 'compare.json'}")
    print(
        "\n  view it in the unified viewer (npm run dev --prefix viewer):"
        "\n    http://localhost:5173/?view=compare"
    )
    print(
        f"  standalone fallback (deprecated): {cmp_dir / 'index.html'}"
        "  (open in Chrome/Edge)"
    )
    print("\n  concept overlap (Jaccard):")
    for k, v in comp["stats"]["jaccard"].items():
        print(f"    {k}: {v}")


def _add_llm_args(sp: argparse.ArgumentParser) -> None:
    """Flags for an OpenAI-compatible chat server (LM Studio, vLLM, llama.cpp,
    an MLX box on the LAN), plus the model-identity and cost-ceiling flags.
    Shared by every subcommand that names clusters, so the four namer
    front-ends cannot drift apart."""
    from .corpus import DEFAULT_MAX_COST_USD as _DEFAULT_MAX_COST_USD
    sp.add_argument(
        "--llm-host",
        default="http://localhost:8050",
        help="OpenAI-compatible chat base URL, e.g. http://<lan-host>:8050",
    )
    sp.add_argument(
        "--llm-model",
        default="",
        help="chat model id, or a fragment to match (default: first model on "
        "the server that is not an embedder/reranker/audio model)",
    )
    sp.add_argument(
        "--llm-api-key",
        default=None,
        help="bearer token for --llm-host (omit for a keyless local server)",
    )
    sp.add_argument(
        "--namer-model",
        default=None,
        help="PIN the naming model: a corpus key (muse-glimmer-30b), an HF repo "
        "or an endpoint slug. Only a backend serving that exact model may run; "
        "if none can, the run FAILS instead of titling the map with a different "
        "model. Omit for 'any reachable namer' — the chain then stamps whatever "
        "answered into meta.namer_model",
    )
    sp.add_argument(
        "--hf-model",
        default="",
        help="model id for --namer hf when nothing is pinned (HF Inference "
        "Providers router, e.g. meta-models/Muse-Glimmer-30B)",
    )
    sp.add_argument(
        "--max-cost-usd",
        type=float,
        default=_DEFAULT_MAX_COST_USD,
        help=f"ceiling on one command's spend at a paid endpoint (default "
        f"${_DEFAULT_MAX_COST_USD:.2f}). Over budget the run is REFUSED with the "
        "estimate and the cheaper corpus alternatives listed; it never downgrades "
        "to a cheaper model on your behalf. A $0.00 endpoint skips the gate",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        prog="nebulai",
        description="Nebul.AI — semantic cloud of a micro model's concept space",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tokens", help="Plan C: token-embedding map")
    t.add_argument("--model", default="gpt2", help="HF model id (default: gpt2)")
    t.add_argument("--out", default="out", help="output directory root")
    t.add_argument(
        "--source",
        choices=["hf", "api"],
        default="hf",
        help="geometry source: hf = the model's own W_E rows (model-internal); "
        "api = a third-party text embedder over the same curated vocab "
        "(labeled as such — NOT model-internal geometry)",
    )
    t.add_argument(
        "--embed-host",
        default=_default_embed_host(),
        help="[--source api] embeddings endpoint base URL "
        "(default: $NEBULAI_EMBED_HOST, else the local ollama server; pass "
        "'auto' to discover the M4 worker's rotating LAN address at run time)",
    )
    t.add_argument(
        "--embed-model",
        default="mxbai-embed-large",
        help="[--source api] embedding model name",
    )
    t.add_argument(
        "--embed-api",
        choices=["ollama", "openai"],
        default="ollama",
        help="[--source api] transport: ollama /api/embed or OpenAI-compatible "
        "/v1/embeddings (bearer key from EMBED_API_KEY or OPENAI_API_KEY)",
    )
    t.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="keep only the N most frequent tokens (default: full curated vocab)",
    )
    t.add_argument(
        "--no-center", action="store_true", help="skip mean-centering W_E"
    )
    t.add_argument(
        "--revision",
        default="main",
        help="HF branch/tag/sha for the weights (default: main; the resolved "
        "commit sha is stamped into the map's meta, so the run stays "
        "reproducible after main moves)",
    )
    t.add_argument(
        "--remote",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="read W_E over HTTP range requests instead of downloading the "
        "checkpoint (default: auto — remote for sharded or >2GB checkpoints, "
        "which is the only way to reach the 24-208GB corpus models)",
    )
    t.add_argument(
        "--unembedding",
        action="store_true",
        help="map W_U (lm_head) instead of W_E — refused on tied models, whose "
        "W_U IS their W_E",
    )
    t.add_argument("--cluster-dim", type=int, default=10)
    t.add_argument("--n-neighbors", type=int, default=30)
    t.add_argument("--min-cluster-size", type=int, default=None)
    t.add_argument("--min-samples", type=int, default=None)
    t.add_argument(
        "--cluster-method",
        choices=["leaf", "eom"],
        default="leaf",
        help="HDBSCAN selection (leaf: fine clusters; eom: coarse, mega-cluster-prone)",
    )
    t.add_argument(
        "--seed",
        type=int,
        default=42,
        help="UMAP seed; -1 = non-deterministic but parallel (faster)",
    )
    t.add_argument(
        "--namer",
        choices=["auto", "openrouter", "hf", "ollama", "openai", "anthropic", "claude-cli", "codex-cli", "none"],
        default="auto",
        help="cluster-naming backend "
        "(auto: ollama -> openai -> openrouter -> centroid)",
    )
    t.add_argument(
        "--openrouter-model",
        default="openai/gpt-oss-120b:free",
        help="OpenRouter slug, e.g. google/gemma-4-31b-it:free, "
        "nvidia/nemotron-3-super-120b-a12b:free, cohere/north-mini-code:free, "
        "poolside/laguna-m.1:free, mistralai/mistral-nemo",
    )
    t.add_argument(
        "--ollama-model",
        default="liquidai/lfm2.5-1.2b-instruct",
        help="preferred ollama model on the worker (falls back to first text model)",
    )
    t.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="ollama base URL (default: local ollama server)",
    )
    t.add_argument("--anthropic-model", default="claude-opus-5")
    t.add_argument(
        "--claude-cli-model",
        default="",
        help="model for --namer claude-cli (e.g. sonnet, opus); empty = the CLI default",
    )
    t.add_argument(
        "--codex-cli-model",
        default="",
        help="model for --namer codex-cli (e.g. gpt-5.6-sol); empty = the CLI default",
    )
    _add_llm_args(t)
    t.add_argument(
        "--env-file",
        default=None,
        help="path to a .env with OPENROUTER_API_KEY (default: ~/.config/nebulai/.env)",
    )
    t.add_argument(
        "--edges",
        choices=["knn", "cluster", "none"],
        default="knn",
        help="similarity edges in the export: knn = cluster edges + per-point "
        "kNN (adds ~4MB), cluster = cluster edges only, none = v2 without edges",
    )
    t.add_argument(
        "--force", action="store_true", help="recompute cached reductions"
    )
    t.set_defaults(fn=_run_tokens)

    s = sub.add_parser("sae", help="Plan A: SAE decoder-direction feature map")
    s.add_argument(
        "--sae-release",
        default="jbloom/GPT2-Small-SAEs-Reformatted",
        help="HF repo holding the SAE weights (sae-lens release gpt2-small-res-jb)",
    )
    s.add_argument(
        "--sae-id",
        default="blocks.8.hook_resid_pre",
        help="SAE id / subfolder in the release (default: mid-late resid stream)",
    )
    s.add_argument(
        "--max-features",
        type=int,
        default=4096,
        help="keep the first N of d_sae features (deterministic MVP subset); "
        "0 = the full dictionary (e.g. all 36864 sparsify latents)",
    )
    s.add_argument(
        "--source",
        choices=["decoder", "label"],
        default="decoder",
        help="geometry source: decoder = the SAE's W_dec rows (model-internal); "
        "label = RESERVED (not implemented — would use label-embedder geometry)",
    )
    s.add_argument(
        "--labels",
        choices=["neuronpedia", "none"],
        default="neuronpedia",
        help="label source: neuronpedia = auto-interp export; none = all "
        "placeholders (offline/dev)",
    )
    s.add_argument(
        "--center",
        action="store_true",
        help="mean-center decoder rows (off by default — directions ARE the "
        "semantics; reduce uses cosine)",
    )
    s.add_argument("--out", default="out", help="output directory root")
    s.add_argument("--cluster-dim", type=int, default=10)
    s.add_argument("--n-neighbors", type=int, default=30)
    s.add_argument("--min-cluster-size", type=int, default=None)
    s.add_argument("--min-samples", type=int, default=None)
    s.add_argument(
        "--cluster-method",
        choices=["leaf", "eom"],
        default="leaf",
        help="HDBSCAN selection (leaf: fine clusters; eom: coarse)",
    )
    s.add_argument(
        "--seed",
        type=int,
        default=42,
        help="UMAP seed; -1 = non-deterministic but parallel (faster)",
    )
    s.add_argument(
        "--namer",
        choices=["auto", "openrouter", "hf", "ollama", "openai", "anthropic", "claude-cli", "codex-cli", "none"],
        default="auto",
        help="cluster-naming backend "
        "(auto: ollama -> openai -> openrouter -> centroid)",
    )
    s.add_argument(
        "--openrouter-model",
        default="openai/gpt-oss-120b:free",
        help="OpenRouter slug for cluster naming",
    )
    s.add_argument(
        "--ollama-model",
        default="liquidai/lfm2.5-1.2b-instruct",
        help="preferred ollama model on the worker (falls back to first text model)",
    )
    s.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="ollama base URL (default: local ollama server)",
    )
    s.add_argument("--anthropic-model", default="claude-opus-5")
    s.add_argument(
        "--claude-cli-model",
        default="",
        help="model for --namer claude-cli (e.g. sonnet, opus); empty = the CLI default",
    )
    s.add_argument(
        "--codex-cli-model",
        default="",
        help="model for --namer codex-cli (e.g. gpt-5.6-sol); empty = the CLI default",
    )
    _add_llm_args(s)
    s.add_argument(
        "--env-file",
        default=None,
        help="path to a .env with OPENROUTER_API_KEY (default: ~/.config/nebulai/.env)",
    )
    s.add_argument(
        "--edges",
        choices=["knn", "cluster", "none"],
        default="knn",
        help="similarity edges in the export (see `tokens --edges`)",
    )
    s.add_argument(
        "--force", action="store_true", help="recompute cached reductions"
    )
    s.set_defaults(fn=_run_sae)

    n = sub.add_parser(
        "neurons",
        help="Plan B: MLP-neuron write-direction map (raw-neuron contrast to sae)",
    )
    n.add_argument(
        "--model",
        default="openai-community/gpt2",
        help="HF repo holding the GPT-2 weights (model.safetensors)",
    )
    n.add_argument(
        "--layer",
        type=int,
        default=8,
        help="MLP layer (default 8 — matched depth to sae's blocks.8.hook_resid_pre)",
    )
    n.add_argument(
        "--max-neurons",
        type=int,
        default=None,
        help="keep the first N of d_mlp neurons (default: all — a full layer is "
        "3072, under sae's 4096 cap)",
    )
    n.add_argument(
        "--source",
        choices=["model", "label"],
        default="model",
        help="geometry source: model = rows of the MLP down-projection (c_proj / "
        "W_out — model-internal); label = RESERVED (not implemented — would use "
        "label-embedder geometry)",
    )
    n.add_argument(
        "--labels",
        choices=["none"],
        default="none",
        help="label source: none = placeholders only (no public raw-neuron "
        "auto-interp export exists for gpt2-small — Neuronpedia's mlp sources "
        "are OpenAI SAEs)",
    )
    n.add_argument(
        "--center",
        action="store_true",
        help="mean-center W_out rows (off by default — directions ARE the "
        "semantics; reduce uses cosine)",
    )
    n.add_argument(
        "--revision",
        default="main",
        help="HF branch/tag/sha for the weights (default: main; the resolved "
        "commit sha is stamped into the map's meta)",
    )
    n.add_argument(
        "--expert",
        default=None,
        help="which write matrix on an MoE layer: an expert index (0..N-1), "
        "'shared' for a shared expert, or 'dense' for the layer's plain "
        "mlp.down_proj. Required on MoE layers — a layer has one write matrix "
        "per expert, and mapping one of 128 under the layer's name would "
        "misreport what was measured",
    )
    n.add_argument(
        "--remote",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="read the layer over HTTP range requests instead of downloading "
        "the checkpoint (default: auto — remote for sharded or >2GB models)",
    )
    n.add_argument("--out", default="out", help="output directory root")
    n.add_argument("--cluster-dim", type=int, default=10)
    n.add_argument("--n-neighbors", type=int, default=30)
    n.add_argument("--min-cluster-size", type=int, default=None)
    n.add_argument("--min-samples", type=int, default=None)
    n.add_argument(
        "--cluster-method",
        choices=["leaf", "eom"],
        default="leaf",
        help="HDBSCAN selection (leaf: fine clusters; eom: coarse)",
    )
    n.add_argument(
        "--seed",
        type=int,
        default=42,
        help="UMAP seed; -1 = non-deterministic but parallel (faster)",
    )
    n.add_argument(
        "--namer",
        choices=["auto", "openrouter", "hf", "ollama", "openai", "anthropic", "claude-cli", "codex-cli", "none"],
        default="auto",
        help="cluster-naming backend "
        "(auto: ollama -> openai -> openrouter -> centroid)",
    )
    n.add_argument(
        "--openrouter-model",
        default="openai/gpt-oss-120b:free",
        help="OpenRouter slug for cluster naming",
    )
    n.add_argument(
        "--ollama-model",
        default="liquidai/lfm2.5-1.2b-instruct",
        help="preferred ollama model on the worker (falls back to first text model)",
    )
    n.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="ollama base URL (default: local ollama server)",
    )
    n.add_argument("--anthropic-model", default="claude-opus-5")
    n.add_argument(
        "--claude-cli-model",
        default="",
        help="model for --namer claude-cli (e.g. sonnet, opus); empty = the CLI default",
    )
    n.add_argument(
        "--codex-cli-model",
        default="",
        help="model for --namer codex-cli (e.g. gpt-5.6-sol); empty = the CLI default",
    )
    _add_llm_args(n)
    n.add_argument(
        "--env-file",
        default=None,
        help="path to a .env with OPENROUTER_API_KEY (default: ~/.config/nebulai/.env)",
    )
    n.add_argument(
        "--edges",
        choices=["knn", "cluster", "none"],
        default="knn",
        help="similarity edges in the export (see `tokens --edges`)",
    )
    n.add_argument(
        "--force", action="store_true", help="recompute cached reductions"
    )
    n.set_defaults(fn=_run_neurons)

    e = sub.add_parser(
        "edges",
        help="backfill similarity edges into existing nebulai.json files "
        "(uses cached reduced.npz — no UMAP recompute)",
    )
    e.add_argument(
        "models",
        nargs="+",
        help="model ids already built with `tokens` (e.g. gpt2 distilgpt2)",
    )
    e.add_argument("--out", default="out", help="output directory root")
    e.add_argument(
        "--mode",
        choices=["knn", "cluster"],
        default="knn",
        help="knn = cluster edges + per-point kNN; cluster = cluster edges only",
    )
    e.set_defaults(fn=_run_edges)

    ip = sub.add_parser(
        "interp",
        help="compute real interp bundles (weight spectra, positional DFT, "
        "forward traces) for the viewer's mechanistic-interpretability features",
    )
    ip.add_argument("--model", default="gpt2", help="HF model id (GPT-2 family)")
    ip.add_argument("--out", default="out", help="output directory root")
    ip.add_argument(
        "--prompts-file",
        default=None,
        help="text file, one prompt per line (default: curated circuit prompts)",
    )
    ip.set_defaults(fn=_run_interp)

    mt = sub.add_parser(
        "metrics",
        help="structural comparison table (silhouette / noise / cluster count) "
        "across several already-built maps",
    )
    mt.add_argument(
        "datasets",
        nargs="+",
        help="dataset dir ids or model ids already built (e.g. "
        "HuggingFaceTB__SmolLM2-135M HuggingFaceTB__SmolLM2-135M__sae__layers.21.mlp)",
    )
    mt.add_argument("--out", default="out", help="output directory root")
    mt.set_defaults(fn=_run_metrics)

    pr = sub.add_parser(
        "probe",
        help="semantic cloud grown from a seed topic — no model weights needed "
        "(LLM proposes concepts, an embedder places them)",
    )
    pr.add_argument("seed", help="seed topic or keyword, e.g. 'grief' or 'photosynthesis'")
    pr.add_argument("--out", default="out", help="output directory root")
    pr.add_argument(
        "--depth", type=int, default=2, help="expansion hops from the seed (default 2)"
    )
    pr.add_argument(
        "--breadth",
        type=int,
        default=12,
        help="concepts requested per term per hop (default 12)",
    )
    pr.add_argument(
        "--sensitivity",
        type=float,
        default=0.35,
        help="cosine floor against the seed: 0 keeps everything, ~0.35 keeps a "
        "recognisable topic, ~0.6 keeps near-synonyms (default 0.35)",
    )
    pr.add_argument(
        "--generator",
        default="auto",
        choices=["auto", "ollama", "openai", "openrouter", "hf", "anthropic"],
        help="which LLM proposes concepts "
        "(auto: ollama -> openai -> openrouter -> anthropic). Pin the exact "
        "model with --namer-model: the generator is half of what a probe cloud "
        "measures, so a silent fall-through to another model is a fabrication",
    )
    pr.add_argument(
        "--namer",
        choices=["auto", "openrouter", "hf", "ollama", "openai", "anthropic", "claude-cli", "codex-cli", "none"],
        default="auto",
        help="cluster-naming backend "
        "(auto: ollama -> openai -> openrouter -> centroid)",
    )
    pr.add_argument("--ollama-host", default="http://localhost:11434")
    pr.add_argument("--ollama-model", default="liquidai/lfm2.5-1.2b-instruct")
    _add_llm_args(pr)
    pr.add_argument("--openrouter-model", default="openai/gpt-oss-120b:free")
    pr.add_argument("--anthropic-model", default="claude-opus-5")
    pr.add_argument(
        "--claude-cli-model",
        default="",
        help="model for --namer claude-cli (e.g. sonnet, opus); empty = the CLI default",
    )
    pr.add_argument(
        "--codex-cli-model",
        default="",
        help="model for --namer codex-cli (e.g. gpt-5.6-sol); empty = the CLI default",
    )
    pr.add_argument("--env-file", default=None)
    pr.add_argument("--embed-host", default=_default_embed_host())
    pr.add_argument("--embed-model", default="mxbai-embed-large")
    pr.add_argument("--embed-api", default="ollama", choices=["ollama", "openai"])
    pr.add_argument(
        "--reuse-terms",
        default=None,
        metavar="DATASET_ID",
        help="re-embed an existing probe map's concepts instead of generating "
        "new ones (holds the point set fixed across a change of embedder; "
        "needs no generator)",
    )
    pr.add_argument("--cluster-dim", type=int, default=10)
    pr.add_argument(
        "--n-neighbors",
        type=int,
        default=None,
        help="UMAP n_neighbors (default: scaled to cloud size)",
    )
    pr.add_argument("--seed-rng", type=int, default=42, help="UMAP seed (default 42)")
    pr.add_argument("--min-cluster-size", type=int, default=None)
    pr.add_argument("--min-samples", type=int, default=None)
    pr.add_argument("--cluster-method", default="leaf", choices=["leaf", "eom"])
    pr.add_argument(
        "--edges", choices=["knn", "cluster", "none"], default="knn"
    )
    pr.set_defaults(fn=_run_probe)

    v = sub.add_parser(
        "validate",
        help="independent validation (trustworthiness / seed stability / null "
        "baseline) — the checks that are NOT the construction procedure",
    )
    v.add_argument(
        "datasets",
        nargs="+",
        help="dataset dir ids already built (e.g. gpt2 gpt2__neurons__h.8.mlp.c_proj)",
    )
    v.add_argument("--out", default="out", help="output directory root")
    v.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 1, 2, 3],
        help="UMAP seeds to score against each other (default: 42 1 2 3)",
    )
    v.add_argument(
        "--trust-neighbors",
        type=int,
        default=15,
        help="k for trustworthiness (default: 15)",
    )
    v.add_argument(
        "--trust-sample",
        type=int,
        default=5000,
        help="max points scored for trustworthiness — it is O(n^2) (default: 5000)",
    )
    v.add_argument(
        "--stability-sample",
        type=int,
        default=4000,
        help="max points for the seed sweep and null baseline; each re-runs "
        "UMAP once per seed (default: 4000)",
    )
    v.add_argument(
        "--skip-stability",
        action="store_true",
        help="trustworthiness (and null) only — skips the per-seed UMAP re-runs",
    )
    v.add_argument(
        "--skip-null",
        action="store_true",
        help="skip the column-shuffled null baseline",
    )
    v.set_defaults(fn=_run_validate)

    r = sub.add_parser(
        "rename",
        help="re-title built maps with a better namer — no rebuild, geometry "
        "untouched",
    )
    r.add_argument(
        "datasets",
        nargs="+",
        help="dataset dir ids already built. Pass `all` for every built map; "
        "maps whose labels are all placeholders are skipped, not invented.",
    )
    r.add_argument("--out", default="out", help="output directory root")
    r.add_argument(
        "--namer",
        choices=["auto", "openrouter", "hf", "ollama", "openai", "anthropic", "claude-cli", "codex-cli", "none"],
        default="claude-cli",
        help="naming backend (default: claude-cli — it runs on an existing "
        "subscription, so re-titling a whole corpus costs no API spend)",
    )
    r.add_argument("--openrouter-model", default="openai/gpt-oss-120b:free")
    r.add_argument("--ollama-model", default="liquidai/lfm2.5-1.2b-instruct")
    r.add_argument("--ollama-host", default="http://localhost:11434")
    r.add_argument("--anthropic-model", default="claude-opus-5")
    r.add_argument(
        "--claude-cli-model",
        default="",
        help="model for --namer claude-cli (e.g. sonnet, opus); empty = the CLI default",
    )
    r.add_argument(
        "--codex-cli-model",
        default="",
        help="model for --namer codex-cli (e.g. gpt-5.6-sol); empty = the CLI default",
    )
    _add_llm_args(r)
    r.add_argument("--env-file", default=None)
    r.set_defaults(fn=_run_rename)

    c = sub.add_parser(
        "compare",
        help="combine several models' clouds into one categorized WebGPU map",
    )
    c.add_argument(
        "models",
        nargs="+",
        help="dataset ids under --out to compare (model ids work for token maps: "
        "gpt2 distilgpt2 EleutherAI/pythia-70m; SAE/neuron/api maps use their "
        "directory name). Pass `all` for every built map.",
    )
    c.add_argument("--out", default="out", help="output directory root")
    c.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="ollama base URL hosting the embed model (default: local ollama server)",
    )
    c.add_argument(
        "--embed-host",
        default=None,
        help="embeddings base URL, overrides $NEBULAI_EMBED_HOST and "
        "--ollama-host (ollama's stock port is 11434, but a host may bind "
        "elsewhere — this project's LAN box uses 11435; an OpenAI-compatible "
        "server needs --embed-api openai). Pass 'auto' to discover the M4 "
        "worker's rotating LAN address at run time.",
    )
    c.add_argument(
        "--embed-api",
        choices=["ollama", "openai"],
        default="ollama",
        help="ollama /api/embed, or any OpenAI-compatible /v1/embeddings",
    )
    c.add_argument("--embed-model", default="mxbai-embed-large")
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(fn=_run_compare)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
