import argparse
import json
import time
from pathlib import Path

import numpy as np


def _timer():
    t0 = time.time()
    return lambda: f"{time.time() - t0:.1f}s"


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

    from .backend.cluster import cluster_units
    from .backend.export import export_json
    from .backend.name import name_clusters
    from .backend.reduce import reduce_vectors
    from .backend.viz import render

    dataset_id = args.model.replace("/", "__")
    if args.source == "api":
        dataset_id += f"__api-{args.embed_model.replace('/', '__')}"
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
            args.model, center=not args.no_center, max_tokens=args.max_tokens
        )
        print(
            f"[1/5] loaded {len(units)} token units from {args.model} "
            f"(vocab {units.meta['vocab_size']}, curated to {units.meta['kept']}) [{t()}]"
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
        env_file=args.env_file,
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
    from .backend.cluster import cluster_units
    from .backend.export import export_json
    from .backend.name import name_clusters
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
        env_file=args.env_file,
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
            f"SAE decoder directions ({args.sae_id}) -> UMAP -> HDBSCAN"
        ),
    )
    _update_index(Path(args.out))
    print(f"[5/5] exported [{t()}]")
    for p in (json_path, png, html):
        print(f"  {p}")


def _run_neurons(args: argparse.Namespace) -> None:
    """Plan B: MLP-neuron write-direction map. Mirrors _run_sae's 5-stage
    structure and its exact `[k/5] ...` prints (build_server parses them)."""
    from .backend.cluster import cluster_units
    from .backend.export import export_json
    from .backend.name import name_clusters
    from .backend.reduce import reduce_vectors
    from .backend.viz import render
    from .frontends.neurons import (
        load_neuron_units,
        neuron_dataset_id,
        neuron_tensor_path,
        placeholder_titles,
    )

    if args.source == "label":
        raise SystemExit(
            "label-space projection is not implemented — it would lay the map "
            "out by the label-embedder's semantics, not the model's geometry; "
            "the MVP ships model space (w_out rows) only"
        )

    t = _timer()
    units = load_neuron_units(
        model_repo=args.model,
        layer=args.layer,
        max_neurons=args.max_neurons,
        center=args.center,
        labels_source=args.labels,
        out_root=Path(args.out),
    )
    tensor_path = neuron_tensor_path(args.layer)
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
            env_file=args.env_file,
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


def _run_compare(args: argparse.Namespace) -> None:
    from .backend.compare import build_comparison, export_comparison
    from .backend.viewer import write_viewer

    out_root = Path(args.out)
    json_paths = []
    for m in args.models:
        jp = out_root / m.replace("/", "__") / "nebulai.json"
        if not jp.exists():
            raise SystemExit(
                f"missing {jp} — run `nebulai tokens --model {m}` first"
            )
        json_paths.append(jp)

    t = _timer()
    comp = build_comparison(
        json_paths,
        embed_host=args.ollama_host,
        embed_model=args.embed_model,
        seed=args.seed,
    )
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
        default="http://192.168.0.200:11434",
        help="[--source api] embeddings endpoint base URL (default: M4 worker)",
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
        choices=["auto", "openrouter", "ollama", "anthropic", "none"],
        default="auto",
        help="cluster-naming backend (auto: ollama -> openrouter -> centroid)",
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
        default="http://192.168.0.200:11434",
        help="ollama base URL (default: M4 worker)",
    )
    t.add_argument("--anthropic-model", default="claude-opus-4-8")
    t.add_argument(
        "--env-file",
        default=None,
        help="path to a .env with OPENROUTER_API_KEY (default: ~/.hermes/.env)",
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
        help="keep the first N of d_sae features (deterministic MVP subset)",
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
        choices=["auto", "openrouter", "ollama", "anthropic", "none"],
        default="auto",
        help="cluster-naming backend (auto: ollama -> openrouter -> centroid)",
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
        default="http://192.168.0.200:11434",
        help="ollama base URL (default: M4 worker)",
    )
    s.add_argument("--anthropic-model", default="claude-opus-4-8")
    s.add_argument(
        "--env-file",
        default=None,
        help="path to a .env with OPENROUTER_API_KEY (default: ~/.hermes/.env)",
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
        choices=["auto", "openrouter", "ollama", "anthropic", "none"],
        default="auto",
        help="cluster-naming backend (auto: ollama -> openrouter -> centroid)",
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
        default="http://192.168.0.200:11434",
        help="ollama base URL (default: M4 worker)",
    )
    n.add_argument("--anthropic-model", default="claude-opus-4-8")
    n.add_argument(
        "--env-file",
        default=None,
        help="path to a .env with OPENROUTER_API_KEY (default: ~/.hermes/.env)",
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

    c = sub.add_parser(
        "compare",
        help="combine several models' clouds into one categorized WebGPU map",
    )
    c.add_argument(
        "models",
        nargs="+",
        help="model ids already built with `tokens` (e.g. gpt2 distilgpt2 EleutherAI/pythia-70m)",
    )
    c.add_argument("--out", default="out", help="output directory root")
    c.add_argument(
        "--ollama-host",
        default="http://192.168.0.200:11434",
        help="ollama base URL hosting the embed model (default: M4 worker)",
    )
    c.add_argument("--embed-model", default="mxbai-embed-large")
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(fn=_run_compare)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
