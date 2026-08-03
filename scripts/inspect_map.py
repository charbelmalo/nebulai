#!/usr/bin/env python3
"""Summarize any exported map: meta line, largest clusters with sample members,
size distribution.

Referenced by the `nebulai` skill as the thing to reach for INSTEAD of writing
another inline JSON-poking snippet. It reads only `nebulai.json`, so it works
on any exported artifact without the model, the reduction cache, or a venv with
umap/sklearn installed.

    scripts/inspect_map.py out/gpt2/nebulai.json
    scripts/inspect_map.py out/gpt2/nebulai.json --top 30 --members 8
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _meta_line(meta: dict) -> str:
    """One line that says what this map actually is — model, unit, how it was
    reduced/clustered/named. Provenance first, because every other number here
    is only interpretable against it."""
    bits = [
        f"model={meta.get('model')}",
        f"unit={meta.get('unit')}",
        f"points={meta.get('n_points')}",
        f"clusters={meta.get('n_clusters')}",
    ]
    if (nf := meta.get("noise_fraction")) is not None:
        bits.append(f"noise={nf:.1%}")
    if (h := meta.get("hdbscan")) is not None:
        bits.append(
            f"hdbscan={h.get('method')}/mcs={h.get('min_cluster_size')}"
            f"/ms={h.get('min_samples')}"
        )
    else:
        bits.append("hdbscan=<not stamped>")
    bits.append(f"namer={meta.get('namer')}")
    if meta.get("labels_source") is not None:
        bits.append(f"labels={meta.get('labels_source')}({meta.get('n_labeled')})")
    return "  ".join(str(b) for b in bits)


def _histogram(sizes: list[int], width: int = 40) -> list[str]:
    """Log-ish size buckets — cluster sizes are heavy-tailed, so linear bins put
    everything in the first bucket and tell you nothing."""
    edges = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 10**9]
    counts = [0] * (len(edges) - 1)
    for s in sizes:
        for i in range(len(edges) - 1):
            if edges[i] <= s < edges[i + 1]:
                counts[i] += 1
                break
    peak = max(counts) if counts else 0
    lines = []
    for i, c in enumerate(counts):
        if c == 0:
            continue
        lo, hi = edges[i], edges[i + 1]
        label = f"{lo}+" if hi >= 10**9 else f"{lo}-{hi - 1}"
        bar = "#" * (round(width * c / peak) if peak else 0)
        lines.append(f"    {label:>9}  {c:>5}  {bar}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", type=Path, help="path to a nebulai.json")
    ap.add_argument("--top", type=int, default=20, help="largest N clusters (default 20)")
    ap.add_argument(
        "--members", type=int, default=6, help="sample members per cluster (default 6)"
    )
    args = ap.parse_args()

    if not args.path.exists():
        sys.exit(f"no such file: {args.path}")
    doc = json.loads(args.path.read_text())
    meta, points, clusters = doc["meta"], doc["points"], doc.get("clusters", [])

    print(_meta_line(meta))

    # recompute sizes from points rather than trusting cluster.size — if the two
    # ever disagree, the points are the map that actually rendered
    by_cluster: dict[int, list[str]] = {}
    for p in points:
        by_cluster.setdefault(int(p["cluster_id"]), []).append(str(p.get("label", "")))
    n_noise = len(by_cluster.get(-1, []))
    titles = {int(c["id"]): str(c.get("title", "")) for c in clusters}

    real = {cid: m for cid, m in by_cluster.items() if cid >= 0}
    sizes = sorted((len(m) for m in real.values()), reverse=True)
    print(
        f"\n  {len(points)} points   {len(real)} clusters   "
        f"{n_noise} noise ({n_noise / max(len(points), 1):.1%})"
    )

    print(f"\n  largest {min(args.top, len(real))} clusters")
    ranked = sorted(real.items(), key=lambda kv: len(kv[1]), reverse=True)
    for cid, members in ranked[: args.top]:
        sample = ", ".join(repr(m) for m in members[: args.members])
        print(f"    [{cid:>4}] {len(members):>5}  {titles.get(cid, '<untitled>')}")
        print(f"           {sample}")

    # a title reused across clusters is the labeling bottleneck showing up as a
    # number instead of a vibe — worth surfacing every time
    dupes = [(t, n) for t, n in Counter(titles.values()).most_common() if n > 1 and t]
    if dupes:
        print(f"\n  repeated titles ({len(dupes)} distinct, top 10)")
        for t, n in dupes[:10]:
            print(f"    {n:>4}x  {t}")

    print("\n  cluster size distribution")
    for line in _histogram(sizes):
        print(line)


if __name__ == "__main__":
    main()
