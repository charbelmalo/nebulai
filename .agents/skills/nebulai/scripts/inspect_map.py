#!/usr/bin/env python3
"""Summarize a nebulai.json map: meta, top clusters with sample members, noise.

Reuse this instead of writing an inline JSON-poking snippet — every session so
far has independently re-written one. Pure stdlib, no deps.

Usage:
    python inspect_map.py out/gpt2/nebulai.json            # top 20 clusters
    python inspect_map.py out/gpt2/nebulai.json --top 40   # top 40
    python inspect_map.py out/gpt2/nebulai.json --members 12 --tail
"""
import argparse
import collections
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="path to nebulai.json")
    ap.add_argument("--top", type=int, default=20, help="how many clusters to show")
    ap.add_argument("--members", type=int, default=10, help="sample members per cluster")
    ap.add_argument("--tail", action="store_true", help="also show the smallest clusters")
    args = ap.parse_args()

    with open(args.path, encoding="utf-8") as f:
        d = json.load(f)

    m = d["meta"]
    print(
        f"model={m.get('model')} unit={m.get('unit')} "
        f"points={m.get('n_points')} clusters={m.get('n_clusters')} "
        f"noise={m.get('noise_fraction')} namer={m.get('namer')} "
        f"centered={m.get('centered')}"
    )

    members = collections.defaultdict(list)
    for p in d["points"]:
        members[p["cluster_id"]].append(p["label"])

    clusters = sorted(d["clusters"], key=lambda c: -c["size"])

    def show(cl):
        for c in cl:
            sample = members.get(c["id"], [])[: args.members]
            print(f"  [{c['id']:>4}] n={c['size']:<5} {c['title']!r}")
            print(f"         {', '.join(repr(s) for s in sample)}")

    print(f"\nTop {min(args.top, len(clusters))} clusters by size:")
    show(clusters[: args.top])

    if args.tail and len(clusters) > args.top:
        print("\nSmallest clusters:")
        show(clusters[-5:])

    sizes = sorted((c["size"] for c in clusters), reverse=True)
    if sizes:
        import statistics

        print(
            f"\nsize: max={sizes[0]} median={int(statistics.median(sizes))} "
            f"min={sizes[-1]} | clusters={len(sizes)} "
            f"| in-cluster points={sum(sizes)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
