#!/usr/bin/env python3
"""Sweep HDBSCAN settings on a cached reduction — no minutes-long UMAP re-run.

The reduction (reduced.npz, produced by the pipeline) is the expensive step;
clustering it is seconds. Use this to pick --cluster-method / --min-cluster-size
/ --min-samples before committing to a full run.

Run inside the project venv so scikit-learn is importable:
    uv run python .claude/skills/nebulai/scripts/sweep_hdbscan.py out/gpt2/reduced.npz

Optionally print sample members per cluster for one setting by pairing with the
matching nebulai.json labels:
    uv run python sweep_hdbscan.py out/gpt2/reduced.npz --labels out/gpt2/nebulai.json
"""
import argparse
import json
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reduced", help="path to reduced.npz")
    ap.add_argument("--labels", help="optional nebulai.json to preview members")
    ap.add_argument(
        "--grid",
        default="leaf:15:5,leaf:15:None,leaf:25:10,eom:15:10,eom:30:15",
        help="comma list of method:min_cluster_size:min_samples (None allowed)",
    )
    args = ap.parse_args()

    from sklearn.cluster import HDBSCAN

    u = np.load(args.reduced)["u_cluster"]
    print(f"u_cluster: {u.shape}\n")

    labels = None
    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            labels = [p["label"] for p in json.load(f)["points"]]

    print(f"{'method':<6} {'mcs':>4} {'ms':>4} | {'clusters':>8} {'noise':>6} | top sizes")
    best = None
    for spec in args.grid.split(","):
        method, mcs, ms = spec.split(":")
        mcs = int(mcs)
        ms = None if ms == "None" else int(ms)
        h = HDBSCAN(
            min_cluster_size=mcs, min_samples=ms, cluster_selection_method=method
        ).fit(u)
        ids = h.labels_
        n = len(set(ids)) - (1 if -1 in ids else 0)
        noise = float((ids == -1).mean())
        top = sorted(np.bincount(ids[ids >= 0]).tolist(), reverse=True)[:3] if n else []
        print(f"{method:<6} {mcs:>4} {str(ms):>4} | {n:>8} {noise:>5.0%} | {top}")
        best = (spec, ids)

    # preview members of the last swept setting, if labels were provided
    if labels is not None and best is not None:
        spec, ids = best
        import collections

        by = collections.defaultdict(list)
        for lab, c in zip(labels, ids):
            if c >= 0:
                by[c].append(lab)
        print(f"\nsample clusters for [{spec}]:")
        for c in sorted(by, key=lambda c: -len(by[c]))[:12]:
            print(f"  n={len(by[c]):<4} {[l for l in by[c][:10]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
