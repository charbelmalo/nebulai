#!/usr/bin/env python3
"""Sweep HDBSCAN settings on a CACHED reduction — never re-run UMAP to try a
different `min_cluster_size`.

UMAP is the minutes-long step and it does not depend on any clustering
parameter, so `reduced.npz` is all you need to explore leaf/eom x
min_cluster_size x min_samples. Referenced by the `nebulai`, `nebulai-tokens`
and `nebulai-neurons` skills; the neurons skill in particular says to re-sweep
before trusting a map, because the back-end defaults were tuned on token
geometry and write-direction space clusters very differently.

    scripts/sweep_hdbscan.py out/gpt2/reduced.npz
    scripts/sweep_hdbscan.py out/gpt2/reduced.npz --method leaf eom \
        --min-cluster-size 15 25 50 --min-samples 5 10

Silhouette here carries the same caveat it does everywhere in this project: it
is computed in `u_cluster`, the space HDBSCAN just clustered, so it ranks
settings against each other but is NOT evidence that the clusters exist in the
model's original space. Use `nebulai validate` for that.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nebulai.backend.cluster import cluster_units  # noqa: E402
from nebulai.backend.metrics import _silhouette  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", type=Path, help="path to a reduced.npz")
    ap.add_argument(
        "--method", nargs="+", default=["leaf", "eom"], choices=["leaf", "eom"]
    )
    ap.add_argument(
        "--min-cluster-size",
        type=int,
        nargs="+",
        default=None,
        help="default: a spread around the n-dependent back-end default",
    )
    ap.add_argument("--min-samples", type=int, nargs="+", default=[5, 10])
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="cluster only the first N points (0 = all); speeds up a coarse first pass",
    )
    args = ap.parse_args()

    if not args.path.exists():
        sys.exit(f"no such file: {args.path}")
    u = np.load(args.path)["u_cluster"]
    if args.sample:
        u = u[: args.sample]
    n = len(u)

    if args.min_cluster_size is None:
        # the back-end default is max(15, n // 1000) — bracket it so the sweep
        # always shows what the shipped map would have done, in context
        d = max(15, n // 1000)
        args.min_cluster_size = sorted({max(2, d // 2), d, d * 2, d * 4})

    print(f"{args.path}  n={n}  dim={u.shape[1]}")
    print(f"  back-end default min_cluster_size for this n: {max(15, n // 1000)}\n")

    headers = ["method", "mcs", "ms", "clusters", "noise", "silhouette", "med.size"]
    rows = []
    for method in args.method:
        for mcs in args.min_cluster_size:
            for ms in args.min_samples:
                ids, _probs = cluster_units(
                    u, min_cluster_size=mcs, min_samples=ms, method=method
                )
                labelled = ids[ids >= 0]
                uniq = sorted(set(labelled.tolist()))
                sizes = np.array([int((ids == c).sum()) for c in uniq], dtype=int)
                sil = _silhouette(u, ids)
                rows.append(
                    [
                        method,
                        str(mcs),
                        str(ms),
                        str(len(uniq)),
                        f"{float((ids < 0).mean()) * 100:.1f}%",
                        "n/a" if sil is None else f"{sil:.4f}",
                        str(int(np.median(sizes))) if len(sizes) else "–",
                    ]
                )

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for r in rows:
        print(fmt(r))

    print(
        "\n  silhouette is measured in the clustering space itself — it ranks these\n"
        "  settings against each other, it does not validate the clusters. Pair a\n"
        "  chosen setting with `nebulai validate` before trusting the map, and\n"
        "  stamp it into meta so the map stays reproducible."
    )


if __name__ == "__main__":
    main()
