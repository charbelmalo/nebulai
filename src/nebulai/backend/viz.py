"""Phase-1 visualization: datamapplot static PNG + interactive HTML.

The interactive map does zoom-dependent labeling, per-point hover, and search;
the static PNG only titles the largest clusters so it stays readable.
"""

from pathlib import Path

import numpy as np


def _names_per_point(
    cluster_ids: np.ndarray,
    titles: dict[int, str],
    max_labels: int | None = None,
) -> np.ndarray:
    keep = set(titles)
    if max_labels is not None and len(titles) > max_labels:
        sizes = {cid: int((cluster_ids == cid).sum()) for cid in titles}
        keep = set(sorted(sizes, key=sizes.get, reverse=True)[:max_labels])
    return np.array(
        [
            titles[int(c)] if int(c) in keep and int(c) >= 0 else "Unlabelled"
            for c in cluster_ids
        ],
        dtype=object,
    )


def render(
    u2: np.ndarray,
    cluster_ids: np.ndarray,
    titles: dict[int, str],
    hover_labels: list[str],
    out_png: Path,
    out_html: Path,
    title: str,
    sub_title: str,
    max_static_labels: int = 60,
) -> None:
    import datamapplot

    static_names = _names_per_point(cluster_ids, titles, max_static_labels)
    fig, _ax = datamapplot.create_plot(
        u2,
        static_names,
        noise_label="Unlabelled",
        title=title,
        sub_title=sub_title,
        darkmode=True,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    hover = np.array([repr(s) for s in hover_labels], dtype=object)
    interactive_names = _names_per_point(cluster_ids, titles)
    plot = datamapplot.create_interactive_plot(
        u2,
        interactive_names,
        noise_label="Unlabelled",
        hover_text=hover,
        title=title,
        sub_title=sub_title,
        darkmode=True,
        enable_search=True,
    )
    plot.save(out_html)
