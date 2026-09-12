#!/usr/bin/env python3
"""W_E vs W_U: do a model's input and output token geometries agree?

Track 2b asks a question the maps alone cannot answer. A model with untied
embeddings learns two matrices over the same vocabulary: W_E, the rows it reads
a token in with, and W_U, the rows it scores a token out with. The token maps in
`out/` are built from W_E. If W_U arranged the vocabulary the same way, the
second map would be redundant and every W_E finding would carry over. If it did
not, then "the model's token geometry" was never one thing, and any claim made
from a W_E map needs the qualifier.

This script measures the agreement three ways over the SAME curated token set:

1. **Neighbourhood overlap.** For each token, the k nearest tokens by cosine in
   W_E and in W_U; the score is |intersection| / k. This is the raw-space
   measure — it does not go through UMAP, so it cannot be an artifact of the
   reduction. The chance baseline is k/(n-1) and is printed beside it, because
   an overlap of 0.05 sounds small until you see that chance is 0.0002.

2. **Cluster agreement.** The adjusted Rand index between the two maps' HDBSCAN
   assignments, plus the mean Jaccard of the two cluster TITLES a token sits
   under. ARI answers "is the partition the same"; title Jaccard answers "would
   a reader of the two maps read the same words", which is the thing a viewer
   actually shows and the thing that can agree while the partition does not.

3. **Which families diverge.** Every token is bucketed by surface form (code
   punctuation, CJK, numerals, byte fragments, whitespace, Latin word pieces)
   and the overlap is reported per bucket. A single mean hides the case where
   the two spaces agree completely on words and not at all on byte fragments.

The degenerate control matters as much as the measurement. For a TIED model
(gemma-4-26b-a4b-it) W_U *is* W_E, so `--control` computes exactly the same
statistic on W_E against itself. It must return 1.0. A pipeline that cannot
score 1.0 on a copy of the data is not measuring similarity, and every untied
number below it would be uninterpretable.

Source vectors are not stored in a map directory (`reduced.npz` holds only the
UMAP output), so they are re-read over HTTP ranges and cached as .npy under
--cache-dir; a second run costs nothing.

Usage:
    python scripts/we_wu_overlap.py --model mistralai/Mistral-Nemo-Instruct-2407 \\
        --max-tokens 5000 --k 50
    python scripts/we_wu_overlap.py --model google/gemma-4-26B-A4B-it \\
        --max-tokens 5000 --control
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nebulai.frontends.tokens import load_token_units  # noqa: E402

# --------------------------------------------------------------------------
# token families
# --------------------------------------------------------------------------

# The byte-fallback forms every BPE tokenizer in the corpus uses for bytes it
# cannot decode. Ġ/Ċ are GPT-2-family space/newline markers; ▁ is SentencePiece.
_BYTE_FALLBACK = re.compile(r"^(<0x[0-9A-Fa-f]{2}>|\\x[0-9a-f]{2})$")
_CODE_CHARS = set("{}[]()<>;=+-*/\\|&^%$#@~`_:!?.,'\"")


def token_family(label: str) -> str:
    """Bucket a token by surface form.

    The buckets are deliberately coarse and mutually exclusive, checked in an
    order that puts the unambiguous tests first. A token that is 50% CJK and 50%
    Latin is rare enough that a finer scheme would buy noise, not resolution.
    """
    s = label.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
    core = s.strip()
    if _BYTE_FALLBACK.match(label.strip()):
        return "byte_fragment"
    if not core:
        return "whitespace"
    if any(_is_cjk(ch) for ch in core):
        return "cjk"
    if all(ch.isdigit() for ch in core):
        return "numeral"
    if any(ch.isdigit() for ch in core) and any(ch.isalpha() for ch in core):
        return "alphanumeric"
    if all(ch in _CODE_CHARS for ch in core):
        return "punctuation"
    if any(ch.isalpha() for ch in core):
        if _is_non_latin_script(core):
            return "non_latin_script"
        return "latin_word"
    return "other"


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF  # CJK unified
        or 0x3400 <= o <= 0x4DBF  # extension A
        or 0x3040 <= o <= 0x30FF  # kana
        or 0xAC00 <= o <= 0xD7AF  # hangul
        or 0xF900 <= o <= 0xFAFF  # compatibility
    )


def _is_non_latin_script(core: str) -> bool:
    for ch in core:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if not name.startswith("LATIN"):
            return True
    return False


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def _unit_rows(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=np.float32)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return V / n


def knn_indices(V: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
    """Row-wise top-k by cosine, excluding self, in chunks.

    The full n x n similarity matrix is never materialised: at n=50000 it would
    be 10 GB, and the whole point of the byte-fragment breakdown is that it has
    to run at the vocabulary sizes the big maps use.
    """
    U = _unit_rows(V)
    n = U.shape[0]
    out = np.empty((n, k), dtype=np.int32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sims = U[lo:hi] @ U.T
        # Drop self by force rather than by hoping it is the argmax: with
        # duplicate rows (which real vocabularies contain) it may not be.
        sims[np.arange(hi - lo), np.arange(lo, hi)] = -np.inf
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1), axis=1
        )
        out[lo:hi] = ordered
    return out


def overlap_per_token(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    k = a.shape[1]
    return np.array(
        [len(set(a[i].tolist()) & set(b[i].tolist())) / k for i in range(a.shape[0])],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------
# cluster agreement
# --------------------------------------------------------------------------


def adjusted_rand(a: list[int], b: list[int]) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def _title_words(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", title.lower()) if w not in _STOP}


_STOP = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "tokens",
    "token",
    "cluster",
    "various",
    "related",
    "terms",
    "words",
    "word",
}


def title_jaccard(t_a: str, t_b: str) -> float:
    wa, wb = _title_words(t_a), _title_words(t_b)
    if not wa and not wb:
        return float("nan")
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def read_map(path: Path) -> dict:
    """Read the two things a shipped map can tell us that the raw matrix cannot:
    which cluster each token landed in, and what that cluster was called.

    Two schema details are load-bearing and were both got wrong once:

    - the per-point cluster key is `cluster_id`, not `cluster`. Defaulting a
      missing key to -1 turns every point into noise, which silently produces a
      *perfect* ARI of 1.0 between two all-noise partitions. A degenerate 1.0 is
      worse than a crash, so this reads the key with no default and refuses.
    - `p["id"]` is the point's index in the export, not the token's id in the
      vocabulary. The token id lives in `unit_ref.index`. Matching two maps on
      the export index would happen to work here — curation is deterministic, so
      both maps enumerate the same tokens in the same order — but it would break
      the moment a map was built with a different cap.
    """
    doc = json.loads(path.read_text())
    titles = {int(c["id"]): str(c.get("title") or "") for c in doc.get("clusters", [])}
    points = doc["points"]
    if points and "cluster_id" not in points[0]:
        raise KeyError(
            f"{path}: points carry no 'cluster_id'. Refusing to treat that as "
            "all-noise: two all-noise partitions score a perfect ARI of 1.0, "
            "which would read as total agreement."
        )
    return {
        "meta": doc["meta"],
        "ids": [int(p["id"]) for p in points],
        "token_ids": [int((p.get("unit_ref") or {}).get("index", p["id"])) for p in points],
        "labels": [str(p["label"]) for p in points],
        "cluster": [int(p["cluster_id"]) for p in points],
        "titles": titles,
    }


# --------------------------------------------------------------------------
# loading (with a cache, because these are HTTP range reads)
# --------------------------------------------------------------------------


def load_vectors(
    model: str, which: str, max_tokens: int, revision: str, cache_dir: Path
) -> tuple[np.ndarray, list[str], list[int], dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = model.replace("/", "__") + f"__{which}__{max_tokens}"
    vec_path = cache_dir / f"{slug}.npy"
    side_path = cache_dir / f"{slug}.json"
    if vec_path.exists() and side_path.exists():
        side = json.loads(side_path.read_text())
        return np.load(vec_path), side["labels"], side["ids"], side["meta"]
    t0 = time.time()
    units = load_token_units(
        model,
        center=True,
        max_tokens=max_tokens,
        revision=revision,
        remote=True,
        which=which,
    )
    print(
        f"  loaded {which:6s} {units.vectors.shape} in {time.time() - t0:.1f}s"
        f"  ({units.meta.get('bytes_fetched', 0) / 1e6:.0f} MB)",
        flush=True,
    )
    np.save(vec_path, units.vectors)
    meta = {k: v for k, v in units.meta.items() if isinstance(v, (str, int, float, bool, type(None)))}
    side_path.write_text(
        json.dumps({"labels": units.labels, "ids": units.ids, "meta": meta})
    )
    return units.vectors, units.labels, units.ids, meta


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=5000)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument(
        "--control",
        action="store_true",
        help="Tied model: score W_E against itself. Must print 1.000.",
    )
    ap.add_argument("--we-map", default=None, help="path to the W_E nebulai.json")
    ap.add_argument("--wu-map", default=None, help="path to the W_U nebulai.json")
    ap.add_argument(
        "--cache-dir",
        default=str(Path(__file__).resolve().parent.parent / ".we_wu_cache"),
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    print(f"{args.model}  (k={args.k}, n={args.max_tokens})", flush=True)

    E, labels, ids, meta_e = load_vectors(
        args.model, "input", args.max_tokens, args.revision, cache
    )

    # The two weight matrices are never needed at the same time: each is read
    # once, by knn_indices, and everything after that works off the (n,k) int32
    # neighbour tables and the token strings. Holding both anyway costs four
    # copies of an (n,d) float32 -- the two matrices plus the unit-normalised
    # copy knn_indices makes of each -- which at Muse-Glimmer-30B's 6656
    # dimensions and n=50000 is 5.3 GB and got this script SIGKILLed twice on a
    # 16 GB machine, after it had already spent 11 minutes fetching W_E. So the
    # untied path computes W_E's table, frees W_E, and only then fetches W_U:
    # same numbers, half the peak, and the expensive download is not repeated
    # because load_vectors caches it.
    t0 = time.time()
    if args.control:
        # The tied control keeps the old shape on purpose. It copies W_E and
        # runs knn_indices a second, independent time, so "1.0000" is evidence
        # that the pipeline returns identity on identical input rather than
        # evidence that a table compared to itself equals itself. That is the
        # whole job of a degenerate control, and it is worth the second copy --
        # a tied model's matrix is the only one that has to be held twice.
        U, labels_u, meta_u = E.copy(), labels, dict(meta_e)
        print("  CONTROL: W_U is W_E (tied). Any score below 1.000 is a bug.")
        nn_e = knn_indices(E, args.k)
        nn_u = knn_indices(U, args.k)
        del U
    else:
        nn_e = knn_indices(E, args.k)
        del E
        U, labels_u, _ids_u, meta_u = load_vectors(
            args.model, "output", args.max_tokens, args.revision, cache
        )
        if labels_u != labels:
            print("REFUSED: the two loads curated different token sets.", file=sys.stderr)
            return 2
        nn_u = knn_indices(U, args.k)
        del U
    ov = overlap_per_token(nn_e, nn_u)
    n = len(labels)
    chance = args.k / (n - 1)
    print(f"  kNN in {time.time() - t0:.1f}s")
    print()
    print(f"  mean neighbourhood overlap   {ov.mean():.4f}")
    print(f"  median                       {np.median(ov):.4f}")
    print(f"  chance baseline (k/(n-1))    {chance:.6f}")
    print(f"  ratio to chance              {ov.mean() / chance:.1f}x")
    print(f"  tokens with zero overlap     {int((ov == 0).sum())} / {n}")
    print(f"  tokens with full overlap     {int((ov == 1).sum())} / {n}")

    rows = []
    fams: dict[str, list[float]] = {}
    for i, lab in enumerate(labels):
        fams.setdefault(token_family(lab), []).append(float(ov[i]))
    print()
    print("  by token family (most divergent last)")
    fam_summary = {}
    for fam, vals in sorted(fams.items(), key=lambda kv: -float(np.mean(kv[1]))):
        fam_summary[fam] = {"n": len(vals), "mean_overlap": float(np.mean(vals))}
        print(f"    {fam:18s} n={len(vals):6d}  overlap {np.mean(vals):.4f}")

    result = {
        "model": args.model,
        "control_tied": bool(args.control),
        "k": args.k,
        "n_tokens": n,
        "mean_overlap": float(ov.mean()),
        "median_overlap": float(np.median(ov)),
        "chance_baseline": chance,
        "ratio_to_chance": float(ov.mean() / chance),
        "zero_overlap_tokens": int((ov == 0).sum()),
        "full_overlap_tokens": int((ov == 1).sum()),
        "by_family": fam_summary,
        "we_meta": {k: meta_e.get(k) for k in ("weight_key", "revision", "vocab_size", "kept")},
        "wu_meta": {k: meta_u.get(k) for k in ("weight_key", "revision", "vocab_size", "kept")},
    }

    # cluster / title agreement, when both maps exist
    if args.we_map and args.wu_map:
        me, mu = read_map(Path(args.we_map)), read_map(Path(args.wu_map))
        # compare the token STRINGS, not the export row numbers. Export ids are
        # just 0..n-1 in both files and would always "match", which is not a
        # check — it is the appearance of one.
        if me["labels"] != mu["labels"]:
            print("\n  maps curate different token sets; skipping cluster agreement")
        else:
            ari = adjusted_rand(me["cluster"], mu["cluster"])
            named = [
                title_jaccard(me["titles"].get(a, ""), mu["titles"].get(b, ""))
                for a, b in zip(me["cluster"], mu["cluster"])
                if a >= 0 and b >= 0
            ]
            named = [v for v in named if not np.isnan(v)]
            jac = float(np.mean(named)) if named else float("nan")
            both_noise = sum(
                1 for a, b in zip(me["cluster"], mu["cluster"]) if a < 0 and b < 0
            )
            print()
            print(f"  cluster ARI (W_E vs W_U)     {ari:.4f}")
            if named:
                print(
                    f"  mean title Jaccard           {jac:.4f}"
                    f"  (over {len(named)} tokens clustered in both)"
                )
            else:
                print(
                    "  mean title Jaccard           not computed"
                    " — no token is clustered in BOTH maps"
                )
            print(f"  noise in both maps           {both_noise} / {n}")
            print(
                f"  noise fraction               W_E {me['meta'].get('noise_fraction')}"
                f"   W_U {mu['meta'].get('noise_fraction')}"
            )
            result["cluster_ari"] = ari
            result["mean_title_jaccard"] = jac if named else None
            result["tokens_clustered_in_both"] = len(named)
            result["noise_in_both"] = both_noise
            result["noise_fraction_we"] = me["meta"].get("noise_fraction")
            result["noise_fraction_wu"] = mu["meta"].get("noise_fraction")
            result["n_clusters_we"] = me["meta"].get("n_clusters")
            result["n_clusters_wu"] = mu["meta"].get("n_clusters")

    print()
    print("  This measures whether the input and output token geometries agree.")
    print("  It is not a quality score and it ranks no model.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
        print(f"  wrote {args.json_out}")
    del rows
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
