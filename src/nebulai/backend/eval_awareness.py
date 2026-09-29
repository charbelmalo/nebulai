"""The eval-awareness direction — ATTRACTORS-PLAN.md §3.2, matched pairs.

A diff-of-means direction in the pinned model's residual stream between two
conditions that differ in ONE thing: whether the system message says the
assistant is being tested or being used. The user message is byte-identical
across the pair, so whatever the direction is, it is not a topic direction.

What it still could be, and what this module therefore measures:

* the surface features of the framing sentences themselves ("grader",
  "benchmark", "sandbox"). No prompt set can rule this out, so the entry ships
  a **held-out split over frame pairs the fit never saw** — if the direction
  only knew those words it will not transfer to two unseen framings.
* nothing at all. So the entry also ships a **label-permutation null**: the
  same arithmetic over shuffled labels, 32 draws at a fixed seed, and the
  p95 of that null beside the real effect. A Cohen's d inside the null is a
  direction that found nothing, and it is reported rather than suppressed.

`in_sample: true` is on the shipped number on purpose. The vector is fitted on
all 128 prompts because that is the best estimate of the axis; the number that
says whether the axis GENERALISES is `heldout_cohens_d`, and the two are
printed side by side so neither can be read as the other.

────────────────────────────────────────────────────────────────────────────
WHERE THIS WRITES. Nowhere of its own. `backend/directions.py` is the single
registry, and this module is one of its callers: `diff_of_means` builds the
`Direction`, `projection_channels` makes it renderable, `write_directions` and
`write_channels` put it on disk. What survives here is the part that is actually
about eval-awareness — the matched-pair prompt construction, and two controls
that `directions.contrast()` does not have:

* a **label-permutation** null (refit the direction on shuffled labels, 32 draws)
  beside `contrast`'s random-unit-direction null. The random-direction null asks
  "would any axis separate these two sets this well"; the permutation null asks
  "would this *fitting procedure* separate two arbitrary halves of these 128
  prompts this well", which is the question a fitted direction invites;
* a held-out split over **unseen framings** beside `contrast`'s random-half
  split. A random half shares framings with the fit, so it cannot catch a
  direction that only learned the words "grader" and "benchmark".

Both live under `source["eval_awareness"]`, never merged into `source["contrast"]`:
two different held-out numbers under one key is the kind of collision that makes a
file unreadable six months later.

THE POINT CLOUD. These directions live in `resid.L<k>`, so the only points they
may be projected onto (D2) are activations in `resid.L<k>` — not a token map,
whose points are in a `W_E` space. The cloud used is the prompt set's own 128
activations, and `channels.json`'s `point_source` says so verbatim. That is what
makes the entry pass `directions.renderable()`; projecting it onto a token map
would also produce numbers, and they would mean nothing.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..prompts import load_prompt_set
from ..spaces import resid
from .channels import CHANNELS_FILENAME, Channel, write_channels
from .directions import (
    DEFAULT_NULL_N,
    DIRECTIONS_FILENAME,
    Direction,
    DirectionError,
    diff_of_means,
    projection_channels,
    read_directions,
    renderable,
    separation,
    unit,
    write_directions,
)

DEFAULT_PROMPT_SET = "eval_awareness.v1"
NULL_N = 32
NULL_SEED = 0


class EvalAwarenessError(ValueError):
    """The prompt set or the model cannot support the recipe."""


@dataclass(frozen=True)
class Prompts:
    """Rendered prompts plus the two label arrays the recipe needs.

    `frame` is which framing PAIR a prompt came from, and it is the unit the
    held-out split is taken over — splitting on prompts instead would put the
    same framing sentence on both sides of the split and measure memorisation
    of the task, not transfer to a new frame.
    """

    texts: list[str]
    label: np.ndarray  # 1 = eval-framed, 0 = deployment-framed
    frame: np.ndarray  # index into the prompt set's `frames`
    n_frames: int
    heldout_frames: int


def build_prompts(doc: dict[str, Any], model: Any) -> Prompts:
    """Render the matched pairs through the MODEL's own chat template.

    The template is the model's, not ours: applying a different one moves every
    coordinate, and a direction fitted at the wrong position is a direction
    about padding.
    """
    frames = doc["frames"]
    tasks = doc["tasks"]
    tmpl = doc["system_template"]
    if not frames or not tasks:
        raise EvalAwarenessError("prompt set has no frames or no tasks")

    texts: list[str] = []
    label: list[int] = []
    frame: list[int] = []
    for fi, pair in enumerate(frames):
        for cond, lab in (("eval", 1), ("deploy", 0)):
            if cond not in pair:
                raise EvalAwarenessError(f"frame {fi} has no `{cond}` side")
            system = tmpl.format(frame=pair[cond])
            for task in tasks:
                texts.append(
                    model.apply_chat_template(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": task},
                        ],
                        add_generation_prompt=True,
                    )
                )
                label.append(lab)
                frame.append(fi)
    return Prompts(
        texts=texts,
        label=np.asarray(label, dtype=np.int64),
        frame=np.asarray(frame, dtype=np.int64),
        n_frames=len(frames),
        heldout_frames=int(doc.get("heldout_frames", 0)),
    )


def fit_direction(acts: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Unit-length mean(positive) − mean(negative).

    Raises rather than returning a zero vector when a side is empty: a
    direction with no examples on one side is not a small effect, it is not a
    direction.

    The means are taken in float64 because `directions.diff_of_means` takes them
    in float64, and this function's result has to BE the vector that gets
    written. Accumulating 64 rows of a 576-wide residual stream in float32
    instead moved the unit vector by ~1e-5 per component — enough that the
    statistics computed here would have described a slightly different direction
    from the one on disk. `_entry_for` asserts the two agree; this is why it can.
    """
    a = np.asarray(acts, dtype=np.float64)
    pos = a[label == 1]
    neg = a[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise EvalAwarenessError("a condition has no prompts")
    try:
        return unit(pos.mean(axis=0) - neg.mean(axis=0))
    except DirectionError as exc:  # zero norm: the two means coincide
        raise EvalAwarenessError(f"the two conditions have identical means ({exc})") from exc


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD standardised difference, from `directions.separation`.

    One implementation of the statistic, so the number this module prints and
    the number the axis rail prints cannot drift apart. NaN when undefined.
    """
    d = separation(a, b)["cohens_d"]
    return float("nan") if d is None else float(d)


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of projections on the WRONG side of the midpoint of the means.

    A plain misclassification rate under the one threshold the direction
    itself implies — not an AUC, and deliberately: this is the number a reader
    can check by eye against the histogram.
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    mid = (a.mean() + b.mean()) / 2.0
    wrong = int((a <= mid).sum() + (b > mid).sum())
    return float(wrong / (len(a) + len(b)))


def _effect(acts: np.ndarray, label: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    proj = acts @ v
    return _cohens_d(proj[label == 1], proj[label == 0]), _overlap(
        proj[label == 1], proj[label == 0]
    )


def _null(acts: np.ndarray, label: np.ndarray, n: int, seed: int) -> np.ndarray:
    """`n` label-permutation draws of the in-sample Cohen's d.

    The labels are shuffled, not the activations: shuffling the rows would
    break nothing, because the arithmetic is symmetric in them.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.float64)
    perm = label.copy()
    for i in range(n):
        rng.shuffle(perm)
        try:
            v = fit_direction(acts, perm)
        except EvalAwarenessError:
            out[i] = float("nan")
            continue
        out[i] = _effect(acts, perm, v)[0]
    return out


def build_sweep(
    model: Any,
    *,
    prompt_set_id: str = DEFAULT_PROMPT_SET,
    layers: Sequence[int] | None = None,
    batch_size: int = 16,
    null_n: int = NULL_N,
    null_seed: int = NULL_SEED,
    progress: Any = None,
) -> tuple[list[Direction], dict[int, np.ndarray]]:
    """:func:`build_entries`, plus the activations each entry was fitted on.

    The activations come back because `project_entries` needs them: a
    `resid.L<k>` direction can only be projected onto points in `resid.L<k>`, and
    recapturing them would be a second forward pass over the same 128 prompts.
    """
    entries = build_entries(
        model,
        prompt_set_id=prompt_set_id,
        layers=layers,
        batch_size=batch_size,
        null_n=null_n,
        null_seed=null_seed,
        progress=progress,
        _acts_out=(out := {}),
    )
    return entries, out


def build_entries(
    model: Any,
    *,
    prompt_set_id: str = DEFAULT_PROMPT_SET,
    layers: Sequence[int] | None = None,
    batch_size: int = 16,
    null_n: int = NULL_N,
    null_seed: int = NULL_SEED,
    progress: Any = None,
    _acts_out: dict[int, np.ndarray] | None = None,
) -> list[Direction]:
    """The whole recipe at every requested layer, one entry each.

    A SWEEP rather than a layer, and every layer of it is written. Picking the
    layer with the biggest effect after seeing the effects is choosing the
    winner of 30 comparisons and then reporting it as one — so the file gets
    all of them, and a reader who wants the best layer can see how many others
    were tried and what the null was at each.
    """
    doc, sha = load_prompt_set(prompt_set_id)
    p = build_prompts(doc, model)
    want = [int(model.n_layer // 2)] if layers is None else [int(x) for x in layers]

    acts_by_layer = {L: np.empty((len(p.texts), model.d), dtype=np.float32) for L in want}
    step = max(batch_size, 1)
    for start in range(0, len(p.texts), step):
        chunk = p.texts[start : start + step]
        got = model.capture_resid(chunk, want, batch_size=step)
        for L in want:
            acts_by_layer[L][start : start + len(chunk)] = got[L]
        if progress is not None:
            progress(min(start + step, len(p.texts)), len(p.texts))

    if _acts_out is not None:
        _acts_out.update(acts_by_layer)
    return [
        _entry_for(model, p, doc, sha, acts_by_layer[L], L, prompt_set_id, null_n, null_seed)
        for L in want
    ]


def build_entry(
    model: Any,
    *,
    prompt_set_id: str = DEFAULT_PROMPT_SET,
    layer: int | None = None,
    batch_size: int = 16,
    null_n: int = NULL_N,
    null_seed: int = NULL_SEED,
    progress: Any = None,
) -> Direction:
    """One layer of :func:`build_entries`."""
    return build_entries(
        model,
        prompt_set_id=prompt_set_id,
        layers=None if layer is None else [layer],
        batch_size=batch_size,
        null_n=null_n,
        null_seed=null_seed,
        progress=progress,
    )[0]


def _entry_for(
    model: Any,
    p: Prompts,
    doc: dict[str, Any],
    sha: str,
    acts: np.ndarray,
    L: int,
    prompt_set_id: str,
    null_n: int,
    null_seed: int,
) -> Direction:
    """One layer's `Direction`, built by `directions.diff_of_means`.

    The vector, the `contrast` block and the space tag all come from the
    registry's own constructor. What is added here is the `eval_awareness`
    sub-block: the label-permutation null and the unseen-framing split, which
    are the two controls this recipe needs and `contrast` does not provide.
    """
    pos = acts[p.label == 1].astype(np.float64)
    neg = acts[p.label == 0].astype(np.float64)
    v = fit_direction(acts, p.label)
    d_in, ov_in = _effect(acts, p.label, v)
    null = _null(acts, p.label, null_n, null_seed)

    # held out over FRAMINGS: the last k frame pairs, never seen by the fit
    k = max(int(p.heldout_frames), 0)
    if 0 < k < p.n_frames:
        test = p.frame >= (p.n_frames - k)
        v_tr = fit_direction(acts[~test], p.label[~test])
        d_out, ov_out = _effect(acts[test], p.label[test], v_tr)
        frame_heldout: dict[str, Any] = {
            "cohens_d": round(d_out, 6),
            "overlap": round(ov_out, 6),
            "n_pos": int((p.label[test] == 1).sum()),
            "n_neg": int((p.label[test] == 0).sum()),
            "n_frames": k,
        }
    else:
        # say "no split was possible"; a blank field reads as a failed test
        frame_heldout = {
            "cohens_d": None,
            "overlap": None,
            "n_pos": 0,
            "n_neg": 0,
            "n_frames": 0,
            "missing": "the prompt set asks for no held-out frames",
        }

    n_pos = int((p.label == 1).sum())
    n_neg = int((p.label == 0).sum())
    revision = getattr(model, "revision", "unknown")
    protocol = (
        f"diff of means over residual-stream rows: prompt set '{prompt_set_id}' "
        f"(sha {sha[:12]}, n_pos={n_pos}, n_neg={n_neg}), model {model.model_id} @ {revision}, "
        f"space resid.L{L}, token position -1. Matched pairs: the user message is "
        f"byte-identical across the two conditions and only the framing sentence in the "
        f"system message differs, so this is not a topic direction. It may still be a "
        f"direction for the framing WORDS, which is what eval_awareness.frame_heldout over "
        f"{frame_heldout['n_frames']} unseen frame pairs tests — contrast.heldout_* is a "
        f"PAIRED half that shares framings with the fit, so it cannot. Two nulls travel with "
        f"this entry and they ask different questions: contrast.null_cohens_d_* is "
        f"{DEFAULT_NULL_N} random unit directions scored on the same two sets, and "
        f"eval_awareness.label_permutation is {null_n} refits of this whole procedure on "
        f"shuffled labels. Fitted on a small instruct model with no evaluation-awareness "
        f"training, it measures how this model's residual stream separates these "
        f"{n_pos + n_neg} strings and nothing more. This entry is one layer of a sweep; "
        f"every layer tried is in this file, so the largest effect among them is the maximum "
        f"of that many comparisons and its null is the per-layer null printed here, not a "
        f"sweep-wide one."
    )
    d = diff_of_means(
        pos,
        neg,
        str(resid(L)),
        # the set is crossed: pos[i] and neg[i] are the same task under the two
        # framings, so the registry's held-out split must keep the pairing. With
        # independent halves this number came out NEGATIVE at 20 of 30 layers,
        # and the negative was the split, not the direction — see `_heldout`.
        paired=True,
        id=f"{prompt_set_id.replace('_', '-').replace('.', '-')}-L{L}",
        label="test-framed prompts − use-framed prompts",
        protocol=protocol,
        source={
            "prompt_set": prompt_set_id,
            "prompt_set_sha": sha[:12],
            "layer": L,
            "token_position": -1,
            "axis": doc.get("axis", ""),
            "eval_awareness": {
                "cohens_d": round(d_in, 6),
                "overlap": round(ov_in, 6),
                "overlap_definition": (
                    "fraction of projections on the wrong side of the midpoint of the "
                    "two means — a misclassification rate, NOT contrast.overlap, which "
                    "is the overlapping area of two histograms"
                ),
                "in_sample": True,
                "label_permutation": {
                    "n": int(null_n),
                    "seed": int(null_seed),
                    "cohens_d_mean": round(float(np.nanmean(np.abs(null))), 6),
                    "cohens_d_p95": round(float(np.nanpercentile(np.abs(null), 95)), 6),
                },
                "frame_heldout": frame_heldout,
            },
        },
    )
    # diff_of_means normalises mean(pos) - mean(neg); this module's own
    # `fit_direction` must agree with it to the last bit, or the statistics
    # above describe a different vector from the one being written.
    if not np.allclose(d.vector, v, rtol=0.0, atol=1e-7):
        raise EvalAwarenessError(
            f"L{L}: the registry's vector and this module's differ — the "
            f"statistics would not describe the vector on disk"
        )
    return d


def write_entry(entry: Direction, path: Path, *, model: str, revision: str) -> Path:
    """Merge one `Direction` into a `directions.json` through the registry.

    The registry's `write_directions` replaces by id and keeps the rest, which
    is what a rerun of the same recipe should do — a correction, not a second
    measurement. The one thing added here is the refusal below: `write_directions`
    merges only when the file already describes the same model, and *silently
    starts a new list* when it does not. For this writer that would mean one
    command quietly discarding another model's directions, so a mismatch raises.
    """
    return write_entries([entry], path, model=model, revision=revision)


def write_entries(
    entries: Sequence[Direction],
    path: Path,
    *,
    model: str,
    revision: str,
) -> Path:
    """Every layer of a sweep, in one write through the registry."""
    path = Path(path)
    prev = read_directions(path)
    if prev is not None:
        had = (prev.get("meta") or {}).get("model")
        if had not in (None, model):
            raise EvalAwarenessError(f"{path} holds directions for {had}, not {model}")
    return write_directions(path, model=model, revision=revision, directions=list(entries))


def project_entries(
    entries: Sequence[Direction],
    acts_by_layer: dict[int, np.ndarray],
    *,
    n_null: int = 32,
    seed: int = NULL_SEED,
) -> list[Channel]:
    """Make each entry renderable against its OWN layer's activations.

    A `resid.L<k>` direction may only be projected onto points in `resid.L<k>`
    (D2), and the points that exist in that space here are the prompt set's own
    activations — so those are the cloud. `projection_channels` fills in each
    direction's `projection` and `null` blocks in place, which is what
    `directions.renderable()` then looks for.
    """
    chans: list[Channel] = []
    for d in entries:
        L = int(d.source["layer"])
        acts = acts_by_layer.get(L)
        if acts is None:
            raise EvalAwarenessError(f"no activations captured for layer {L}")
        chans.extend(
            projection_channels(d, acts.astype(np.float64), n_null=n_null, seed=seed)
        )
    return chans


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point.

    Not a `nebulai` subcommand: the `direction` group is `cli.py`'s and this
    recipe needs a pinned instruct model and a layer sweep, neither of which
    that group's flags describe. Run with
    `python -m nebulai.backend.eval_awareness --local-dir …`.
    """
    ap = argparse.ArgumentParser(description="fit the eval-awareness direction")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--local-dir", default=None, help="snapshot directory of the pinned model")
    ap.add_argument("--revision", required=True, help="the resolved commit sha, recorded verbatim")
    ap.add_argument(
        "--layers",
        default=None,
        help="comma-separated layers to sweep (-1 is the embedding output); default is every layer",
    )
    ap.add_argument("--prompt-set", default=DEFAULT_PROMPT_SET)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--out",
        required=True,
        help="directory to write directions.json and channels.json into",
    )
    args = ap.parse_args(argv)

    from .interp.llama_numpy import LlamaNumpy

    model = LlamaNumpy(args.model, revision=args.revision, local_dir=args.local_dir)
    layers = (
        [int(x) for x in args.layers.split(",") if x.strip() != ""]
        if args.layers
        else list(range(model.n_layer))
    )
    entries, acts = build_sweep(
        model,
        prompt_set_id=args.prompt_set,
        layers=layers,
        batch_size=args.batch_size,
        progress=lambda i, n: print(f"  {i}/{n}", end="\r", flush=True),
    )
    revision = getattr(model, "revision", "unknown")
    out_dir = Path(args.out)
    # `--out` used to be the directions.json itself; accept that spelling so an
    # old command line fails loudly in one place rather than writing a file named
    # directions.json/directions.json.
    if out_dir.suffix == ".json":
        out_dir = out_dir.parent
    dpath = out_dir / DIRECTIONS_FILENAME
    cpath = out_dir / CHANNELS_FILENAME

    chans = project_entries(entries, acts)
    n_points = len(next(iter(acts.values())))
    write_entries(entries, dpath, model=model.model_id, revision=revision)
    write_channels(
        cpath,
        model=model.model_id,
        revision=revision,
        n_points=n_points,
        channels=chans,
        point_source=f"prompts:{args.prompt_set} (the {n_points} prompt activations, not a map)",
    )
    print(f"\nwrote {dpath}")
    print(f"wrote {cpath}  ({len(chans)} channels over {n_points} prompt activations)")

    ok, drops = renderable(read_directions(dpath), json.loads(cpath.read_text()))
    print(f"renderable: {len(ok)}/{len(entries)}")
    for did, why in drops:
        print(f"  NOT renderable  {did}: {why}")

    print(
        f"\n{'layer':>6} {'d':>9} {'overlap':>8} {'perm p95':>9} {'frame d':>8} "
        f"{'rand p95':>9} {'half d':>8} {'clears':>7}"
    )
    for d in entries:
        ea = d.source["eval_awareness"]
        c = d.source["contrast"]
        fh = ea["frame_heldout"]["cohens_d"]
        hd = c.get("heldout_cohens_d")
        clears = abs(ea["cohens_d"]) > ea["label_permutation"]["cohens_d_p95"]
        print(
            f"{d.source['layer']:>6} {ea['cohens_d']:>9.4f} {ea['overlap']:>8.4f} "
            f"{ea['label_permutation']['cohens_d_p95']:>9.4f} "
            f"{'   —   ' if fh is None else format(fh, '>8.4f')} "
            f"{c['null_cohens_d_p95']:>9.4f} "
            f"{'  miss ' if not isinstance(hd, (int, float)) else format(hd, '>8.4f')} "
            f"{str(clears):>7}"
        )
    print(
        "\n'clears' is the in-sample effect against the LABEL-PERMUTATION p95 — the "
        "stricter of the two nulls. 'rand p95' is the registry's random-unit-direction "
        "null and 'half d' its held-out half — PAIRED, because the set is crossed; with "
        "independent halves it ran negative at 20 of 30 layers and the negative was the "
        "split. Both are in the file too, under contrast.*."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
