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
FOR THE MERGE: this is a duplicate §3.2 writer.

`backend/directions.py` is another agent's file and does not exist on this
branch; the schema below was matched field-for-field against a `directions.json`
that file produced (`out/gpt2/directions.json`: `meta` + `directions[]`, each
with `source.kind/protocol/...` and the same `contrast` keys). When the two
branches meet, this module's `fit_direction` / `_cohens_d` / `_overlap` /
`_null` should collapse into that file's equivalents and only the prompt-set
loading and the protocol string should survive here. Nothing in the viewer
reads this module, so the collapse is a backend-only edit.
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
    """
    pos = acts[label == 1]
    neg = acts[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise EvalAwarenessError("a condition has no prompts")
    v = pos.mean(axis=0) - neg.mean(axis=0)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise EvalAwarenessError("the two conditions have identical means")
    return (v / n).astype(np.float32)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD standardised difference. NaN when it is undefined."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    sd = float(np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)))
    if sd == 0.0:
        return float("nan")
    return float((a.mean() - b.mean()) / sd)


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


def build_entries(
    model: Any,
    *,
    prompt_set_id: str = DEFAULT_PROMPT_SET,
    layers: Sequence[int] | None = None,
    batch_size: int = 16,
    null_n: int = NULL_N,
    null_seed: int = NULL_SEED,
    progress: Any = None,
) -> list[dict[str, Any]]:
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
) -> dict[str, Any]:
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
) -> dict[str, Any]:
    v = fit_direction(acts, p.label)
    d_in, ov_in = _effect(acts, p.label, v)
    null = _null(acts, p.label, null_n, null_seed)

    # held-out: the LAST k frame pairs, never seen by the fit
    k = max(int(p.heldout_frames), 0)
    held: dict[str, Any] = {}
    if k > 0 and k < p.n_frames:
        test = p.frame >= (p.n_frames - k)
        train = ~test
        v_tr = fit_direction(acts[train], p.label[train])
        d_out, ov_out = _effect(acts[test], p.label[test], v_tr)
        held = {
            "heldout_cohens_d": round(d_out, 6),
            "heldout_overlap": round(ov_out, 6),
            "heldout_n_pos": int((p.label[test] == 1).sum()),
            "heldout_n_neg": int((p.label[test] == 0).sum()),
            "heldout_frames": k,
        }
    else:
        # no split is possible; say so rather than shipping a blank field that
        # reads as a failed generalisation test
        held = {
            "heldout_cohens_d": None,
            "heldout_overlap": None,
            "heldout_n_pos": 0,
            "heldout_n_neg": 0,
            "heldout_frames": 0,
            "heldout_missing": "the prompt set asks for no held-out frames",
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
        f"direction for the framing WORDS, which is what heldout_cohens_d over "
        f"{held.get('heldout_frames', 0)} unseen frame pairs tests. Fitted on a small "
        f"instruct model with no evaluation-awareness training, it measures how this "
        f"model's residual stream separates these {n_pos + n_neg} strings and nothing more. "
        f"This entry is one layer of a sweep; every layer tried is in this file, so the "
        f"largest effect among them is the maximum of that many comparisons and its null "
        f"is the per-layer null printed here, not a sweep-wide one."
    )
    return {
        "id": f"{prompt_set_id.replace('_', '-').replace('.', '-')}-L{L}",
        "label": "test-framed prompts − use-framed prompts",
        "space": f"resid.L{L}",
        "method": "diff_of_means",
        "d": int(model.d),
        "vector": [round(float(x), 6) for x in v],
        "source": {
            "kind": "computed",
            "protocol": protocol,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prompt_set": prompt_set_id,
            "prompt_set_sha": sha[:12],
            "layer": L,
            "token_position": -1,
            "axis": doc.get("axis", ""),
            "contrast": {
                "cohens_d": round(d_in, 6),
                "overlap": round(ov_in, 6),
                "in_sample": True,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "null_n": int(null_n),
                "null_seed": int(null_seed),
                "null_cohens_d_mean": round(float(np.nanmean(np.abs(null))), 6),
                "null_cohens_d_p95": round(float(np.nanpercentile(np.abs(null), 95)), 6),
                **held,
            },
        },
    }


def write_entry(entry: dict[str, Any], path: Path, *, model: str, revision: str) -> Path:
    """Merge one entry into a `directions.json`, replacing by id.

    Replacing rather than appending: two entries with one id would let a reader
    pick the stale vector, and a rerun of the same recipe is a correction, not
    a second measurement.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any]
    if path.exists():
        doc = json.loads(path.read_text())
        if doc.get("meta", {}).get("model") not in (None, model):
            raise EvalAwarenessError(
                f"{path} holds directions for {doc['meta']['model']}, not {model}"
            )
    else:
        doc = {"meta": {}, "directions": []}
    doc["meta"] = {
        "model": model,
        "revision": revision,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    kept = [d for d in doc.get("directions", []) if d.get("id") != entry["id"]]
    kept.append(entry)
    doc["directions"] = kept
    path.write_text(json.dumps(doc, indent=1))
    return path


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point.

    Not a `nebulai` subcommand: `cli.py`'s `direction` group belongs to another
    agent's branch and reflowing it here would collide at the merge. Run with
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
    ap.add_argument("--out", required=True, help="directions.json to write or merge into")
    args = ap.parse_args(argv)

    from .interp.llama_numpy import LlamaNumpy

    model = LlamaNumpy(args.model, revision=args.revision, local_dir=args.local_dir)
    layers = (
        [int(x) for x in args.layers.split(",") if x.strip() != ""]
        if args.layers
        else list(range(model.n_layer))
    )
    entries = build_entries(
        model,
        prompt_set_id=args.prompt_set,
        layers=layers,
        batch_size=args.batch_size,
        progress=lambda i, n: print(f"  {i}/{n}", end="\r", flush=True),
    )
    path = Path(args.out)
    for entry in entries:
        path = write_entry(
            entry,
            path,
            model=model.model_id,
            revision=getattr(model, "revision", "unknown"),
        )
    print(f"\nwrote {path}")
    print(f"{'layer':>6} {'d':>9} {'overlap':>8} {'null p95':>9} {'held d':>8} {'clears':>7}")
    for entry in entries:
        c = entry["source"]["contrast"]
        hd = c["heldout_cohens_d"]
        clears = abs(c["cohens_d"]) > c["null_cohens_d_p95"]
        print(
            f"{c['layer'] if 'layer' in c else entry['source']['layer']:>6} "
            f"{c['cohens_d']:>9.4f} {c['overlap']:>8.4f} {c['null_cohens_d_p95']:>9.4f} "
            f"{'   —   ' if hd is None else format(hd, '>8.4f')} {str(clears):>7}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
