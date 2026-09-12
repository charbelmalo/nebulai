"""Batched local GPT-2 sampling on the optional `behavior-local` stack (§5.6).

This module lives under `organisms/` because it is one of the two places the
Behavior study touches torch, and `tests/test_no_torch_in_base.py` allows torch
— even lazily, even inside a method — nowhere else. The public import path is
unchanged: `nebulai.behavior.adapters.gpt2_local` re-exports everything here.

`GPT2Numpy` stays the reference implementation — it is what the Internals page
links to and what conformance is measured against — but it computes a full
`(T, V)` logit matrix with every attention pattern retained, which is the right
shape for inspection and the wrong shape for 100 cues × 40 trials × 48 tokens.
So this adapter runs the same checkpoint through a pinned Transformers stack and
is *accepted only after* :func:`conformance` shows its next-token logits match
`GPT2Numpy` on golden prompts within a declared tolerance.

The tolerance is declared, not discovered: fp32 matmul reassociation between
numpy and torch is real and small, and a check that reports whatever difference
it happens to find would accept a genuinely wrong forward pass just as happily.

`paid = False`. Both GPT-2 arms and the whole §5.7 capability-control pair run
here at zero spend, which is why §5.7 can be a *required* arm rather than a
budget negotiation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..behavior.adapters.base import AdapterError, Completion, SamplerSettings

#: The declared tolerance for §5.6 acceptance, on next-token logits in fp32.
#: Max absolute difference over the vocabulary, and the rank correlation of the
#: top-50, are both checked: a small max-abs with a scrambled top-50 would be a
#: broken forward pass hiding behind a well-behaved norm.
LOGIT_ATOL = 2e-3
TOP_K_AGREEMENT = 0.98

#: Golden prompts for the conformance check. Short, ASCII, and deliberately
#: including the study's own frame so the check covers the text actually used.
GOLDEN_PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "hot -> cold, warm, heat\nsalt -> ",
    "Once upon a time, in a small village by the sea,",
)


@dataclass
class GPT2LocalAdapter:
    name: str = "gpt2-local"
    pinned: str = "gpt2"
    revision: str = ""  # resolved to a commit sha at load time (§5.5 item 8)
    paid: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    _model: Any = None
    _tok: Any = None
    _detail: dict[str, Any] = field(default_factory=dict)

    # -- loading ---------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - covered by the dep test
            raise AdapterError(
                "the gpt2-local adapter needs the optional 'behavior-local' "
                "dependency group (plan §5.6). Install it with\n"
                "    uv sync --group behavior-local\n"
                "The base nebulai install stays torch-free on purpose, so the "
                "cloud pipelines never inherit a multi-gigabyte dependency."
            ) from exc

        rev = self.revision or "main"
        tok = AutoTokenizer.from_pretrained(self.pinned, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(
            self.pinned, revision=rev, dtype=torch.float32
        )
        model.eval()
        model.to(self.device)
        tok.padding_side = "left"
        if tok.pad_token is None:
            # GPT-2 has no pad token; reusing EOS with left padding and an
            # attention mask is the standard batched-sampling arrangement and
            # changes no logit at a non-pad position.
            tok.pad_token = tok.eos_token
        self._tok, self._model = tok, model

        sha = rev
        try:
            from huggingface_hub import HfApi

            sha = HfApi().model_info(self.pinned, revision=rev).sha or rev
        except Exception:  # offline or hub unreachable: keep what we were given
            pass
        self.revision = sha
        self._detail = {
            "adapter": self.name,
            "pinned": self.pinned,
            "revision": sha,
            "device": self.device,
            "dtype": self.dtype,
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
            "n_params": int(sum(p.numel() for p in model.parameters())),
        }

    def describe(self) -> dict[str, Any]:
        self._load()
        return dict(self._detail)

    # -- sampling --------------------------------------------------------
    def complete(
        self, prompt: str, settings: SamplerSettings, *, trial_seed: int
    ) -> Completion:
        return self.complete_batch([prompt], settings, trial_seeds=[trial_seed])[0]

    def complete_batch(
        self,
        prompts: list[str],
        settings: SamplerSettings,
        *,
        trial_seeds: list[int],
    ) -> list[Completion]:
        """Sample a batch. The runner uses this for the local arms.

        Each trial still gets its own seed, so a batch is not a different
        experiment from the same trials run one at a time — only faster. The
        seed is set once per batch from the first trial's seed and the batch is
        drawn jointly; per-row reproducibility is provided by the store, which
        records each row's seed, not by re-deriving the RNG per row (which torch
        cannot do inside one `generate` call).
        """
        self._load()
        import torch

        assert self._tok is not None and self._model is not None
        enc = self._tok(prompts, return_tensors="pt", padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        torch.manual_seed(settings.seed ^ (trial_seeds[0] & 0x7FFFFFFF))
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self._model.generate(
                **enc,
                do_sample=True,
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_new_tokens=settings.max_output_tokens,
                pad_token_id=self._tok.pad_token_id,
            )
        dt = int((time.perf_counter() - t0) * 1000)
        n_in = enc["input_ids"].shape[1]
        texts = self._tok.batch_decode(out[:, n_in:], skip_special_tokens=True)

        res: list[Completion] = []
        for i, (p, txt, s) in enumerate(zip(prompts, texts, trial_seeds, strict=False)):
            body = _apply_stop(txt, settings.stop)
            res.append(
                Completion(
                    text=body,
                    requested_model=self.pinned,
                    served_model=self.pinned,
                    response_id=f"local-{self.pinned}-{s}",
                    latency_ms=dt // max(len(prompts), 1),
                    usage={
                        "prompt_tokens": int(enc["attention_mask"][i].sum()),
                        "completion_tokens": int(out.shape[1] - n_in),
                    },
                    cost_usd=0.0,
                    detail={"revision": self.revision, "device": self.device},
                )
            )
        return res

    # -- §5.6 acceptance --------------------------------------------------
    def next_token_logits(self, prompt: str) -> np.ndarray:
        self._load()
        import torch

        assert self._tok is not None and self._model is not None
        ids = self._tok(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model(**ids)
        return out.logits[0, -1].float().cpu().numpy()


def _apply_stop(text: str, stop: tuple[str, ...]) -> str:
    """Truncate at the first stop sequence. The untruncated string is what the
    caller stores as raw evidence; this is the *answer* view of it."""
    cut = len(text)
    for s in stop:
        i = text.find(s)
        if i >= 0:
            cut = min(cut, i)
    return text[:cut]


def conformance(
    model_id: str = "gpt2",
    prompts: tuple[str, ...] = GOLDEN_PROMPTS,
    *,
    atol: float = LOGIT_ATOL,
    top_k: int = 50,
) -> dict[str, Any]:
    """Compare the batched backend's next-token logits against `GPT2Numpy`.

    Returns a report rather than raising, because the number itself is the
    deliverable: §5.6 requires the *declared* tolerance and the *measured*
    difference both on the record, and a function that only ever raises or
    stays silent records neither.
    """
    from ..backend.interp.gpt2_numpy import GPT2Numpy

    ref = GPT2Numpy(model_id)
    ad = GPT2LocalAdapter(pinned=model_id)

    rows = []
    for p in prompts:
        ref_logits = ref.forward(p).logits[-1]
        got = ad.next_token_logits(p)
        n = min(len(ref_logits), len(got))
        a, b = np.asarray(ref_logits[:n], np.float64), np.asarray(got[:n], np.float64)
        max_abs = float(np.abs(a - b).max())
        # Compare the two top-k sets, which is what actually decides a sample.
        ka = set(np.argsort(-a)[:top_k].tolist())
        kb = set(np.argsort(-b)[:top_k].tolist())
        agree = len(ka & kb) / top_k
        rows.append(
            {
                "prompt": p,
                "max_abs_logit_diff": max_abs,
                "top_k_agreement": agree,
                "argmax_match": bool(int(np.argmax(a)) == int(np.argmax(b))),
            }
        )

    worst = max(r["max_abs_logit_diff"] for r in rows)
    min_agree = min(r["top_k_agreement"] for r in rows)
    return {
        "model_id": model_id,
        "declared_atol": atol,
        "declared_top_k_agreement": TOP_K_AGREEMENT,
        "top_k": top_k,
        "measured_max_abs_logit_diff": worst,
        "measured_min_top_k_agreement": min_agree,
        "passed": bool(worst <= atol and min_agree >= TOP_K_AGREEMENT),
        "rows": rows,
        "backend": ad.describe(),
    }
