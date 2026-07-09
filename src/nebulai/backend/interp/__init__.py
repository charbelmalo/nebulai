"""Interp back-end — real model internals for the Phase-2 viewer's 25 features.

Everything here is computed from the model's actual weights (safetensors, loaded
as numpy) with NO torch / transformer_lens dependency. A pure-numpy forward pass
(`gpt2_numpy`) yields honest activations, attention, residual-stream trajectories
and logits; weight-only analyses (SVD spectra, positional DFT, write directions)
need no forward pass at all. The point of this package is the project's honesty
guardrail made literal: every viewer feature is backed by a real computed
quantity, and the exact provenance is stamped into each exported bundle.

Feature → data-source classification lives in docs/INTERP_FEATURES.md.
"""

from .gpt2_numpy import GPT2Numpy, Trace

__all__ = ["GPT2Numpy", "Trace"]
