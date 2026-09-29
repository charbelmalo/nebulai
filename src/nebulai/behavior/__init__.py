"""Behavioral semantic divergence — repeated, controlled output sampling.

`docs/BEHAVIORAL-DIVERGENCE-PLAN.md` is the specification. This package is
deliberately OUTSIDE the `Units` → reduce → cluster → name → export pipeline
(plan §9.1): a repeated behavioral experiment has a different unit of evidence
(a trial, not a weight row), a different provenance model (a served model id and
a response id, not a checkpoint sha), and a different validation contract
(reliability and false-discovery control, not projection trustworthiness).
Routing trials through `Units` would make the two look comparable when they are
not.

Nothing here claims anything about a model's internals, and nothing here ranks
models. The strongest sentence the package may produce is the plan's §1.1:

    Under protocol P, exact model deployments A and B produced reliably
    different association distributions for cue C.

Import cost: this module imports nothing at package-import time beyond the
standard library, because `nebulai`'s CLI imports lazily and the optional
`behavior-local` stack (torch / transformers / sentence-transformers) must not
become a hard dependency of the base install.
"""

#: Bumped when `behavior.json` changes shape in a way an older reader cannot
#: parse. The viewer fails closed on an unknown version (plan §9.5) rather than
#: rendering a partial study.
BEHAVIOR_SCHEMA_VERSION = 1
