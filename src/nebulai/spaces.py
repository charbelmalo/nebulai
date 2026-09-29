"""The closed set of vector spaces a channel or a direction may live in.

Every per-point scalar (`out/<model>/channels.json`) and every direction
(`out/<model>/directions.json`) carries a `space` tag. The tag is not decoration:
it is what makes D2 — *refuse* cross-space projection rather than warn about it —
mechanical. A refusal that lives in a UI can be routed around by the next view;
a refusal that lives in the type of the data cannot.

The set is CLOSED. `parse()` raises on anything it does not recognise, so a typo
("resid.14", "W_E") fails at write time in Python rather than silently producing
a channel nothing will ever agree to plot against.

    W_E.raw                the model's input embedding rows, as stored
    W_E.centered           the same rows after the map's mean-centering
    W_U.raw                the output (unembedding) rows, as stored
    resid.L<k>             the residual stream entering block k (k = -1 is the
                           embedding output before block 0)
    mlp_out.L<k>           layer k's MLP write directions (rows of c_proj)
    sae.L<k>.<repo>        one SAE's decoder space at layer k
    text-embed.<model>     a foreign text embedder's space — NOT model-internal
    persona-pca.<space_id> a fixed persona coordinate system (phase 2)

`W_E.raw` and `W_E.centered` are deliberately DIFFERENT spaces even though one
is an affine image of the other. The whole SolidGoldMagikarp experiment is that
a glitch token's row never moved from initialisation, which is a fact about the
raw rows; the map's own geometry is centred. Letting the two tags compare equal
would let the low-norm knot be plotted against a direction found in the centred
cloud, and the resulting picture would be about the centring, not the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class UnknownSpaceError(ValueError):
    """Raised for a space tag outside the closed set."""


class SpaceFamily(str, Enum):
    """The eight families. A tag is a family plus zero or more parameters."""

    WE_RAW = "W_E.raw"
    WE_CENTERED = "W_E.centered"
    WU_RAW = "W_U.raw"
    RESID = "resid"
    MLP_OUT = "mlp_out"
    SAE = "sae"
    TEXT_EMBED = "text-embed"
    PERSONA_PCA = "persona-pca"


#: families that take no parameters — the tag IS the family value
_BARE = {SpaceFamily.WE_RAW, SpaceFamily.WE_CENTERED, SpaceFamily.WU_RAW}

_LAYER = re.compile(r"^L(-?\d+)$")
# an HF repo id or a local slug: letters, digits and - _ . / only. Deliberately
# narrow — a space tag ends up in a filename-adjacent id and in a URL fragment.
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


@dataclass(frozen=True)
class Space:
    """A parsed space tag. Compare these, never the raw strings."""

    family: SpaceFamily
    #: residual / mlp_out / sae layer index; None for the other families
    layer: int | None = None
    #: sae repo, text embedder id, or persona space_id; None otherwise
    ref: str | None = None

    def __str__(self) -> str:
        if self.family in _BARE:
            return self.family.value
        if self.family is SpaceFamily.SAE:
            return f"sae.L{self.layer}.{self.ref}"
        if self.family in (SpaceFamily.RESID, SpaceFamily.MLP_OUT):
            return f"{self.family.value}.L{self.layer}"
        return f"{self.family.value}.{self.ref}"

    @property
    def model_internal(self) -> bool:
        """False for the one family that is a foreign embedder's geometry.

        R7 ("foreign data wears foreign clothes") reads this: a text-embedder
        position and a model-internal one never share a glyph.
        """
        return self.family is not SpaceFamily.TEXT_EMBED


def parse(tag: str) -> Space:
    """Parse a space tag, or raise `UnknownSpaceError`.

    Never guesses. `"resid.14"` (missing the `L`) is an error, not `resid.L14`,
    because a channel written under a tag nothing else uses is a channel that
    silently never renders.
    """
    if not isinstance(tag, str) or not tag:
        raise UnknownSpaceError(f"space tag must be a non-empty string, got {tag!r}")

    for fam in _BARE:
        if tag == fam.value:
            return Space(fam)

    head, sep, rest = tag.partition(".")
    if not sep:
        raise UnknownSpaceError(
            f"unknown space {tag!r}: not one of {sorted(f.value for f in _BARE)} "
            f"and carries no parameter"
        )

    if head in ("resid", "mlp_out"):
        m = _LAYER.match(rest)
        if not m:
            raise UnknownSpaceError(
                f"unknown space {tag!r}: {head} takes a layer like {head}.L8 "
                f"(L-1 is the embedding output before block 0)"
            )
        fam = SpaceFamily.RESID if head == "resid" else SpaceFamily.MLP_OUT
        return Space(fam, layer=int(m.group(1)))

    if head == "sae":
        layer_part, sep2, repo = rest.partition(".")
        m = _LAYER.match(layer_part)
        if not (sep2 and m and _REPO.match(repo)):
            raise UnknownSpaceError(
                f"unknown space {tag!r}: sae takes a layer and a repo, like "
                f"sae.L8.jbloom/GPT2-Small-SAEs-Reformatted"
            )
        return Space(SpaceFamily.SAE, layer=int(m.group(1)), ref=repo)

    if head == "text-embed":
        if not _REPO.match(rest):
            raise UnknownSpaceError(
                f"unknown space {tag!r}: text-embed takes an embedder id, like "
                f"text-embed.mxbai-embed-large"
            )
        return Space(SpaceFamily.TEXT_EMBED, ref=rest)

    if head == "persona-pca":
        if not _REPO.match(rest):
            raise UnknownSpaceError(
                f"unknown space {tag!r}: persona-pca takes a space_id, like "
                f"persona-pca.smollm2-360m-instruct@abc1234.v1"
            )
        return Space(SpaceFamily.PERSONA_PCA, ref=rest)

    raise UnknownSpaceError(
        f"unknown space {tag!r}: the closed set is W_E.raw, W_E.centered, "
        f"W_U.raw, resid.L<k>, mlp_out.L<k>, sae.L<k>.<repo>, "
        f"text-embed.<model>, persona-pca.<space_id>"
    )


def is_known(tag: str) -> bool:
    """True when `tag` parses. Never raises."""
    try:
        parse(tag)
    except UnknownSpaceError:
        return False
    return True


def compatible(a: str | Space, b: str | Space) -> bool:
    """May a quantity in space `a` be plotted against one in space `b`?

    Only when they are the SAME space. There is no partial compatibility and no
    "close enough": `resid.L13` and `resid.L14` are different bases, `W_E.raw`
    and `W_E.centered` differ by a translation the experiment is about, and two
    SAEs at the same layer are two different dictionaries. D2 approved the hard
    refusal precisely so that a picture you cannot make is one you cannot
    mistake for a finding.

    An unparseable tag is incompatible with everything, including itself — an
    unknown space is not a space.
    """
    try:
        pa = a if isinstance(a, Space) else parse(a)
        pb = b if isinstance(b, Space) else parse(b)
    except UnknownSpaceError:
        return False
    return pa == pb


def refusal_reason(a: str | Space, b: str | Space) -> str | None:
    """Why `a` and `b` may not be compared, or None when they may.

    The UI states this rather than inventing its own wording, so the browser and
    the CLI refuse in the same words.
    """
    if compatible(a, b):
        return None
    for tag in (a, b):
        if isinstance(tag, str) and not is_known(tag):
            return f"unknown space {tag!r}"
    return f"space mismatch: {a} vs {b} — different bases, never comparable"


def we_space(centered: bool) -> Space:
    """The token-map space for a given centering flag."""
    return Space(SpaceFamily.WE_CENTERED if centered else SpaceFamily.WE_RAW)


def resid(layer: int) -> Space:
    """`resid.L<k>`; layer -1 is the embedding output before block 0."""
    return Space(SpaceFamily.RESID, layer=layer)
