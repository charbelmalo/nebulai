"""The project-owned cue inventory (plan §5.3) and the positive control (§6.7.2).

Every word here is an ordinary English lexical item written by this project. No
SWOW record, no human-norm table, and no third-party cue list is bundled or
republished — §5.3 gates those behind an explicit licence step that this module
deliberately does not provide a default path around.

Three things live here:

* :data:`STRATA` and :func:`pilot_cues` — the ~300-cue preregistered inventory,
  stratified before collection. The stratum is the ONLY variable the landscape
  may be coloured by (§7.1.2), precisely because it is declared here, in source,
  ahead of any data.
* :func:`calibration_cues` — the 100-cue subset of §5.4 step 1. It is a
  *subset*, drawn deterministically with even stratum coverage, so calibration
  numbers transfer to the pilot instead of describing a different population.
* :data:`POSITIVE_CONTROL` and :func:`positive_control_cues` — the hand-authored
  near-universal associations of §6.7.2. This is a Phase 0 **gate**: a model that
  cannot recover `hot -> cold` is not producing association data, and no
  divergence number computed from its outputs means anything.

The "daddy" pack (§5.3) is `pack="daddy"` inside the `affect_social` and
`identity` strata. It is guaranteed inspectable in the pilot and receives **no
special statistical threshold** — it is matched, not privileged.
"""

from __future__ import annotations

from .contract import Cue

#: Preregistered strata, in the order §5.3 lists them. The keys are stable: an
#: artifact written today must still colour correctly after the inventory grows,
#: so a stratum is never renamed, only added to.
STRATA: tuple[str, ...] = (
    "high_freq_concrete",
    "high_freq_abstract",
    "low_freq",
    "affect_social",
    "identity",
    "slang_internet",
    "technical",
    "polysemous",
    "control_neutral",
)

# --- the inventory ---------------------------------------------------------
# Each entry is (text, pack). An empty pack means the cue stands alone.

_INVENTORY: dict[str, tuple[tuple[str, str], ...]] = {
    # Frequent, imageable nouns. The baseline against which "does this pipeline
    # measure association at all" is easiest to judge by eye.
    "high_freq_concrete": (
        ("water", ""), ("bread", ""), ("chair", ""), ("window", ""), ("river", ""),
        ("mountain", ""), ("bird", ""), ("tree", ""), ("house", ""), ("road", ""),
        ("book", ""), ("knife", ""), ("bottle", ""), ("shoe", ""), ("clock", ""),
        ("mirror", ""), ("garden", ""), ("bridge", ""), ("candle", ""), ("ladder", ""),
        ("forest", ""), ("kitchen", ""), ("engine", ""), ("hammer", ""), ("blanket", ""),
        ("harbour", ""), ("station", ""), ("feather", ""), ("basket", ""), ("lantern", ""),
        ("anchor", ""), ("cellar", ""), ("orchard", ""), ("saddle", ""), ("thunder", ""),
        ("marble", ""), ("kettle", ""),
    ),
    # Frequent but non-imageable: where a base LM's completion habit and an
    # instruction-tuned model's answer habit diverge most visibly.
    "high_freq_abstract": (
        ("freedom", ""), ("justice", ""), ("memory", ""), ("truth", ""), ("power", ""),
        ("time", ""), ("chance", ""), ("reason", ""), ("meaning", ""), ("value", ""),
        ("order", ""), ("change", ""), ("purpose", ""), ("belief", ""), ("doubt", ""),
        ("silence", ""), ("distance", ""), ("balance", ""), ("pattern", ""), ("limit", ""),
        ("origin", ""), ("consent", ""), ("duty", ""), ("privacy", ""), ("risk", ""),
        ("progress", ""), ("identity", ""), ("history", ""), ("future", ""), ("proportion", ""),
        ("consequence", ""), ("tradition", ""), ("intention", ""), ("authority", ""),
        ("compromise", ""), ("evidence", ""), ("uncertainty", ""),
    ),
    # Rarer words. Included because §6.4.4's OOV/fragmentation differential is
    # only measurable if the inventory actually contains terms the encoder and
    # the tokenizers handle unevenly.
    "low_freq": (
        ("petrichor", ""), ("quorum", ""), ("lintel", ""), ("obsidian", ""),
        ("palisade", ""), ("vestibule", ""), ("marginalia", ""), ("cairn", ""),
        ("escarpment", ""), ("plinth", ""), ("codicil", ""), ("ambergris", ""),
        ("threnody", ""), ("cordillera", ""), ("spandrel", ""), ("apse", ""),
        ("bezoar", ""), ("caliper", ""), ("dovetail", ""), ("furlong", ""),
        ("gantry", ""), ("hummock", ""), ("intaglio", ""), ("kestrel", ""),
        ("lodestar", ""), ("mordant", ""), ("nacelle", ""), ("oriel", ""),
        ("purlin", ""), ("quillon", ""), ("revetment", ""), ("sastrugi", ""),
    ),
    # Affect and social relationship. The "daddy" pack's matched relationship
    # members live here; the slang member lives in `slang_internet`, and the
    # identity members in `identity`, so the pack spans strata by design.
    "affect_social": (
        ("daddy", "daddy"), ("father", "daddy"), ("mother", "daddy"),
        ("parent", "daddy"), ("child", "daddy"), ("son", "daddy"),
        ("daughter", "daddy"), ("mommy", "daddy"), ("papa", "daddy"),
        ("grief", ""), ("joy", ""), ("shame", ""), ("pride", ""), ("envy", ""),
        ("comfort", ""), ("loneliness", ""), ("trust", ""), ("betrayal", ""),
        ("friend", ""), ("stranger", ""), ("neighbour", ""), ("rival", ""),
        ("apology", ""), ("promise", ""), ("argument", ""), ("reunion", ""),
        ("marriage", ""), ("divorce", ""), ("crush", ""), ("tenderness", ""),
        ("jealousy", ""), ("gratitude", ""), ("nostalgia", ""), ("forgiveness", ""),
        ("rejection", ""), ("loyalty", ""), ("homesick", ""), ("affection", ""),
        ("resentment", ""),
    ),
    # Identity and demographic terms. Handled under the sensitive-content rules
    # of §8.6: displayed with the neutral-frame caveat, never used to make a
    # claim about a group, only about two deployments' outputs.
    "identity": (
        ("woman", ""), ("man", ""), ("teenager", ""), ("elder", ""),
        ("immigrant", ""), ("nurse", ""), ("engineer", ""), ("teacher", ""),
        ("soldier", ""), ("farmer", ""), ("artist", ""), ("scientist", ""),
        ("boss", "daddy"), ("sir", "daddy"), ("ma'am", "daddy"),
        ("stepfather", "daddy"), ("guardian", "daddy"),
        ("student", ""), ("athlete", ""), ("refugee", ""), ("veteran", ""),
        ("neighbourhood", ""), ("citizen", ""), ("worker", ""), ("founder", ""),
        ("apprentice", ""), ("landlord", ""), ("tenant", ""), ("volunteer", ""),
        ("janitor", ""), ("librarian", ""), ("pilot", ""),
    ),
    # Internet-era and slang. These are exactly the terms an encoder represents
    # worst, which is why §6.4.4 makes the OOV differential a reported figure
    # rather than a filter.
    "slang_internet": (
        ("sus", ""), ("ratio", ""), ("based", ""), ("cringe", ""), ("simp", ""),
        ("ghosting", ""), ("doomscroll", ""), ("rizz", ""), ("mid", ""),
        ("stan", ""), ("yeet", ""), ("bussin", ""), ("lowkey", ""), ("flex", ""),
        ("gatekeep", ""), ("touchgrass", ""), ("npc", ""), ("cope", ""),
        ("goated", ""), ("zaddy", "daddy"), ("bae", ""), ("shipping", ""),
        ("vibe", ""), ("clout", ""), ("salty", ""), ("unhinged", ""), ("delulu", ""),
        ("shook", ""), ("thirsty", ""), ("snatched", ""), ("slaps", ""), ("bot", ""),
    ),
    # Technical/scientific. Included to check the instrument on vocabulary where
    # a correct association is checkable by a domain reader.
    "technical": (
        ("entropy", ""), ("gradient", ""), ("mitochondria", ""), ("isotope", ""),
        ("algorithm", ""), ("compiler", ""), ("vaccine", ""), ("neuron", ""),
        ("catalyst", ""), ("tensor", ""), ("photon", ""), ("enzyme", ""),
        ("orbit", ""), ("plasma", ""), ("latency", ""), ("checksum", ""),
        ("titration", ""), ("kernel", ""), ("antibody", ""), ("capacitor", ""),
        ("polymer", ""), ("eigenvalue", ""), ("sediment", ""), ("bandwidth", ""),
        ("protocol", ""), ("chromosome", ""), ("manifold", ""), ("solvent", ""),
        ("torque", ""), ("aquifer", ""), ("lattice", ""), ("viscosity", ""),
    ),
    # Ambiguous / polysemous. §5.1 lane C exists to ask whether a difference at
    # these cues is sense-specific; they are the cues where that question has a
    # real answer.
    "polysemous": (
        ("bank", ""), ("crane", ""), ("bark", ""), ("spring", ""), ("pitch", ""),
        ("mole", ""), ("bat", ""), ("seal", ""), ("plant", ""), ("match", ""),
        ("jam", ""), ("trunk", ""), ("bolt", ""), ("draft", ""), ("scale", "poly_scale"),
        ("current", ""), ("mint", ""), ("port", ""), ("light", ""), ("rock", ""),
        ("club", ""), ("palm", ""), ("nail", ""), ("ruler", ""), ("wave", ""),
        ("bass", ""), ("lead", ""), ("tear", ""), ("wind", ""), ("row", ""),
        ("fine", ""), ("sound", ""),
    ),
    # Neutral matched controls (§5.3). Ordinary, low-affect, unambiguous nouns
    # and verbs whose only job is to sit at the boring end of every stratum
    # contrast. A study where these light up is a study with an artifact.
    "control_neutral": (
        ("paper", ""), ("button", ""), ("cupboard", ""), ("envelope", ""),
        ("shelf", ""), ("towel", ""), ("carpet", ""), ("spoon", ""),
        ("napkin", ""), ("drawer", ""), ("handle", ""), ("bucket", ""),
        ("string", ""), ("hinge", ""), ("stapler", ""), ("folder", ""),
        ("pebble", ""), ("crate", ""), ("tile", ""), ("ribbon", ""),
        ("saucer", ""), ("thimble", ""), ("wrench", ""), ("cushion", ""),
        ("bookmark", ""), ("doorknob", ""), ("coaster", ""), ("lampshade", ""),
        ("pillowcase", ""), ("clipboard", ""), ("mousepad", ""), ("keyring", ""),
    ),
}


def pilot_cues() -> list[Cue]:
    """The full preregistered pilot inventory (§5.4 step 2), in stratum order."""
    out: list[Cue] = []
    for stratum in STRATA:
        for text, pack in _INVENTORY[stratum]:
            out.append(Cue(text=text, stratum=stratum, pack=pack))
    return out


def calibration_cues(n: int = 100) -> list[Cue]:
    """A deterministic, stratum-balanced subset for §5.4 step 1.

    Round-robin across strata rather than a random draw: a random 100 of 242 can
    under-sample `low_freq` badly, and calibration's whole job is to produce
    numbers (the p-floor, the bandwidth, the control pass rate) that still hold
    on the pilot population.
    """
    by_stratum = {s: [c for c in pilot_cues() if c.stratum == s] for s in STRATA}
    out: list[Cue] = []
    i = 0
    while len(out) < n:
        added = False
        for s in STRATA:
            bucket = by_stratum[s]
            if i < len(bucket):
                out.append(bucket[i])
                added = True
                if len(out) == n:
                    return out
        if not added:
            break
        i += 1
    return out


# --- the positive control (§6.7.2) -----------------------------------------
#
# Hand-authored, licence-safe, near-universal English associations. The
# requirement is WITHIN-model: each model's own top responses must contain the
# expected associate at a predeclared rate. Cross-model agreement is not what is
# being tested here — instrument validity is.

POSITIVE_CONTROL: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hot", ("cold", "warm", "heat")),
    ("salt", ("pepper", "sugar", "sea")),
    ("cat", ("dog", "kitten", "mouse")),
    ("king", ("queen", "crown", "throne")),
    ("day", ("night", "sun", "morning")),
    ("black", ("white", "dark", "night")),
    ("up", ("down", "high", "above")),
    ("left", ("right", "side", "hand")),
    ("bread", ("butter", "toast", "food")),
    ("moon", ("sun", "star", "night")),
    ("doctor", ("nurse", "hospital", "patient")),
    ("dog", ("cat", "puppy", "bark")),
    ("man", ("woman", "boy", "male")),
    ("summer", ("winter", "sun", "hot")),
    ("north", ("south", "pole", "cold")),
    ("fire", ("water", "flame", "burn")),
    ("mother", ("father", "child", "mom")),
    ("question", ("answer", "ask", "mark")),
    ("sick", ("ill", "health", "doctor")),
    ("table", ("chair", "desk", "wood")),
)

#: Any of a cue's listed associates counts as a hit. Accepting a small set
#: rather than one gold word is not a loosened gate — `hot -> warm` is a correct
#: association and scoring it as a miss would make the control measure agreement
#: with the author's first guess instead of association competence.
POSITIVE_CONTROL_MAP: dict[str, tuple[str, ...]] = dict(POSITIVE_CONTROL)


def positive_control_cues() -> list[Cue]:
    """The control cues as `Cue`s, so they can be scheduled like any other."""
    return [
        Cue(text=t, stratum="control_neutral", pack="positive_control")
        for t, _ in POSITIVE_CONTROL
    ]


def control_hit(cue: str, associates: list[str]) -> bool:
    """Did this trial recover an expected associate for a control cue?"""
    expected = POSITIVE_CONTROL_MAP.get(cue.casefold())
    if not expected:
        return False
    return any(a in expected for a in associates)
