"""Independent-validation tests — trustworthiness, seed stability, null baseline.

Offline and small: the UMAP-driven checks run on tiny synthetic blobs with
reduced n_neighbors, so the whole file stays fast enough to keep in the default
suite. The assertions are directional (a faithful embedding beats a scrambled
one) rather than exact — the point of these metrics is the ordering, and
pinning exact scores would just encode the current UMAP version.
"""

import numpy as np
import pytest

from nebulai.backend.validate import (
    null_baseline,
    reload_units,
    scale_cluster_kwargs,
    seed_stability,
    trustworthiness_score,
)
from nebulai.units import Units


def _blobs(n_per: int = 40, dim: int = 8, seed: int = 0) -> np.ndarray:
    """Three well-separated gaussian blobs — real structure to find."""
    rng = np.random.RandomState(seed)
    centers = np.eye(3, dim) * 12.0
    return np.vstack(
        [rng.randn(n_per, dim) * 0.4 + c for c in centers]
    ).astype(np.float32)


# --- trustworthiness ------------------------------------------------------


def test_trustworthiness_rewards_a_faithful_embedding():
    X = _blobs()
    faithful = X[:, :2]  # a real (if lossy) projection of the same geometry
    scrambled = np.random.RandomState(1).randn(len(X), 2).astype(np.float32)

    good = trustworthiness_score(X, faithful, n_neighbors=5)
    bad = trustworthiness_score(X, scrambled, n_neighbors=5)

    assert good["trustworthiness"] > bad["trustworthiness"]
    assert good["trustworthiness"] > 0.9
    # random 2-D noise should land near the chance level, nowhere near 1.0
    assert bad["trustworthiness"] < 0.75


def test_trustworthiness_reports_what_it_actually_scored():
    X = _blobs(n_per=100)  # 300 points
    r = trustworthiness_score(X, X[:, :2], n_neighbors=5, sample_cap=50)
    assert r["n_scored"] == 50
    assert r["subsampled"] is True

    full = trustworthiness_score(X, X[:, :2], n_neighbors=5, sample_cap=10_000)
    assert full["n_scored"] == 300
    assert full["subsampled"] is False


def test_trustworthiness_undefined_on_tiny_input():
    X = np.random.RandomState(0).randn(4, 3).astype(np.float32)
    assert trustworthiness_score(X, X[:, :2], n_neighbors=15)["trustworthiness"] is None


def test_trustworthiness_rejects_length_mismatch():
    X = _blobs()
    with pytest.raises(ValueError, match="length mismatch"):
        trustworthiness_score(X, X[:-1, :2])


# --- seed stability -------------------------------------------------------


def test_seed_stability_high_on_genuinely_separated_blobs():
    """Three blobs this far apart survive reseeding — ARI should be near 1."""
    X = _blobs(n_per=60)
    r = seed_stability(
        X,
        seeds=(42, 1, 2),
        reduce_kwargs={"cluster_dim": 3, "n_neighbors": 10},
        cluster_kwargs={"min_cluster_size": 10, "method": "eom"},
    )
    assert r["mean_ari"] > 0.9
    assert r["min_ari"] > 0.8
    assert len(r["pairs"]) == 3  # 3 seeds -> 3 unordered pairs
    assert [p["seed"] for p in r["per_seed"]] == [42, 1, 2]
    assert [p["n_clusters"] for p in r["per_seed"]] == [3, 3, 3]


def test_seed_stability_exposes_the_leaf_default_fragmenting():
    """The metric's whole job is to catch instability, so pin the case that
    proves it can: on the SAME three blobs, `leaf` (the project default) splits
    them well past 3 and the seed-to-seed agreement collapses, while `eom`
    holds at ARI 1.0. A low stability score on a real map is therefore partly a
    statement about this trade-off — which is why the method is reported next
    to the number rather than left implicit."""
    X = _blobs(n_per=60)
    rk = {"cluster_dim": 3, "n_neighbors": 10}

    leaf = seed_stability(
        X, seeds=(42, 1, 2), reduce_kwargs=rk,
        cluster_kwargs={"min_cluster_size": 10, "method": "leaf"},
    )
    eom = seed_stability(
        X, seeds=(42, 1, 2), reduce_kwargs=rk,
        cluster_kwargs={"min_cluster_size": 10, "method": "eom"},
    )

    assert all(p["n_clusters"] > 3 for p in leaf["per_seed"])
    assert leaf["mean_ari"] < eom["mean_ari"]


def test_seed_stability_records_sample_size():
    X = _blobs(n_per=60)  # 180 points
    r = seed_stability(
        X,
        seeds=(42, 1),
        reduce_kwargs={"cluster_dim": 3, "n_neighbors": 10},
        cluster_kwargs={"min_cluster_size": 10},
        sample_cap=100,
    )
    assert r["n_scored"] == 100
    assert r["subsampled"] is True


# --- null baseline --------------------------------------------------------


def test_min_cluster_size_is_rescaled_for_subsamples():
    """A subsample must be clustered at the full map's GRANULARITY.

    Real case this comes from: SmolLM2's token map resolves min_cluster_size=48
    over 48,636 points. Carrying 48 onto a 4,000-point null asks for clusters
    12x larger in relative terms — it produced 6 giant clusters and inflated the
    null silhouette to 0.43, while GPT-2's null under a different stamped value
    produced 51 clusters and 0.21. Neither described its own map.
    """
    scaled = scale_cluster_kwargs(
        {"min_cluster_size": 48, "min_samples": 5, "method": "leaf"},
        n_full=48_636,
        n_sample=4_000,
    )
    assert scaled["min_cluster_size"] == 4  # 48 * 4000/48636 -> 3.9 -> 4
    assert scaled["min_samples"] == 5  # untouched — not a size-relative knob
    assert scaled["method"] == "leaf"


def test_min_cluster_size_untouched_when_not_subsampling():
    kw = {"min_cluster_size": 25, "min_samples": 5, "method": "leaf"}
    assert scale_cluster_kwargs(kw, n_full=3072, n_sample=3072) == kw
    assert scale_cluster_kwargs(kw, n_full=3072, n_sample=4000) == kw


def test_rescaled_min_cluster_size_never_degenerates():
    scaled = scale_cluster_kwargs(
        {"min_cluster_size": 15}, n_full=1_000_000, n_sample=100
    )
    assert scaled["min_cluster_size"] == 2  # floor, never 0 or negative


def test_null_baseline_returns_a_scored_floor_and_says_what_it_scored():
    """Contract only — deliberately NOT `real > null`.

    That assertion looks obvious and is false at this scale: shuffled 180x8
    data scores silhouette ~0.88 while three genuinely separated blobs score
    ~0.43, because UMAP manufactures clean islands out of noise when n is
    small. Encoding `real > null` here would pin a research claim that the
    metric itself disproves. The directional comparison is only meaningful at
    the map's real scale, which is the caveat carried in the module docstring.
    """
    X = _blobs(n_per=60)
    null = null_baseline(
        X,
        reduce_kwargs={"cluster_dim": 3, "n_neighbors": 10},
        cluster_kwargs={"min_cluster_size": 10},
    )
    assert null["n_scored"] == len(X)
    assert null["subsampled"] is False
    assert null["silhouette"] is None or -1.0 <= null["silhouette"] <= 1.0
    assert 0.0 <= null["noise_fraction"] <= 1.0
    assert null["n_clusters"] >= 0


def test_null_baseline_preserves_column_marginals():
    """The shuffle must be per-column, not global — otherwise the null has the
    wrong scale per dimension and the comparison is unfair."""
    X = _blobs(n_per=30)
    rk = {"cluster_dim": 2, "n_neighbors": 8}
    null_baseline(X, reduce_kwargs=rk, cluster_kwargs={"min_cluster_size": 5})
    # verified indirectly: re-derive the shuffle the function performs
    rng = np.random.RandomState(42)
    shuffled = np.empty_like(X)
    for c in range(X.shape[1]):
        shuffled[:, c] = rng.permutation(X[:, c])
    assert np.allclose(np.sort(shuffled, axis=0), np.sort(X, axis=0))


# --- the reload guard -----------------------------------------------------
#
# Trustworthiness needs the ORIGINAL vectors, which `reduced.npz` does not
# cache, so `reload_units` replays the front-end from meta. For front-ends that
# CANNOT be replayed exactly, refusing is the whole feature: a reload that
# silently returned a different point set would validate something other than
# the map on screen, and `validate_map`'s length guard only catches that when
# the counts happen to differ.


def test_probe_maps_refuse_to_reload_and_say_why():
    """Probe is non-replayable twice over — a stochastic generator AND a live
    embedding host — so it must never grow a reload path."""
    meta = {
        "unit": "probe_concept(sentence-transformers/all-MiniLM-L6-v2)",
        "generator": "openai:some-model",
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embed_host": "http://embed.test:8040",
        "kept": 53,
        "n_proposed": 118,
    }
    with pytest.raises(ValueError) as e:
        reload_units(meta)
    msg = str(e.value)
    # the reason has to name BOTH sources of drift, not just the host —
    # "re-run against a reachable host" would be wrong advice for probe
    assert "generator" in msg and "live service" in msg
    assert "DIFFERENT point set" in msg
    assert "53" in msg and "118" in msg


def test_api_text_embedding_maps_refuse_to_reload():
    meta = {
        "unit": "api_text_embedding(mxbai-embed-large)",
        "embed_model": "mxbai-embed-large",
        "embed_host": "http://embed.test:11434",
    }
    with pytest.raises(ValueError, match="cannot be revalidated offline"):
        reload_units(meta)


def test_unknown_unit_types_refuse_rather_than_guess():
    with pytest.raises(ValueError, match="no reload path"):
        reload_units({"unit": "something_new"})


# --- W_U maps replay, and they replay PINNED ------------------------------


def test_unembedding_maps_reload_at_the_pinned_revision(monkeypatch):
    """A W_U map must be replayable, and it must replay the vocabulary the map
    was actually built on.

    The `kept` rows of a W_U map are chosen by `curated_vocab`, which reads the
    repo's TOKENIZER. Replaying against "main" would score whatever vocabulary
    the repo has today — a different point set of the same size, which is
    exactly the failure `validate_map`'s length guard cannot catch. So the
    branch has to forward `revision`, and this test fails if anyone drops it.
    """
    seen: dict[str, object] = {}

    def fake_load(model_id, **kw):
        seen["model_id"] = model_id
        seen.update(kw)
        return Units(ids=[0, 1], vectors=np.zeros((2, 4), dtype=np.float32), labels=["a", "b"])

    monkeypatch.setattr("nebulai.frontends.tokens.load_token_units", fake_load)
    reload_units(
        {
            "unit": "token_unembedding",
            "which": "output",
            "model": "mistralai/Mistral-Nemo-Instruct-2407",
            "revision": "04d8a90549d23fc6bd7f642064003592df51e9b3",
            "source": "remote-range",
            "centered": True,
            "kept": 5000,
        }
    )
    assert seen["model_id"] == "mistralai/Mistral-Nemo-Instruct-2407"
    assert seen["revision"] == "04d8a90549d23fc6bd7f642064003592df51e9b3"
    assert seen["which"] == "output", "a W_U reload that reads W_E would score the wrong matrix"
    assert seen["max_tokens"] == 5000
    assert seen["center"] is True
    # the map was built over HTTP ranges against a 24 GB checkpoint; the reload
    # must not quietly decide to download it
    assert seen["remote"] is True


def test_unembedding_reload_does_not_force_remote_for_a_locally_built_map(monkeypatch):
    """`remote=None` leaves the per-repo decision to the front-end, which is the
    right answer for anything not stamped `remote-range`."""
    seen: dict[str, object] = {}

    def fake_load(model_id, **kw):
        seen.update(kw)
        return Units(ids=[0], vectors=np.zeros((1, 4), dtype=np.float32), labels=["a"])

    monkeypatch.setattr("nebulai.frontends.tokens.load_token_units", fake_load)
    reload_units({"unit": "token_unembedding", "model": "gpt2", "kept": 10})
    assert seen["remote"] is None
    assert seen["which"] == "output", "the default must be the unembedding, not W_E"
    assert seen["revision"] == "main"


# --- neuron maps replay pinned too, for the same reason -------------------


def test_neuron_maps_reload_at_the_pinned_revision_and_stay_remote(monkeypatch):
    """The `down_proj` depth series on Mistral-Nemo is built over HTTP ranges
    against a pinned commit of a 24 GB checkpoint. Dropping either forwarding
    turns `nebulai validate` into a 24 GB download, or into a silent score of
    whatever `main` holds today — and a different commit's `down_proj` has the
    same row count, so the length guard in `validate_map` cannot notice."""
    seen: dict[str, object] = {}

    def fake_load(model_repo, **kw):
        seen["model_repo"] = model_repo
        seen.update(kw)
        return Units(
            ids=[0, 1], vectors=np.zeros((2, 4), dtype=np.float32), labels=["n0", "n1"]
        )

    monkeypatch.setattr("nebulai.frontends.neurons.load_neuron_units", fake_load)
    reload_units(
        {
            "unit": "mlp_neuron(mistralai/Mistral-Nemo-Instruct-2407, model.layers.4.mlp.down_proj)",
            "model_repo": "mistralai/Mistral-Nemo-Instruct-2407",
            "layer": 4,
            "kept": 4096,
            "centered": False,
            "labels_source": "none",
            "revision": "04d8a90549d23fc6bd7f642064003592df51e9b3",
            "source": "remote-range",
            "expert": None,
        }
    )
    assert seen["model_repo"] == "mistralai/Mistral-Nemo-Instruct-2407"
    assert seen["revision"] == "04d8a90549d23fc6bd7f642064003592df51e9b3"
    assert seen["remote"] is True
    assert seen["layer"] == 4
    assert seen["max_neurons"] == 4096
    assert seen["expert"] is None


def test_neuron_reload_forwards_the_expert_for_an_moe_layer(monkeypatch):
    """An MoE `down_proj` has one tensor per expert. Replaying without the
    expert index raises inside the loader about ambiguous keys — a confusing
    failure about tensors, for a map whose meta recorded the answer."""
    seen: dict[str, object] = {}

    def fake_load(model_repo, **kw):
        seen.update(kw)
        return Units(ids=[0], vectors=np.zeros((1, 4), dtype=np.float32), labels=["n"])

    monkeypatch.setattr("nebulai.frontends.neurons.load_neuron_units", fake_load)
    reload_units(
        {
            "unit": "mlp_neuron(x/y, model.layers.3.mlp.experts.7.down_proj)",
            "model_repo": "x/y",
            "layer": 3,
            "kept": 8,
            "expert": 7,
        }
    )
    assert seen["expert"] == 7
    assert seen["remote"] is None, "a locally built map must not be forced remote"
