"""Semantic-cloud front-end tests — dedup, BFS shape, the sensitivity gate, and
the honesty stamps.

Fully offline: the generator and the embedder are both fakes. The fake embedder
places every term at a controlled cosine from the seed, which is what makes the
`--sensitivity` behaviour testable at all — with a real embedder the threshold's
effect depends on the embedder's opinions, which is precisely the thing this
front-end must never hide.
"""

import json

import numpy as np
import pytest

from nebulai import llm as llm_mod
from nebulai.backend import embed as embed_mod
from nebulai.frontends import probe as probe_mod
from nebulai.frontends.probe import _clean, _norm, load_probe_units


# --- pure helpers ---------------------------------------------------------


def test_norm_folds_case_and_edge_punctuation_only():
    assert _norm("  Grief.  ") == "grief"
    assert _norm("Stages of Grief") == "stages of grief"
    # distinct wordings stay distinct — merging them is a judgement the map
    # is supposed to display, not silently make
    assert _norm("car") != _norm("automobile")


def test_clean_dedups_against_seen_and_respects_limit():
    seen = {"grief"}
    out = _clean(["Grief", "mourning", "Mourning", "loss", "denial"], seen, limit=2)
    assert out == ["mourning", "loss"]  # 'Grief' already seen, 'Mourning' a dupe
    assert "mourning" in seen and "loss" in seen


def test_clean_drops_empty_and_overlong_terms():
    out = _clean(["", "   ", "x" * 200, "valid"], set(), limit=10)
    assert out == ["valid"]


# --- the cloud ------------------------------------------------------------


def _fake_embedder(sim_for):
    """Return an embed_texts stand-in placing term i at cosine `sim_for(term)`
    from the seed (term 0), in 2-D: seed=[1,0], term=[s, sqrt(1-s^2)]."""

    def embed_texts(texts, **_kw):
        vecs = []
        for i, t in enumerate(texts):
            s = 1.0 if i == 0 else float(sim_for(t))
            s = max(-1.0, min(1.0, s))
            vecs.append([s, float(np.sqrt(max(0.0, 1.0 - s * s)))])
        return np.asarray(vecs, dtype=np.float32)

    return embed_texts


@pytest.fixture
def fake_stack(monkeypatch):
    """Install a deterministic generator + embedder. Terms are named
    'near-<n>' / 'far-<n>' so the sensitivity gate has something to bite on."""

    def install(sim_for, breadth_terms=None):
        calls = []

        def expand(term, n):
            calls.append((term, n))
            if breadth_terms is not None:
                return breadth_terms(term, n)
            return [f"near-{term}-{i}" for i in range(n)]

        monkeypatch.setattr(
            probe_mod, "_make_expander", lambda *a, **k: (expand, "fake:gen-1")
        )
        monkeypatch.setattr(embed_mod, "embed_texts", _fake_embedder(sim_for))
        return calls

    return install


def test_bfs_expands_breadth_first_to_the_requested_depth(fake_stack):
    calls = fake_stack(sim_for=lambda t: 0.9)
    units = load_probe_units("grief", depth=2, breadth=3, sensitivity=0.0)

    # depth 1: 1 call on the seed -> 3 terms. depth 2: 3 calls -> 9 terms.
    assert len(calls) == 4
    assert calls[0] == ("grief", 3)
    assert len(units) == 1 + 3 + 9
    assert units.meta["max_depth_reached"] == 2
    assert units.labels[0] == "grief"


def test_sensitivity_gates_on_similarity_to_the_seed(fake_stack):
    """Terms below the floor are dropped and COUNTED, not silently discarded."""
    fake_stack(
        sim_for=lambda t: 0.8 if "keep" in t else 0.1,
        breadth_terms=lambda term, n: [f"keep-{term}-{i}" for i in range(n // 2)]
        + [f"drop-{term}-{i}" for i in range(n - n // 2)],
    )
    units = load_probe_units("grief", depth=1, breadth=4, sensitivity=0.5)

    assert all("drop" not in lab for lab in units.labels)
    assert units.meta["n_proposed"] == 5  # seed + 4
    assert units.meta["kept"] == len(units)
    assert units.meta["n_dropped"] == units.meta["n_proposed"] - units.meta["kept"]
    assert units.meta["n_dropped"] == 2


def test_seed_is_never_gated_out(fake_stack):
    """The seed anchors the map; a high threshold must not delete it."""
    fake_stack(sim_for=lambda t: 0.99)
    units = load_probe_units("grief", depth=1, breadth=6, sensitivity=0.95)
    assert units.labels[0] == "grief"
    assert units.meta["seed_similarity_min"] >= 0.95


def test_meta_declares_the_geometry_is_not_model_internal(fake_stack):
    fake_stack(sim_for=lambda t: 0.9)
    units = load_probe_units(
        "grief",
        depth=1,
        breadth=6,
        sensitivity=0.0,
        embed_host="http://embed.internal:8040",
        embed_model="fake-embedder",
    )
    m = units.meta
    assert "NOT model-internal" in m["geometry"]
    assert m["generator"] == "fake:gen-1"
    assert m["embed_model"] == "fake-embedder"
    assert m["embed_host"] == "remote"
    assert "embed.internal" not in json.dumps(m)
    assert m["probe_seed"] == "grief"
    assert m["sensitivity"] == 0.0
    assert m["unit"].startswith("probe_concept")


def test_units_contract_holds(fake_stack):
    fake_stack(sim_for=lambda t: 0.9)
    units = load_probe_units("grief", depth=2, breadth=3, sensitivity=0.0)
    assert len(units.ids) == len(units.labels) == units.vectors.shape[0]
    assert units.vectors.dtype == np.float32


def test_too_aggressive_sensitivity_fails_loudly(fake_stack):
    fake_stack(sim_for=lambda t: 0.1)
    with pytest.raises(RuntimeError, match="kept only"):
        load_probe_units("grief", depth=1, breadth=6, sensitivity=0.9)


def test_a_generator_that_returns_nothing_fails_loudly(fake_stack):
    fake_stack(sim_for=lambda t: 0.9, breadth_terms=lambda term, n: [])
    with pytest.raises(RuntimeError, match="only 1 terms"):
        load_probe_units("grief", depth=2, breadth=6, sensitivity=0.0)


def test_probe_maps_get_a_readable_label_in_the_metrics_table():
    from nebulai.backend.compare import _source_label

    label = _source_label(
        {"model": "grief", "unit": "probe_concept(mxbai-embed-large)"}
    )
    assert label == "grief · probe concepts"


# --- generator availability check ------------------------------------------
# _make_expander sends one throwaway expansion to prove a backend works. A
# reasoning model that rambles past the token ceiling on that call has proved
# exactly the opposite of "unavailable" — it generated too much, not nothing.


def test_a_rambling_generator_is_kept_not_retired(monkeypatch, capsys):
    from nebulai.llm import ChatTruncated

    calls: list[str] = []

    def fake_expand(term, n, host, model, key):
        calls.append(term)
        if len(calls) == 1:
            raise ChatTruncated("m truncated its reply at 16384 tokens")
        return ["a", "b"]

    # _make_expander calls the picker through the shared llm module, so patch
    # it where it lives
    monkeypatch.setattr(llm_mod, "openai_pick_model", lambda *a, **k: "m")
    monkeypatch.setattr(probe_mod, "_expand_openai", fake_expand)
    expand, label = probe_mod._make_expander(
        "openai", "http://o", "", "orm", "am", None, llm_host="http://h"
    )
    assert label == "openai:m"
    assert "rambles" in capsys.readouterr().out
    assert expand("grief", 2) == ["a", "b"]  # the real calls still work


def test_a_generator_that_cannot_generate_is_still_retired(monkeypatch):
    # _make_expander calls the picker through the shared llm module, so patch
    # it where it lives
    monkeypatch.setattr(llm_mod, "openai_pick_model", lambda *a, **k: "m")
    monkeypatch.setattr(
        probe_mod,
        "_expand_openai",
        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("refused")),
    )
    with pytest.raises(RuntimeError, match="no expansion backend available"):
        probe_mod._make_expander(
            "openai", "http://o", "", "orm", "am", None, llm_host="http://h"
        )


def test_the_availability_probe_uses_a_concrete_term(monkeypatch):
    # "test" gives a reasoning model nothing to expand and it spirals
    seen: list[str] = []
    # _make_expander calls the picker through the shared llm module, so patch
    # it where it lives
    monkeypatch.setattr(llm_mod, "openai_pick_model", lambda *a, **k: "m")
    monkeypatch.setattr(
        probe_mod,
        "_expand_openai",
        lambda term, *a, **k: (seen.append(term), ["a", "b"])[1],
    )
    probe_mod._make_expander(
        "openai", "http://o", "", "orm", "am", None, llm_host="http://h"
    )
    assert seen == [probe_mod._PROBE_TERM] and probe_mod._PROBE_TERM != "test"


# --- generator identity ----------------------------------------------------
# The generator is half of what a probe cloud measures: a cloud attributed to
# Glimmer but actually grown by whatever local model happened to be up is a
# fabrication about what Glimmer associates with the seed. So the same rule the
# namer follows applies here — pin means pin, and `auto` must stamp what ran.


@pytest.fixture(autouse=True)
def _no_probe_network(monkeypatch, tmp_path):
    """Keep a real HF token on the developer's box from turning a
    'nothing serves it' test into a live call."""
    for var in ("OPENROUTER_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm_mod, "HF_TOKEN_FILE", str(tmp_path / "absent-token"))
    monkeypatch.setattr(llm_mod, "DEFAULT_ENV_FILE", str(tmp_path / "absent.env"))


def test_a_pinned_generator_no_backend_serves_refuses(monkeypatch):
    """Every backend is up — for a different model — which is exactly when the
    fall-through chain would grow a cloud and mislabel whose it is."""
    ran: list[str] = []
    monkeypatch.setattr(llm_mod, "ollama_tags", lambda host: ["llama3.2:3b"])
    monkeypatch.setattr(
        llm_mod, "openai_list_models", lambda host, key=None: ["Qwen3.6-35B"]
    )
    monkeypatch.setattr(
        probe_mod,
        "_expand_ollama",
        lambda *a, **k: (ran.append("ollama"), ["a", "b"])[1],
    )
    monkeypatch.setattr(
        probe_mod,
        "_expand_openai",
        lambda *a, **k: (ran.append("openai"), ["a", "b"])[1],
    )

    with pytest.raises(llm_mod.IdentityError) as exc:
        probe_mod._make_expander(
            "auto",
            "http://o",
            "",
            "orm",
            "am",
            None,
            llm_host="http://h",
            model="muse-glimmer-30b",
        )
    msg = str(exc.value)
    assert "muse-glimmer-30b" in msg
    assert "llama3.2:3b" in msg and "Qwen3.6-35B" in msg
    assert ran == [], f"a different model grew concepts: {ran}"


def test_a_pinned_generator_that_is_served_runs_and_is_stamped(monkeypatch):
    """The positive control for the refusal above."""
    monkeypatch.setattr(llm_mod, "ollama_tags", lambda host: [])
    monkeypatch.setattr(
        llm_mod,
        "openai_list_models",
        lambda host, key=None: ["meta-models/Muse-Glimmer-30B"],
    )
    asked: list[str] = []
    monkeypatch.setattr(
        probe_mod,
        "_expand_openai",
        lambda term, n, host, model, key: (asked.append(model), ["a", "b"])[1],
    )

    stamp: dict = {}
    _, label = probe_mod._make_expander(
        "auto",
        "http://o",
        "",
        "orm",
        "am",
        None,
        llm_host="http://h",
        model="meta/muse-glimmer-30b",  # the OpenRouter spelling of the same model
        stamp=stamp,
    )
    assert label == "openai:meta-models/Muse-Glimmer-30B"
    assert asked == ["meta-models/Muse-Glimmer-30B"]
    assert stamp["identity"] == "pinned"
    assert stamp["model"] == "meta-models/Muse-Glimmer-30B"
    assert stamp["backend"] == "openai"


def test_the_probe_cost_gate_refuses_a_big_expansion_without_downgrading(monkeypatch):
    """A depth-3 breadth-20 probe is 1 + 20 + 400 BFS calls plus the one
    throwaway availability probe = 422. The gate must be able to trip on that,
    and must not quietly pick the free model instead."""
    monkeypatch.setattr(llm_mod, "ollama_tags", lambda host: [])
    monkeypatch.setattr(llm_mod, "openai_list_models", lambda host, key=None: [])

    assert probe_mod.expansion_calls(3, 20) == 1 + 20 + 400 + 1
    with pytest.raises(llm_mod.BudgetError) as exc:
        probe_mod._make_expander(
            "openrouter",
            "http://o",
            "",
            "orm",
            "am",
            None,
            model="muse-glimmer-30b",
            max_cost_usd=0.10,
            depth=3,
            breadth=20,
        )
    msg = str(exc.value)
    assert "$0.10 ceiling" in msg
    assert "NOT selected" in msg and "gemma-4-26b" in msg


def test_a_free_generator_is_never_gated(monkeypatch):
    """Gemma-4 is $0.00, so no expansion size can put it over any ceiling."""
    seen: list[str] = []
    monkeypatch.setattr(
        probe_mod,
        "_expand_openrouter",
        lambda term, n, model, env, **k: (seen.append(model), ["a", "b"])[1],
    )
    stamp: dict = {}
    _, label = probe_mod._make_expander(
        "openrouter",
        "http://o",
        "",
        "google/gemma-4-26b-a4b-it:free",
        "am",
        None,
        max_cost_usd=0.0,
        depth=4,
        breadth=30,
        stamp=stamp,
    )
    assert label == "openrouter:google/gemma-4-26b-a4b-it:free"
    assert seen == ["google/gemma-4-26b-a4b-it:free"]
    assert stamp["identity"] == "auto"


def test_the_cloud_stamps_the_resolved_generator_identity(monkeypatch, fake_stack):
    """meta must carry the exact model id, not only the 'backend:model' blob."""
    fake_stack(sim_for=lambda t: 0.9)

    def expander(*a, **k):
        if k.get("stamp") is not None:
            k["stamp"].update(
                {
                    "backend": "hf",
                    "model": "meta-models/Muse-Glimmer-30B",
                    "identity": "pinned",
                    "cost_usd": 0.00042,
                }
            )
        return (lambda t, n: [f"c-{t}-{i}" for i in range(n)], "hf:meta-models/Muse-Glimmer-30B")

    monkeypatch.setattr(probe_mod, "_make_expander", expander)
    units = load_probe_units("grief", depth=1, breadth=4, sensitivity=0.0)
    m = units.meta
    assert m["generator"] == "hf:meta-models/Muse-Glimmer-30B"
    assert m["generator_backend"] == "hf"
    assert m["generator_model"] == "meta-models/Muse-Glimmer-30B"
    assert m["generator_identity"] == "pinned"
    assert m["generator_cost_usd"] == 0.00042


def test_reused_terms_claim_no_generator_identity(monkeypatch, fake_stack):
    """A rebuild from an exported term list ran no generator, so it must not
    inherit or invent one."""
    fake_stack(sim_for=lambda t: 0.9)
    units = load_probe_units(
        "grief",
        sensitivity=0.0,
        reuse_terms=["grief", "mourning", "loss"],
        reused_from="probe__grief",
    )
    assert units.meta["generator"] == "reused:probe__grief"
    assert units.meta["generator_identity"] == "none"
    assert units.meta["generator_model"] == ""


def test_the_hf_expander_posts_to_the_router_with_the_shared_prompt(
    monkeypatch, tmp_path
):
    tok = tmp_path / "token"
    tok.write_text("hf_xyz\n")
    monkeypatch.setattr(llm_mod, "HF_TOKEN_FILE", str(tok))
    seen: dict = {}

    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._p

    def fake(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return _Resp(
            json.dumps(
                {
                    "model": "meta-models/Muse-Glimmer-30B",
                    "choices": [
                        {"message": {"content": json.dumps({"concepts": ["tide", "reef"]})}}
                    ],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 30},
                }
            ).encode()
        )

    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", fake)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    got = probe_mod._expand_hf(
        "ocean",
        5,
        "meta-models/Muse-Glimmer-30B",
        None,
        expect_model="meta-models/Muse-Glimmer-30B",
        usage=usage,
    )

    assert got == ["tide", "reef"]
    assert seen["url"] == "https://router.huggingface.co/v1/chat/completions"
    assert seen["auth"] == "Bearer hf_xyz"
    assert seen["body"]["model"] == "meta-models/Muse-Glimmer-30B"
    # the same system prompt every other generator gets — a transport-specific
    # prompt would make a cross-generator comparison meaningless
    assert seen["body"]["messages"][0]["content"] == probe_mod._SYSTEM
    assert usage == {"prompt_tokens": 120, "completion_tokens": 30}


def test_a_router_serving_another_model_is_rejected_not_stamped(monkeypatch, tmp_path):
    tok = tmp_path / "token"
    tok.write_text("hf_xyz\n")
    monkeypatch.setattr(llm_mod, "HF_TOKEN_FILE", str(tok))

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {
                    "model": "google/gemma-4-12b-it",
                    "choices": [
                        {"message": {"content": json.dumps({"concepts": ["tide"]})}}
                    ],
                }
            ).encode()

    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", lambda *a, **k: _Resp())
    with pytest.raises(llm_mod.IdentityError, match="answered as"):
        probe_mod._expand_hf(
            "ocean",
            5,
            "meta-models/Muse-Glimmer-30B",
            None,
            expect_model="meta-models/Muse-Glimmer-30B",
        )
