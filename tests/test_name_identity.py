"""Model identity and the cost gate — the two rules that stop a namer from
quietly becoming a different namer.

Fully offline. An autouse fixture replaces urlopen with a function that RAISES,
so a test which forgets to install its own transport fails here instead of
billing a real endpoint; each test then opts back in with a fake of its own.

The properties worth protecting are the ones that produce a map which looks
perfectly plausible and is wrong about what it is evidence of:

  1. a pinned model that no backend serves must REFUSE, not reach the nearest
     reachable model — a cheaper endpoint serving a different model is a
     different model, and its titles are not the pinned model's semantics;
  2. `auto` may still fall through (the caller pinned nothing) but must stamp
     which model actually answered, or the two cases are indistinguishable
     afterwards;
  3. over budget the gate refuses and LISTS cheaper models without selecting
     one — auto-downgrading to fit a ceiling is the same substitution bug
     arriving from the money side.
"""

import io
import json

import numpy as np
import pytest

from nebulai import llm as llm_mod
from nebulai.backend import name as name_mod
from nebulai.units import Units


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _units(n_clusters: int) -> tuple[Units, np.ndarray]:
    """n_clusters clusters of 2 members each."""
    n = n_clusters * 2
    units = Units(
        ids=list(range(n)),
        vectors=np.zeros((n, 3), dtype=np.float32),
        labels=[f"tok{i}" for i in range(n)],
        meta={"model": "m", "unit": "token_embedding"},
    )
    return units, np.repeat(np.arange(n_clusters), 2)


def _completion(titles: list[dict], model: str = "", usage: dict | None = None) -> bytes:
    payload: dict = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"titles": titles}),
                },
            }
        ]
    }
    if model:
        payload["model"] = model
    if usage is not None:
        payload["usage"] = usage
    return json.dumps(payload).encode()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch, tmp_path):
    """No test in this file may reach the network, and none may pick up a real
    credential from the developer's machine — this box has a live HF token in
    ~/.cache/huggingface/token, which would otherwise turn a 'no backend can
    serve it' test into a live paid call."""

    def forbidden(req, timeout=None):
        url = getattr(req, "full_url", req)
        raise AssertionError(f"test reached the network: {url}")

    monkeypatch.setattr(name_mod.urllib.request, "urlopen", forbidden)
    for var in ("OPENROUTER_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm_mod, "HF_TOKEN_FILE", str(tmp_path / "absent-token"))
    monkeypatch.setattr(llm_mod, "DEFAULT_ENV_FILE", str(tmp_path / "absent.env"))


# --- what counts as the same model ----------------------------------------


def test_same_model_accepts_case_and_an_ollama_tag_but_nothing_else():
    # measured: the HF router serves `google/gemma-4-26B-A4B-it` while the
    # corpus writes it lowercase — one repo, so a case-exact compare would
    # refuse a model that IS being served
    assert llm_mod.same_model("google/gemma-4-26B-A4B-it", "google/gemma-4-26b-a4b-it")
    # a quantisation is a build of the model, not another model
    assert llm_mod.same_model("mistral-nemo:q4_K_M", "mistral-nemo")
    # and these are the substitutions a pin exists to forbid
    assert not llm_mod.same_model("qwen3.6-35b-instruct", "qwen")
    assert not llm_mod.same_model("google/gemma-4-12b-it", "google/gemma-4-26b-a4b-it")
    assert not llm_mod.same_model("", "anything")


def test_a_corpus_model_is_recognised_under_every_one_of_its_spellings():
    """A pin written as the OpenRouter slug must match a backend serving the HF
    repo — they are one model, and treating them as two would refuse a model
    that is in fact available."""
    for spelling in (
        "muse-glimmer-30b",
        "meta-models/Muse-Glimmer-30B",
        "meta/muse-glimmer-30b",
    ):
        assert llm_mod.corpus_entry(spelling).key == "muse-glimmer-30b"
    assert llm_mod.corpus_entry("some/model-nobody-mapped") is None


# --- pinning refuses rather than substituting ------------------------------


def test_a_pin_no_backend_serves_refuses_and_never_names_with_another_model(
    monkeypatch,
):
    """The headline case. Every backend is up and answering — for a DIFFERENT
    model — which is exactly when a fall-through chain would produce a map
    labelled Glimmer and titled by something else."""
    answered: list[str] = []

    monkeypatch.setattr(llm_mod, "ollama_tags", lambda host: ["llama3.2:3b"])
    monkeypatch.setattr(
        llm_mod, "openai_list_models", lambda host, key=None: ["Qwen3.6-35B-A3B"]
    )
    # both would have happily produced titles if they were ever reached
    monkeypatch.setattr(
        name_mod,
        "_name_with_ollama",
        lambda *a, **k: (answered.append("ollama"), {0: "x", 1: "y"})[1],
    )
    monkeypatch.setattr(
        name_mod,
        "_name_with_openai",
        lambda *a, **k: (answered.append("openai"), {0: "x", 1: "y"})[1],
    )

    units, cids = _units(2)
    with pytest.raises(name_mod.NamerIdentityError) as exc:
        name_mod.name_clusters(units, cids, namer="auto", model="muse-glimmer-30b")

    msg = str(exc.value)
    assert "muse-glimmer-30b" in msg
    # it has to say what it tried and why, not just that it failed
    for backend in ("ollama", "openai", "hf", "openrouter"):
        assert backend in msg, f"{backend} missing from the refusal"
    assert "llama3.2:3b" in msg and "Qwen3.6-35B-A3B" in msg
    assert answered == [], f"a different model was asked for titles: {answered}"
    # and nothing may claim an identity the map does not have
    assert "namer_model" not in units.meta


def test_a_pin_does_not_fall_back_to_centroid_either():
    """centroid is not the pinned model — it is four token strings joined by a
    dot. Offering it under a pin would satisfy the chain and violate the ask."""
    units, cids = _units(2)
    with pytest.raises(name_mod.NamerIdentityError):
        name_mod.name_clusters(units, cids, namer="openrouter", model="muse-glimmer-30b")
    assert units.meta.get("namer_backend") != "centroid"


def test_hf_refuses_a_corpus_model_no_provider_serves(monkeypatch, tmp_path):
    """Ling's corpus row has hf_endpoint=None. That is a refusal, not an
    invitation to route the request to whatever the provider does serve."""
    tok = tmp_path / "token"
    tok.write_text("hf_test\n")
    monkeypatch.setattr(llm_mod, "HF_TOKEN_FILE", str(tok))

    units, cids = _units(2)
    with pytest.raises(name_mod.NamerIdentityError) as exc:
        name_mod.name_clusters(units, cids, namer="hf", model="ling-2.6-flash")
    assert "no HF inference provider serves ling-2.6-flash" in str(exc.value)


def test_a_pin_the_local_server_does_serve_runs_and_stamps_the_served_id(
    monkeypatch,
):
    """The positive control: the same pin, on a box that has it, must actually
    run — otherwise the refusal above proves nothing."""
    monkeypatch.setattr(llm_mod, "ollama_tags", lambda host: ["nope:1b"])
    monkeypatch.setattr(
        llm_mod,
        "openai_list_models",
        lambda host, key=None: ["meta-models/Muse-Glimmer-30B"],
    )
    seen: dict = {}

    def fake_name(reps, host, model, api_key=None, batch_size=20):
        seen["model"] = model
        return {cid: f"title-{cid}" for cid in reps}

    monkeypatch.setattr(name_mod, "_name_with_openai", fake_name)

    units, cids = _units(3)
    titles, label = name_mod.name_clusters(
        units, cids, namer="auto", model="meta/muse-glimmer-30b"
    )

    assert titles == {0: "title-0", 1: "title-1", 2: "title-2"}
    assert label == "openai:meta-models/Muse-Glimmer-30B"
    # the pin was written as the OpenRouter slug; the id actually sent is the
    # one the server advertises
    assert seen["model"] == "meta-models/Muse-Glimmer-30B"
    assert units.meta["namer_backend"] == "openai"
    assert units.meta["namer_model"] == "meta-models/Muse-Glimmer-30B"
    assert units.meta["namer_identity"] == "pinned"


def test_a_pinned_ollama_build_stamps_the_TAG_that_answered(monkeypatch):
    """`mistral-nemo:q4_K_M` is the model, at a quantisation. The quantisation
    is part of what produced the titles, so the tag is what gets recorded — not
    the tidier string the human typed."""
    monkeypatch.setattr(llm_mod, "ollama_tags", lambda host: ["mistral-nemo:q4_K_M"])
    monkeypatch.setattr(
        name_mod, "_name_with_ollama", lambda reps, h, m: {cid: "t" for cid in reps}
    )

    units, cids = _units(2)
    _, label = name_mod.name_clusters(units, cids, namer="ollama", model="mistral-nemo")
    assert label == "ollama:mistral-nemo:q4_K_M"
    assert units.meta["namer_model"] == "mistral-nemo:q4_K_M"


# --- auto keeps falling through, but says what answered --------------------


def test_auto_still_falls_through_and_stamps_the_identity_that_answered(
    monkeypatch, capsys
):
    monkeypatch.setattr(llm_mod, "ollama_tags", lambda host: [])
    monkeypatch.setattr(
        llm_mod, "openai_list_models", lambda host, key=None: ["Qwen3.6-35B-A3B"]
    )
    monkeypatch.setattr(
        name_mod,
        "_name_with_openai",
        lambda reps, *a, **k: {cid: f"t{cid}" for cid in reps},
    )

    units, cids = _units(3)
    titles, label = name_mod.name_clusters(units, cids, namer="auto")

    assert titles == {0: "t0", 1: "t1", 2: "t2"}
    assert label == "openai:Qwen3.6-35B-A3B"
    assert "namer 'ollama' unavailable" in capsys.readouterr().out
    assert units.meta["namer_identity"] == "auto"
    assert units.meta["namer_backend"] == "openai"
    assert units.meta["namer_model"] == "Qwen3.6-35B-A3B"


def test_even_the_centroid_fallback_stamps_an_identity(monkeypatch):
    """A map titled by no model at all must say so in the same field a real
    namer would have filled — an absent stamp reads as 'nobody checked'."""
    monkeypatch.setattr(llm_mod, "ollama_tags", lambda host: [])
    monkeypatch.setattr(llm_mod, "openai_list_models", lambda host, key=None: [])

    units, cids = _units(2)
    _, label = name_mod.name_clusters(units, cids, namer="auto")
    assert label == "centroid"
    assert units.meta["namer_backend"] == "centroid"
    assert units.meta["namer_model"] == "centroid"
    assert units.meta["namer_identity"] == "auto"
    assert units.meta["namer_cost_usd"] is None


# --- the cost gate ---------------------------------------------------------


def test_the_gate_passes_at_realistic_sizes():
    """Measured reference points: a 250-cluster map on Glimmer is ~$0.019 and a
    17-map corpus re-name ~$0.33, both well under the $1.00 default. A gate that
    tripped here would be unusable, so pin the numbers."""
    assert llm_mod.cost_gate("meta/muse-glimmer-30b", 250, 1.00) == pytest.approx(
        0.0191, abs=5e-4
    )
    assert llm_mod.cost_gate("meta/muse-glimmer-30b", 4250, 1.00) == pytest.approx(
        0.3195, abs=5e-4
    )


def test_a_free_endpoint_skips_the_gate_entirely():
    """Gemma-4 is the only genuinely $0 corpus endpoint. A million clusters at a
    $0.00 ceiling still passes, because zero times anything is zero."""
    assert llm_mod.cost_gate("google/gemma-4-26b-a4b-it:free", 1_000_000, 0.0) == 0.0
    # recognised through the HF spelling and its casing too
    assert llm_mod.cost_gate("google/gemma-4-26B-A4B-it", 1_000_000, 0.0) == 0.0


def test_the_gate_refuses_over_budget_and_names_alternatives_without_picking_one():
    with pytest.raises(llm_mod.BudgetError) as exc:
        llm_mod.cost_gate("meta/muse-glimmer-30b", 20_000, 1.00)
    msg = str(exc.value)

    assert "$1.5008" in msg and "$1.00 ceiling" in msg
    # the numbers behind the estimate, so the human can check it
    assert "1334 batches x 15 clusters" in msg
    assert "$0.35/M in and $1.5/M out" in msg
    # cheaper models are OFFERED, and the message says why they were not taken
    assert "gemma-4-26b" in msg and "ling-2.6-flash" in msg
    assert "NOT selected" in msg
    # a more expensive corpus model is not proposed as an "alternative"
    assert "muse-glimmer-30b" not in msg.split("NOT selected")[1]


def test_over_budget_refuses_the_run_instead_of_downgrading_or_falling_back(
    monkeypatch,
):
    """End to end: the gate must be terminal. Falling through to centroid would
    hide the refusal; falling through to a cheaper model would BE the bug."""
    monkeypatch.setattr(llm_mod, "load_openrouter_key", lambda env: "sk-test")
    units, cids = _units(3)

    with pytest.raises(llm_mod.BudgetError):
        name_mod.name_clusters(
            units,
            cids,
            namer="openrouter",
            openrouter_model="meta/muse-glimmer-30b",
            max_cost_usd=0.001,  # 3 clusters = 1 batch = $0.001125
        )
    # the autouse fixture would have raised on any request, and no fallback ran
    assert units.meta == {"model": "m", "unit": "token_embedding"}


def test_an_unpriceable_model_is_flagged_rather_than_silently_gated(capsys):
    """Most OpenRouter slugs have no corpus row, so the ceiling genuinely cannot
    be enforced for them. Saying so beats both blocking them and pretending."""
    assert llm_mod.cost_gate("some/unmapped-model", 500, 1.00) is None
    assert "no corpus price" in capsys.readouterr().out


# --- the HF Inference Providers backend ------------------------------------


def test_the_hf_backend_builds_the_router_request(monkeypatch, tmp_path):
    tok = tmp_path / "token"
    tok.write_text("hf_abc123\n")
    monkeypatch.setattr(llm_mod, "HF_TOKEN_FILE", str(tok))
    seen: dict = {}

    def fake(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return _FakeResponse(
            _completion([{"id": 0, "title": "animals"}, {"id": 1, "title": "colours"}])
        )

    monkeypatch.setattr(name_mod.urllib.request, "urlopen", fake)
    titles = name_mod._name_with_hf(
        {0: ["cat", "dog"], 1: ["red", "blue"]},
        "meta-models/Muse-Glimmer-30B",
        None,
    )

    assert seen["url"] == "https://router.huggingface.co/v1/chat/completions"
    assert seen["auth"] == "Bearer hf_abc123"  # from the CLI's own token file
    assert seen["body"]["model"] == "meta-models/Muse-Glimmer-30B"
    # same structured-output contract as the OpenRouter leg, not a looser one
    assert seen["body"]["response_format"]["json_schema"]["schema"] == name_mod._SCHEMA
    assert seen["body"]["response_format"]["json_schema"]["strict"] is True
    assert titles == {0: "animals", 1: "colours"}


def test_hf_and_openrouter_send_the_identical_body(monkeypatch, tmp_path):
    """They differ only in URL and auth. If the prompts drift, a cross-endpoint
    comparison stops being a comparison of models."""
    tok = tmp_path / "token"
    tok.write_text("hf_abc\n")
    monkeypatch.setattr(llm_mod, "HF_TOKEN_FILE", str(tok))
    monkeypatch.setattr(llm_mod, "load_openrouter_key", lambda env: "sk-test")
    bodies: list[dict] = []

    def fake(req, timeout=None):
        bodies.append(json.loads(req.data))
        return _FakeResponse(_completion([{"id": 0, "title": "a"}]))

    monkeypatch.setattr(name_mod.urllib.request, "urlopen", fake)
    reps = {0: ["cat", "dog"]}
    name_mod._name_with_hf(reps, "same/model", None)
    name_mod._name_with_openrouter(reps, "same/model", None)
    assert bodies[0] == bodies[1]


def test_a_router_that_answers_as_a_different_model_is_rejected(monkeypatch, tmp_path):
    """Both routers echo the model they ran. A provider quietly serving a
    neighbour would otherwise be invisible — and its titles would be exported as
    the pinned model's."""
    tok = tmp_path / "token"
    tok.write_text("hf_abc\n")
    monkeypatch.setattr(llm_mod, "HF_TOKEN_FILE", str(tok))
    monkeypatch.setattr(
        name_mod.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(
            _completion([{"id": 0, "title": "a"}], model="google/gemma-4-12b-it")
        ),
    )
    with pytest.raises(name_mod.NamerIdentityError, match="answered as"):
        name_mod._name_with_hf(
            {0: ["cat"]},
            "meta-models/Muse-Glimmer-30B",
            None,
            expect_model="meta-models/Muse-Glimmer-30B",
        )


# --- actual cost, not the estimate -----------------------------------------


def test_actual_cost_is_priced_from_the_reported_usage():
    usage = {"prompt_tokens": 1500, "completion_tokens": 400}
    # 1500 * $0.35/M + 400 * $1.50/M
    assert llm_mod.actual_cost("meta/muse-glimmer-30b", usage) == pytest.approx(
        0.001125
    )
    # a provider's own number wins over our arithmetic when it reports one
    assert llm_mod.actual_cost("meta/muse-glimmer-30b", {**usage, "cost": 0.002}) == (
        0.002
    )
    # unpriceable stays honest rather than defaulting to zero
    assert llm_mod.actual_cost("some/unmapped-model", usage) is None
    assert llm_mod.actual_cost("meta/muse-glimmer-30b", {}) is None


def test_the_stamped_cost_is_the_measured_one_not_the_estimate(monkeypatch, tmp_path):
    """The estimate is an upper bound by construction, so stamping it would
    overstate every run. Here the endpoint reports a tenth of the estimate and
    the tenth is what lands in meta."""
    tok = tmp_path / "token"
    tok.write_text("hf_abc\n")
    monkeypatch.setattr(llm_mod, "HF_TOKEN_FILE", str(tok))
    monkeypatch.setattr(
        llm_mod, "openai_list_models", lambda host, key=None: []
    )
    monkeypatch.setattr(llm_mod, "ollama_tags", lambda host: [])

    def fake(req, timeout=None):
        asked = [
            int(line.split()[1].rstrip(":"))
            for line in json.loads(req.data)["messages"][1]["content"].splitlines()
            if line.startswith("cluster ")
        ]
        return _FakeResponse(
            _completion(
                [{"id": c, "title": f"t{c}"} for c in asked],
                model="meta-models/Muse-Glimmer-30B",
                usage={"prompt_tokens": 150, "completion_tokens": 40},
            )
        )

    monkeypatch.setattr(name_mod.urllib.request, "urlopen", fake)

    units, cids = _units(3)
    titles, label = name_mod.name_clusters(
        units, cids, namer="hf", model="muse-glimmer-30b"
    )

    assert titles == {0: "t0", 1: "t1", 2: "t2"}
    assert label == "hf:meta-models/Muse-Glimmer-30B"
    assert units.meta["namer_identity"] == "pinned"
    assert units.meta["namer_tokens"] == {"prompt": 150, "completion": 40}
    # 150 * 0.35/M + 40 * 1.50/M = 0.0001125, a tenth of the $0.001125 estimate
    assert units.meta["namer_cost_usd"] == pytest.approx(0.0001125)
    assert units.meta["namer_cost_usd"] < llm_mod.estimate_naming_cost(
        3, "muse-glimmer-30b"
    )


def test_usage_accumulates_across_batches(monkeypatch, tmp_path):
    tok = tmp_path / "token"
    tok.write_text("hf_abc\n")
    monkeypatch.setattr(llm_mod, "HF_TOKEN_FILE", str(tok))

    def fake(req, timeout=None):
        asked = [
            int(line.split()[1].rstrip(":"))
            for line in json.loads(req.data)["messages"][1]["content"].splitlines()
            if line.startswith("cluster ")
        ]
        return _FakeResponse(
            _completion(
                [{"id": c, "title": f"t{c}"} for c in asked],
                usage={"prompt_tokens": 100, "completion_tokens": 10},
            )
        )

    monkeypatch.setattr(name_mod.urllib.request, "urlopen", fake)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    reps = {i: [f"tok{i}"] for i in range(10)}
    titles = name_mod._name_with_hf(reps, "m", None, batch_size=4, usage=usage)

    assert len(titles) == 10
    assert usage == {"prompt_tokens": 300, "completion_tokens": 30}  # 3 batches


# --- the neutral embedder is under the same rule ---------------------------


def test_an_embedder_that_answers_as_a_different_model_is_rejected():
    """compare's whole premise is ONE neutral space. A multi-model server that
    ignores the `model` field and serves whatever it has loaded would produce a
    map stamped mxbai-embed-large and positioned by something else."""
    from nebulai.backend.embed import EmbedIdentityError, parse_embed_response

    payload = {"model": "nomic-embed-text-v1.5", "embeddings": [[0.1, 0.2]]}
    with pytest.raises(EmbedIdentityError, match="different embedder"):
        parse_embed_response(payload, "ollama", "http://h", "mxbai-embed-large")


def test_an_embedder_that_confirms_the_model_or_stays_silent_is_accepted():
    """A server that reports the right model passes; one that reports nothing
    is taken at its word, because refusing it would break every endpoint that
    simply does not echo the field."""
    from nebulai.backend.embed import parse_embed_response

    confirmed = {"model": "mxbai-embed-large:latest", "embeddings": [[0.1, 0.2]]}
    assert parse_embed_response(confirmed, "ollama", "h", "mxbai-embed-large") == [
        [0.1, 0.2]
    ]
    silent = {"embeddings": [[0.3]]}
    assert parse_embed_response(silent, "ollama", "h", "mxbai-embed-large") == [[0.3]]
