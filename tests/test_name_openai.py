"""OpenAI-compatible namer backend — reply parsing, model picking, truncation.

Fully offline: urlopen is monkeypatched, so nothing here touches the network.
The properties worth protecting are the three that produced real, confusing
failures against an MLX server on the LAN:

  1. a reasoning model spends max_tokens on its scratchpad before emitting any
     content, so a tight budget truncates mid-fence and surfaces as a JSON
     syntax error that says nothing about the real cause;
  2. /v1/models on a multi-purpose box lists rerankers, whisper, TTS, embedders
     and a diffusion checkpoint, and auto-picking one of those as the "chat"
     model fails much later and much more obscurely;
  3. a hallucinated cluster id must not invent a cluster that HDBSCAN never
     produced.
"""

import io
import json

import pytest

from nebulai.backend import name as name_mod


def _reps(n: int) -> dict[int, list[str]]:
    return {i: [f"tok{i}a", f"tok{i}b"] for i in range(n)}


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _completion(content: str, finish_reason: str = "stop") -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ]
        }
    ).encode()


# --- json_object ----------------------------------------------------------


def test_json_object_parses_bare_json():
    assert name_mod.json_object('{"titles": []}') == {"titles": []}


def test_json_object_strips_code_fences():
    raw = '```json\n{"concepts": ["a", "b"]}\n```'
    assert name_mod.json_object(raw) == {"concepts": ["a", "b"]}


def test_json_object_recovers_object_after_prose():
    raw = 'Sure! Here you go:\n{"concepts": ["a"]}\nHope that helps.'
    assert name_mod.json_object(raw) == {"concepts": ["a"]}


def test_json_object_ignores_braces_inside_strings():
    raw = '{"title": "a } brace", "n": 1}'
    assert name_mod.json_object(raw) == {"title": "a } brace", "n": 1}


def test_json_object_rejects_a_reply_with_no_object():
    # the exact shape of a budget-truncated reasoning reply
    with pytest.raises(ValueError):
        name_mod.json_object("```json")


# --- model picking --------------------------------------------------------


def _fake_models(monkeypatch, ids: list[str]) -> None:
    payload = json.dumps({"data": [{"id": i} for i in ids]}).encode()
    monkeypatch.setattr(
        name_mod.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(payload),
    )


def test_pick_model_prefers_an_exact_id(monkeypatch):
    _fake_models(monkeypatch, ["chat-a", "chat-b"])
    assert name_mod._openai_pick_model("http://h", "chat-b") == "chat-b"


def test_pick_model_matches_a_fragment(monkeypatch):
    _fake_models(monkeypatch, ["Qwen3.6-35B-Instruct-MLX", "other"])
    assert (
        name_mod._openai_pick_model("http://h", "qwen") == "Qwen3.6-35B-Instruct-MLX"
    )


def test_pick_model_never_auto_picks_a_non_chat_model(monkeypatch):
    # the real /v1/models listing from the M4, chat model last
    _fake_models(
        monkeypatch,
        [
            "Qwen3-Reranker-8B-mxfp8",
            "whisper-large-v3",
            "nomic-embed-text-v1.5",
            "Qwen3-TTS-12Hz-0.6B-Base-8bit",
            "models--Lakonik-AsymFLUX.2-klein-9B",
            "nsfw_image_detection",
            "Qwen3.6-35B-A3B-Reasoning",
        ],
    )
    assert name_mod._openai_pick_model("http://h", "") == "Qwen3.6-35B-A3B-Reasoning"


def test_pick_model_returns_none_when_only_non_chat_models_exist(monkeypatch):
    _fake_models(monkeypatch, ["all-MiniLM-L6-v2", "whisper-large-v3"])
    assert name_mod._openai_pick_model("http://h", "") is None


def test_pick_model_returns_none_when_unreachable(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(name_mod.urllib.request, "urlopen", boom)
    assert name_mod._openai_pick_model("http://h", "") is None


# --- truncation retry -----------------------------------------------------


def test_a_length_finish_is_retried_with_a_doubled_budget(monkeypatch):
    seen: list[int] = []

    def fake(req, timeout=None):
        seen.append(json.loads(req.data)["max_tokens"])
        if len(seen) == 1:
            return _FakeResponse(_completion("```json", finish_reason="length"))
        return _FakeResponse(_completion('{"titles": [{"id": 0, "title": "ok"}]}'))

    monkeypatch.setattr(name_mod.urllib.request, "urlopen", fake)
    got = name_mod._chat_openai(
        "http://h", "m", "sys", "usr", name_mod._SCHEMA, "cluster_titles"
    )
    assert got["titles"][0]["title"] == "ok"
    assert seen[1] == seen[0] * 2


def test_truncation_escalates_twice_then_names_the_real_cause(monkeypatch):
    seen: list[int] = []

    def fake(req, timeout=None):
        seen.append(json.loads(req.data)["max_tokens"])
        return _FakeResponse(_completion("```json", finish_reason="length"))

    monkeypatch.setattr(name_mod.urllib.request, "urlopen", fake)
    with pytest.raises(name_mod.ChatTruncated, match="truncated"):
        name_mod._chat_openai(
            "http://h", "m", "sys", "usr", name_mod._SCHEMA, "cluster_titles"
        )
    # a distilled reasoner burned 8192 on one ordinary expansion, so the ladder
    # has to reach past that before giving up
    assert seen == [4096, 8192, 16384]


def test_chat_truncated_is_still_a_runtime_error():
    # callers that only want "the chat failed" must keep working unchanged
    assert issubclass(name_mod.ChatTruncated, RuntimeError)


def test_no_request_is_budgeted_below_the_reasoning_floor(monkeypatch):
    seen: list[int] = []

    def fake(req, timeout=None):
        seen.append(json.loads(req.data)["max_tokens"])
        return _FakeResponse(_completion('{"titles": []}'))

    monkeypatch.setattr(name_mod.urllib.request, "urlopen", fake)
    name_mod._chat_openai(
        "http://h", "m", "sys", "usr", name_mod._SCHEMA, "cluster_titles", max_tokens=10
    )
    assert seen == [name_mod._MIN_CHAT_TOKENS]


# --- naming ---------------------------------------------------------------


def test_names_every_cluster_across_several_batches(monkeypatch):
    def fake(req, timeout=None):
        body = json.loads(req.data)
        asked = [
            int(line.split()[1].rstrip(":"))
            for line in body["messages"][1]["content"].splitlines()
            if line.startswith("cluster ")
        ]
        return _FakeResponse(
            _completion(
                json.dumps({"titles": [{"id": c, "title": f"t{c}"} for c in asked]})
            )
        )

    monkeypatch.setattr(name_mod.urllib.request, "urlopen", fake)
    titles = name_mod._name_with_openai(_reps(45), "http://h", "m", batch_size=20)
    assert titles == {i: f"t{i}" for i in range(45)}


def test_a_hallucinated_cluster_id_is_discarded(monkeypatch):
    monkeypatch.setattr(
        name_mod.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(
            _completion(
                json.dumps(
                    {
                        "titles": [
                            {"id": 0, "title": "real"},
                            {"id": 1, "title": "real"},
                            {"id": 99, "title": "invented"},
                        ]
                    }
                )
            )
        ),
    )
    titles = name_mod._name_with_openai(_reps(2), "http://h", "m")
    assert titles == {0: "real", 1: "real"}


def test_a_short_reply_fails_loudly_rather_than_naming_half_the_map(monkeypatch):
    """A map where some clusters silently export an empty title looks fully
    named — the failure has to surface here instead."""
    monkeypatch.setattr(
        name_mod.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(
            _completion(json.dumps({"titles": [{"id": 0, "title": "only-one"}]}))
        ),
    )
    with pytest.raises(RuntimeError, match="named 1/3"):
        name_mod._name_with_openai(_reps(3), "http://h", "m")
