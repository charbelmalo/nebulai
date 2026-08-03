"""Anthropic Batch API naming — routing, out-of-order results, partial failure.

Fully offline: `anthropic.Anthropic` is monkeypatched, so nothing here touches
the network or needs a key. The property worth protecting is the one that
cannot be caught by eye later — batch results arrive in arbitrary order, and
attaching a chunk's titles to the wrong clusters would produce a map that looks
perfectly plausible and is wrong everywhere.
"""

import json
import types

import numpy as np
import pytest

from nebulai.backend import name as name_mod
from nebulai.units import Units


def _reps(n: int) -> dict[int, list[str]]:
    """n clusters, each with a member that encodes its own id."""
    return {i: [f"tok{i}a", f"tok{i}b"] for i in range(n)}


def _titles_payload(cids: list[int]) -> str:
    """What the model returns for a chunk: title carries the cluster id so a
    mis-keyed result is detectable."""
    return json.dumps({"titles": [{"id": c, "title": f"title-{c}"} for c in cids]})


def _fake_message(cids: list[int]):
    block = types.SimpleNamespace(type="text", text=_titles_payload(cids))
    return types.SimpleNamespace(content=[block])


class _FakeBatches:
    def __init__(self, order="forward", fail: set[str] | None = None):
        self.order = order
        self.fail = fail or set()
        self.submitted: list[dict] = []
        self._by_id: dict[str, list[int]] = {}

    def create(self, *, requests):
        self.submitted = list(requests)
        for r in requests:
            # recover the chunk's cluster ids from the rendered prompt, exactly
            # as the real API would only know them via custom_id
            self._by_id[r["custom_id"]] = r["_cids"]
        return types.SimpleNamespace(id="batch_test", processing_status="ended")

    def retrieve(self, _id):
        return types.SimpleNamespace(id=_id, processing_status="ended")

    def results(self, _id):
        items = list(self._by_id.items())
        if self.order == "reversed":
            items = items[::-1]
        for custom_id, cids in items:
            if custom_id in self.fail:
                yield types.SimpleNamespace(
                    custom_id=custom_id,
                    result=types.SimpleNamespace(type="errored"),
                )
            else:
                yield types.SimpleNamespace(
                    custom_id=custom_id,
                    result=types.SimpleNamespace(
                        type="succeeded", message=_fake_message(cids)
                    ),
                )


class _FakeClient:
    def __init__(self, batches):
        self.messages = types.SimpleNamespace(batches=batches, create=self._create)
        self.sync_calls = 0

    def _create(self, **params):
        self.sync_calls += 1
        # echo back titles for whatever cluster ids the prompt mentioned
        content = params["messages"][0]["content"]
        # mirrors _batch_lines: "cluster <id>: 'tok', 'tok'"
        cids = [
            int(line.split(":")[0].removeprefix("cluster ").strip())
            for line in content.splitlines()
            if line.startswith("cluster ")
        ]
        return _fake_message(cids)


@pytest.fixture
def patched(monkeypatch):
    """Install a fake Anthropic client and record the chunk->cids mapping the
    batch path builds, so results can be replayed in any order."""
    holder = {}

    def install(order="forward", fail=None):
        batches = _FakeBatches(order=order, fail=fail)
        client = _FakeClient(batches)
        holder["client"] = client

        real_chunk_params = name_mod._chunk_params

        def spy(reps, batch, model):
            p = real_chunk_params(reps, batch, model)
            p["_cids"] = list(batch)
            return p

        monkeypatch.setattr(name_mod, "_chunk_params", spy)

        # the batch path stores params under "params"; lift _cids up to the
        # request so the fake can key on custom_id like the real service does
        real_batch = name_mod._name_with_anthropic_batch

        def wrapper(reps, model, batch_size=15):
            return real_batch(reps, model, batch_size)

        monkeypatch.setattr(name_mod, "_name_with_anthropic_batch", wrapper)
        monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: client)
        monkeypatch.setattr(name_mod, "_BATCH_POLL_SECONDS", 0)
        return client

    holder["install"] = install
    return holder


def _run_batch(client, reps, batch_size=5):
    """Drive the batch path, moving _cids from params onto the request dict the
    way the fake service needs it."""
    orig_create = client.messages.batches.create

    def create(*, requests):
        for r in requests:
            r["_cids"] = r["params"].pop("_cids")
        return orig_create(requests=requests)

    client.messages.batches.create = create
    return name_mod._name_with_anthropic_batch(reps, "claude-opus-5", batch_size)


def test_batch_results_are_keyed_by_custom_id_not_position(patched):
    """The load-bearing test: replay the SAME results in reverse order and every
    cluster must still get its own title."""
    client = patched["install"](order="reversed")
    reps = _reps(20)
    titles = _run_batch(client, reps, batch_size=5)

    assert len(titles) == 20
    for cid in range(20):
        assert titles[cid] == f"title-{cid}", f"cluster {cid} got {titles[cid]!r}"


def test_batch_forward_order_matches_reversed(patched):
    client = patched["install"](order="forward")
    fwd = _run_batch(client, _reps(20), batch_size=5)
    client2 = patched["install"](order="reversed")
    rev = _run_batch(client2, _reps(20), batch_size=5)
    assert fwd == rev


def test_partial_batch_failure_returns_what_succeeded(patched):
    client = patched["install"](order="forward", fail={"chunk-000000"})
    titles = _run_batch(client, _reps(20), batch_size=5)
    # first chunk (clusters 0-4) dropped, the other 15 survive
    assert len(titles) == 15
    assert 0 not in titles
    assert titles[5] == "title-5"


def test_total_batch_failure_raises_so_the_chain_falls_back(patched):
    client = patched["install"](
        order="forward", fail={f"chunk-{i:06d}" for i in (0, 5, 10, 15)}
    )
    with pytest.raises(RuntimeError, match="every batch request failed"):
        _run_batch(client, _reps(20), batch_size=5)


def test_small_maps_use_the_synchronous_path(patched):
    client = patched["install"]()
    reps = _reps(10)  # well under _BATCH_API_MIN_CLUSTERS
    titles = name_mod._name_with_anthropic(reps, "claude-opus-5", batch_size=5)
    assert client.sync_calls == 2  # 10 clusters / 5 per chunk
    assert not client.messages.batches.submitted
    assert titles[3] == "title-3"


def test_partial_coverage_is_stamped_into_the_namer_label(monkeypatch):
    """A short result set must be visible in `meta.namer`, not silently exported
    as empty titles."""
    units = Units(
        ids=list(range(6)),
        vectors=np.zeros((6, 3), dtype=np.float32),
        labels=[f"t{i}" for i in range(6)],
        meta={"model": "m", "unit": "token_embedding"},
    )
    cluster_ids = np.array([0, 0, 1, 1, 2, 2])
    monkeypatch.setattr(
        name_mod, "_name_with_anthropic", lambda *a, **k: {0: "only one"}
    )
    titles, backend = name_mod.name_clusters(units, cluster_ids, namer="anthropic")
    assert titles == {0: "only one"}
    assert "partial:1/3" in backend
