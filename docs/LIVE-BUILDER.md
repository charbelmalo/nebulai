# The live semantic cloud builder

**What it is:** a panel in the viewer's Settings tab that runs the *real*
`nebulai` pipeline as a subprocess and hot-swaps the finished map into the scene
when it lands. Nothing about it is simulated — the progress bar is the
pipeline's own stage output, and the artifact it produces is byte-identical to
what you would get by running the CLI yourself.

It has three build sources:

| Source | Subcommand | Needs | What the geometry means |
|---|---|---|---|
| `hf` — model geometry | `nebulai tokens` | HF weights (downloaded once) | the model's own `W_E`, model-internal |
| `api` — API embeddings | `nebulai tokens --source api` | an embedding server | a *third-party* embedding space, **not** model-internal |
| `probe` — seed concept | `nebulai probe` | a chat LLM **and** an embedding server | the joint opinion of a generator and an embedder; **no model under study at all** |

The `probe` source is the one that answers "grow me a cloud about X" without
any model weights. It is also the one that was non-functional until the
OpenAI-compatible backend landed — see §5.

---

## 1. Start the build server

The viewer talks to a small stdlib HTTP server. It is **not** started by the
dev server; you start it yourself, from the repo root:

```bash
.venv/bin/python -m nebulai.backend.build_server --port 8124
```

Leave it running. Health check:

```bash
curl -s http://localhost:8124/build/models | python -m json.tool | head -20
```

You should see `sources` containing `hf`, `api`, **and** `probe`. If `probe` is
missing you are running an older build server — restart it.

The viewer finds it at the URL in **Settings → Build server**
(default `http://127.0.0.1:8124`). The health dot next to it must be green;
the Build button stays disabled while it is not.

---

## 2. Turn on the models (M4 Worker, `<m4-host>`)

`probe` needs **two** services, and neither is bundled. The M4 exposes them
through the LAN activation API on `:8100`. There is no SSH to `.200` — HTTP
only.

```bash
# what's running, and how much RAM is free
curl -s http://<m4-host>:8100/v1/status/opus
curl -s http://<m4-host>:8100/v1/status/embedding

# turn them on (activate is async — poll status until running:true)
curl -s -X POST http://<m4-host>:8100/v1/activate/opus
curl -s -X POST http://<m4-host>:8100/v1/activate/embedding
```

| Role | Service | Port | `need_gb` | Endpoint style |
|---|---|---|---|---|
| chat / generator / namer | `opus` | 8050 | 21.8 | OpenAI-compatible `/v1/chat/completions` |
| embedder | `embedding` | 8040 | 0.3 | OpenAI-compatible `/v1/embeddings` |

Confirm what each one actually serves before pointing the viewer at it — the
model ids are long and you need the embedding one exactly:

```bash
curl -s http://<m4-host>:8050/v1/models   # chat model id
curl -s http://<m4-host>:8040/v1/models   # -> sentence-transformers/all-MiniLM-L6-v2
```

**RAM is the real constraint.** The box has 48 GB and runs one heavy request at
a time. `opus` (21.8 GB) and `ollama` (23.5 GB) are mutually exclusive — you
cannot have both. Check `free_gb` before activating anything large, and
deactivate only what you turned on.

> **Ollama on the M4 is currently down** and cannot be fixed from here: the
> launchd daemon on `.200` is not running, and the activation API can only load
> a model *into* an already-running ollama. Restart returns
> `Native Ollama not reachable on 127.0.0.1:11435`. This is why every field
> below points at the **OpenAI-compatible** endpoints instead. Fixing it needs
> hands on `.200`.

---

## 3. Fill in the viewer panel

**Settings → Live build.** Under **Geometry source** pick
*Seed concept — no model: an LLM grows the cloud, an embedder places it*, then:

| Field | Value |
|---|---|
| Seed concept | e.g. `glassblowing` |
| Concept generator | `openai` |
| Chat server | `http://<m4-host>:8050` |
| Chat model | *(leave blank — it auto-picks the served chat model)* |
| Embed API | `openai` |
| Embed host | `http://<m4-host>:8040` |
| Embed model | `sentence-transformers/all-MiniLM-L6-v2` |
| Cluster namer | `openai` |
| Depth | `2` |
| Breadth | `10`–`12` |
| Sensitivity | `0.28`–`0.35` |

Press **Grow cloud**. Progress updates roughly every LLM call
(`depth 2: expanding 5/10 'lampworking'…`); a depth-2 × breadth-10 run is about
two minutes against the M4. When it finishes, the map is written to
`out/probe__<slug>/` and the viewer swaps to it automatically.

You can leave the defaults in `.env` so the fields come pre-filled:

```
VITE_LLM_HOST=http://<m4-host>:8050
VITE_EMBED_HOST=http://<m4-host>:8040
```

### The same thing from the CLI

```bash
.venv/bin/python -m nebulai.cli probe "glassblowing" \
  --generator openai --namer openai \
  --llm-host http://<m4-host>:8050 \
  --embed-api openai \
  --embed-host http://<m4-host>:8040 \
  --embed-model sentence-transformers/all-MiniLM-L6-v2 \
  --depth 2 --breadth 10 --sensitivity 0.28 \
  --n-neighbors 12 --min-cluster-size 4 --cluster-method leaf
```

Note `--seed-rng` (not `--seed`) for the RNG: the positional argument is
already the seed *phrase*.

---

## 4. Picking knobs that produce a usable cloud

The two failure modes are both about size.

- **Too few points.** Reduction needs more than `cluster_dim + 1` points
  (11 by default). Below that you get a clear
  `only N points to reduce — raise --breadth or --depth, or lower
  --sensitivity`. Depth 2 × breadth 6 at sensitivity 0.3 lands here.
  Depth 2 × breadth 10 gives ~35 kept concepts, which clusters fine.
- **Too slow.** Expansion is one sequential LLM call per frontier term, so cost
  is roughly `1 + breadth` calls at depth 2 and `1 + breadth + breadth²` at
  depth 3. Against the M4 each call is 15–40 s. Depth 3 is a long wait; start
  at depth 2.

`--sensitivity` is cosine-to-**seed**, not to parent, at every depth — so
raising it tightens the whole cloud around the topic uniformly instead of
letting depth-3 terms drift while each individual hop still looks reasonable.

---

## 5. Why it didn't work before, and what to expect now

The M4 serves **only** OpenAI-compatible endpoints. `nebulai`'s generator and
namer spoke ollama, OpenRouter and Anthropic — and there are no API keys in this
environment. So every probe failed at "no expansion backend available" no matter
what was running. The fix was an OpenAI-compatible backend in both
`backend/name.py` and `frontends/probe.py`, plus `--embed-api` on `compare`.

Three behaviours that follow from the M4's model being a *reasoning distill*,
worth knowing so the logs read correctly:

1. **It writes a scratchpad before the answer**, unpredictably long — 300–700
   tokens on trivial prompts, but 8192 on one ordinary 12-concept expansion.
   Token budgets therefore start at 4096 and escalate ×2 twice on a `length`
   finish. Retries are only ever paid on an actual truncation.
2. **A truncated availability check is not a dead backend.** The generator
   probe sends one throwaway expansion; if the model rambles past the ceiling,
   that proves it *can* generate. You will see
   `generator openai:… rambles (…) — using it anyway` and the run continues.
   Before this, a perfectly working generator was being retired.
3. **A single expansion can still fail** after all three attempts. It is logged
   (`expansion failed for 'counseling' (…)`) and that one term is skipped
   rather than killing the run.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build button disabled | build server down, or no seed typed | start `build_server`; check the health dot |
| `no chat model on <host>` | host unreachable, or it serves only embedding/rerank models | `curl <host>/v1/models`; activate `opus` |
| `no expansion backend available` | every backend in the chain failed — reasons are printed above it, one line each | read those lines; they name the actual cause |
| `only N points to reduce` | cloud too small | raise `--breadth`/`--depth` or lower `--sensitivity` |
| Progress frozen on one message for minutes | a single LLM call is slow | expected at 15–40 s/call; the counter advances per term |
| `Native Ollama not reachable on 127.0.0.1:11435` | launchd daemon on `.200` is down | needs hands on `.200`; use the OpenAI-compatible path meanwhile |
| `insufficient_ram, N GB short` | `opus` and `ollama` can't coexist | deactivate one before activating the other |

---

## 7. Housekeeping

Deactivate whatever you activated when you're done — but **only** what you
turned on:

```bash
curl -s -X POST http://<m4-host>:8100/v1/deactivate/opus
curl -s -X POST http://<m4-host>:8100/v1/deactivate/embedding
```

Probe maps land in `out/probe__<slug>/` and are picked up by `out/index.json`
on the next catalog refresh, so they appear in the viewer's dataset list
alongside the model maps. They are **not** added to the `compare` view by
default: `compare` measures concept overlap between *model geometries*, and a
topic cloud grown from a seed word is a different kind of object.
