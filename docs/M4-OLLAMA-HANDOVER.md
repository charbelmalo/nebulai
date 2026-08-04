# RESOLVED — `ollama` service on the M4 Worker would not start

**Host:** `<m4-host>` (M4 Worker, 48 GB) · **Service:** `ollama` / port **11435**
**Raised:** 2026-08-04 · **Fixed:** 2026-08-04 by the M4-side operator
**Status:** fixed, persistent, and the three blocked nebulai probe maps are rebuilt.

This file was originally a handover request. It is kept as the incident record
because most of its diagnosis was **wrong**, and the wrong parts are the kind
that would waste the next person's afternoon.

---

## Root cause (from the M4 side — authoritative)

Two compounding faults, neither of them anything this end guessed:

1. **`ollama-load` never starts a server.** It is `load_ollama_model()` in
   `openwebui-manager.sh` — a `/api/generate` warm-up call against an
   *already-running* server. With nothing listening it prints
   `Native Ollama not reachable on 127.0.0.1:11435 — is the launchd server up?`
   and returns 1. That line appears 3× in `lan-activate-actions.log`, once per
   activation attempt.

2. **The launchd agent was in launchd's persistent `disabled` list.**
   `com.charbelmalo.ollama-native` was disabled, so `launchctl bootstrap`
   refused with `5: Input/output error`. Last successful serve: 2026-07-03.

The fix was `launchctl enable` + `bootstrap` (MDM did not block it), and
`KeepAlive` was then tested for real with a `kill -9` — launchd respawned in 2 s.
It now survives reboots.

**The `:8100` API masked all of this.** It discards both stderr and the exit
code of the verb it runs, so a command that exited 1 was reported as
`{"activated":true,"error":""}`. That is the bug that made a one-line failure
look like a silent multi-hour stall.

## What this end got wrong — corrections that matter

- **`opus` does NOT need to be swapped out.** This was the most consequential
  error here. The registry's `need_gb: 23.5` gates the *coding* model;
  `mxbai-embed-large` is **0.67 GB** and runs happily alongside a running
  `opus`. The original plan of stopping the generator to make room was never
  necessary. All three rebuilds ran with `opus` untouched.
- **Port drift was a red herring.** This file's leading hypothesis was a
  mismatch between the launcher's bind port and the registry's polled port. The
  plist has always said `OLLAMA_HOST=0.0.0.0:11435`, matching the registry. The
  `11434` in the 2026-07-16 map is simply stale.
- **"ollama is a slow starter" was also wrong.** An intermediate reading here
  guessed the service was merely slow to bind, because a background waiter on
  `:11435` returned success ~40 min after the activation attempt. It did not
  start slowly — it never started at all, and the waiter fired the moment the
  operator started it by hand. Do not carry that theory forward.

## Still true, and worth keeping

- **`free_gb` lags a deactivate by several seconds.** Immediately after
  `POST /v1/deactivate/opus` the API reported `free_gb: 21.5` and
  `insufficient_ram`; seconds later, `31.1` and `loadable`. Anything gating on a
  `free_gb` read taken right after a stop decides wrong.
- **`keep_alive`.** `ollama-embed-load` warms with `keep_alive: 5m`. It reloads
  on demand in ~1 s, but a long batch should pass its own
  `"keep_alive":"2h"` in the `/api/embeddings` body to avoid churn between
  batches. (The nebulai probe rebuilds are 35–55 texts each and never hit this.)
- **`:8050` also registers `nomic-embed-text-v1.5`** — a second third-party
  embedding space on this box, if a future map wants one.

## Verification

```
GET  http://<m4-host>:11435/api/tags        → mxbai-embed-large:latest (0.67 GB, 1024-dim)
POST http://<m4-host>:11435/api/embeddings  → dim=1024
GET  http://<m4-host>:8100/v1/status/ollama → "state":"running","running":true
```

## Outstanding (M4 side, not nebulai)

The `:8100` API reporting `activated:true` for a verb that exited 1 is unfixed.
Propagating the exit code would turn this class of failure into an immediate
error instead of an opaque stall. Flagged, not actioned.
