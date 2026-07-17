# Handover — static Nebul.AI deploy at `research.elysiumsolutions.net/psychiX/nebulai-maps`

**Audience:** the agent operating the self-hosted `research.elysiumsolutions.net`
server. **Goal:** serve the Nebul.AI viewer as a **fully static, pre-baked**
site under the sub-path `/psychiX/nebulai-maps/` with **zero server-side
computation** and **no "data not available" gaps** for any selection.

This branch (`deploy/static`) is the source of truth for the deploy. `main` is
the working library; do not build the public site from `main` (its data base
and live-endpoint defaults are tuned for local dev, not sub-path hosting).

---

## 0. TL;DR

The viewer is a static SPA. Every view (Atlas / Chord / Hierarchical / Compare
and all 25 Internals panels) is a plain `fetch()` of a **pre-computed JSON file**
under `out/`. Nothing is computed at request time. To deploy you only:

1. `git clone` this repo, checkout `deploy/static`, and **build the SPA** with the
   sub-path base.
2. **Copy the baked `out/` data tree** (~320 MB) next to the built app. `out/` is
   **git-ignored — it is NOT in the repo** and must be transferred out-of-band.
3. Serve both as static files, with the data tree at `<app>/out/`.

No Python, no Node, no GPU, no model weights, and no live backend run on the
server. WebGPU/WebGL runs entirely in the visitor's browser.

---

## 1. What "every possible selection" resolves to (verified complete)

The dataset catalog is `out/index.json` (11 datasets). Verified on the build
machine: **every referenced artifact exists on disk — zero missing files**, so no
selection can hit a "data not available" state as long as you ship the whole
`out/` tree.

| Selection | File(s) fetched (relative to `<app>/out/`) |
|---|---|
| Boot / dataset list | `index.json` |
| A dataset's Atlas/Chord/Hierarchy | `<dataset-id>/nebulai.json` (Chord & Hierarchy reuse the loaded columns — no extra fetch) |
| Internals panels (#1–#25) | `<model>/interp/index.json` + the bundle for each panel (`weights.json`, `embed.json`, `neurons.json`, `sae.json`, `trace_<slug>.json`, … — 3 models have interp: `gpt2`, `distilgpt2`, `gpt2-medium`) |
| Compare view | `compare/compare.json` |

Ship `out/` verbatim (the user chose "everything as-is"). `out/neuronpedia/` and
the `*.npz` reduction caches are **build-time only** and never fetched by the
browser — harmless to include, safe to omit if you want to trim ~90 MB.

---

## 2. Getting the code

```sh
git clone https://github.com/charbelmalo/nebulai.git
cd nebulai
git checkout deploy/static
```

(Forgejo mirror: `https://git.charbelmalo.online/charbelmalo/nebulai.git` — same
branches.)

---

## 3. Building the SPA (sub-path aware)

Requires Node ≥ 20. The base path is injected via the `VITE_BASE` env var so the
app's own assets **and** the baked-data root resolve under the sub-path:

```sh
cd viewer
npm ci
VITE_BASE=/psychiX/nebulai-maps/ npm run build   # -> viewer/dist/
```

Why this matters: `viewer/src/data/base.ts` derives the data root
(`DATA_BASE`) from Vite's `BASE_URL`. With `VITE_BASE=/psychiX/nebulai-maps/`,
every data fetch targets `/psychiX/nebulai-maps/out/…`. **If you forget
`VITE_BASE`, data requests fall back to `./out` and will 404 under the sub-path.**
Confirm after building:

```sh
grep -o '/psychiX/nebulai-maps/assets/[^"]*' viewer/dist/index.html   # should print asset paths
```

The trailing slash in `VITE_BASE=/psychiX/nebulai-maps/` is required.

---

## 4. Getting the baked data (`out/`, ~320 MB) — the one real logistics step

`out/` is git-ignored and lives only on the build machine. Move it to the server
by whichever of these you have access to:

**Option A — rsync from the build machine (preferred if you have SSH):**
```sh
# run on the machine that has ~/Developer/nebulai/out
rsync -avz --delete ~/Developer/nebulai/out/ \
  user@research.elysiumsolutions.net:/var/www/nebulai-maps/out/
```

**Option B — tarball, then transfer + extract:**
```sh
# build machine
tar -czf nebulai-out.tar.gz -C ~/Developer/nebulai out       # ~120–150 MB compressed
# server
mkdir -p /var/www/nebulai-maps && tar -xzf nebulai-out.tar.gz -C /var/www/nebulai-maps
```

Either way the tree must end up at `<webroot>/psychiX/nebulai-maps/out/` (see §5).

> **Updating the maps later:** re-run the pipeline on the build machine
> (`uv run nebulai tokens …` / `sae` / `neurons` / `interp` / `compare`), then
> re-rsync `out/`. No rebuild of the SPA is needed unless viewer code changed.
> `nebulai.json` files compress ~4× (gpt2: 13.8 MB → 3.0 MB gzip), so keep server
> compression on (§6).

---

## 5. On-server layout

Serve a single directory as the site root; the app and its data sit under the
sub-path together:

```
/var/www/nebulai-maps/                      <- webroot (server root, or an alias)
└── psychiX/
    └── nebulai-maps/
        ├── index.html                      <- from viewer/dist/
        ├── assets/                         <- from viewer/dist/assets/
        └── out/                            <- the baked data tree (§4)
            ├── index.json
            ├── gpt2/nebulai.json
            ├── gpt2/interp/*.json
            ├── compare/compare.json
            └── … (all 11 datasets)
```

Put `viewer/dist/*` into `…/psychiX/nebulai-maps/` and the `out/` tree beside it.

---

## 6. Web-server config

The app uses **query-param routing** (`?view=…`), not path routing, so **no SPA
history fallback is needed** — `index.html` is the only HTML entry. The only
must-haves: correct `application/json` MIME for `.json`, and compression (the
data is large but highly compressible).

**Nginx** (sub-path via `alias`):
```nginx
location /psychiX/nebulai-maps/ {
    alias /var/www/nebulai-maps/psychiX/nebulai-maps/;
    index index.html;

    types { application/json json; text/html html; image/png png; }
    default_type application/octet-stream;

    gzip on;
    gzip_types application/json application/javascript text/css;
    gzip_min_length 1024;
    # brotli on; brotli_types application/json application/javascript text/css;  # if ngx_brotli present

    # hashed build assets are immutable; data can revalidate
    location ~* /assets/.*\.(js|css)$ { expires 1y; add_header Cache-Control "public, immutable"; }
    location ~* /out/.*\.json$        { add_header Cache-Control "public, max-age=3600"; }
}
```

**Caddy** (equivalent, compression + JSON MIME are automatic):
```
research.elysiumsolutions.net {
    handle_path /psychiX/nebulai-maps/* {
        root * /var/www/nebulai-maps/psychiX/nebulai-maps
        encode zstd gzip
        file_server
    }
}
```

No CORS headers are needed — the app and data are same-origin. Serve over HTTPS
(WebGPU requires a secure context; `localhost` is exempt but the public host must
be TLS).

---

## 7. Post-deploy verification (mirror of what was tested locally)

```sh
BASE=https://research.elysiumsolutions.net/psychiX/nebulai-maps
for p in "" out/index.json out/gpt2/nebulai.json out/gpt2/interp/index.json \
         out/gpt2/interp/weights.json out/compare/compare.json; do
  printf '%s -> ' "$p"; curl -s -o /dev/null -w '%{http_code}\n' "$BASE/$p"
done
# all should be 200
```

Then in a browser at `…/psychiX/nebulai-maps/`:
- [ ] Semantic map renders; status bar shows `… pts · … clusters · gpu: webgpu`.
- [ ] Dataset dropdown switches models (loads `<id>/nebulai.json`).
- [ ] **Internals** tab on `gpt2` shows "25 of 25 live" and a panel renders
      (e.g. #21 Weight Spectrum draws curves).
- [ ] View dropdown → **Compare** renders (loads `compare/compare.json`).
- [ ] DevTools console is clean; every `…/out/…` request is `200`.

This exact flow was verified on a deployment-shaped local server before handover.

---

## 8. Live features = bring-your-own-endpoint (no backend required)

On this branch the optional live features (the #25 "Live Nebula" driver,
"+ your prompt" trace/SAE re-derive, on-demand build, and the model probe) ship
with **blank default endpoints**. They are inert until a visitor pastes their own
OpenAI-compatible / nebulai server URL under **Settings → Model Probing**. The
static site contacts **no** backend on its own. You do **not** need to run any
server-side process for the site to be fully functional as a map viewer.

(If you ever want the live panels active, that's a separate opt-in: run
`python -m nebulai.backend.interp.live_server` somewhere reachable and have users
point Settings at it. Not part of this static deploy.)

---

## 9. Privacy / safety notes

- The public repo was sanitized: no private IPs, no `~/.hermes` path, no keys.
  History was scanned — no secrets were ever committed.
- The baked `out/` JSON contains only public micro-model artifacts (token
  strings, cluster titles, coordinates, SVD spectra). No credentials, no PII.
- Everything renders client-side; visitor prompts in the (opt-in, off-by-default)
  live features never leave their browser unless they configure their own server.
