# Handover — static Nebul.AI deploy at `research.elysiumsystems.net/psychiX/nebulai-maps`

**Audience:** the agent operating the self-hosted `research.elysiumsystems.net`
server. **Goal:** serve the Nebul.AI viewer as a **fully static, pre-baked**
site under the sub-path `/psychiX/nebulai-maps/` with **zero server-side
computation** and **no "data not available" gaps** for any selection.

> **Deploy host, in one line** (verified 2026-08-12): the server is the Mac mini
> at **`192.168.0.199`** (`Digitals-Mac-mini.local`), account **`digitalcharbel`**,
> and the live webroot is
> `/Users/digitalcharbel/Documents/digiCharbel/data/www/research/`.
> It is normally **already SMB-mounted on the build machine**, which makes the
> data transfer in §4 a *local file copy* — see §4 Option A.

Build from **`main`** — it is the single source of truth. There is no separate
deploy branch: the sub-path base and the blank live-endpoint defaults are both
selected at **build time** via `VITE_*` env vars (§3), so the same `main` tree
serves local dev (loopback defaults) and this static deploy (blanked) with no
code divergence.

---

## 0. TL;DR

The viewer is a static SPA. Every view (Atlas / Chord / Hierarchical / Compare
and all 25 Internals panels) is a plain `fetch()` of a **pre-computed JSON file**
under `out/`. Nothing is computed at request time. To deploy you only:

1. `git clone` this repo (branch `main`) and **build the SPA** with the sub-path
   base and blanked live endpoints (§3).
2. **Copy the baked `out/` data tree** (~385 MB on disk; ~378 MB actually
   shipped — see the exclusions in §4) next to the built app. `out/` is
   **git-ignored — it is NOT in the repo** and must be transferred out-of-band.
3. Serve both as static files, with the data tree at `<app>/out/`.

No Python, no Node, no GPU, no model weights, and no live backend run on the
server. WebGPU/WebGL runs entirely in the visitor's browser.

---

## 1. What "every possible selection" resolves to (verified complete)

The dataset catalog is `out/index.json` (**18 datasets** as of 2026-08-12; this
number grows as the pipeline is re-run — re-check rather than trusting it, and
note the `out/` tree also holds non-catalog dirs such as `compare/` and
`neuronpedia/`). Verified on the build
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
cd nebulai            # branch main — the deploy source of truth
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
VITE_BASE=/psychiX/nebulai-maps/ \
VITE_LIVE_URL= VITE_BUILD_URL= VITE_EMBED_HOST= VITE_SEER_URL= \
  npm run build                                  # -> viewer/dist/
```

Two groups of build-time env vars, both required for this deploy:

- **`VITE_BASE=/psychiX/nebulai-maps/`** — the sub-path. `viewer/src/data/base.ts`
  derives the data root (`DATA_BASE`) from Vite's `BASE_URL` and resolves it to an
  **absolute** URL against the page origin (this is load-bearing: the data parse
  runs in a Web Worker, and a relative base would resolve against the worker
  script, not the page). With this set, every data fetch targets
  `https://…/psychiX/nebulai-maps/out/…`. **If you omit `VITE_BASE`, the base
  defaults to `./` and data 404s under the sub-path.** The trailing slash is
  required.
- **`VITE_LIVE_URL= VITE_BUILD_URL= VITE_EMBED_HOST= VITE_SEER_URL=`** (empty) —
  blanks the optional live/build/embed/seer endpoints so the static site is
  bring-your-own-endpoint and contacts no backend on its own (§8). Omitting these
  bakes in the local-dev loopback defaults (`127.0.0.1:8123/8124/8125`,
  `localhost:11434`) instead — **don't**, for a public deploy. `VITE_SEER_URL` is
  the one most easily forgotten because it arrived last: left unset, every public
  visitor's browser tries to reach a SessionSeer collector on *their own*
  `127.0.0.1:8125`. Blanked, the Seer page says no server is configured, which is
  the truth for a static visitor.

Confirm the base after building:

```sh
grep -o '/psychiX/nebulai-maps/assets/[^"]*' viewer/dist/index.html   # should print asset paths
```

---

## 4. Getting the baked data (`out/`, ~385 MB) — the one real logistics step

`out/` is git-ignored and lives only on the build machine. The tree must end up
at `<webroot>/psychiX/nebulai-maps/out/` (see §5), which on the real host is:

```
/Users/digitalcharbel/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps/out/
```

**Option A — local rsync over the SMB mount (what actually works; no SSH).**
The Mac mini's home is normally already mounted on the build machine at
`/Volumes/digitalcharbel`, so the "transfer" is a plain local copy:

```sh
# verify the mount first — if this path is missing, re-mount before anything else
ls -d /Volumes/digitalcharbel/Documents/digiCharbel/data/www/research

/opt/homebrew/bin/rsync -rlt --delete \
  --exclude='.DS_Store' --exclude='._*' --exclude='.backup-pre-recuration/' \
  ~/Developer/nebulai/out/ \
  /Volumes/digitalcharbel/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps/out/
```

Three details in that command are load-bearing, all learned the hard way:

- **Use Homebrew rsync 3.x** (`/opt/homebrew/bin/rsync`), not `/usr/bin/rsync` —
  macOS ships *openrsync* (protocol 29, "2.6.9 compatible"), which lacks flags
  used here and in the verification below.
- **`-rlt`, not `-a`.** `-a` implies `-pog` (perms/owner/group); SMB cannot
  preserve those and the run fills with errors. `-rlt` keeps recursion,
  symlinks and mtimes — mtimes are what make re-deploys incremental.
- **Keep the excludes.** `.backup-pre-recuration/` is a local pipeline backup
  (~6.6 MB) that would otherwise be **published on a public site**; `.DS_Store`
  and `._*` are macOS noise. Excluded paths are protected from `--delete`, so
  they are not pruned from the server either.

Expect a re-deploy to move only what changed (a 2026-08-12 sync moved 155 MB of
the 396 MB tree in ~1m40s and deleted nothing).

**Option B — SSH/rsync from elsewhere.** The doc previously assumed this. Note
that SSH to `192.168.0.199` as `charbelmalo` is **refused (publickey)** — the
account on that box is **`digitalcharbel`**. Port 22 is open (OpenSSH 10.2);
port 2222 also appears in `known_hosts` (it is the `Digitals-Mac-mini.local`
git endpoint in `~/.ssh/config`) but was closed when last checked. If you do
have a key authorized for `digitalcharbel`:

```sh
rsync -rlt --delete --exclude='.DS_Store' --exclude='._*' \
  --exclude='.backup-pre-recuration/' ~/Developer/nebulai/out/ \
  digitalcharbel@192.168.0.199:'/Users/digitalcharbel/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps/out/'
```

**Option C — tarball with a checksum**, if neither of the above is available:
```sh
# build machine — pack and record an integrity checksum
tar -czf nebulai-out.tar.gz -C ~/Developer/nebulai --exclude='.backup-pre-recuration' out
shasum -a 256 nebulai-out.tar.gz | tee nebulai-out.tar.gz.sha256
# transfer both files, then on the server verify BEFORE extracting
shasum -a 256 -c nebulai-out.tar.gz.sha256                   # must print: OK
tar -xzf nebulai-out.tar.gz -C /Users/digitalcharbel/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps
```

rsync is the more reliable choice for this single self-hosted target — it is
incremental, and `--delete` keeps the server tree exactly in sync on re-deploys.
(A GitHub release asset would only be worth the public-upload overhead if many
independent consumers or a CI job needed to pull the data — not the case for one
server you control.)

> **Do not confuse the clone with the webroot.** There is also a checkout at
> `/Volumes/digitalcharbel/Developer/nebulai/` **with its own `out/`**. That is
> *not* what Caddy serves — writing there deploys nothing. Only the
> `…/digiCharbel/data/www/research/…` path above is live.

> **Updating the maps later:** re-run the pipeline on the build machine
> (`uv run nebulai tokens …` / `sae` / `neurons` / `interp` / `compare`), then
> re-run the Option A rsync. No rebuild of the SPA is needed unless viewer code
> changed. `nebulai.json` files compress ~4× (gpt2: 13.8 MB → 3.0 MB gzip), so
> keep server compression on (§6).

---

## 5. On-server layout

Serve a single directory as the site root; the app and its data sit under the
sub-path together:

```
…/digiCharbel/data/www/research/           <- webroot  (= /srv/www/research in the container)
├── index.html                             <- the research dashboard (not ours)
├── assets/
└── psychiX/
    └── nebulai-maps/
        ├── index.html                     <- from viewer/dist/
        ├── assets/                        <- from viewer/dist/assets/
        └── out/                           <- the baked data tree (§4)
            ├── index.json
            ├── gpt2/nebulai.json
            ├── gpt2/interp/*.json
            ├── compare/compare.json
            └── … (every dataset in index.json)
```

Put `viewer/dist/*` into `…/psychiX/nebulai-maps/` and the `out/` tree beside it.

**Why that path — the chain, so you can re-derive it when something moves.**
Caddy runs as the `caddy` service in the `macmini-homelab` compose stack and its
`root` for this vhost is `/srv/www/research`, which is a *container* path. It
maps to the host like this:

1. `~/.cloudflared/config.yml` routes every public hostname — including
   `research.elysiumsystems.net` — to `http://127.0.0.1:8080`.
2. `Documents/digiCharbel/compose/compose.yaml` binds that port
   (`127.0.0.1:${CADDY_HTTP_PORT:-8080}:80`) and mounts
   `${SERVER_DATA_PATH}/www:/srv/www:rw`.
3. `Documents/digiCharbel/caddy/Caddyfile` sets `root * /srv/www/research` for
   `{$RESEARCH_HOST}`.
4. `Documents/digiCharbel/.env.local` sets
   `SERVER_DATA_PATH='/Users/digitalcharbel/Documents/digiCharbel/data'` and
   `RESEARCH_HOST=research.elysiumsystems.net`.

So `/srv/www/research` → `/Users/digitalcharbel/Documents/digiCharbel/data/www/research`.
Because the whole webroot is a bind mount, **files written on the host are live
immediately — no container restart and no Caddy reload is needed** after a data
re-sync.

---

## 6. Web-server config

The app uses **query-param routing** (`?view=…`), not path routing, so **no SPA
history fallback is needed** — `index.html` is the only HTML entry. The only
must-haves: correct `application/json` MIME for `.json`, and compression (the
data is large but highly compressible).

**This is already configured on the live host — you should not need to touch
it.** The block below is what `Documents/digiCharbel/caddy/Caddyfile` actually
contains; the nginx variant is kept only for porting to a different server.

**Caddy** (live config, compression + JSON MIME are automatic):
```
http://{$RESEARCH_HOST} {          # RESEARCH_HOST=research.elysiumsystems.net
    encode zstd gzip
    root * /srv/www/research       # bind-mounted from the host — see §5
    file_server

    @bakeddata path /psychiX/nebulai-maps/out/*
    header @bakeddata Cache-Control "public, max-age=3600"

    import public_security_headers
}
```

It listens on plain `http://` **by design**: TLS is terminated by the Cloudflare
tunnel in front of it, so Caddy never sees port 443 and issues no certificate.
The site is still HTTPS to the visitor, which is what WebGPU's secure-context
requirement needs. Note this vhost serves the **whole** research site from one
root — the nebulai viewer is just the `/psychiX/nebulai-maps/` subtree, so do not
repoint `root`.

**Nginx** (only if porting elsewhere; sub-path via `alias`):
```nginx
location /psychiX/nebulai-maps/ {
    alias /srv/www/research/psychiX/nebulai-maps/;
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

No CORS headers are needed — the app and data are same-origin. The public URL
must be HTTPS (WebGPU requires a secure context; `localhost` is exempt) — on
this host that is satisfied by the Cloudflare tunnel, not by Caddy.

---

## 7. Post-deploy verification (mirror of what was tested locally)

First confirm the copy itself is complete, by re-running the **same rsync with
`--dry-run`**. A second run that reports `0` created / `0` deleted / `0`
transferred is proof the tree is fully in sync — and unlike a bare `ls`, it can
actually fail:

```sh
/opt/homebrew/bin/rsync -rlt --delete --dry-run --stats \
  --exclude='.DS_Store' --exclude='._*' --exclude='.backup-pre-recuration/' \
  ~/Developer/nebulai/out/ \
  /Volumes/digitalcharbel/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps/out/ \
  | grep -E 'Number of (created|deleted|regular)'
```

Then check it over the wire:

```sh
BASE=https://research.elysiumsystems.net/psychiX/nebulai-maps
for p in "" out/index.json out/gpt2/nebulai.json out/gpt2/interp/index.json \
         out/gpt2/interp/weights.json out/compare/compare.json; do
  printf '%s -> ' "$p"; curl -s -o /dev/null -w '%{http_code}\n' "$BASE/$p"
done
# all should be 200
```

**Add at least one dataset that the server did *not* already have** to that
loop. The paths above return `200` whether or not your sync landed — they
existed before it — so on their own they cannot distinguish a successful deploy
from a no-op. A newly-added `out/<new-dataset>/nebulai.json` returning `200` is
the check that can actually fail, and is therefore the one worth trusting.

Two things that look like failures but are not: `192.168.0.199` does **not**
answer `ping` (ICMP is filtered) and has ports 80/443 **closed** on the LAN — it
is reachable publicly only through the Cloudflare tunnel. Also, immediately
after a large rsync the SMB mount can report a **stale, inflated** directory
count (a 22-entry dir listed as 25); re-stat before believing a diff.

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

Built with the empty `VITE_LIVE_URL/VITE_BUILD_URL/VITE_EMBED_HOST/VITE_SEER_URL`
(§3), the optional live features (the #25 "Live Nebula" driver, "+ your prompt"
trace/SAE re-derive, on-demand build, the model probe, and the SessionSeer page)
ship with **blank default
endpoints**. They are inert until a visitor pastes their own OpenAI-compatible /
nebulai server URL under **Settings → Model Probing**. SessionSeer is inert in a
different way and deliberately so: it captures agent runs on the machine the
collector runs on, so a static visitor has nothing for it to watch and the page
says so rather than probing for one. The static site contacts
**no** backend on its own — every map, chord, hierarchy, compare and Internals
panel is served from the baked `out/` tree. You do **not** need to run any
server-side process for the site to be fully functional as a map viewer.

One honest caveat: if a visitor *actively enables* the live #25 driver without
first pasting a URL, the driver falls back to probing `http://127.0.0.1:8123` —
i.e. **their own** loopback, not any server of ours (this is a generic dev
default; it reveals no infrastructure and leaks no data off their machine). It
simply fails to connect. Nothing is contacted unless the visitor opts in.

(If you ever want the live panels active for real, that's a separate opt-in: run
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
