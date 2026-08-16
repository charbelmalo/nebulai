# Handover — static psychiX deploy at `research.elysiumsystems.net/psychiX/`

**Audience:** the agent operating the self-hosted `research.elysiumsystems.net`
server. **Goal:** serve the branded PsychiX shell and its **two** fully static,
pre-baked instruments under `/psychiX/`, with **zero server-side computation**:

| URL | What | Needs baked `out/` data? |
|---|---|---|
| `/psychiX/` | the separately built, branded **PsychiX shell** whose current instrument boundary enters Nebul.AI | no |
| `/psychiX/nebulai-maps/` | **Nebul.AI** — "map what a model knows" — **ALREADY LIVE; do not rename this path** | **yes** (~385 MB) |
| `/psychiX/seer/` | **Seer** — "map what an agent did" | no |

Nebul.AI and Seer used to be two tabs of one document; they are now two
separate instruments (`viewer/index.html` and `viewer/seer.html`) that share
one chrome and one `@psychix/viz` package but ship as **independent builds**.
This repository also builds a generic, JS-free hub for standalone deployments.
The current Elysium deployment does **not** use that hub: its branded PsychiX
shell is maintained and deployed separately. None of the three public surfaces
needs anything computed at request time.

> **Deploy target** (verified 2026-08-13): the live webroot is
> `/Users/digitalcharbel/Documents/digiCharbel/data/www/research/`.
> It is available directly on the deployment machine, which makes the data
> transfer in §4 a *local file copy* — see §4 Option A. Keep private network
> addresses and account details out of this repository.

Build the two instruments from **`main`** — it is their single source of truth.
There is no separate deploy branch: every sub-path base, cross-instrument link
and blanked live-endpoint default is selected at **build time** via `VITE_*`
env vars (§3). The separately maintained PsychiX shell has its own source and
deployment workflow.

---

## 0. TL;DR

All three public surfaces are static. Nebul.AI and Seer are SPAs; the branded
shell is built in its own project. Nebul.AI's views (Atlas / Chord /
Hierarchical / Compare and all 25 Internals panels) are each a plain `fetch()` of a
**pre-computed JSON file** under `out/`. **Seer needs no such tree — it ships
with zero baked artifacts and fetches nothing at boot**; it only ever talks to
a Live capture server if a visitor points it at one. To deploy:

1. `git clone` this repo (branch `main`) and **build both instruments plus the
   standalone hub validation artifact** with
   `npm run build:deploy` — one command that runs `build:nebulai`,
   `build:seer` and `build:hub` in turn, each with its own sub-path base and
   its own `dist/<name>/` output (§3).
2. **Copy the baked `out/` data tree** (~385 MB on disk; ~378 MB actually
   shipped — see the exclusions in §4) next to the built Nebul.AI app only.
   `out/` is **git-ignored — it is NOT in the repo** and must be transferred
   out-of-band. Seer and the hub need no equivalent step.
3. Serve both instrument trees under `/psychiX/`, with the data tree at
   `<app>/psychiX/nebulai-maps/out/`. Preserve the separately deployed branded
   shell at the parent path.

No Python, no Node, no GPU, no model weights, and no live backend run on the
server for any of the three. WebGPU/WebGL runs entirely in the visitor's
browser.

---

## 1. What "every possible selection" resolves to (Nebul.AI only, verified complete)

This section is **specific to `/psychiX/nebulai-maps/`** — the dataset catalog
below is a Nebul.AI concept and neither Seer nor the hub has an analogous
manifest to verify.

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

Seer, by contrast, boots with `window.__store.getState().datasets` empty and
`dataset` null and makes **zero** requests under `out/` — proved by
`viewer/tests/e2e/seer/seer-entry.spec.ts`, which records every request the
document makes and asserts none of them touch `/out/`.

---

## 2. Getting the code

```sh
git clone https://github.com/charbelmalo/nebulai.git
cd nebulai            # branch main — the deploy source of truth
```

(Forgejo mirror: `https://git.charbelmalo.online/charbelmalo/nebulai.git` — same
branches.)

---

## 3. Building the three SPAs (sub-path aware)

Requires Node ≥ 20. Vite's `base` is a property of the *build*, not of an
entry, so three sub-path deploys are three separate builds — one per tree —
each with its own base, its own HTML input, and its own `outDir`. Run all
three at once:

```sh
cd viewer
npm ci
npm run build:deploy   # -> dist/nebulai/, dist/seer/, dist/hub/
```

which is exactly `npm run build:nebulai && npm run build:seer && npm run
build:hub`. The three underlying commands, spelled out (see `package.json`):

```sh
PSYCHIX_ENTRY=nebulai VITE_BASE=/psychiX/nebulai-maps/ \
  VITE_NEBULAI_APP_URL=/psychiX/nebulai-maps/ VITE_SEER_APP_URL=/psychiX/seer/ VITE_HUB_URL=/psychiX/ \
  VITE_LIVE_URL= VITE_BUILD_URL= VITE_EMBED_HOST= VITE_SEER_URL= \
  vite build --outDir dist/nebulai

PSYCHIX_ENTRY=seer VITE_BASE=/psychiX/seer/ \
  VITE_NEBULAI_APP_URL=/psychiX/nebulai-maps/ VITE_SEER_APP_URL=/psychiX/seer/ VITE_HUB_URL=/psychiX/ \
  VITE_LIVE_URL= VITE_BUILD_URL= VITE_EMBED_HOST= VITE_SEER_URL= \
  vite build --outDir dist/seer

PSYCHIX_ENTRY=hub VITE_BASE=/psychiX/ vite build --outDir dist/hub
```

Each `dist/<name>/` comes out as a self-contained, drop-in root: `index.html`
plus a hashed `assets/` folder. (`PSYCHIX_ENTRY=seer` and `=hub` compile
`seer.html` / `hub.html` respectively but a Vite plugin renames the *emitted*
asset to `index.html` inside the bundle, because each deploy tree must answer
a bare directory request with its own document — verified on disk: `dist/
seer/index.html` and `dist/hub/index.html` are what land, not `seer.html` /
`hub.html`.) `dist/nebulai/` needs no such rename — `index.html` was always
its input's name.

Four groups of build-time env vars matter here:

- **`VITE_BASE`** — the sub-path for *this* build (`/psychiX/nebulai-maps/`,
  `/psychiX/seer/`, or `/psychiX/` for the hub). `viewer/src/data/base.ts`
  derives the data root (`DATA_BASE`) from Vite's `BASE_URL` and resolves it
  to an **absolute** URL against the page origin (this is load-bearing: the
  data parse runs in a Web Worker, and a relative base would resolve against
  the worker script, not the page). With this set, Nebul.AI's data fetches
  target `https://…/psychiX/nebulai-maps/out/…`. **Omit it and the base
  defaults to `./`, and data 404s under the sub-path.** The trailing slash is
  required.
- **`VITE_NEBULAI_APP_URL` / `VITE_SEER_APP_URL` / `VITE_HUB_URL`** — where the
  *other* instrument and the hub live, for the one cross-instrument link each
  top bar renders (`viewer/src/chrome/apps/nav.ts`). Their **default** (unset)
  is the relative sibling — `./seer.html` / `./index.html`, no hub link at
  all — which is correct only for the combined `npm run build` (both
  documents emitted side by side into one `dist/`, used by dev, `vite
  preview`, and the e2e suite). It is **wrong** for these per-app deploys:
  Nebul.AI and Seer land in **sibling directories**
  (`/psychiX/nebulai-maps/` and `/psychiX/seer/`), where a relative
  `./seer.html` would resolve to `/psychiX/nebulai-maps/seer.html` and 404.
  So `build:nebulai` / `build:seer` pass the absolute sub-paths explicitly, in
  **both** directions — each build needs to know where the *other* one lives,
  not just where it itself will sit. Verified on disk: `dist/nebulai/assets/
  index-*.js` contains the literal strings `/psychiX/`, `/psychiX/
  nebulai-maps` and `/psychiX/seer`.
- **`VITE_LIVE_URL= VITE_BUILD_URL= VITE_EMBED_HOST=`** (empty) — blanks
  Nebul.AI's three optional live-probing endpoints (the #25 "Live Nebula"
  driver, "+ your prompt" trace/SAE re-derive, on-demand build, the model
  probe — all in `viewer/src/app/slices/probing.ts`) so the static site is
  bring-your-own-endpoint and contacts no backend on its own (§8). Omitting
  these bakes in the local-dev loopback defaults (`127.0.0.1:8123/8124`,
  `localhost:11434`) instead — **don't**, for a public deploy.
- **`VITE_SEER_URL=`** (empty) — blanks the default address of the Seer
  **capture server** (`seer serve`, `:8125`) that Seer's own Live page talks
  to (`viewer/src/app/slices/seer.ts`). **This is not the same variable as
  `VITE_SEER_APP_URL` above** — that one is the URL of the Seer *web app* on
  this site; this one is the URL of a *collector process* a visitor would run
  themselves. Setting one when you meant the other is the obvious mistake,
  hence the different suffix. Left unset, every public visitor's browser
  would try to reach a collector on *their own* `127.0.0.1:8125`; blanked,
  Seer's Live page says no server is configured, which is the truth for a
  static visitor. (The env var name itself did not change across the
  segmentation — only the store field that holds it moved, from
  `probing.seerUrl` to `seer.serverUrl`, because Seer is now its own slice
  rather than a corner of Nebul.AI's probing state.)

Confirm the bases after building:

```sh
grep -o '/psychiX/nebulai-maps/assets/[^"]*' dist/nebulai/index.html   # should print asset paths
grep -o '/psychiX/seer/assets/[^"]*'         dist/seer/index.html
grep -o '/psychiX/assets/[^"]*'              dist/hub/index.html
```

---

## 4. Getting the baked data (`out/`, ~385 MB) — Nebul.AI only; the one real logistics step

**Seer and the hub need nothing from this section.** Only Nebul.AI reads a
baked artifact tree, and only its own sub-path needs one.

`out/` is git-ignored and is not supplied by a fresh clone. On the current
deployment machine, `~/Developer/nebulai/out/` is the local working mirror and
the live webroot is the authoritative complete corpus. Pull the live tree into
that mirror before frontend QA so Playwright does not validate stale datasets
or comparisons. A checksum dry run should be silent:

```sh
rsync -rcn --delete --exclude='.DS_Store' --itemize-changes \
  /Users/digitalcharbel/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps/out/ \
  ~/Developer/nebulai/out/
```

The tree must end up at `<webroot>/psychiX/nebulai-maps/out/` (see §5), which on
the real host is:

```
/Users/digitalcharbel/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps/out/
```

**Option A — local rsync on the deployment machine (current path).** The source
checkout and live webroot are available on the same machine, so the transfer is
a plain local copy:

```sh
# resolve the exact target before allowing --delete
ls -d /Users/digitalcharbel/Documents/digiCharbel/data/www/research

rsync -rlt --delete \
  --exclude='.DS_Store' --exclude='._*' --exclude='.backup-pre-recuration/' \
  ~/Developer/nebulai/out/ \
  /Users/digitalcharbel/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps/out/
```

Four details in that command are load-bearing, all learned the hard way:

- **Use only the portable flags shown here.** The current deployment machine's
  `/usr/bin/rsync` is openrsync and supports this command. Verify the installed
  version before adding GNU-rsync-only options.
- **`-rlt`, not `-a`.** `-a` implies `-pog` (perms/owner/group); SMB cannot
  preserve those and the run fills with errors. `-rlt` keeps recursion,
  symlinks and mtimes — mtimes are what make re-deploys incremental.
- **Keep the excludes.** `.backup-pre-recuration/` is a local pipeline backup
  (~6.6 MB) that would otherwise be **published on a public site**; `.DS_Store`
  and `._*` are macOS noise. Excluded paths are protected from `--delete`, so
  they are not pruned from the server either.
- **Keep the trailing slash on `~/Developer/nebulai/out/`.** It means “copy the
  contents of this directory into the deployed `out/` directory.” Without it,
  rsync creates an extra nested `out/out/` level and every dataset URL 404s.

Expect a re-deploy to move only what changed (a 2026-08-12 sync moved 155 MB of
the 396 MB tree in ~1m40s and deleted nothing).

**Option B — SSH/rsync from elsewhere.** Resolve the private hostname and
deployment account from the operator's local configuration; do not commit them
to this repository. If that account has an authorized key:

```sh
rsync -rlt --delete --exclude='.DS_Store' --exclude='._*' \
  --exclude='.backup-pre-recuration/' ~/Developer/nebulai/out/ \
  deploy-user@research-host:'/Users/digitalcharbel/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps/out/'
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

> **Do not confuse the clone with the webroot.** Editing or building the
> checkout deploys nothing by itself. Only the explicitly resolved
> `…/digiCharbel/data/www/research/…` path above is live.

> **Updating the maps later:** re-run the pipeline on the build machine
> (`uv run nebulai tokens …` / `sae` / `neurons` / `interp` / `compare`), then
> re-run the Option A rsync. No rebuild of the SPA is needed unless viewer code
> changed. `nebulai.json` files compress ~4× (gpt2: 13.8 MB → 3.0 MB gzip), so
> keep server compression on (§6).

---

## 5. On-server layout

Serve a single directory as the site root. The separately built PsychiX shell
owns the parent path; the two instrument builds are its siblings, with
Nebul.AI's data tree nested inside its own subtree only:

```
…/digiCharbel/data/www/research/           <- webroot  (= /srv/www/research in the container)
├── index.html                             <- the research dashboard (not ours)
├── assets/
└── psychiX/
    ├── index.html                         <- from the branded PsychiX shell
    ├── assets/                            <- from the branded PsychiX shell
    ├── nebulai-maps/                      <- ALREADY LIVE — do not rename this directory
    │   ├── index.html                     <- from dist/nebulai/
    │   ├── assets/                        <- from dist/nebulai/assets/
    │   └── out/                           <- the baked data tree (§4)
    │       ├── index.json
    │       ├── gpt2/nebulai.json
    │       ├── gpt2/interp/*.json
    │       ├── compare/compare.json
    │       └── … (every dataset in index.json)
    └── seer/
        ├── index.html                     <- from dist/seer/
        └── assets/                        <- from dist/seer/assets/ — no out/, ever
```

Put each instrument build into its matching directory under `psychiX/`.
`--delete` is correct for those self-contained subtrees only.

```sh
# Nebul.AI app code. `out/` is deliberately excluded: it is the separate,
# heavier §4 data tree and is not present in dist/nebulai/. Without this
# exclusion, --delete would erase every deployed map.
rsync -rlt --delete --exclude='out/' \
  ~/Developer/nebulai/viewer/dist/nebulai/ \
  /Users/digitalcharbel/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps/

# Seer — self-contained, no out/ tree ever lands here
rsync -rlt --delete \
  ~/Developer/nebulai/viewer/dist/seer/ \
  /Users/digitalcharbel/Documents/digiCharbel/data/www/research/psychiX/seer/
```

(Swap in the Option B / Option C mechanics from §4 if the copy is remote — same
`-rlt`, the same portable flags, and the same required `out/` exclusion
for the Nebul.AI app-code sync.)

> Do not copy `dist/hub/` over the current Elysium shell. The generic hub is
> only for a standalone deployment where PsychiX has no external landing page.

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
re-sync, and that holds for all three `psychiX/` subtrees identically.

---

## 6. Web-server config

All three sites use in-document routing — Nebul.AI and Seer via a
`location.hash` permalink (`#page=…`) plus a legacy `?view=` query param for
e2e/handoff, the hub via nothing at all — **never** path-based routing. So **no
SPA history fallback is needed anywhere**: each of the three sub-paths has
exactly one HTML entry at its own root (`psychiX/index.html`,
`psychiX/nebulai-maps/index.html`, `psychiX/seer/index.html`), and no request
for a *deeper* path within any of them needs to be rewritten back to it. (The
old version of this doc claimed "index.html is the only HTML entry" for the
whole deploy — that was true when there was one tree; now there are three, one
per subtree, each still self-contained in exactly the same sense.) The only
must-haves: correct `application/json` MIME for `.json`, and compression (the
Nebul.AI data is large but highly compressible).

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
root — the psychiX instruments are just the `/psychiX/` subtree (itself now
three sibling subtrees, per §5), so do not repoint `root`, and the
`@bakeddata` cache rule only needs to name Nebul.AI's `out/` path — Seer and
the hub have no data tree of their own to add a rule for.

**Nginx** (only if porting elsewhere; sub-path via `alias`):
```nginx
location /psychiX/ {
    alias /srv/www/research/psychiX/;
    index index.html;

    types { application/json json; text/html html; image/png png; }
    default_type application/octet-stream;

    gzip on;
    gzip_types application/json application/javascript text/css;
    gzip_min_length 1024;
    # brotli on; brotli_types application/json application/javascript text/css;  # if ngx_brotli present

    # hashed build assets are immutable; data can revalidate. Applies under
    # every subtree — psychiX/assets/, psychiX/nebulai-maps/assets/ and
    # psychiX/seer/assets/ — because the regex matches on suffix, not depth.
    location ~* /assets/.*\.(js|css)$ { expires 1y; add_header Cache-Control "public, immutable"; }
    location ~* /nebulai-maps/out/.*\.json$ { add_header Cache-Control "public, max-age=3600"; }
}
```

No CORS headers are needed — the app and data are same-origin. The public URL
must be HTTPS (WebGPU requires a secure context; `localhost` is exempt) — on
this host that is satisfied by the Cloudflare tunnel, not by Caddy.

---

## 7. Post-deploy verification (mirror of what was tested locally)

First confirm the copy itself is complete, by re-running the **same rsync with
`--dry-run`** for both instrument subtrees. Verify the external shell through
its own project workflow. For Nebul.AI's data tree, a second run that reports
`0` created / `0` deleted / `0` transferred is proof the tree is fully in sync
— and unlike a bare `ls`, it can actually fail:

```sh
rsync -rlt --delete --dry-run --stats \
  --exclude='.DS_Store' --exclude='._*' --exclude='.backup-pre-recuration/' \
  ~/Developer/nebulai/out/ \
  /Users/digitalcharbel/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps/out/ \
  | grep -E 'Number of (created|deleted|regular)'
```

Then check all three sites over the wire:

```sh
BASE=https://research.elysiumsystems.net/psychiX
for p in "" "nebulai-maps/" "seer/" \
         "nebulai-maps/out/index.json" "nebulai-maps/out/gpt2/nebulai.json" \
         "nebulai-maps/out/gpt2/interp/index.json" "nebulai-maps/out/gpt2/interp/weights.json" \
         "nebulai-maps/out/compare/compare.json"; do
  printf '%s -> ' "$p"; curl -s -o /dev/null -w '%{http_code}\n' "$BASE/$p"
done
# all should be 200
```

**Add at least one dataset that the server did *not* already have** to that
loop. The Nebul.AI paths above return `200` whether or not your data sync
landed — they existed before it — so on their own they cannot distinguish a
successful deploy from a no-op. A newly-added
`nebulai-maps/out/<new-dataset>/nebulai.json` returning `200` is the check
that can actually fail, and is therefore the one worth trusting. The shell and
Seer have no equivalent "did the data actually move" question — their `200`s
above are the whole check.

The public hostname is served through the Cloudflare tunnel, so a failed LAN
ping or closed LAN HTTP port does not establish that the public route is down.
After a large filesystem copy, re-stat before believing a cached directory
count.

Then in a browser:

**`…/psychiX/`** (the branded PsychiX shell):
- [ ] The branded shell renders and the browser console is clean.
- [ ] "Enter NebulAI" → `…/psychiX/nebulai-maps/`. Seer is deliberately not
      duplicated here; it remains reachable from Nebul.AI's cross-instrument link.

**`…/psychiX/nebulai-maps/`** (Nebul.AI):
- [ ] Semantic map renders; status bar shows `… pts · … clusters · gpu: webgpu`.
- [ ] Dataset dropdown switches models (loads `<id>/nebulai.json`).
- [ ] **Internals** tab on `gpt2` shows "25 of 25 live" and a panel renders
      (e.g. #21 Weight Spectrum draws curves).
- [ ] View dropdown → **Compare** renders (loads `compare/compare.json`).
- [ ] The top bar's sibling link goes to `…/psychiX/seer/`, and its hub link
      goes to `…/psychiX/`.
- [ ] DevTools console is clean; every `…/out/…` request is `200`.

**`…/psychiX/seer/`** (Seer):
- [ ] Boots straight to the **Live** page with no dataset selector anywhere —
      there is nothing to select.
- [ ] **Transcripts** and **Topics** pills switch pages; no request under
      `/out/` appears in DevTools' Network tab at any point.
- [ ] The top bar's sibling link goes to `…/psychiX/nebulai-maps/`, and its
      hub link goes to `…/psychiX/`.
- [ ] DevTools console is clean.

The HTTP surface was reverified against the live host on 2026-08-13. Repeat the
rendered checklist after each deploy; an HTTP `200` cannot prove that WebGPU,
navigation, or responsive chrome rendered correctly.

---

## 8. Live features = bring-your-own-endpoint (no backend required)

Both instruments ship with **blank default endpoints** when built per §3, and
both are bring-your-own-endpoint in the same sense but for different features:

- **Nebul.AI**, built with empty `VITE_LIVE_URL` / `VITE_BUILD_URL` /
  `VITE_EMBED_HOST`: the optional live features (the #25 "Live Nebula" driver,
  "+ your prompt" trace/SAE re-derive, on-demand build, the model probe) are
  inert until a visitor pastes their own OpenAI-compatible / nebulai server
  URL under **Settings → Model Probing**.
- **Seer**, built with empty `VITE_SEER_URL`: its **Live** page is inert in a
  different way and deliberately so. It captures agent runs on the machine
  the collector runs on, so a static visitor has nothing for it to watch, and
  the page says so rather than probing for one.

Neither instrument contacts **any** backend on its own. Every Nebul.AI map,
chord, hierarchy, compare and Internals panel is served from the baked `out/`
tree; every Seer page renders only a transcript a visitor drops in, or a
capture server they name themselves. You do **not** need to run any
server-side process for either site to be fully functional as shipped.

One honest caveat: if a visitor *actively enables* Nebul.AI's live #25 driver
without first pasting a URL, the driver falls back to probing
`http://127.0.0.1:8123` — i.e. **their own** loopback, not any server of ours
(this is a generic dev default; it reveals no infrastructure and leaks no data
off their machine). It simply fails to connect. The same is true of Seer's
Live page against its own `127.0.0.1:8125` default. Nothing is contacted
unless the visitor opts in.

(If you ever want the live panels active for real, that's a separate opt-in:
run `python -m nebulai.backend.interp.live_server` — for Nebul.AI's probing
features — or `seer serve` — for Seer's Live page — somewhere reachable, and
have users point Settings / the Live page at it. Not part of this static
deploy.)

---

## 9. Privacy / safety notes

- The public repo was sanitized: no private IPs, no `~/.hermes` path, no keys.
  History was scanned — no secrets were ever committed.
- The baked `out/` JSON (Nebul.AI only) contains only public micro-model
  artifacts (token strings, cluster titles, coordinates, SVD spectra). No
  credentials, no PII. Seer and the hub ship no comparable data tree at all.
- Everything renders client-side; visitor prompts in the (opt-in,
  off-by-default) live features — Nebul.AI's probing panels, Seer's Live page
  — never leave their browser unless they configure their own server.
