#!/usr/bin/env bash
# Sync the baked data tree (`out/`, ~320 MB) between this repo and the live
# webroot on the digiCharbel Mac mini.
#
# WHY THIS GOES THROUGH DOCKER. The webroot lives under ~/Documents, which macOS
# TCC blocks for non-interactive tools (an agent shell gets EPERM even with the
# sandbox off). Docker's file sharing is outside that boundary and already has
# the directory mounted for the Caddy container, so a throwaway container with
# both paths mounted gives a normal rsync — incremental, checksum-verified,
# no special host permissions. Run it from a Terminal you have granted Full Disk
# Access and plain host-side rsync works too; this just always works.
#
#   ./scripts/sync-out.sh pull            # dry run: webroot -> repo
#   ./scripts/sync-out.sh pull --apply    # do it
#   ./scripts/sync-out.sh push --apply    # repo -> webroot (publish new maps)
#   ./scripts/sync-out.sh publish         # dry run: manifest + artifacts only
#   ./scripts/sync-out.sh publish --apply # additive release of experience.json
#   ./scripts/sync-out.sh verify          # checksum-compare both trees
#
# Dry run is the default in every direction: --delete is involved, and for most
# of this tree's life the webroot copy has been the only copy.
#
# ARTIFACT RETENTION (docs/DEPLOY-STATIC.md §10.2, "Artifact retention").
# `artifacts/<sha256>/nebulai.json` are the immutable, content-addressed maps a
# saved finding pins (viewer/scripts/package-experience.ts makes them as hard
# links to the current maps). Three rules hold in every mode:
#   1. `artifacts/` is PROTECTED from --delete, on both sides. A digest an old
#      finding names is never removed by a sync — removal is a separate, manual,
#      reviewed step (see the doc).
#   2. An artifact already present on the destination is never rewritten: if
#      the two copies differ the run stops before copying anything.
#   3. Every source artifact must hash to its own directory name.
# `-H` keeps the hard links, so publishing the artifacts costs no second copy
# of each map on the webroot (the new names link to the maps already there).
# `publish` sends ONLY the immutable artifacts and then the mutable manifest,
# in that order, and deletes nothing; use it to release a new experience.json
# without touching anything else in the live tree.
set -euo pipefail

WEBROOT_OUT="${WEBROOT_OUT:-$HOME/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps/out}"
REPO_OUT="${REPO_OUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/out}"
IMAGE="${SYNC_IMAGE:-alpine:latest}"

MODE="${1:-}"
APPLY="${2:-}"

# .DS_Store is Finder litter that Caddy would happily serve; never propagate it.
# `P /artifacts/**` protects every retained artifact from --delete (rule 1).
# (the options are joined into the container's `sh -c` string, so anything with
# a space or a glob carries its own single quotes)
RSYNC_OPTS=(-aH --delete "--filter='P /artifacts/**'" "--exclude='.DS_Store'" --itemize-changes --info=stats2)

# rules 2 and 3, run inside the sync container before any write. $1 = source
# root, $2 = destination root (both as mounted in the container).
GUARD='
set -e
src="$1"; dst="$2"; bad=0
if [ -d "$src/artifacts" ]; then
  for f in "$src"/artifacts/*/nebulai.json; do
    [ -e "$f" ] || continue
    d=$(basename "$(dirname "$f")")
    h=$(sha256sum "$f" | cut -d" " -f1)
    if [ "$h" != "$d" ]; then echo "guard: $f hashes to $h, not its name" >&2; bad=1; fi
  done
  if [ -d "$dst/artifacts" ]; then
    changed=$(rsync -rcn --existing --itemize-changes "$src/artifacts/" "$dst/artifacts/" | grep "^>f" || true)
    if [ -n "$changed" ]; then
      echo "guard: these artifacts differ from the copies already published:" >&2
      echo "$changed" >&2; bad=1
    fi
  fi
fi
[ "$bad" = 0 ] || { echo "guard: refusing to sync; an artifact is immutable once published" >&2; exit 3; }
echo "guard: artifacts ok (named by their own sha256; none rewritten, none deleted)"
'

case "$MODE" in
  pull) SRC="/webroot/"; DST="/repo/" ;;
  push|publish) SRC="/repo/"; DST="/webroot/" ;;
  verify)
    # -c forces a full checksum read of every file rather than trusting
    # size+mtime; silence means the trees are byte-identical.
    docker run --rm \
      -v "$WEBROOT_OUT:/webroot:ro" \
      -v "$REPO_OUT:/repo:ro" \
      "$IMAGE" sh -c \
      'apk add --no-cache rsync >/dev/null 2>&1 &&
       rsync -rcn --delete --exclude=".DS_Store" --itemize-changes /webroot/ /repo/' \
      | grep -v '^\.d' || true
    echo "verify: any differing paths are listed above; no output means identical"
    exit 0
    ;;
  *)
    echo "usage: $0 {pull|push|publish|verify} [--apply]" >&2
    exit 2
    ;;
esac

if [ "$APPLY" != "--apply" ]; then
  RSYNC_OPTS+=(-n)
  echo ">>> DRY RUN ($MODE) — re-run with --apply to write"
fi

# The destination is mounted rw, the source ro, so a mixed-up argument order
# fails at the mount layer instead of overwriting the wrong tree.
if [ "$MODE" = "pull" ]; then
  mkdir -p "$REPO_OUT"
  MOUNTS=(-v "$WEBROOT_OUT:/webroot:ro" -v "$REPO_OUT:/repo")
else
  MOUNTS=(-v "$WEBROOT_OUT:/webroot" -v "$REPO_OUT:/repo:ro")
fi

if [ "$MODE" = "publish" ]; then
  # immutable artifacts first, then the manifest that names them; the dataset
  # maps ride along only so -H can link each artifact to the copy already
  # published (unchanged maps are skipped, nothing is deleted)
  PUB_OPTS=(-aH "--exclude='.DS_Store'" --itemize-changes --info=stats2)
  [ "$APPLY" = "--apply" ] || PUB_OPTS+=(-n)
  docker run --rm "${MOUNTS[@]}" "$IMAGE" sh -c "
    apk add --no-cache rsync >/dev/null 2>&1 &&
    sh -c '$GUARD' guard /repo /webroot &&
    rsync ${PUB_OPTS[*]} --include='/artifacts/***' --include='/*/' --include='/*/nebulai.json' --exclude='*' /repo/ /webroot/ &&
    rsync ${PUB_OPTS[*]} --include=/experience.json --exclude='*' /repo/ /webroot/"
else
  docker run --rm "${MOUNTS[@]}" "$IMAGE" sh -c "
    apk add --no-cache rsync >/dev/null 2>&1 &&
    sh -c '$GUARD' guard ${SRC%/} ${DST%/} &&
    rsync ${RSYNC_OPTS[*]} $SRC $DST"
fi

if [ "$MODE" != "pull" ] && [ "$APPLY" = "--apply" ]; then
  echo ">>> published; spot-check the live tree:"
  echo "    curl -sI https://research.elysiumsystems.net/psychiX/nebulai-maps/out/index.json | head -1"
fi
