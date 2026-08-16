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
#   ./scripts/sync-out.sh verify          # checksum-compare both trees
#
# Dry run is the default in both directions: --delete is involved, and for most
# of this tree's life the webroot copy has been the only copy.
set -euo pipefail

WEBROOT_OUT="${WEBROOT_OUT:-$HOME/Documents/digiCharbel/data/www/research/psychiX/nebulai-maps/out}"
REPO_OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/out"
IMAGE="${SYNC_IMAGE:-alpine:latest}"

MODE="${1:-}"
APPLY="${2:-}"

# .DS_Store is Finder litter that Caddy would happily serve; never propagate it.
RSYNC_OPTS=(-a --delete --exclude=".DS_Store" --info=stats2)

case "$MODE" in
  pull) SRC="/webroot/"; DST="/repo/" ;;
  push) SRC="/repo/";    DST="/webroot/" ;;
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
    echo "usage: $0 {pull|push|verify} [--apply]" >&2
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

docker run --rm "${MOUNTS[@]}" "$IMAGE" sh -c \
  "apk add --no-cache rsync >/dev/null 2>&1 && rsync ${RSYNC_OPTS[*]} $SRC $DST"

if [ "$MODE" = "push" ] && [ "$APPLY" = "--apply" ]; then
  echo ">>> published; spot-check the live tree:"
  echo "    curl -sI https://research.elysiumsystems.net/psychiX/nebulai-maps/out/index.json | head -1"
fi
