"""Replace a published file without rewriting it in place.

`out/artifacts/<sha256>/nebulai.json` is a HARD LINK to the map it was
packaged from (viewer/scripts/package-experience.ts), because the mini has no
disk to spare for a second copy of every map. A hard link is only immutable
while nobody writes through it: `Path.write_text` truncates the existing
inode, which would silently change the "immutable" artifact along with the
map. Writing a sibling temp file and renaming it over the target gives the
map a NEW inode and leaves the artifact's bytes exactly as they were hashed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o777)
        else:
            os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
