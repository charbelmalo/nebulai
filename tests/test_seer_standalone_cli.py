"""`seer` is its own console script, not a `nebulai` subcommand grafted on.

Two properties are the whole point of the split and are worth pinning down
with a real test rather than trusting them to stay true:

* `seer --help` builds and prints from `seer`'s own top-level parser — the
  standalone entry point works without ever going through `nebulai.cli`.
* Importing `nebulai.cli` does not, even transitively, import `nebulai.seer`.
  This is the decoupling the split exists for: without a test the graft that
  used to live at the end of `nebulai.cli.main()` could silently grow back.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nebulai.seer.cli import main

# Every sub-subcommand `_add_subcommands` registers, so a rename or removal
# in the wiring shows up here rather than only in a human reading `--help`.
_SUBCOMMANDS = (
    "run", "attach", "reconcile", "protocol", "list", "show", "compare",
    "export", "analyze", "serve", "reindex", "delete", "install",
    "uninstall", "watch", "import-spool",
)


def test_seer_help_standalone_lists_subcommands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main()` builds its own `argparse.ArgumentParser(prog="seer")` — this
    is `seer` as a top-level command, not `nebulai seer` grafted on."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage: seer" in out
    for cmd in _SUBCOMMANDS:
        assert cmd in out, f"{cmd!r} missing from standalone `seer --help`"


def test_root_is_an_argument_of_the_seer_level_not_the_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`seer --root X serve`, not `serve --root X` — `--root` must parse
    before the sub-subcommand name at the top level too, the same quirk
    documented for the grafted form in SESSIONSEER-HANDOVER.md."""
    # Wrong order: `serve` has no `--root` of its own, so this is refused.
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--root", "/tmp/wherever"])
    assert exc.value.code != 0
    assert "unrecognized arguments" in capsys.readouterr().err

    # Right order: `--root` belongs to the `seer` level, parsed before the
    # sub-subcommand name — this must reach `serve`'s own parser and succeed.
    with pytest.raises(SystemExit) as exc:
        main(["--root", "/tmp/wherever", "serve", "--help"])
    assert exc.value.code == 0
    assert "usage: seer serve" in capsys.readouterr().out


def test_importing_nebulai_cli_does_not_import_seer() -> None:
    """The graft is severed: plain `import nebulai.cli` must not pull in
    `nebulai.seer` or any of its submodules.

    This has to run in a fresh interpreter. Inside the same pytest session,
    other `test_seer_*.py` modules import `nebulai.seer.*` directly, which
    would already be sitting in `sys.modules` by the time this test runs and
    would make an in-process check pass for the wrong reason.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import nebulai.cli\n"
            "hits = sorted(\n"
            "    m for m in sys.modules\n"
            "    if m == 'nebulai.seer' or m.startswith('nebulai.seer.')\n"
            ")\n"
            "print(repr(hits))\n",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", (
        f"importing nebulai.cli pulled in seer modules: {proc.stdout.strip()}\n"
        f"{proc.stderr}"
    )


def test_nebulai_cli_source_has_no_seer_graft() -> None:
    """Belt-and-suspenders on the above: no reference to the seer package
    survives in `nebulai/cli.py` at all, grafted or otherwise."""
    import nebulai.cli as nebulai_cli

    src = Path(nebulai_cli.__file__).read_text(encoding="utf-8")
    assert "seer" not in src
