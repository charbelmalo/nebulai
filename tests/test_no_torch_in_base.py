"""The base install must stay torch-free (Attractors P5 / D1b exit criterion).

D1 chose "both": a numpy forward pass for everything that ships, and torch only
for the one study that needs gradients. That choice is worth exactly as much as
it is enforced. The failure mode it guards against is not dramatic — someone
adds `import torch` at the top of a backend module because it was convenient
for one function, and six months later `pip install nebulai` pulls a couple of
gigabytes and the static deploy has a dependency it cannot satisfy.

So this file checks the boundary three ways:

* **statically**, by parsing every module in the package — this catches an
  import even in a venv where torch happens to be installed, and it catches
  lazy imports inside functions, which a smoke test never would;
* **dynamically**, in a fresh interpreter with torch made unimportable, because
  a static scan cannot see an import that arrives through a string; and
* **in the metadata**, because the whole point is what a user's resolver does.

The third one is the one that actually describes `pip install nebulai`.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "nebulai"
ORGANISMS = PKG / "organisms"

# The gradient stack. `datasets` and `peft` are here for the same reason torch
# is: they are the organisms extra's dependencies and nothing that ships may
# reach for them.
HEAVY = {"torch", "transformers", "peft", "datasets", "accelerate"}


def _modules() -> list[Path]:
    return sorted(p for p in PKG.rglob("*.py") if ORGANISMS not in p.parents)


def _imported_roots(path: Path) -> set[str]:
    """Every top-level module name this file imports, at any nesting depth.

    `ast.walk` rather than a scan of the module body on purpose: an import
    hidden inside a function is still an import, and "it's lazy" is not a
    defence when the dependency is two gigabytes and the extra exists.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # a relative import is ours
                roots.add(node.module.split(".")[0])
    return roots


# ── the static scan ──────────────────────────────────────────────────────────


def test_no_shipped_module_imports_the_gradient_stack() -> None:
    offenders = {}
    for path in _modules():
        hits = _imported_roots(path) & HEAVY
        if hits:
            offenders[str(path.relative_to(REPO))] = sorted(hits)
    assert offenders == {}, (
        f"these shipped modules import the organisms extra's dependencies: {offenders}. "
        "Move the code under src/nebulai/organisms/ and call require_torch()."
    )


def test_the_scan_actually_covers_the_package() -> None:
    """A scan over an empty list passes every assertion in this file."""
    paths = _modules()
    assert len(paths) > 30
    names = {p.name for p in paths}
    assert {"cli.py", "weights.py", "absorbing.py", "persona.py"} <= names
    # and it really does see imports — numpy is everywhere, so if the parser
    # were silently returning nothing this would notice
    assert any("numpy" in _imported_roots(p) for p in paths)


def test_the_scan_would_catch_a_lazy_import(tmp_path: Path) -> None:
    """The guard's own guard: a function-level `import torch` must be found."""
    f = tmp_path / "sneaky.py"
    f.write_text("def train():\n    import torch\n    return torch\n")
    assert "torch" in _imported_roots(f)


def test_the_organisms_package_is_the_one_place_that_may_import_torch() -> None:
    """It is excluded from the scan above; check the exclusion is not vacuous —
    the package must exist, and importing it must not itself need torch."""
    assert (ORGANISMS / "__init__.py").is_file()
    assert "torch" not in _imported_roots(ORGANISMS / "__init__.py")


# ── the fresh interpreter ────────────────────────────────────────────────────

_BLOCKER = """
import sys

class _NoTorch:
    BANNED = {heavy!r}

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BANNED:
            raise ImportError("blocked by test_no_torch_in_base: " + name)
        return None

sys.meta_path.insert(0, _NoTorch())
sys.path.insert(0, {src!r})

import nebulai.cli
import nebulai.seer.cli
import nebulai.backend.absorbing
import nebulai.backend.persona
import nebulai.backend.interp.llama_numpy
import nebulai.organisms

assert not any(m.split(".")[0] in _NoTorch.BANNED for m in sys.modules)
assert nebulai.organisms.have_torch() is False
print("ok")
"""


def test_the_clis_import_with_torch_made_unimportable() -> None:
    """Run in a subprocess so the blocker sees a genuinely cold import.

    In-process this would be theatre: pytest has already imported half the
    package by the time this file runs, and `sys.modules` would serve every one
    of these lines without consulting the blocker at all.
    """
    code = _BLOCKER.format(heavy=sorted(HEAVY), src=str(REPO / "src"))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert proc.stdout.strip().endswith("ok")


def test_require_torch_gives_the_install_line_not_a_traceback() -> None:
    from nebulai.organisms import OrganismsNotInstalled, have_torch, require_torch

    if have_torch():
        pytest.skip("torch is installed in this venv; the refusal path cannot be exercised")
    with pytest.raises(OrganismsNotInstalled) as e:
        require_torch()
    assert "nebulai[organisms]" in str(e.value)


def test_have_torch_does_not_import_torch() -> None:
    """Asking the question must not pay the price of the answer."""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import nebulai.organisms as o; o.have_torch();"
        "print('torch' in sys.modules)" % str(REPO / "src")
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "False"


# ── the metadata, which is what `pip install nebulai` actually reads ─────────


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def test_torch_is_an_extra_and_not_a_core_dependency() -> None:
    doc = _pyproject()
    core = " ".join(doc["project"]["dependencies"]).lower()
    for name in HEAVY:
        assert name not in core, f"{name} must not be a core dependency"


def test_the_organisms_extra_declares_the_gradient_stack() -> None:
    doc = _pyproject()
    extras = doc["project"]["optional-dependencies"]
    assert "organisms" in extras
    names = {req.split("[")[0].split(">")[0].split("=")[0].strip().lower() for req in extras["organisms"]}
    # the plan's list (docs/ATTRACTORS-PLAN.md, Phase 5)
    assert {"torch", "transformers", "peft", "datasets"} <= names
