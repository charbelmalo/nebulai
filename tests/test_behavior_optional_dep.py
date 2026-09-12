"""The base install stays torch-free (BEHAVIORAL-DIVERGENCE-PLAN.md §5.6).

The Behavior study needs a real transformer stack for two things: the pinned
sentence encoder that defines the semantic metric (§6.3.1) and the local GPT-2
capability-control arm (§5.7). Both are heavy — torch alone is a few hundred
megabytes — and neither is needed to build a token map, run the viewer's
exporters, or run the rest of this test suite.

So they live in an optional dependency group, `behavior-local`, and the rule
that makes the group meaningful is: **nothing under `src/nebulai/` may import
torch, transformers or sentence-transformers at module scope.** A single
top-level import would make the whole package unimportable without the group,
which is the failure mode the group exists to prevent — and it is exactly the
kind of thing that gets added by accident while debugging and never noticed,
because the machine doing the debugging has the group installed.

The test is an AST walk rather than an import-time check on purpose: it holds
whether or not torch happens to be installed on the machine running it, so it
cannot pass for the wrong reason.
"""

import ast
import tomllib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "nebulai"
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

#: Heavy optional roots. A module-scope import of any of these is the failure.
HEAVY = {"torch", "transformers", "sentence_transformers", "accelerate", "safetensors_torch"}

PY_FILES = sorted(SRC.rglob("*.py"))


def _module_scope_imports(tree: ast.Module) -> set[str]:
    """Roots imported at module scope only.

    An import inside a function or method body is deliberate laziness and is
    what the optional group asks for; an import inside `if TYPE_CHECKING:` never
    executes at runtime. Both are allowed. Anything else at depth 0 is not.
    """
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING:` blocks do not execute; anything else at module
            # scope does, so its imports count.
            test = ast.unparse(node.test)
            if "TYPE_CHECKING" in test:
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Import):
                    roots |= {a.name.split(".")[0] for a in inner.names}
                elif isinstance(inner, ast.ImportFrom) and inner.level == 0 and inner.module:
                    roots.add(inner.module.split(".")[0])
    return roots


def test_there_is_something_to_check():
    """Guard against the walk silently finding no files and passing."""
    assert len(PY_FILES) > 30


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_module_imports_the_heavy_stack_at_module_scope(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offending = _module_scope_imports(tree) & HEAVY
    assert not offending, (
        f"{path.relative_to(SRC)} imports {sorted(offending)} at module scope. "
        f"The base install must stay torch-free (§5.6): move the import inside "
        f"the function that needs it and raise EmbedderUnavailable / "
        f"AdapterError with the `uv sync --group behavior-local` instruction "
        f"when it fails."
    )


# --------------------------------------------------------------------------
# the group itself
# --------------------------------------------------------------------------


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_the_behavior_local_group_exists_and_is_not_a_base_dependency():
    pp = _pyproject()
    groups = pp.get("dependency-groups", {})
    assert "behavior-local" in groups, "§5.6's optional group is missing"
    base = " ".join(pp["project"]["dependencies"]).lower()
    for heavy in ("torch", "transformers", "sentence-transformers"):
        assert heavy not in base, f"{heavy} must not be a base dependency"


@pytest.mark.parametrize("pkg", ["torch", "transformers", "sentence-transformers"])
def test_every_member_of_the_group_is_pinned_to_an_exact_version(pkg):
    """`>=` here would make the semantic judge a moving target.

    §6.3.1 requires the encoder's identity AND revision in the manifest. A
    floating torch can change kernel-level numerics between runs, so the same
    pinned checkpoint would stop producing the same vectors — the pin would be
    on the weights only, and the study's distances would drift underneath it.
    """
    spec = [s for s in _pyproject()["dependency-groups"]["behavior-local"] if s.startswith(pkg)]
    assert spec, f"{pkg} is not in the behavior-local group"
    assert "==" in spec[0], f"{spec[0]!r} is not an exact pin"


def test_importing_the_behavior_package_needs_none_of_it():
    """The package must import on a base install. If torch happens to be
    present this still passes — the AST test above is what makes it meaningful."""
    import nebulai.behavior.analyze  # noqa: F401
    import nebulai.behavior.cli  # noqa: F401
    import nebulai.behavior.embed  # noqa: F401
    import nebulai.behavior.export  # noqa: F401
    import nebulai.behavior.runner  # noqa: F401
    import nebulai.behavior.stats  # noqa: F401


def test_the_unavailable_encoder_refuses_with_an_install_instruction(monkeypatch):
    """No fallback to the LAN worker and none to HashEmbedder (§6.3.1)."""
    import builtins

    from nebulai.behavior.embed import EmbedderUnavailable, LocalSentenceEmbedder

    real = builtins.__import__

    def blocked(name, *a, **kw):
        if name.split(".")[0] in {"torch", "sentence_transformers"}:
            raise ImportError(f"blocked for the test: {name}")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(EmbedderUnavailable) as exc:
        LocalSentenceEmbedder()._load()
    msg = str(exc.value)
    assert "uv sync --group behavior-local" in msg
    assert "no fallback" in msg.lower()
