"""Git snapshots at run start and run end.

Two facts per snapshot: the HEAD sha, and a SHA-256 of `git status --porcelain`.
Either alone is misleading. HEAD says which commit the run was measured
against; the status hash says whether the working tree moved while the run
happened — which is what most agent runs do and what a HEAD-only record cannot
show.

The failure this file mostly guards against is subtle and one line wide.
`git status --porcelain` prints nothing for a clean tree *and* prints nothing
when it errors. A helper that returned a bare string would make "the tree is
clean" and "we could not ask" the same value, and every run captured outside a
repository would be recorded as having changed nothing.

Offline and deterministic: every test builds its own repository in `tmp_path`
with `/usr/bin/git`, and none of them touch the checkout the suite runs in.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nebulai.seer.contract import Fidelity
from nebulai.seer.runner import GIT_BIN, _repo_context, git_snapshot

pytestmark = pytest.mark.skipif(
    not Path(GIT_BIN).exists(), reason=f"{GIT_BIN} is not installed"
)


def git(root: Path, *args: str) -> None:
    subprocess.run(
        [GIT_BIN, "-C", str(root), *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with one commit, built from nothing.

    `-c` rather than `git config` so nothing is read from or written to the
    developer's global config, and `--initial-branch` so the default-branch
    setting of the host cannot change what this test sees.
    """
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(
        [GIT_BIN, "-C", str(root), "init", "--initial-branch", "main", "-q"],
        cwd=str(root),
        check=True,
        capture_output=True,
    )
    (root / "a.txt").write_text("one\n")
    git(root, "add", "a.txt")
    git(
        root,
        "-c", "user.name=t",
        "-c", "user.email=t@example.invalid",
        "commit", "-q", "-m", "first",
    )
    return root


class TestInsideARepository:
    def test_a_clean_tree_reports_a_sha_and_a_hash(self, repo: Path) -> None:
        s = git_snapshot(repo, "start")
        assert s["phase"] == "start"
        assert len(s["head"]) == 40
        assert s["head_fidelity"] == Fidelity.DETERMINISTIC.value
        assert len(s["status_hash"]) == 64
        assert s["status_fidelity"] == Fidelity.DETERMINISTIC.value
        assert s["status_lines"] == 0
        assert s["dirty"] is False
        assert "note" not in s

    def test_an_untracked_file_moves_the_status_hash(self, repo: Path) -> None:
        """The whole reason the status hash exists: an agent that edits without
        committing leaves HEAD alone, and a HEAD-only record would say the run
        changed nothing."""
        before = git_snapshot(repo, "start")
        (repo / "b.txt").write_text("two\n")
        after = git_snapshot(repo, "end")
        assert after["head"] == before["head"]
        assert after["status_hash"] != before["status_hash"]
        assert after["dirty"] is True
        assert after["status_lines"] == 1

    def test_the_hash_is_not_the_paths(self, repo: Path) -> None:
        """A researcher's dirty file names are not something a captured run
        needs to carry. Equality is all the comparison needs."""
        (repo / "secret-plan.txt").write_text("x\n")
        s = git_snapshot(repo, "end")
        assert "secret-plan" not in str(s)

    def test_the_same_tree_hashes_the_same_twice(self, repo: Path) -> None:
        assert git_snapshot(repo, "start")["status_hash"] == (
            git_snapshot(repo, "end")["status_hash"]
        )

    def test_a_commit_moves_head_and_settles_the_tree(self, repo: Path) -> None:
        before = git_snapshot(repo, "start")
        (repo / "b.txt").write_text("two\n")
        git(repo, "add", "b.txt")
        git(
            repo,
            "-c", "user.name=t",
            "-c", "user.email=t@example.invalid",
            "commit", "-q", "-m", "second",
        )
        after = git_snapshot(repo, "end")
        assert after["head"] != before["head"]
        assert after["dirty"] is False

    def test_a_subdirectory_resolves_to_the_repository_root(self, repo: Path) -> None:
        sub = repo / "deep" / "deeper"
        sub.mkdir(parents=True)
        assert git_snapshot(sub, "start")["root_id"] == git_snapshot(repo, "start")[
            "root_id"
        ]


class TestMissingIsNotClean:
    def test_outside_a_repository_everything_is_missing(self, tmp_path: Path) -> None:
        """The honesty rule this item exists for. Not `""`, not `False`, not a
        hash of the empty string — absent, with a stated reason."""
        plain = tmp_path / "plain"
        plain.mkdir()
        s = git_snapshot(plain, "start")
        assert s["head"] is None
        assert s["head_fidelity"] == Fidelity.MISSING.value
        assert s["status_hash"] is None
        assert s["status_fidelity"] == Fidelity.MISSING.value
        assert s["status_lines"] is None
        assert s["note"]

    def test_a_missing_tree_is_not_reported_as_a_clean_one(
        self, tmp_path: Path, repo: Path
    ) -> None:
        """The one-line bug, stated as a test: `dirty` is `None` outside a
        repository and `False` inside a clean one, and the two must not be the
        same value."""
        plain = tmp_path / "plain"
        plain.mkdir()
        assert git_snapshot(plain, "start")["dirty"] is None
        assert git_snapshot(repo, "start")["dirty"] is False

    def test_a_repository_with_no_commits_has_no_head_but_has_a_tree(
        self, tmp_path: Path
    ) -> None:
        """`rev-parse HEAD` fails here because there is nothing to resolve. The
        status is still readable, so half the snapshot is present and the
        missing half says so — rather than the whole thing collapsing."""
        root = tmp_path / "empty"
        root.mkdir()
        subprocess.run(
            [GIT_BIN, "-C", str(root), "init", "--initial-branch", "main", "-q"],
            cwd=str(root),
            check=True,
            capture_output=True,
        )
        s = git_snapshot(root, "start")
        assert s["head"] is None
        assert s["head_fidelity"] == Fidelity.MISSING.value
        assert s["note"]
        assert s["status_fidelity"] == Fidelity.DETERMINISTIC.value

    def test_a_path_that_does_not_exist_is_missing_not_a_crash(
        self, tmp_path: Path
    ) -> None:
        s = git_snapshot(tmp_path / "nope", "start")
        assert s["head"] is None and s["note"]

    def test_git_is_addressed_by_absolute_path(self) -> None:
        """The agent's environment is passed through unchanged by design, so
        its `PATH` is not ours to trust for a provenance record."""
        assert GIT_BIN.startswith("/")


class TestRepoContext:
    def test_it_carries_the_branch_and_the_status_hash(self, repo: Path) -> None:
        ctx = _repo_context(repo)
        assert ctx is not None
        assert ctx["branch"] == "main"
        assert ctx["root_id"] == str(repo.resolve())
        assert len(ctx["status_hash"]) == 64
        assert ctx["dirty"] is False

    def test_outside_a_repository_it_is_none(self, tmp_path: Path) -> None:
        """`store._index` and `attach.py` both read `None` as "no repo". That
        contract is older than this change and is not allowed to move."""
        plain = tmp_path / "plain"
        plain.mkdir()
        assert _repo_context(plain) is None


class TestRunnerEmitsBothSnapshots:
    def test_a_captured_run_brackets_itself_with_two_snapshots(
        self, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end through `Runner`, with a scripted agent so the test stays
        offline: the run emits `git.snapshot` at start and at end, and the two
        are told apart by `phase` alone.

        The agent is `/bin/sh -c` printing two real `codex exec --json` lines,
        so the runner's own stream handling is exercised rather than stubbed.
        """
        from nebulai.seer.contract import EventType
        from nebulai.seer import runner as R
        from nebulai.seer.store import EventStore

        lines = (
            '{"type":"thread.started","thread_id":"th_1"}\n'
            '{"type":"turn.started"}\n'
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n'
        )
        monkeypatch.setattr(
            R, "build_command", lambda *a, **k: ["/bin/sh", "-c", f"printf '{lines}'"]
        )
        monkeypatch.setattr(R, "agent_version", lambda agent: "test")

        r = R.Runner(
            agent="codex",
            prompt="noop",
            cwd=repo,
            store=EventStore(tmp_path / "store"),
        )
        result = r.run(timeout_s=30)
        assert result.exit_code == 0

        snaps = [
            e for e in r.store.read(r.run_id)
            if e.event_type is EventType.GIT_SNAPSHOT
        ]
        assert [e.payload["phase"] for e in snaps] == ["start", "end"]
        assert all(len(e.payload["head"]) == 40 for e in snaps)
        assert all(len(e.payload["status_hash"]) == 64 for e in snaps)
        # A snapshot must survive a metadata-level export, or a data-quality
        # panel would lose the provenance it exists to show.
        assert all(e.privacy["content_level"] == "metadata" for e in snaps)

    def test_an_edit_during_the_run_shows_up_between_the_two(
        self, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What the pair is for. The scripted agent writes a file, and the end
        snapshot's status hash differs from the start's while HEAD does not
        move — the shape of almost every real agent run."""
        from nebulai.seer.contract import EventType
        from nebulai.seer import runner as R
        from nebulai.seer.store import EventStore

        monkeypatch.setattr(
            R,
            "build_command",
            lambda *a, **k: [
                "/bin/sh", "-c",
                "echo two > b.txt; "
                "printf '{\"type\":\"thread.started\",\"thread_id\":\"t\"}\n'",
            ],
        )
        monkeypatch.setattr(R, "agent_version", lambda agent: "test")

        r = R.Runner(
            agent="codex", prompt="noop", cwd=repo,
            store=EventStore(tmp_path / "store"),
        )
        r.run(timeout_s=30)
        snaps = [
            e.payload for e in r.store.read(r.run_id)
            if e.event_type is EventType.GIT_SNAPSHOT
        ]
        start, end = snaps
        assert start["head"] == end["head"]
        assert start["status_hash"] != end["status_hash"]
        assert start["dirty"] is False and end["dirty"] is True
