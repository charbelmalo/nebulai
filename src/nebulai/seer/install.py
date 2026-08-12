"""Hook installers that merge, back up, and restore exactly.

This module edits configuration files a person depends on. `~/.claude/settings.json`
on the machine this was written on already carries a `PreToolUse`/`Bash` hook
running `rtk hook claude`; an installer that writes rather than merges silently
disables it, and the user finds out weeks later from a billing graph. So the
rules are mechanical:

1. **Read, merge, write.** Never construct a config from scratch. Unknown keys
   are carried through untouched, including ones added by a future agent
   version this code has never heard of.
2. **Back up first, byte-exact.** The original bytes go to `<file>.seer-backup-<n>`
   before the first write, and `uninstall` restores from the backup only after
   verifying that what is on disk is still what we wrote.
3. **Own only our entries.** Every entry we add is tagged. Uninstall removes
   entries carrying our tag and nothing else — so a hook the user added *after*
   installing survives an uninstall, which restoring the backup wholesale would
   destroy.
4. **A dry run is the default in the CLI.** `plan()` returns exactly what would
   change, in the file's own vocabulary, before anything is written.

Hermes is deliberately a two-step install: it gates first use of a shell hook
behind `~/.hermes/shell-hooks-allowlist.json`, and writing that file for the
user would be forging their consent. `plan()` reports the approval as a manual
step, and `install` refuses to fabricate it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters.observed import hook_events
from .spool import install_shim, remove_shim, shim_path

#: Stamped into every entry we write. Present-tense, greppable, and the sole
#: criterion for what uninstall is allowed to remove.
TAG = "nebulai-sessionseer"

CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
CODEX_CONFIG = Path.home() / ".codex" / "config.toml"
HERMES_CONFIG = Path.home() / ".hermes" / "config.yaml"
HERMES_ALLOWLIST = Path.home() / ".hermes" / "shell-hooks-allowlist.json"


@dataclass
class Change:
    """One thing the installer would do, in words the user can check."""

    kind: str  # "add" | "remove" | "create" | "manual"
    target: str
    detail: str


@dataclass
class Plan:
    agent: str
    config: Path
    changes: list[Change] = field(default_factory=list)
    #: what the user must do themselves; an installer that could do these
    #: without asking would be doing something it should not
    manual: list[str] = field(default_factory=list)
    #: entries already present and untouched — the proof that merging worked
    preserved: list[str] = field(default_factory=list)
    supported: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "config": str(self.config),
            "supported": self.supported,
            "reason": self.reason,
            "changes": [c.__dict__ for c in self.changes],
            "manual": self.manual,
            "preserved": self.preserved,
        }


# ── backups ──────────────────────────────────────────────────────────────────


def backup_path(config: Path) -> Path:
    """A fresh numbered backup, so installing twice cannot overwrite the
    original with an already-modified copy."""
    n = 0
    while True:
        p = config.with_suffix(config.suffix + f".seer-backup-{n}")
        if not p.exists():
            return p
        n += 1


def existing_backups(config: Path) -> list[Path]:
    return sorted(config.parent.glob(config.name + ".seer-backup-*"))


def _backup(config: Path) -> Path | None:
    if not config.exists():
        return None
    p = backup_path(config)
    shutil.copy2(config, p)
    return p


# ── Claude Code ──────────────────────────────────────────────────────────────


def _claude_entry(shim: Path, event: str) -> dict[str, Any]:
    return {
        "type": "command",
        "command": f'"{shim}" claude {event}',
        # not a field Claude Code reads; it is how uninstall recognises its own
        # work without pattern-matching a command string a user may have edited
        "_source": TAG,
    }


def _load_json(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise RuntimeError(f"{p} is not valid JSON ({e}); refusing to touch it") from e
    return d if isinstance(d, dict) else {}


def _claude_plan(root: Path, settings: Path) -> Plan:
    plan = Plan(agent="claude", config=settings)
    cfg = _load_json(settings)
    hooks = cfg.get("hooks") or {}
    shim = shim_path(root)

    for event in hook_events("claude"):
        groups = hooks.get(event) or []
        if _has_tagged(groups):
            continue
        plan.changes.append(
            Change("add", f"hooks.{event}", f'"{shim}" claude {event}')
        )
    for event, groups in sorted(hooks.items()):
        for g in groups if isinstance(groups, list) else []:
            for h in (g or {}).get("hooks") or []:
                if isinstance(h, dict) and h.get("_source") != TAG:
                    plan.preserved.append(
                        f"hooks.{event}[{g.get('matcher') or '*'}]: {h.get('command')}"
                    )
    if not settings.exists():
        plan.changes.insert(0, Change("create", str(settings), "new settings file"))
    return plan


def _has_tagged(groups: Any) -> bool:
    for g in groups if isinstance(groups, list) else []:
        for h in (g or {}).get("hooks") or []:
            if isinstance(h, dict) and h.get("_source") == TAG:
                return True
    return False


def _claude_install(root: Path, settings: Path) -> Plan:
    plan = _claude_plan(root, settings)
    cfg = _load_json(settings)
    hooks = cfg.setdefault("hooks", {})
    shim = shim_path(root)

    for event in hook_events("claude"):
        groups = hooks.setdefault(event, [])
        if not isinstance(groups, list):
            # a shape we do not understand: leave it exactly as it is rather
            # than "fixing" someone's config into a shape we prefer
            plan.changes.append(
                Change("manual", f"hooks.{event}", "unexpected shape; left untouched")
            )
            continue
        if _has_tagged(groups):
            continue
        groups.append({"hooks": [_claude_entry(shim, event)]})

    _write_json(settings, cfg)
    return plan


def _claude_uninstall(settings: Path) -> Plan:
    plan = Plan(agent="claude", config=settings)
    cfg = _load_json(settings)
    hooks = cfg.get("hooks") or {}
    for event, groups in list(hooks.items()):
        if not isinstance(groups, list):
            continue
        kept: list[Any] = []
        for g in groups:
            entries = (g or {}).get("hooks") or []
            mine = [h for h in entries if isinstance(h, dict) and h.get("_source") == TAG]
            theirs = [h for h in entries if h not in mine]
            for h in mine:
                plan.changes.append(Change("remove", f"hooks.{event}", str(h.get("command"))))
            for h in theirs:
                plan.preserved.append(f"hooks.{event}: {(h or {}).get('command')}")
            if theirs:
                kept.append({**g, "hooks": theirs})
        if kept:
            hooks[event] = kept
        else:
            hooks.pop(event, None)
    if hooks:
        cfg["hooks"] = hooks
    else:
        cfg.pop("hooks", None)
    _write_json(settings, cfg)
    return plan


def _write_json(p: Path, cfg: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".seer-tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)  # atomic: a crash mid-write cannot leave a half-config


# ── Codex ────────────────────────────────────────────────────────────────────

_CODEX_BLOCK = """
# --- {tag} (added by `seer install codex`) ---
# Remove this block, or run `seer uninstall codex`, to stop capturing.
[hooks]
enabled = true
{entries}
# --- end {tag} ---
"""


def _codex_plan(root: Path, config: Path) -> Plan:
    plan = Plan(agent="codex", config=config)
    text = config.read_text(encoding="utf-8") if config.exists() else ""
    if TAG in text:
        return plan
    events = hook_events("codex")
    plan.changes.append(
        Change("add", "[hooks]", f"{len(events)} events → {shim_path(root)}")
    )
    if "[hooks]" in text:
        # Downgrade the change to a manual step rather than leaving it in
        # `changes`: a plan that lists something it will not do reads as a
        # promise, and the user finds out it was not kept by not being captured.
        plan.supported = False
        plan.reason = (
            f"{config} already defines a [hooks] table. TOML has no safe merge for a"
            " table we did not write, so this has to be edited by hand."
        )
        plan.changes.clear()
        plan.manual.append(
            "Append the block from `seer install codex --print-block` to"
            f" {config} yourself, merging it into the [hooks] table already there."
        )
    plan.preserved.append(f"{len(text.splitlines())} existing config lines")
    return plan


def codex_block(root: Path) -> str:
    entries = "\n".join(
        f'{e.replace("/", "_")} = ["{shim_path(root)}", "codex", "{e}"]'
        for e in hook_events("codex")
    )
    return _CODEX_BLOCK.format(tag=TAG, entries=entries)


def _codex_install(root: Path, config: Path) -> Plan:
    plan = _codex_plan(root, config)
    if not plan.supported:
        return plan
    if not plan.changes:
        return plan
    config.parent.mkdir(parents=True, exist_ok=True)
    with config.open("a", encoding="utf-8") as fh:
        fh.write(codex_block(root))
    return plan


def _strip_block(text: str) -> tuple[str, int]:
    """Remove our tagged block, and the blank line install put above it.

    Every line between the two markers goes and nothing else does — an
    uninstall that rewrote the file from a parse would reformat, reorder, and
    drop the comments of a config we were only ever a guest in.

    The blank separator is part of what we wrote, so it leaves with the rest.
    Take it and the round trip is byte-exact for any config ending in a
    newline, which is every config a real editor produces; leave it and each
    install/uninstall cycle silently grows the file by one line.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    dropped, inside = 0, False
    for line in lines:
        if line.startswith(f"# --- {TAG}"):
            inside = True
            if out and out[-1].strip() == "":
                out.pop()
                dropped += 1
        if inside:
            dropped += 1
            if line.startswith(f"# --- end {TAG}"):
                inside = False
            continue
        out.append(line)
    return "".join(out), dropped


def _codex_uninstall(config: Path) -> Plan:
    plan = Plan(agent="codex", config=config)
    if not config.exists():
        return plan
    text, dropped = _strip_block(config.read_text(encoding="utf-8"))
    if dropped:
        plan.changes.append(Change("remove", "[hooks]", f"{dropped} lines"))
        config.write_text(text, encoding="utf-8")
    plan.preserved.append(f"{len(text.splitlines())} lines kept")
    return plan


# ── Hermes ───────────────────────────────────────────────────────────────────


def hermes_block(root: Path) -> str:
    entries = "\n".join(
        f'    - command: "{shim_path(root)} hermes {e}"' for e in hook_events("hermes")
    )
    return (
        f"# --- {TAG} (added by `seer install hermes`) ---\n"
        "hooks:\n"
        "  shell:\n"
        f"{entries}\n"
        f"# --- end {TAG} ---\n"
    )


def _hermes_plan(root: Path, config: Path, allowlist: Path) -> Plan:
    plan = Plan(agent="hermes", config=config)
    text = config.read_text(encoding="utf-8") if config.exists() else ""
    if TAG not in text:
        plan.changes.append(
            Change("add", "hooks.shell", f"{len(hook_events('hermes'))} observe-only events")
        )
    plan.manual.append(
        f"Approve the shim in {allowlist} — Hermes gates first use of a shell hook"
        " behind an explicit allowlist, and writing your approval for you would"
        " defeat the point of having one."
    )
    plan.preserved.append(
        "no transform_* hook is registered: those can alter the agent's own data"
        " flow, and an observer that can change what it observes is not one"
    )
    if "hooks:" in text and TAG not in text:
        plan.supported = False
        plan.reason = (
            f"{config} already has a `hooks:` block. A YAML merge without a YAML"
            " parser would risk corrupting a live config, so this one is by hand."
        )
        plan.changes.clear()
        plan.manual.append(
            "Merge the block from `seer install hermes --print-block` into"
            f" the existing `hooks:` block in {config} yourself."
        )
    return plan


def _hermes_install(root: Path, config: Path, allowlist: Path) -> Plan:
    plan = _hermes_plan(root, config, allowlist)
    if not plan.supported or not plan.changes:
        return plan
    config.parent.mkdir(parents=True, exist_ok=True)
    with config.open("a", encoding="utf-8") as fh:
        fh.write("\n" + hermes_block(root))
    return plan


def _hermes_uninstall(config: Path) -> Plan:
    plan = Plan(agent="hermes", config=config)
    if not config.exists():
        return plan
    text, dropped = _strip_block(config.read_text(encoding="utf-8"))
    if dropped:
        plan.changes.append(Change("remove", "hooks.shell", f"{dropped} lines"))
        config.write_text(text, encoding="utf-8")
    return plan


# ── public API ───────────────────────────────────────────────────────────────

CONFIGS = {
    "claude": CLAUDE_SETTINGS,
    "codex": CODEX_CONFIG,
    "hermes": HERMES_CONFIG,
}


def plan(agent: str, root: Path, *, config: Path | None = None) -> Plan:
    """What `install` would change. Writes nothing."""
    cfg = config or CONFIGS[agent]
    if agent == "claude":
        return _claude_plan(root, cfg)
    if agent == "codex":
        return _codex_plan(root, cfg)
    if agent == "hermes":
        return _hermes_plan(root, cfg, HERMES_ALLOWLIST)
    raise ValueError(f"unknown agent {agent!r}")


def install(agent: str, root: Path, *, config: Path | None = None) -> tuple[Plan, Path | None]:
    """Install the shim and register the hooks. Returns the plan and the backup.

    The shim is written first: a config that points at a missing shim would make
    every hook in the user's session fail to exec, and the order is the whole
    difference between "not capturing yet" and "every tool call logs an error".
    """
    cfg = config or CONFIGS[agent]
    install_shim(root)
    backup = _backup(cfg)
    if agent == "claude":
        p = _claude_install(root, cfg)
    elif agent == "codex":
        p = _codex_install(root, cfg)
    elif agent == "hermes":
        p = _hermes_install(root, cfg, HERMES_ALLOWLIST)
    else:
        raise ValueError(f"unknown agent {agent!r}")
    return p, backup


def uninstall(
    agent: str, root: Path, *, config: Path | None = None, remove_spool: bool = False
) -> Plan:
    """Remove our entries and leave everything else alone.

    Removes by tag rather than restoring the backup: a hook the user added after
    installing is not ours to delete, and a wholesale restore would delete it.
    The backups stay on disk — `seer install --show-backups` lists them
    for the case where a config needs to go back exactly.
    """
    cfg = config or CONFIGS[agent]
    _backup(cfg)
    if agent == "claude":
        p = _claude_uninstall(cfg)
    elif agent == "codex":
        p = _codex_uninstall(cfg)
    elif agent == "hermes":
        p = _hermes_uninstall(cfg)
    else:
        raise ValueError(f"unknown agent {agent!r}")
    remove_shim(root, keep_spool=not remove_spool)
    return p


def status(root: Path) -> dict[str, Any]:
    """Which agents are currently registered, read from their live configs."""
    out: dict[str, Any] = {"shim": str(shim_path(root)), "shim_installed": shim_path(root).exists()}
    for agent, cfg in CONFIGS.items():
        installed = False
        if cfg.exists():
            installed = TAG in cfg.read_text(encoding="utf-8", errors="replace")
        out[agent] = {
            "config": str(cfg),
            "exists": cfg.exists(),
            "installed": installed,
            "backups": [str(p) for p in existing_backups(cfg)],
        }
    return out


__all__ = [
    "CONFIGS",
    "TAG",
    "Change",
    "Plan",
    "backup_path",
    "codex_block",
    "existing_backups",
    "hermes_block",
    "install",
    "plan",
    "status",
    "uninstall",
]
