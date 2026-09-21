"""`seer keyword` — attributing a word to whoever put it in the context window.

The fixture `claude-session.jsonl` is not a tidy example. Every line in it
reproduces one shape that made a naive count wrong when measured against real
transcripts in `~/.claude/projects`:

* line 1 files a CLAUDE.md sentence twice — once under `attachment`, once as the
  harness's `rendered[]` copy;
* line 4 files one tool result twice — the block the model saw, and the
  harness's structured `toolUseResult.stdout` copy of the same bytes;
* every message line repeats the project path in `cwd`, and that path is
  `/Users/x/Developer/face-kit`, so a search for "face" hits it on every line;
* line 3 carries a base64 thinking signature that happens to contain `/face/`;
* line 6 carries both "Hugging Face" (a real word-bounded hit that means
  nothing about the task) and "huggingface-cli" (a compound a substring grep
  would have scored);
* line 12 is a line kind this module has no rule for, and must surface as
  `unclassified` rather than vanish;
* lines 13-15 are one queued message filed three times — the client's queue
  ledger writes it on `enqueue` and again on `remove`, and only the middle line
  delivers it to the model;
* line 16 arrives in the same shape as a queued message but is a background-task
  notification the harness wrote, and line 17 is one sent by another session —
  neither is the operator's word and neither may be counted as such.

The arithmetic those traps produce is the whole test: 13 counted occurrences out
of 42 raw matches in 4.2 KB.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nebulai.seer.sessionlog import (
    _inflections_of,
    CHANNELS,
    DEFAULT_PROJECTS_ROOT,
    Occurrence,
    Origin,
    Sense,
    SessionLogError,
    SessionRef,
    build_pattern,
    classify,
    discover_sessions,
    enabled_channels,
    refine_attachment,
    refine_line_type,
    resolve_project,
    scan,
    session_title,
)

FIXTURES = Path(__file__).parent / "fixtures" / "seer"
TRANSCRIPT = FIXTURES / "claude-session.jsonl"


@pytest.fixture
def ref() -> SessionRef:
    return SessionRef(TRANSCRIPT, "ses-fixture", "Face exclusion work",
                      TRANSCRIPT.stat().st_size)


@pytest.fixture
def report(ref: SessionRef):
    return scan("face", [ref])


def sense_regex(body: str) -> Sense:
    import re
    return Sense("s", re.compile(body, re.IGNORECASE))


# ── the headline arithmetic ──────────────────────────────────────────────────


def test_counts_only_the_text_that_reached_the_model_once(report):
    """13 counted, from 42 raw substring matches in the file."""
    raw = TRANSCRIPT.read_text().lower().count("face")
    assert raw == 42, f"fixture drifted: {raw} raw substring matches"
    assert report.total == 13


def test_origin_rollup_is_the_diagnosis(report):
    assert report.by_origin == {
        "standing": 2,      # CLAUDE.md + the invoked skill's body
        "human": 2,         # the operator's own turn, and their queued message
        "model": 3,         # prose, thinking, tool argument
        "environment": 3,   # tool result, attached file, the peer's message
        "harness": 3,       # skill listing, system prompt, task notification
    }


def test_every_counted_occurrence_names_a_known_channel(report):
    for occ in report.occurrences:
        assert occ.channel in CHANNELS
        assert CHANNELS[occ.channel].origin is occ.origin


# ── trap 1: the rendered duplicate ───────────────────────────────────────────


def test_rendered_copy_is_suppressed_not_counted(report):
    """Every attachment stores its payload twice; counting both doubles it."""
    assert "meta.rendered_copy" not in report.by_channel
    # one per attachment line that mentions the word: 1, 5, 6, 7, 10, 11, 16
    assert report.suppressed["meta.rendered_copy"] == 7


def test_the_rendered_copy_is_caught_twice_over(ref):
    """Turning the rendered channel ON must still not inflate the total.

    Two independent defences cover this duplicate. The channel rule is the
    cheap one and reports what it held; the snippet fold is the backstop for a
    rendered copy the rule failed to recognise. Measured on a real 283-line
    session, all 16 rendered copies fold exactly, so the two agree — this test
    pins that agreement on the fixture, because a future rendered shape that
    the rule missed AND the fold missed is the one way this module could start
    double-counting silently.
    """
    louder = scan("face", [ref],
                  channels=enabled_channels(include=["meta.rendered_copy"]))
    assert louder.stats.duplicate_fields_folded == 6   # 5 rendered + 1 tool result
    # Two survive, and neither is a double-count. One is the reminder line,
    # whose own channel is off by default, so with nothing left to fold against
    # its rendered twin stands alone. The other is the task notification, whose
    # rendered copy wraps the text in a "NOT USER INPUT" banner close enough to
    # the match to change the snippet — different bytes are not a duplicate, and
    # the fold is right to keep both. Every rendered copy that really was the
    # same sentence as an enabled channel's folded away.
    assert louder.by_channel["meta.rendered_copy"] == 2
    assert louder.total == report_total(ref) + 2


def report_total(ref) -> int:
    return scan("face", [ref]).total


# ── trap 2: the tool result filed twice ──────────────────────────────────────


def test_one_tool_result_filed_under_two_paths_folds_to_one(report):
    """`message.content[].content` and `toolUseResult.stdout`, same bytes."""
    assert report.stats.duplicate_fields_folded == 1
    # the model-facing record is the one kept, per _FOLD_PRIORITY
    assert report.by_channel["env.tool_result"] == 1
    assert "env.stdout" not in report.by_channel


def test_identical_text_from_the_same_path_is_kept(ref, tmp_path):
    """The fold must not swallow genuine repetition within one field."""
    line = {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "face"}, {"type": "text", "text": "face"}]}}
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(line) + "\n")
    rep = scan("face", [SessionRef(p, "s", None, p.stat().st_size)])
    assert rep.total == 2
    assert rep.stats.duplicate_fields_folded == 0


# ── trap 3: bookkeeping that is not content ──────────────────────────────────


def test_the_project_path_in_cwd_is_never_counted(report):
    """`cwd` repeats the project path on every message line."""
    assert "meta.cwd" not in report.by_channel
    assert report.suppressed["meta.cwd"] == 10


def test_a_term_that_lives_only_in_cwd_counts_zero_and_says_why(ref):
    rep = scan("Developer", [ref])
    assert rep.total == 0
    assert rep.suppressed["meta.cwd"] == 10
    payload = rep.to_dict()
    assert payload["totals"]["counted"] == 0
    assert payload["totals"]["suppressed"] == 10


def test_base64_signature_is_classified_as_opaque(report):
    assert report.suppressed["meta.opaque"] == 1


def test_harness_reminders_are_off_by_default(report):
    assert report.suppressed["harness.reminder"] == 1


# ── trap 4: the queue files one message three times ──────────────────────────


def test_a_queued_message_is_counted_once_where_it_reached_the_model(report):
    """Enqueue, deliver, remove — three lines, one message, one count."""
    assert report.by_channel["human.queued"] == 1
    assert report.suppressed["meta.queue_ledger"] == 2
    queued = [o for o in report.occurrences if o.channel == "human.queued"]
    assert "keep the face unblocked" in queued[0].snippet
    assert queued[0].path == ".attachment.prompt"


def test_counting_the_queue_ledger_trebles_the_message(ref):
    """Why the ledger is a suppressed channel and not a per-line fold.

    The three copies are on three *different* lines, so `_fold_duplicate_fields`
    — which only ever looks within one line — cannot see them. Turning the
    ledger on is the measurement that proves it: the same sentence is counted
    three times over.
    """
    louder = scan("face", [ref],
                  channels=enabled_channels(include=["meta.queue_ledger"]))
    assert louder.by_channel["meta.queue_ledger"] == 2
    assert louder.total == scan("face", [ref]).total + 2
    assert louder.stats.duplicate_fields_folded == 1   # unchanged: not one line


def test_a_background_notification_is_the_harness_not_the_operator(report):
    """It arrives as a queued message. The operator did not write it."""
    assert report.by_channel["harness.task_notification"] == 1
    hit = next(o for o in report.occurrences
               if o.channel == "harness.task_notification")
    assert hit.origin is Origin.HARNESS
    assert "Background command" in hit.snippet


def test_a_peer_sessions_message_is_not_the_operators_either(report):
    assert report.by_channel["env.peer_message"] == 1
    hit = next(o for o in report.occurrences if o.channel == "env.peer_message")
    assert hit.origin is Origin.ENVIRONMENT


def test_refine_attachment_splits_the_queue_by_who_sent_it():
    def q(**att):
        return refine_attachment({"attachment": {"type": "queued_command", **att}})
    assert q(commandMode="prompt", origin={"kind": "human"}) == "queued_command"
    assert q(commandMode="prompt") == "queued_command"
    assert q() == "queued_command"
    assert q(commandMode="task-notification") == "queued_command:notification"
    assert q(commandMode="hook-event") == "queued_command:notification"
    assert q(commandMode="prompt", origin={"kind": "peer"}) == "queued_command:peer"
    # An author this module has never seen is not quietly the operator's.
    assert q(commandMode="prompt", origin={"kind": "future"}) == "queued_command:peer"
    # Any other attachment type passes through untouched.
    assert refine_attachment({"attachment": {"type": "file"}}) == "file"
    assert refine_attachment({"attachment": None}) is None
    assert refine_attachment({}) is None


# ── trap 5: substrings lie ───────────────────────────────────────────────────


def test_word_boundary_rejects_compounds_and_reports_them(report):
    """`huggingface-cli` is not a hit, and the caller is told it was rejected."""
    assert "huggingface" in report.compounds_rejected
    assert report.compounds_rejected["huggingface"] == 2  # attachment + rendered
    assert set(report.forms) == {"face"}


def test_the_compound_tally_is_what_a_substring_grep_would_have_matched(report):
    """Deliberately spans suppressed channels too — a grep has no channels."""
    assert sum(report.compounds_rejected.values()) >= 2


def test_regex_mode_uses_the_term_verbatim(ref):
    rep = scan(r"exclu\w+", [ref], regex=True)
    forms = set(rep.forms)
    assert forms == {"exclude", "excluded", "exclusion"}, forms


def test_case_sensitivity_is_honoured(ref):
    assert scan("Face", [ref], case_sensitive=True).total == 1   # "Hugging Face"
    assert scan("Face", [ref], case_sensitive=False).total == 13


# ── who wrote it ─────────────────────────────────────────────────────────────


def test_a_tool_result_is_not_attributed_to_the_operator(report):
    """Claude Code files tool results under `type: "user"`. Counting them as the
    operator's words would put the machine's output in their mouth."""
    human = [o for o in report.occurrences if o.origin is Origin.HUMAN]
    assert {o.channel for o in human} == {"human.prompt", "human.queued"}
    assert "keep the face visible" in human[0].snippet


def test_refine_line_type_splits_user_by_who_wrote_it():
    assert refine_line_type({"type": "user", "message": {"content": "hi"}}) == "user-prompt"
    assert refine_line_type({"type": "user", "toolUseResult": {}}) == "user-result"
    assert refine_line_type({"type": "user", "isMeta": True}) == "user-result"
    assert refine_line_type({"type": "user", "message": {"content": [
        {"type": "tool_result", "content": "x"}]}}) == "user-result"
    assert refine_line_type({"type": "assistant"}) == "assistant"


def test_the_operators_echoed_prompt_is_not_counted_twice(report):
    assert report.suppressed["human.prompt_echo"] == 1
    assert report.by_channel.get("human.prompt") == 1


def test_claude_md_and_a_skill_body_are_standing_not_harness(report):
    standing = {o.channel for o in report.occurrences if o.origin is Origin.STANDING}
    assert standing == {"standing.project_instructions", "standing.skill_body"}


def test_the_agents_own_words_are_split_three_ways(report):
    model = {o.channel for o in report.occurrences if o.origin is Origin.MODEL}
    assert model == {"model.text", "model.thinking", "model.tool_input"}


def test_an_attached_files_content_counts_but_its_path_does_not(report):
    assert report.by_channel["env.file_attachment"] == 1
    assert "env.file_path" not in report.by_channel
    assert report.suppressed["env.file_path"] == 2  # filename + filePath


# ── the visible-gap promise ──────────────────────────────────────────────────


def test_an_unknown_transcript_field_surfaces_as_unclassified(report):
    """A new line kind must be reported, not silently miscounted."""
    assert report.suppressed["unclassified"] == 1
    assert "unclassified" not in report.by_channel


def test_classify_falls_back_to_unclassified():
    assert classify("some-future-kind", None, ".whatever") == "unclassified"


# ── senses and exclusions ────────────────────────────────────────────────────


def test_senses_classify_and_never_drop(ref):
    rep = scan("face", [ref], senses=[
        Sense("idiom", build_pattern("face value")[0]),
        sense_regex("exclu|mask"),
    ])
    assert rep.by_sense["idiom"] == 1
    assert rep.by_sense["s"] >= 3
    # everything the senses did not match is still counted, under its own name
    assert sum(rep.by_sense.values()) == rep.total
    assert rep.by_sense.get("unclassified", 0) == rep.total - rep.by_sense["idiom"] - rep.by_sense["s"]


def test_the_first_matching_sense_wins(ref):
    import re
    rep = scan("face", [ref], senses=[
        Sense("first", re.compile("e", re.I)),
        Sense("second", re.compile("exclu", re.I)),
    ])
    assert "second" not in rep.by_sense


def test_exclude_reports_exactly_what_it_removed(ref):
    import re
    rep = scan("face", [ref], excludes=[("idiom", re.compile("face value"))])
    assert rep.excluded == {"idiom": 1}
    assert rep.total == 12
    assert rep.to_dict()["totals"]["excluded"] == 1


# ── channel selection ────────────────────────────────────────────────────────


def test_only_channel_replaces_the_default_set(ref):
    rep = scan("face", [ref], channels=enabled_channels(only=["standing.project_instructions"]))
    assert rep.by_channel == {"standing.project_instructions": 1}
    assert rep.total == 1


def test_a_typo_in_a_channel_name_is_an_error_not_a_silent_zero():
    with pytest.raises(SessionLogError) as e:
        enabled_channels(include=["standing.claude_md"])
    assert "unknown channel" in str(e.value)


def test_the_default_set_is_exactly_the_channels_marked_default():
    assert enabled_channels() == {c for c, ch in CHANNELS.items() if ch.default}


# ── the report contract ──────────────────────────────────────────────────────


def test_the_report_refuses_to_claim_context_persistence(report):
    """How long a block stayed in the window is not in the transcript."""
    fid = report.to_dict()["fidelity"]
    assert fid["context_persistence"] == "missing"
    assert fid["counts"] == "deterministic"
    assert fid["sense_classification"] == "heuristic"


def test_the_report_is_json_serialisable(report):
    payload = json.dumps(report.to_dict())
    assert json.loads(payload)["schema"] == "seer.keyword/1"


def test_suppressed_channels_carry_their_reason(report):
    rows = report.to_dict()["suppressed_channels"]
    assert rows == sorted(rows, key=lambda r: -r["hits"])
    for row in rows:
        assert row["why"] and row["label"]


def test_samples_are_quota_limited_per_channel_and_sense(ref):
    rep = scan("face", [ref], samples_per_channel=1)
    seen = [(s.channel, s.sense) for s in rep.samples]
    assert len(seen) == len(set(seen))


def test_scanning_nothing_is_not_an_error():
    rep = scan("face", [])
    assert rep.total == 0
    assert rep.to_dict()["scanned"]["sessions"] == 0


def test_a_term_absent_from_the_corpus_reports_zero(ref):
    rep = scan("tarantula", [ref])
    assert rep.total == 0
    assert rep.forms == {}
    assert rep.compounds_rejected == {}


# ── discovery on disk ────────────────────────────────────────────────────────


def mkproject(root: Path, slug: str, sessions: dict[str, str | None]) -> Path:
    d = root / slug
    d.mkdir(parents=True)
    for sid, title in sessions.items():
        (d / f"{sid}.jsonl").write_text(json.dumps({"type": "user"}) + "\n")
        if title is not None:
            side = d / sid
            side.mkdir()
            (side / "custom-title.json").write_text(json.dumps({"customTitle": title}))
    return d


def test_session_title_comes_from_the_sidecar(tmp_path):
    d = mkproject(tmp_path, "-Users-x-proj", {"aaa": "Fix the grade", "bbb": None})
    assert session_title(d / "aaa.jsonl") == "Fix the grade"
    assert session_title(d / "bbb.jsonl") is None


def test_discover_sessions_orders_by_size(tmp_path):
    d = mkproject(tmp_path, "-Users-x-proj", {"aaa": None, "bbb": None})
    (d / "bbb.jsonl").write_text("x" * 500)
    refs = discover_sessions(d)
    assert [r.session_id for r in refs] == ["bbb", "aaa"]


def test_resolve_project_encodes_a_working_directory(tmp_path):
    mkproject(tmp_path, "-Users-x-Developer-gesture-sorcery-game-kit", {"a": None})
    got = resolve_project("/Users/x/Developer/gesture_sorcery_game_kit", tmp_path)
    assert got.slug == "-Users-x-Developer-gesture-sorcery-game-kit"
    assert got.matched_how == "encoded working directory"


def test_resolve_project_prefers_the_slug_that_ends_with_the_name(tmp_path):
    """A project and its `-app` sibling both contain the name as a substring."""
    mkproject(tmp_path, "-Users-x-Developer-gesture-sorcery-game-kit", {"a": None})
    mkproject(tmp_path, "-Users-x-Developer-gesture-sorcery-game-kit-app", {"b": None})
    got = resolve_project("gesture_sorcery_game_kit", tmp_path)
    assert got.slug.endswith("game-kit")
    assert "ends with" in got.matched_how


def test_resolve_project_refuses_real_ambiguity_and_lists_the_candidates(tmp_path):
    mkproject(tmp_path, "-Users-x-alpha-kit-one", {"a": None})
    mkproject(tmp_path, "-Users-x-alpha-kit-two", {"b": None})
    with pytest.raises(SessionLogError) as e:
        resolve_project("alpha-kit", tmp_path)
    assert "-Users-x-alpha-kit-one" in str(e.value)
    assert "-Users-x-alpha-kit-two" in str(e.value)


def test_resolve_project_is_explicit_when_nothing_matches(tmp_path):
    mkproject(tmp_path, "-Users-x-proj", {"a": None})
    with pytest.raises(SessionLogError, match="no project directory matches"):
        resolve_project("nope", tmp_path)


def test_resolve_project_accepts_a_slug_verbatim(tmp_path):
    mkproject(tmp_path, "-Users-x-proj", {"a": None})
    assert resolve_project("-Users-x-proj", tmp_path).matched_how == "directory name"


def test_a_missing_projects_root_is_an_error(tmp_path):
    with pytest.raises(SessionLogError, match="no projects directory"):
        resolve_project("anything", tmp_path / "absent")


def test_the_default_projects_root_is_claude_codes_own():
    assert DEFAULT_PROJECTS_ROOT.parts[-2:] == (".claude", "projects")


# ── robustness ───────────────────────────────────────────────────────────────


def test_an_unparsable_line_is_counted_not_fatal(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"type":"user","message":{"content":"face"}}\n'
                 '{"type":"user","message":{"content":"face"\n'   # truncated
                 '["face"]\n')                                    # not an object
    rep = scan("face", [SessionRef(p, "s", None, p.stat().st_size)])
    assert rep.total == 1
    assert rep.stats.lines_unparsable == 2


def test_the_prefilter_examines_only_lines_that_could_match(ref):
    rep = scan("face", [ref])
    assert rep.stats.lines == 17
    assert rep.stats.lines_prefiltered == 17  # this fixture is dense on purpose
    quiet = scan("tarantula", [ref])
    assert quiet.stats.lines_prefiltered == 0


def test_build_pattern_escapes_regex_metacharacters():
    pat, compound = build_pattern("a.b")
    assert pat.search("a.b")
    assert not pat.search("axb")
    assert compound.search("za.bz")


# ── gaps the module found in itself ──────────────────────────────────────────


def test_the_on_the_wire_tool_input_copy_is_recognised(tmp_path):
    """A third duplicate, found by this module's own `unclassified` tally.

    `assistant.wireToolInputs.<tool_use_id>.command` repeats the arguments
    already recorded in `message.content[].input.command`. Before it had a
    rule, 163 real occurrences landed in `unclassified` — which is the failure
    mode working as designed: a gap that reports itself.
    """
    line = {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tu-9", "name": "Bash",
                 "input": {"command": "grep face src/"}}]},
            "wireToolInputs": {"tu-9": {"command": "grep face src/"}}}
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(line) + "\n")
    ref = SessionRef(p, "s", None, p.stat().st_size)

    rep = scan("face", [ref])
    assert "unclassified" not in rep.suppressed
    assert rep.suppressed["meta.wire_copy"] == 1
    assert rep.by_channel == {"model.tool_input": 1}
    assert rep.total == 1


def test_inflections_of_the_term_are_reported_not_buried(tmp_path):
    """`faces` is correctly not `face` — but the caller wanted it, so say so."""
    line = {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "the face and the faces and faced and a facet"}]}}
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(line) + "\n")
    rep = scan("face", [SessionRef(p, "s", None, p.stat().st_size)])

    assert rep.total == 1
    assert rep.inflections_rejected == {"faces": 1, "faced": 1}
    assert "facet" in rep.compounds_rejected          # a different word
    assert "facet" not in rep.inflections_rejected


def test_a_letter_dropping_inflection_is_declared_unreadable_not_zero(tmp_path):
    """`facing` does not contain `face`, so this scan never sees it.

    The prefilter is built from the term. A line whose only match is `facing`
    is not read at all, and reporting it as absent would be a lie about
    coverage — so the report says the tally cannot see those forms.
    """
    assert "facing" not in _inflections_of("face")
    line = {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "facing the wall, refacing nothing"}]}}
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(line) + "\n")
    rep = scan("face", [SessionRef(p, "s", None, p.stat().st_size)])
    assert rep.total == 0
    assert rep.compounds_rejected == {}
    assert "never read" in rep.to_dict()["inflections_note"]


def test_regex_mode_makes_no_inflection_claim(tmp_path):
    """The caller wrote the pattern; guessing at its inflections would be noise."""
    line = {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "face and faces"}]}}
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(line) + "\n")
    rep = scan("face", [SessionRef(p, "s", None, p.stat().st_size)], regex=True)
    assert rep.inflections_rejected == {}


def test_the_compounds_note_states_its_own_scope(report):
    note = report.to_dict()["compounds_note"]
    assert "suppressed" in note and "grep has no" in note


def test_a_base64_blob_is_counted_but_not_listed_as_a_word(tmp_path):
    """The long ones are real grep matches and they are not words.

    On the real corpus a dozen 200-character signature strings containing the
    term's letters took a dozen places in the top-40 list, pushing out the
    neighbouring words the list exists to show. They are still counted.
    """
    blob = "aGVsbG8" + "x" * 60 + "face" + "y" * 40
    line = {"type": "assistant", "message": {"content": [
        {"type": "text", "text": f"the face, an interface, and {blob}"}]}}
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(line) + "\n")
    rep = scan("face", [SessionRef(p, "s", None, p.stat().st_size)])

    assert rep.total == 1
    assert list(rep.compounds_rejected) == ["interface"]
    assert rep.unwordlike == (1, 1)
    d = rep.to_dict()
    assert d["compounds_unwordlike"]["hits"] == 1
    assert "base64" in d["compounds_unwordlike"]["note"]
