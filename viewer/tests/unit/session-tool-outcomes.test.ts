/** The per-tool outcome split in the session fold.
 *
 *  These lock the one decision that makes the split worth showing: a `tool_use`
 *  with no `tool_result` is `unresolved`, NOT `ok`. Folding it into ok would
 *  make every truncated log read as a clean run. They also lock the second
 *  refusal — a top-level audit-format `is_error` carries no `tool_use_id`, so
 *  it cannot be charged to a tool and is surfaced separately instead of being
 *  smeared over the histogram. */

import { describe, expect, it } from "vitest";
import { parseSessionTranscript, type SessionAnalysis, type ToolOutcome } from "../../src/chrome/sessionlog";
import { normalizeSessionAnalysis } from "../../src/chrome/sessionStore";

type Block = Record<string, unknown>;

let seq = 0;
const stamp = () => new Date(Date.UTC(2026, 0, 1, 0, 0, seq++)).toISOString();

/** One assistant response issuing the given tool calls. */
function assistant(id: string, calls: { id: string; name: string }[]): string {
  return JSON.stringify({
    type: "assistant",
    uuid: `u-${id}`,
    timestamp: stamp(),
    message: {
      id,
      model: "claude-fable-5",
      usage: { input_tokens: 10, output_tokens: 5 },
      content: calls.map((c) => ({ type: "tool_use", id: c.id, name: c.name, input: {} })),
    },
  });
}

/** A user line carrying tool_result blocks answering earlier calls. */
function results(rs: { id: string; isError?: boolean }[]): string {
  const content: Block[] = rs.map((r) => ({
    type: "tool_result",
    tool_use_id: r.id,
    content: "…",
    ...(r.isError ? { is_error: true } : {}),
  }));
  return JSON.stringify({ type: "user", uuid: `r-${seq}`, timestamp: stamp(), message: { content } });
}

const by = (os: ToolOutcome[], tool: string) => os.find((o) => o.tool === tool);

describe("toolOutcomes", () => {
  it("splits a tool's calls into ok / failed / unresolved", () => {
    const raw = [
      assistant("m1", [
        { id: "t1", name: "Read" },
        { id: "t2", name: "Read" },
        { id: "t3", name: "Read" },
      ]),
      results([{ id: "t1" }, { id: "t2", isError: true }]),
      // t3 never answered
    ].join("\n");

    const a = parseSessionTranscript(raw, "s");
    expect(by(a.toolOutcomes, "Read")).toEqual({
      tool: "Read",
      ok: 1,
      failed: 1,
      unresolved: 1,
      total: 3,
    });
  });

  it("never folds an unanswered call into ok", () => {
    const raw = assistant("m1", [{ id: "t1", name: "Bash" }]);
    const o = by(parseSessionTranscript(raw, "s").toolOutcomes, "Bash")!;
    expect(o.ok).toBe(0);
    expect(o.unresolved).toBe(1);
  });

  it("counts a failure once, on the tool that made the call", () => {
    const raw = [
      assistant("m1", [
        { id: "t1", name: "Bash" },
        { id: "t2", name: "Read" },
      ]),
      results([{ id: "t1", isError: true }, { id: "t2" }]),
    ].join("\n");

    const a = parseSessionTranscript(raw, "s");
    expect(by(a.toolOutcomes, "Bash")!.failed).toBe(1);
    expect(by(a.toolOutcomes, "Read")!.failed).toBe(0);
    expect(a.errorCount).toBe(1);
  });

  it("charges results to the right tool across turn boundaries", () => {
    // A call issued in one response is answered on a later line — the id map
    // has to survive the turn fold or the split lands on the wrong tool.
    const raw = [
      assistant("m1", [{ id: "t1", name: "Grep" }]),
      assistant("m2", [{ id: "t2", name: "Edit" }]),
      results([{ id: "t1" }]),
      results([{ id: "t2", isError: true }]),
    ].join("\n");

    const a = parseSessionTranscript(raw, "s");
    expect(by(a.toolOutcomes, "Grep")).toMatchObject({ ok: 1, failed: 0, unresolved: 0 });
    expect(by(a.toolOutcomes, "Edit")).toMatchObject({ ok: 0, failed: 1, unresolved: 0 });
  });

  it("totals agree with the tool histogram", () => {
    const raw = [
      assistant("m1", [
        { id: "t1", name: "Read" },
        { id: "t2", name: "Read" },
        { id: "t3", name: "Bash" },
      ]),
      results([{ id: "t1" }, { id: "t3", isError: true }]),
    ].join("\n");

    const a = parseSessionTranscript(raw, "s");
    const hist = new Map(a.toolHistogram);
    for (const o of a.toolOutcomes) {
      expect(o.ok + o.failed + o.unresolved).toBe(o.total);
      expect(o.total).toBe(hist.get(o.tool));
    }
    expect(a.toolOutcomes.reduce((s, o) => s + o.total, 0)).toBe(a.toolTotal);
  });

  it("sorts by total descending, ties broken by name", () => {
    const raw = assistant("m1", [
      { id: "t1", name: "Zeta" },
      { id: "t2", name: "Read" },
      { id: "t3", name: "Read" },
      { id: "t4", name: "Alpha" },
    ]);
    expect(parseSessionTranscript(raw, "s").toolOutcomes.map((o) => o.tool)).toEqual([
      "Read",
      "Alpha",
      "Zeta",
    ]);
  });

  it("keeps unattributable failures out of the split and reports them separately", () => {
    // The audit format's top-level flag has no tool_use_id. It is a real
    // failure — it must reach errorCount — but charging it to a tool would be
    // a guess, so it surfaces on its own counter instead.
    const raw = [
      assistant("m1", [
        { id: "t1", name: "Read" },
        { id: "t2", name: "Bash" },
      ]),
      results([{ id: "t1" }, { id: "t2", isError: true }]),
      JSON.stringify({
        type: "user",
        uuid: "x",
        _audit_timestamp: stamp(),
        message: { content: [] },
        tool_use_result: { is_error: true },
      }),
    ].join("\n");

    const a = parseSessionTranscript(raw, "s");
    expect(a.unattributedErrors).toBe(1);
    expect(a.errorCount).toBe(2);
    expect(by(a.toolOutcomes, "Read")).toMatchObject({ ok: 1, failed: 0 });
    // the attributed failures alone do not explain errorCount — the view has
    // to show the remainder, which is exactly what unattributedErrors is for
    const attributed = a.toolOutcomes.reduce((s, o) => s + o.failed, 0);
    expect(attributed + a.unattributedErrors).toBe(a.errorCount);
  });

  it("counts a replayed line once, not twice", () => {
    // Resuming a session appends its earlier lines back into the same file:
    // same uuid, same message.id, same tool_use id, hundreds of lines later.
    // Measured on a real 36 MB transcript — 102 of 1392 tool_use blocks were
    // replays. Counting them made a resumed session read 8% busier than it was.
    const call = assistant("m1", [
      { id: "t1", name: "Read" },
      { id: "t2", name: "Bash" },
    ]);
    const answer = results([{ id: "t1" }, { id: "t2", isError: true }]);
    const once = parseSessionTranscript([call, answer].join("\n"), "s");
    const replayed = parseSessionTranscript(
      [call, answer, assistant("m9", [{ id: "t9", name: "Edit" }]), call, answer].join("\n"),
      "s",
    );

    expect(by(replayed.toolOutcomes, "Read")).toEqual(by(once.toolOutcomes, "Read"));
    expect(by(replayed.toolOutcomes, "Bash")).toEqual(by(once.toolOutcomes, "Bash"));
    expect(replayed.errorCount).toBe(once.errorCount);
    expect(replayed.toolTotal).toBe(once.toolTotal + 1); // the one genuinely new call
  });

  it("counts an id-less call as unresolved rather than dropping it", () => {
    // A tool_use with no id can never be answered, so it cannot be classified
    // by a result — but it did happen, and the totals have to keep adding up.
    const raw = JSON.stringify({
      type: "assistant",
      uuid: "u1",
      timestamp: stamp(),
      message: {
        id: "m1",
        usage: {},
        content: [
          { type: "tool_use", name: "Read", input: {} },
          { type: "tool_use", name: "Read", input: {} },
        ],
      },
    });
    const a = parseSessionTranscript(raw, "s");
    expect(by(a.toolOutcomes, "Read")).toMatchObject({ ok: 0, unresolved: 2, total: 2 });
    expect(a.toolTotal).toBe(2);
  });

  it("is empty, not fabricated, for a session with no tool calls", () => {
    const raw = JSON.stringify({
      type: "assistant",
      uuid: "u1",
      timestamp: stamp(),
      message: { id: "m1", usage: {}, content: [{ type: "text", text: "hi" }] },
    });
    const a = parseSessionTranscript(raw, "s");
    expect(a.toolOutcomes).toEqual([]);
    expect(a.unattributedErrors).toBe(0);
  });
});

describe("normalizeSessionAnalysis", () => {
  it("fills the fields an older record predates, so the page cannot throw", () => {
    // Records are persisted without a schema version, so a row written by the
    // previous build reaches the UI missing whatever the fold has since
    // learned to compute. Reading `.length` off that absence blanks the whole
    // Sessions page, which is how this guard came to exist.
    const old = { toolTotal: 12, errorCount: 1 } as unknown as SessionAnalysis;
    const a = normalizeSessionAnalysis(old);
    expect(a.toolOutcomes).toEqual([]);
    expect(a.unattributedErrors).toBe(0);
  });

  it("fills with empty rather than reconstructing from what survived", () => {
    // The histogram stores counts, not outcomes, and the raw transcript is
    // deliberately dropped — so there is nothing to recompute from. A record
    // with 12 calls must come back with NO outcome rows, not 12 invented ok's.
    const old = {
      toolTotal: 12,
      toolHistogram: [["Read", 12]],
      errorCount: 3,
    } as unknown as SessionAnalysis;
    expect(normalizeSessionAnalysis(old).toolOutcomes).toEqual([]);
  });

  it("leaves a current record untouched", () => {
    const fresh = parseSessionTranscript(
      [
        assistant("m1", [{ id: "t1", name: "Read" }]),
        results([{ id: "t1", isError: true }]),
      ].join("\n"),
      "s",
    );
    const before = JSON.stringify(fresh.toolOutcomes);
    expect(JSON.stringify(normalizeSessionAnalysis(fresh).toolOutcomes)).toBe(before);
    expect(fresh.unattributedErrors).toBe(0);
  });
});
