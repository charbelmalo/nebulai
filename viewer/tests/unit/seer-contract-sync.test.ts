/** contract.py ↔ contract.ts drift guard.
 *
 *  The Python side writes the append-only log; this file proves the viewer's
 *  mirror knows every member it can receive. Without it, a new Action or
 *  Fidelity added in Python arrives in the UI as an unstyled, unlabelled value
 *  — and an unlabelled fidelity is exactly the failure this whole subsystem
 *  exists to prevent.
 *
 *  Parses the Python enums textually rather than importing them (no Python in
 *  the vitest run), which is the same trade tokens-sync.test.ts makes. */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  ACTIONS,
  CAPTURE_MODES,
  EFFECTS,
  FIDELITIES,
  OUTCOMES,
  SCHEMA_VERSION,
  SESSION_STATES,
  TOKEN_CATEGORIES,
  formatMeasured,
  isAbsent,
  isDelta,
} from "../../src/seer/contract";

const py = readFileSync(
  new URL("../../../src/nebulai/seer/contract.py", import.meta.url),
  "utf8",
);

/** Pull the string values out of one `class X(str, Enum):` block. */
function pyEnum(name: string): string[] {
  const start = py.indexOf(`class ${name}(str, Enum):`);
  if (start === -1) throw new Error(`class ${name} missing from contract.py`);
  // the block ends at the next top-level `class ` or a `# ──` section rule
  const rest = py.slice(start + 1);
  const endRel = rest.search(/\n(class |# ──|@dataclass|def )/);
  const block = endRel === -1 ? rest : rest.slice(0, endRel);
  return [...block.matchAll(/^\s{4}[A-Z_0-9]+\s*=\s*"([a-z_]+)"/gm)].map((m) => m[1]!);
}

describe("contract.py ↔ contract.ts", () => {
  it("mirrors Fidelity, including the two absent kinds", () => {
    expect(pyEnum("Fidelity").sort()).toEqual([...FIDELITIES].sort());
    expect(FIDELITIES).toContain("missing");
    expect(FIDELITIES).toContain("dropped_by_policy");
  });

  it("mirrors CaptureMode", () => {
    expect(pyEnum("CaptureMode").sort()).toEqual([...CAPTURE_MODES].sort());
  });

  it("mirrors the 9-type Action taxonomy", () => {
    const fromPy = pyEnum("Action");
    expect(fromPy).toHaveLength(9);
    expect(fromPy.sort()).toEqual([...ACTIONS].sort());
  });

  it("mirrors Effect", () => {
    expect(pyEnum("Effect").sort()).toEqual([...EFFECTS].sort());
  });

  it("mirrors TokenCategory", () => {
    expect(pyEnum("TokenCategory").sort()).toEqual([...TOKEN_CATEGORIES].sort());
  });

  it("mirrors SessionState, and stalled is not one of them", () => {
    expect(pyEnum("SessionState").sort()).toEqual([...SESSION_STATES].sort());
    expect(SESSION_STATES).not.toContain("stalled" as never);
  });

  it("mirrors Outcome", () => {
    expect(pyEnum("Outcome").sort()).toEqual([...OUTCOMES].sort());
  });

  it("agrees on the schema version", () => {
    const m = py.match(/SCHEMA_VERSION\s*=\s*"([^"]+)"/);
    expect(m?.[1]).toBe(SCHEMA_VERSION);
  });
});

describe("absent values never render as zero", () => {
  it("treats missing and dropped_by_policy as absent", () => {
    expect(isAbsent({ value: null, fidelity: "missing" })).toBe(true);
    expect(isAbsent({ value: 0, fidelity: "dropped_by_policy" })).toBe(true);
    expect(isAbsent({ value: 0, fidelity: "native" })).toBe(false);
  });

  it("formats an absent value as an em dash, not 0", () => {
    expect(formatMeasured({ value: null, fidelity: "missing" })).toBe("—");
    expect(formatMeasured({ value: 0, fidelity: "dropped_by_policy" })).toBe("—");
    // a real zero is still a zero — the point is to distinguish them
    expect(formatMeasured({ value: 0, fidelity: "native" })).toBe("0");
  });
});

describe("deltas are identifiable from the wire", () => {
  it("flags exactly the streaming families", () => {
    expect(isDelta("message.assistant_delta")).toBe(true);
    expect(isDelta("tool.output_delta")).toBe(true);
    expect(isDelta("tool.completed")).toBe(false);
    expect(isDelta("turn.completed")).toBe(false);
  });
});
