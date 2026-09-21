/** The local-session picker's pure half, and the sniff that made a real
 *  transcript readable by the Snapshot Map at all.
 *
 *  Four properties are pinned here, each of which was wrong at some point while
 *  this was written:
 *
 *  * `projectLabel` shortens an encoded working directory by dropping the home
 *    prefix. A naive last-segment split labelled `…-gesture-sorcery-game-kit`
 *    as "kit", which is not the project and is not unique either;
 *  * `byNewest` sorts a missing mtime LAST, so a transcript whose stat failed
 *    can never masquerade as the session you just ran;
 *  * `localSessionId` includes the project, because a session id is unique only
 *    inside its own directory — and it is what makes re-picking refresh a
 *    session in place instead of duplicating it;
 *  * `parseConversationText` accepts newline-delimited JSON whose first
 *    character is `{`. Every real Claude Code transcript is exactly that, and
 *    the old `!startsWith("{")` sniff sent all of them into `JSON.parse`, which
 *    threw. The format was documented as supported and was unreachable.
 */

import { describe, expect, it } from "vitest";
import { parseConversationText } from "../../src/chrome/snapshot";
import { ago, byNewest, fmtBytes, localSessionId, projectLabel } from "../../src/seer/transcripts";

const ROOT = "/Users/someone/.claude/projects";

describe("projectLabel", () => {
  it("drops the home prefix and keeps what distinguishes the project", () => {
    const slug = "-Users-someone-Developer-gesture-sorcery-game-kit";
    expect(projectLabel(slug, ROOT)).toBe("Developer-gesture-sorcery-game-kit");
  });

  it("keeps every segment of a slug that is not under the reported root", () => {
    // only the leading dash goes — it is the encoded root `/`, never a segment
    expect(projectLabel("-Volumes-Work-clients-acme", ROOT)).toBe("Volumes-Work-clients-acme");
  });

  it("never returns an empty label", () => {
    // the projects root itself, encoded — nothing left after the prefix
    const slug = "-Users-someone";
    expect(projectLabel(slug, ROOT)).toBe(slug);
  });

  it("falls back to the raw slug when the root is unknown", () => {
    expect(projectLabel("-Users-someone-Developer-nebulai")).toBe(
      "Users-someone-Developer-nebulai",
    );
  });
});

describe("byNewest", () => {
  it("orders newest first and sorts an unknown date last", () => {
    const rows = [
      { modified: 1_700_000_000, id: "old" },
      { modified: 0, id: "unknown" },
      { modified: 1_800_000_000, id: "new" },
    ];
    expect(byNewest(rows).map((r) => r.id)).toEqual(["new", "old", "unknown"]);
  });

  it("does not mutate the caller's array", () => {
    const rows = [{ modified: 1 }, { modified: 2 }];
    byNewest(rows);
    expect(rows.map((r) => r.modified)).toEqual([1, 2]);
  });
});

describe("localSessionId", () => {
  it("is stable across calls, so a re-pick replaces rather than duplicates", () => {
    expect(localSessionId("-Users-someone-p", "abc-123")).toBe(
      localSessionId("-Users-someone-p", "abc-123"),
    );
  });

  it("separates the same session id in two projects", () => {
    expect(localSessionId("proj-a", "abc")).not.toBe(localSessionId("proj-b", "abc"));
  });
});

describe("ago / fmtBytes", () => {
  it("says the date is unknown instead of calling a missing mtime 'just now'", () => {
    expect(ago(0)).toBe("date unknown");
  });

  it("reads in the unit a person would use", () => {
    const now = 1_800_000_000_000;
    expect(ago(now / 1000 - 30, now)).toBe("just now");
    expect(ago(now / 1000 - 3 * 3600, now)).toBe("3h ago");
    expect(ago(now / 1000 - 4 * 86400, now)).toBe("4d ago");
    expect(fmtBytes(900)).toBe("900 B");
    expect(fmtBytes(125 << 20)).toBe("125 MB");
  });
});

describe("parseConversationText on a real Claude Code transcript", () => {
  const transcript = [
    `{"type":"user","timestamp":"2026-07-09T18:00:00Z","message":{"role":"user","content":"Clear the face guard."}}`,
    `{"type":"assistant","timestamp":"2026-07-09T18:00:12Z","message":{"role":"assistant","content":[{"type":"text","text":"Reading the scene first."}]}}`,
  ].join("\n");

  it("parses newline-delimited JSON that starts with a brace", () => {
    const log = parseConversationText(transcript, "t");
    expect(log.turns.map((t) => t.role)).toEqual(["user", "assistant"]);
    expect(log.turns[1]!.text).toContain("Reading the scene");
  });

  it("still parses a pretty-printed JSON document", () => {
    const pretty = JSON.stringify({ messages: [{ role: "user", content: "hi there" }] }, null, 2);
    expect(pretty).toContain("\n");
    expect(parseConversationText(pretty, "t").turns).toHaveLength(1);
  });

  it("takes the caller's id when given one, and mints one otherwise", () => {
    expect(parseConversationText(transcript, "t", "local:p/s").id).toBe("local:p/s");
    expect(parseConversationText(transcript, "t").id).toMatch(/^log-/);
  });
});
