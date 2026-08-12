/** The seer HTTP/SSE client — the parts that are easy to get quietly wrong.
 *
 *  Three properties are pinned here:
 *
 *  * every `Fidelity` has a mark and a title, so a new one added in Python can
 *    never reach the screen as an unexplained blank;
 *  * `connectLive` listens for *named* SSE events. The server sends `hello` /
 *    `event` / `run_finished`, and `onmessage` fires for none of them — a
 *    regression there produces a page that looks connected and never updates,
 *    which no type checker catches;
 *  * the error path surfaces the server's own message rather than "HTTP 400",
 *    because "unknown run" is actionable and a status code is not.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FIDELITIES } from "../../src/seer/contract";
import {
  $link,
  $linkError,
  FIDELITY_MARK,
  FIDELITY_TITLE,
  connectLive,
  exportUrl,
  fetchComparison,
  fetchRun,
  fetchRuns,
  fmtSeconds,
  fmtUsd,
  seerBase,
} from "../../src/seer/client";
import { appStore } from "../../src/app/store";

const BASE = "http://seer.test:8125";

beforeEach(() => {
  appStore.getState().setSeerConfig("serverUrl", BASE);
  $link.value = "unknown";
  $linkError.value = null;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ── fidelity ─────────────────────────────────────────────────────────────────

describe("fidelity marks", () => {
  it("covers every fidelity the contract can produce", () => {
    for (const f of FIDELITIES) {
      expect(FIDELITY_MARK, f).toHaveProperty(f);
      expect(FIDELITY_TITLE[f], f).toBeTruthy();
    }
  });

  it("marks only the fidelities a reader must not difference", () => {
    // if native and deterministic carried marks, nearly every number would
    // wear one and the two that matter would stop standing out
    expect(FIDELITY_MARK.native).toBe("");
    expect(FIDELITY_MARK.deterministic).toBe("");
    expect(FIDELITY_MARK.estimated).toBe("~");
    expect(FIDELITY_MARK.heuristic).toBe("?");
  });

  it("distinguishes what we did not capture from what was never reported", () => {
    expect(FIDELITY_TITLE.missing).not.toBe(FIDELITY_TITLE.dropped_by_policy);
  });
});

// ── URLs ─────────────────────────────────────────────────────────────────────

describe("URL building", () => {
  it("tolerates a trailing slash in the configured base", () => {
    appStore.getState().setSeerConfig("serverUrl", `${BASE}///`);
    expect(seerBase()).toBe(BASE);
    expect(exportUrl("run_1")).toBe(`${BASE}/seer/export?run_id=run_1&format=jsonl`);
  });

  it("escapes a run id rather than splicing it in raw", () => {
    expect(exportUrl("a b/c")).toBe(`${BASE}/seer/export?run_id=a%20b%2Fc&format=jsonl`);
  });

  it("defaults to the lossless format", () => {
    // csv is spans-only. A default that silently handed back the lossy one
    // would make "I exported the run" mean two different things.
    expect(exportUrl("r")).toContain("format=jsonl");
    expect(exportUrl("r", "parquet")).toContain("format=parquet");
  });
});

// ── fetch paths ──────────────────────────────────────────────────────────────

function stubFetch(impl: (url: string) => Response | Promise<Response>) {
  const spy = vi.fn((u: RequestInfo | URL) => Promise.resolve(impl(String(u))));
  vi.stubGlobal("fetch", spy);
  return spy;
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("fetching", () => {
  it("asks for runs and comparisons at the documented routes", async () => {
    const spy = stubFetch((u) =>
      u.includes("/seer/runs") ? json({ runs: [{ run_id: "a" }] }) : json({ runs: [] }),
    );
    await fetchRuns(7);
    expect(spy.mock.calls[0]![0]).toBe(`${BASE}/seer/runs?limit=7`);

    await fetchComparison(["a", "b"]);
    expect(spy.mock.calls[1]![0]).toBe(`${BASE}/seer/compare?runs=a,b`);
  });

  it("reports the server's reason, not the status code", async () => {
    stubFetch(() => json({ error: "unknown run 'run_nope'" }, 404));
    await expect(fetchRun("run_nope")).rejects.toThrow("unknown run 'run_nope'");
  });

  it("falls back to the status when the body is not JSON", async () => {
    stubFetch(() => new Response("<html>502</html>", { status: 502 }));
    await expect(fetchRun("r")).rejects.toThrow("HTTP 502");
  });

  it("refuses to guess a base URL when none is configured", async () => {
    appStore.getState().setSeerConfig("serverUrl", "");
    await expect(fetchRuns()).rejects.toThrow(/no seer server/);
  });
});

// ── the live stream ──────────────────────────────────────────────────────────

class FakeEventSource {
  static last: FakeEventSource | null = null;
  readonly listeners = new Map<string, ((e: MessageEvent) => void)[]>();
  onerror: (() => void) | null = null;
  closed = false;

  constructor(readonly url: string) {
    FakeEventSource.last = this;
  }

  addEventListener(type: string, fn: (e: MessageEvent) => void) {
    const l = this.listeners.get(type) ?? [];
    l.push(fn);
    this.listeners.set(type, l);
  }

  close() {
    this.closed = true;
  }

  /** What the server actually sends: a *named* event with a JSON data frame. */
  emit(type: string, data: unknown) {
    for (const fn of this.listeners.get(type) ?? [])
      fn({ data: JSON.stringify(data) } as MessageEvent);
  }
}

describe("connectLive", () => {
  it("subscribes to the named events the server sends", () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const onEvent = vi.fn();
    const onRunFinished = vi.fn();
    const stop = connectLive({ onEvent, onRunFinished });
    const es = FakeEventSource.last!;

    expect(es.url).toBe(`${BASE}/seer/live`);
    expect($link.value).toBe("connecting");

    es.emit("hello", { schema_version: "1.0" });
    expect($link.value).toBe("live");

    es.emit("event", { event: { event_id: "e1", run_id: "r" } });
    expect(onEvent).toHaveBeenCalledWith({ event_id: "e1", run_id: "r" });

    es.emit("run_finished", { run_id: "r", view: { run_id: "r" } });
    expect(onRunFinished).toHaveBeenCalledWith("r", { run_id: "r" });

    stop();
    expect(es.closed).toBe(true);
    expect($link.value).toBe("unknown");
  });

  it("says it is disconnected instead of retrying invisibly", () => {
    vi.useFakeTimers();
    vi.stubGlobal("EventSource", FakeEventSource);
    // the client schedules its own retry through `window` (browser timer ids);
    // vitest runs in node, so point it at the faked globals
    vi.stubGlobal("window", globalThis);
    const stop = connectLive({ onEvent: vi.fn() });
    const first = FakeEventSource.last!;

    first.onerror!();
    expect($link.value).toBe("down");
    expect($linkError.value).toBe("stream closed");

    vi.advanceTimersByTime(2000);
    expect(FakeEventSource.last).not.toBe(first);
    expect($link.value).toBe("connecting");

    stop();
    vi.useRealTimers();
  });

  it("stops reconnecting once disposed", () => {
    vi.useFakeTimers();
    vi.stubGlobal("EventSource", FakeEventSource);
    // the client schedules its own retry through `window` (browser timer ids);
    // vitest runs in node, so point it at the faked globals
    vi.stubGlobal("window", globalThis);
    const stop = connectLive({ onEvent: vi.fn() });
    const first = FakeEventSource.last!;
    first.onerror!();
    stop();
    vi.advanceTimersByTime(10_000);
    expect(FakeEventSource.last).toBe(first);
    vi.useRealTimers();
  });
});

// ── formatting ───────────────────────────────────────────────────────────────

describe("formatting", () => {
  it("keeps sub-second durations legible as milliseconds", () => {
    expect(fmtSeconds(0.06)).toBe("60ms");
    expect(fmtSeconds(9.04)).toBe("9.0s");
    expect(fmtSeconds(125)).toBe("2m 5s");
  });

  it("does not round a fraction-of-a-cent run down to $0.00", () => {
    expect(fmtUsd(0.0003)).toBe("$0.0003");
    expect(fmtUsd(0.031)).toBe("$0.03");
  });
});
