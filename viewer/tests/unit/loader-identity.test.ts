import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  __resetDatasetCache,
  __setWorkerFactory,
  cacheKey,
  evictDataset,
  loadDataset,
  LoadError,
} from "../../src/data/loader";
import { columnarize } from "../../src/data/columns";
import { v2Doc } from "./fixtures";

const BASE = "http://example.test/out";
const SHA = "f".repeat(64);

interface FakeWorker {
  onmessage: ((ev: { data: unknown }) => void) | null;
  onerror: ((e: { message: string }) => void) | null;
  posted: unknown[];
  terminated: boolean;
  postMessage(m: unknown): void;
  terminate(): void;
}

let workers: FakeWorker[] = [];

beforeEach(() => {
  __resetDatasetCache();
  workers = [];
  __setWorkerFactory(() => {
    const w: FakeWorker = {
      onmessage: null,
      onerror: null,
      posted: [],
      terminated: false,
      postMessage(m) {
        this.posted.push(m);
      },
      terminate() {
        this.terminated = true;
      },
    };
    workers.push(w);
    return w as unknown as Worker;
  });
});
afterEach(() => __resetDatasetCache());

const done = (w: FakeWorker, sha: string | null = SHA) =>
  w.onmessage!({ data: { type: "done", columns: columnarize(v2Doc()), hulls: [], ms: 1, bytes: 10, sha256: sha } });

describe("cache identity", () => {
  it("keys by absolute URL plus digest", () => {
    expect(cacheKey("a/nebulai.json", BASE, SHA)).toBe(`${BASE}/a/nebulai.json#${SHA}`);
    expect(cacheKey("a/nebulai.json", BASE)).toBe(`${BASE}/a/nebulai.json#unpinned`);
    expect(cacheKey("a/nebulai.json", `${BASE}/`)).toBe(cacheKey("a/nebulai.json", BASE));
  });

  it("passes the expected digest to the worker and marks the result pinned", async () => {
    const p = loadDataset("a/nebulai.json", { base: BASE, expectedSha256: SHA });
    expect(workers[0]!.posted[0]).toEqual({ url: `${BASE}/a/nebulai.json`, noCache: undefined, expectedSha256: SHA });
    done(workers[0]!);
    const ds = await p;
    expect(ds.pinned).toBe(true);
    expect(ds.sha256).toBe(SHA);
    expect(workers[0]!.terminated).toBe(true);
  });

  it("an unpinned load also answers a pinned request for the same bytes", async () => {
    const p = loadDataset("a/nebulai.json", { base: BASE });
    done(workers[0]!);
    expect((await p).pinned).toBe(false);
    const again = await loadDataset("a/nebulai.json", { base: BASE, expectedSha256: SHA });
    expect(workers).toHaveLength(1);
    expect(again.pinned).toBe(true);
  });

  it("never lets a pinned cache entry answer for another digest", async () => {
    const p = loadDataset("a/nebulai.json", { base: BASE, expectedSha256: SHA });
    done(workers[0]!);
    await p;
    void loadDataset("a/nebulai.json", { base: BASE, expectedSha256: "0".repeat(64) }).catch(() => void 0);
    expect(workers).toHaveLength(2);
  });

  it("evicts every identity of a path", async () => {
    const p = loadDataset("a/nebulai.json", { base: BASE });
    done(workers[0]!);
    await p;
    evictDataset("a/nebulai.json", BASE);
    void loadDataset("a/nebulai.json", { base: BASE, expectedSha256: SHA }).catch(() => void 0);
    expect(workers).toHaveLength(2);
  });
});

describe("failure and cancellation", () => {
  it("surfaces a digest mismatch with both digests", async () => {
    const p = loadDataset("a/nebulai.json", { base: BASE, expectedSha256: SHA });
    workers[0]!.onmessage!({ data: { type: "error", kind: "digest", message: "mismatch", expected: SHA, actual: "0".repeat(64) } });
    const e = await p.catch((x) => x);
    expect(e).toBeInstanceOf(LoadError);
    expect(e).toMatchObject({ kind: "digest", expected: SHA, actual: "0".repeat(64) });
  });

  it("abort terminates the worker and ignores its late answer", async () => {
    const ctrl = new AbortController();
    const seen: number[] = [];
    const p = loadDataset("a/nebulai.json", { base: BASE, signal: ctrl.signal, onProgress: (l) => seen.push(l) });
    workers[0]!.onmessage!({ data: { type: "progress", loaded: 1, total: 10 } });
    ctrl.abort();
    workers[0]!.onmessage!({ data: { type: "progress", loaded: 5, total: 10 } });
    done(workers[0]!);
    const e = await p.catch((x) => x);
    expect(e).toMatchObject({ kind: "aborted" });
    expect(workers[0]!.terminated).toBe(true);
    expect(seen).toEqual([1]);
    // nothing was cached from the abandoned request
    void loadDataset("a/nebulai.json", { base: BASE }).catch(() => void 0);
    expect(workers).toHaveLength(2);
  });

  it("an already-aborted signal never starts a worker", async () => {
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(loadDataset("a/nebulai.json", { base: BASE, signal: ctrl.signal })).rejects.toMatchObject({ kind: "aborted" });
    expect(workers).toHaveLength(0);
  });
});
