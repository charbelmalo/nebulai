/** Main-thread data API: dataset discovery via /out/index.json, dataset loads
 *  via the parse worker, and a column cache so switching back to a dataset is
 *  instant.
 *
 *  Cache identity is the normalized ABSOLUTE artifact URL plus the expected
 *  digest (or an explicit unpinned marker) — never the relative path alone.
 *  Two bases that both contain `gpt2/nebulai.json` are two different files,
 *  and a pinned request must never be answered by bytes nobody verified.
 *
 *  Every load settles exactly once: resolved with a Dataset, or rejected with
 *  a LoadError whose `kind` says why (fetch, parse, digest mismatch, aborted).
 *  An aborted load terminates its worker, fires no further progress, and never
 *  enters the cache — even if the worker's answer was already in flight. */

import type { Columns } from "./columns";
import type { ClusterHull } from "./hulls";
import type { ParseResponse } from "./parse.worker";
import type { DatasetIndex } from "./schema";
import { DATA_BASE } from "./base";

export interface Dataset {
  columns: Columns;
  hulls: ClusterHull[];
  /** fetch + hash + parse + columnarize, inside the worker */
  parseMs: number;
  /** absolute URL the bytes came from */
  url: string;
  /** path relative to DATA_BASE, as requested */
  path: string;
  /** decoded payload size */
  bytes: number;
  /** SHA-256 of those exact bytes; null when hashing was impossible */
  sha256: string | null;
  hashError?: string;
  /** true when the request pinned a digest and the bytes matched it */
  pinned: boolean;
}

export type LoadErrorKind = "fetch" | "parse" | "digest" | "aborted";

export class LoadError extends Error {
  constructor(
    message: string,
    readonly kind: LoadErrorKind,
    readonly expected?: string,
    readonly actual?: string,
  ) {
    super(message);
    this.name = "LoadError";
  }
}

export interface LoadOptions {
  /** total = 0 means the size is unknown — show bytes, not a percentage */
  onProgress?: (loaded: number, total: number) => void;
  base?: string;
  /** skip both the column cache and the browser HTTP cache */
  noCache?: boolean;
  /** lowercase SHA-256 the bytes must match */
  expectedSha256?: string;
  signal?: AbortSignal;
}

const cache = new Map<string, Dataset>();

export function artifactUrl(path: string, base = DATA_BASE): string {
  return new URL(path, `${base.replace(/\/+$/, "")}/`).href;
}

export function cacheKey(path: string, base = DATA_BASE, expectedSha256?: string): string {
  return `${artifactUrl(path, base)}#${expectedSha256 ?? "unpinned"}`;
}

export async function loadIndex(base = DATA_BASE, noCache = false): Promise<DatasetIndex> {
  const res = await fetch(`${base}/index.json`, noCache ? { cache: "no-store" } : undefined);
  const type = res.headers.get("content-type") ?? "";
  // a dev-server SPA fallback answers 200 with index.html: that is "no index"
  if (!res.ok || !type.includes("json")) {
    throw new Error(`no dataset index at ${base}/index.json (${res.status})`);
  }
  return res.json();
}

/** Worker factory, swappable by unit tests (node has no module Worker). */
let makeWorker = (): Worker =>
  new Worker(new URL("./parse.worker.ts", import.meta.url), { type: "module" });

export function __setWorkerFactory(f: () => Worker): void {
  makeWorker = f;
}

export function loadDataset(path: string, opts: LoadOptions = {}): Promise<Dataset> {
  const base = opts.base ?? DATA_BASE;
  const url = artifactUrl(path, base);
  const key = cacheKey(path, base, opts.expectedSha256);
  const { signal, onProgress } = opts;
  if (signal?.aborted) return Promise.reject(new LoadError(`load of ${path} aborted`, "aborted"));
  const cached = opts.noCache ? undefined : cache.get(key);
  if (cached) return Promise.resolve(cached);

  return new Promise((resolve, reject) => {
    const worker = makeWorker();
    let settled = false;
    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      worker.terminate();
      signal?.removeEventListener("abort", onAbort);
      fn();
    };
    const onAbort = () =>
      settle(() => reject(new LoadError(`load of ${path} aborted`, "aborted")));
    signal?.addEventListener("abort", onAbort, { once: true });

    worker.onmessage = (ev: MessageEvent<ParseResponse>) => {
      if (settled) return;
      const msg = ev.data;
      if (msg.type === "progress") {
        onProgress?.(msg.loaded, msg.total);
      } else if (msg.type === "done") {
        settle(() => {
          const ds: Dataset = {
            columns: msg.columns,
            hulls: msg.hulls,
            parseMs: msg.ms,
            url,
            path,
            bytes: msg.bytes,
            sha256: msg.sha256,
            ...(msg.hashError ? { hashError: msg.hashError } : {}),
            pinned: !!opts.expectedSha256 && msg.sha256 === opts.expectedSha256,
          };
          cache.set(key, ds);
          // an unpinned load whose digest is now known also answers a later
          // pinned request for the same bytes at the same URL
          if (!opts.expectedSha256 && ds.sha256) {
            cache.set(cacheKey(path, base, ds.sha256), { ...ds, pinned: true });
          }
          resolve(ds);
        });
      } else {
        settle(() => reject(new LoadError(msg.message, msg.kind, msg.expected, msg.actual)));
      }
    };
    worker.onerror = (e) => settle(() => reject(new LoadError(e.message || "worker failed", "parse")));
    worker.postMessage({
      url,
      noCache: opts.noCache,
      ...(opts.expectedSha256 ? { expectedSha256: opts.expectedSha256 } : {}),
    });
  });
}

/** Drop every cached identity for this path under this base. */
export function evictDataset(path: string, base = DATA_BASE): void {
  const prefix = `${artifactUrl(path, base)}#`;
  for (const k of [...cache.keys()]) if (k.startsWith(prefix)) cache.delete(k);
}

export function __resetDatasetCache(): void {
  cache.clear();
}
