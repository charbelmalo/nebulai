/// <reference lib="webworker" />
/** Off-main-thread data path: fetch (with byte progress) → SHA-256 over the
 *  exact decoded response bytes → JSON.parse → columnarize → hulls →
 *  postMessage with transferables. The ~13MB single-line JSON never blocks the
 *  UI thread.
 *
 *  The digest is taken over the bytes the response body yields — after HTTP
 *  content-encoding is undone, before any text decoding — which are the same
 *  bytes as the `.json` file on disk. Never a re-serialisation of the parsed
 *  object. When the caller pins an expected digest (a manifest artifact or a
 *  finding), a mismatch is a typed failure: those bytes never become a Dataset.
 *  When hashing is impossible (no SubtleCrypto outside a secure context) the
 *  map still loads, with `sha256: null` and the reason, so exploration works
 *  and only exact export/replay is disabled. */

import { columnarize, transferables, type Columns } from "./columns";
import { computeHulls, type ClusterHull } from "./hulls";

export interface ParseRequest {
  url: string;
  /** bypass the browser HTTP cache — used after a rebuild overwrites the file */
  noCache?: boolean;
  /** lowercase SHA-256 the bytes must match, or undefined = unpinned */
  expectedSha256?: string;
}

export type ParseResponse =
  /** total = 0 when the size is not known (compressed transfer, no header) */
  | { type: "progress"; loaded: number; total: number }
  | {
      type: "done";
      columns: Columns;
      hulls: ClusterHull[];
      ms: number;
      bytes: number;
      sha256: string | null;
      hashError?: string;
    }
  | {
      type: "error";
      message: string;
      kind: "fetch" | "parse" | "digest";
      expected?: string;
      actual?: string;
    };

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");
}

self.onmessage = async (ev: MessageEvent<ParseRequest>) => {
  const t0 = performance.now();
  const { url, noCache, expectedSha256 } = ev.data;
  let stage: "fetch" | "parse" = "fetch";
  try {
    const res = await fetch(url, noCache ? { cache: "reload" } : undefined);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);

    // Content-Length is the ENCODED size. Under gzip/br the body yields more
    // bytes than that, so a percentage would pass 100% — report unknown.
    const encoded = (res.headers.get("Content-Encoding") ?? "").trim() !== "";
    const header = Number(res.headers.get("Content-Length") ?? 0);
    const total = !encoded && Number.isFinite(header) && header > 0 ? header : 0;

    const chunks: Uint8Array[] = [];
    let loaded = 0;
    if (res.body) {
      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        postMessage({
          type: "progress",
          loaded,
          total: total >= loaded ? total : 0,
        } satisfies ParseResponse);
      }
    } else {
      const buf = new Uint8Array(await res.arrayBuffer());
      chunks.push(buf);
      loaded = buf.byteLength;
    }
    const bytes = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) {
      bytes.set(c, off);
      off += c.byteLength;
    }
    chunks.length = 0;

    let sha256: string | null = null;
    let hashError: string | undefined;
    try {
      if (!globalThis.crypto?.subtle) throw new Error("SubtleCrypto is unavailable (insecure context)");
      sha256 = await sha256Hex(bytes);
    } catch (e) {
      hashError = e instanceof Error ? e.message : String(e);
    }
    if (expectedSha256) {
      if (sha256 === null) {
        postMessage({
          type: "error",
          kind: "digest",
          message: `cannot verify ${url}: ${hashError}`,
          expected: expectedSha256,
        } satisfies ParseResponse);
        return;
      }
      if (sha256 !== expectedSha256) {
        postMessage({
          type: "error",
          kind: "digest",
          message: `digest mismatch for ${url}`,
          expected: expectedSha256,
          actual: sha256,
        } satisfies ParseResponse);
        return;
      }
    }

    stage = "parse";
    const text = new TextDecoder().decode(bytes);
    const columns = columnarize(JSON.parse(text));
    const hulls = computeHulls(columns.pos2, columns.clusterId);
    const msg: ParseResponse = {
      type: "done",
      columns,
      hulls,
      ms: performance.now() - t0,
      bytes: loaded,
      sha256,
      ...(hashError ? { hashError } : {}),
    };
    postMessage(msg, {
      transfer: [...transferables(columns), ...hulls.map((h) => h.ring.buffer as ArrayBuffer)],
    });
  } catch (e) {
    postMessage({
      type: "error",
      kind: stage,
      message: e instanceof Error ? e.message : String(e),
    } satisfies ParseResponse);
  }
};
