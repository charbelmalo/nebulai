/// <reference lib="webworker" />
/** Off-main-thread data path: fetch (with byte progress) → JSON.parse →
 *  columnarize → hulls → postMessage with transferables. The ~13MB
 *  single-line JSON never blocks the UI thread. */

import { columnarize, transferables, type Columns } from "./columns";
import { computeHulls, type ClusterHull } from "./hulls";

export interface ParseRequest {
  url: string;
  /** bypass the browser HTTP cache — used after a rebuild overwrites the file */
  noCache?: boolean;
}

export type ParseResponse =
  | { type: "progress"; loaded: number; total: number }
  | { type: "done"; columns: Columns; hulls: ClusterHull[]; ms: number }
  | { type: "error"; message: string };

self.onmessage = async (ev: MessageEvent<ParseRequest>) => {
  const t0 = performance.now();
  try {
    const res = await fetch(ev.data.url, ev.data.noCache ? { cache: "reload" } : undefined);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${ev.data.url}`);

    const total = Number(res.headers.get("Content-Length") ?? 0);
    let text: string;
    if (res.body && total > 0) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      const parts: string[] = [];
      let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        loaded += value.byteLength;
        parts.push(decoder.decode(value, { stream: true }));
        postMessage({ type: "progress", loaded, total } satisfies ParseResponse);
      }
      parts.push(decoder.decode());
      text = parts.join("");
    } else {
      text = await res.text();
    }

    const columns = columnarize(JSON.parse(text));
    const hulls = computeHulls(columns.pos2, columns.clusterId);
    const msg: ParseResponse = {
      type: "done",
      columns,
      hulls,
      ms: performance.now() - t0,
    };
    postMessage(msg, {
      transfer: [...transferables(columns), ...hulls.map((h) => h.ring.buffer as ArrayBuffer)],
    });
  } catch (e) {
    postMessage({
      type: "error",
      message: e instanceof Error ? e.message : String(e),
    } satisfies ParseResponse);
  }
};
