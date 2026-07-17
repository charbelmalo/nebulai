/** Main-thread data API: dataset discovery via /out/index.json, dataset loads
 *  via the parse worker, and a per-dataset column cache so switching datasets
 *  after first load is instant. */

import type { Columns } from "./columns";
import type { ClusterHull } from "./hulls";
import type { ParseResponse } from "./parse.worker";
import type { DatasetIndex } from "./schema";
import { DATA_BASE } from "./base";

export interface Dataset {
  columns: Columns;
  hulls: ClusterHull[];
  parseMs: number;
}

const cache = new Map<string, Dataset>();

export async function loadIndex(base = DATA_BASE, noCache = false): Promise<DatasetIndex> {
  const res = await fetch(`${base}/index.json`, noCache ? { cache: "no-store" } : undefined);
  if (!res.ok) throw new Error(`no dataset index at ${base}/index.json (${res.status})`);
  return res.json();
}

export function loadDataset(
  path: string,
  onProgress?: (loaded: number, total: number) => void,
  base = DATA_BASE,
  noCache = false,
): Promise<Dataset> {
  const cached = noCache ? undefined : cache.get(path);
  if (cached) return Promise.resolve(cached);

  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./parse.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (ev: MessageEvent<ParseResponse>) => {
      const msg = ev.data;
      if (msg.type === "progress") {
        onProgress?.(msg.loaded, msg.total);
      } else if (msg.type === "done") {
        worker.terminate();
        const ds: Dataset = { columns: msg.columns, hulls: msg.hulls, parseMs: msg.ms };
        cache.set(path, ds);
        resolve(ds);
      } else {
        worker.terminate();
        reject(new Error(msg.message));
      }
    };
    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(e.message));
    };
    worker.postMessage({ url: `${base}/${path}`, noCache });
  });
}

export function evictDataset(path: string): void {
  cache.delete(path);
}
