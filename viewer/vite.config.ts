/// <reference types="vitest/config" />
import { createReadStream, existsSync, renameSync, statSync } from "node:fs";
import { resolve } from "node:path";
import preact from "@preact/preset-vite";
import { defineConfig, type Plugin } from "vite";

/** Which entries this build emits. Vite's `base` is a property of the BUILD,
 *  not of an entry, so three sub-path deploys (/psychiX/, /psychiX/nebulai-maps/,
 *  /psychiX/seer/) are three builds — one per tree — each with its own base,
 *  its own input, and its own outDir. `PSYCHIX_ENTRY` is what selects the
 *  input; it is deliberately NOT `VITE_`-prefixed, because it decides what
 *  rollup compiles and has no business being readable from `import.meta.env`
 *  inside the app.
 *
 *  Unset (the default) means BOTH entries at base "./" into dist/ — exactly
 *  what `npm run build` has always done, which dev, `vite preview`, the e2e
 *  suite and CI all still depend on. */
type EntryId = "all" | "nebulai" | "seer" | "hub";

const ENTRY = (process.env.PSYCHIX_ENTRY ?? "all") as EntryId;

const INPUTS: Record<EntryId, Record<string, string>> = {
  // Two instruments, one repo, one build. Rollup treats each HTML file as
  // an independent entry and only shares a chunk between them when both
  // actually import it — which is how `npm run build` produces the
  // evidence that the split is real: Seer's graph reaches the chrome and
  // @psychix/viz, and reaches no atlas or Internals driver at all.
  all: {
    index: resolve(__dirname, "index.html"),
    seer: resolve(__dirname, "seer.html"),
  },
  nebulai: { index: resolve(__dirname, "index.html") },
  seer: { seer: resolve(__dirname, "seer.html") },
  hub: { hub: resolve(__dirname, "hub.html") },
};

/** The one HTML file a single-entry deploy tree must serve at its ROOT.
 *  `/psychiX/seer/` has to answer with Seer for a bare directory request, so
 *  that tree needs `index.html`, not `seer.html` — and the same for the hub.
 *
 *  Doing it as a build step — rather than by hand after the fact — is what
 *  makes it survive: every `npm run build:seer` re-does it, so there is no
 *  edited artifact for a rebuild to silently undo. It is safe for the asset
 *  URLs *because both names sit at the same depth* — outDir root either way —
 *  so the hrefs Vite already computed (absolute under a `VITE_BASE`, relative
 *  under "./") stay correct verbatim. Moving an HTML file between DIRECTORIES
 *  here would not be safe, and this deliberately never does.
 *
 *  It runs in `writeBundle` against the written file rather than renaming the
 *  asset inside `generateBundle`, because Vite 8's rolldown ignores mutations
 *  to the bundle object ("This plugin assigns to bundle variable … will be
 *  ignored") — that route silently produced a tree with NO html in it. */
const ROOT_HTML: Partial<Record<EntryId, string>> = {
  seer: "seer.html",
  hub: "hub.html",
};

function writeHtmlAsIndex(from: string): Plugin {
  return {
    name: "psychix-html-as-index",
    enforce: "post",
    writeBundle(options) {
      const dir = options.dir ?? resolve(__dirname, "dist");
      const src = resolve(dir, from);
      if (!existsSync(src)) {
        throw new Error(`psychix-html-as-index: ${src} was not emitted`);
      }
      renameSync(src, resolve(dir, "index.html"));
    },
  };
}

/** Serve the repo's out/ artifacts at /out during dev (no second server). */
function serveOut(): Plugin {
  const root = resolve(__dirname, "..", "out");
  const types: Record<string, string> = {
    ".json": "application/json",
    ".png": "image/png",
    ".html": "text/html",
  };
  return {
    name: "nebulai-serve-out",
    configureServer(server) {
      server.middlewares.use("/out", (req, res, next) => {
        const url = (req.url ?? "/").split("?")[0];
        const file = resolve(root, "." + url);
        if (!file.startsWith(root) || !existsSync(file) || !statSync(file).isFile()) {
          return next();
        }
        const ext = file.slice(file.lastIndexOf("."));
        res.setHeader("Content-Type", types[ext] ?? "application/octet-stream");
        res.setHeader("Content-Length", statSync(file).size);
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  // Dev + relative-hosting default is "./"; a sub-path static deploy sets
  // VITE_BASE (e.g. "/psychiX/nebulai-maps/") so BASE_URL — and thus DATA_BASE
  // in src/data/base.ts — resolve the baked out/ tree under that sub-path.
  base: process.env.VITE_BASE ?? "./",
  plugins: [
    preact(),
    serveOut(),
    ...(ROOT_HTML[ENTRY] ? [writeHtmlAsIndex(ROOT_HTML[ENTRY]!)] : []),
  ],
  resolve: { alias: { "@psychix/viz": resolve(__dirname, "src/viz") } },
  server: { port: Number(process.env.PORT) || 5173 },
  build: {
    target: "es2022",
    rollupOptions: { input: INPUTS[ENTRY] },
  },
  // tests/e2e belongs to Playwright, not vitest
  test: { include: ["tests/unit/**/*.test.ts"] },
});
