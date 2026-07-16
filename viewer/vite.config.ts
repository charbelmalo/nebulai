/// <reference types="vitest/config" />
import { createReadStream, existsSync, statSync } from "node:fs";
import { resolve } from "node:path";
import preact from "@preact/preset-vite";
import { defineConfig, type Plugin } from "vite";

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
  base: "./",
  plugins: [preact(), serveOut()],
  server: { port: Number(process.env.PORT) || 5173 },
  build: { target: "es2022" },
  // tests/e2e belongs to Playwright, not vitest
  test: { include: ["tests/unit/**/*.test.ts"] },
});
