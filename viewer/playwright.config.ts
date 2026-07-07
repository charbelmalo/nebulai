/** E2e battery — two rungs of the fallback ladder as Playwright projects:
 *  - webgl: deterministic (TSL transpiles to GLSL everywhere) — goldens,
 *    perf budgets, a11y, interactions.
 *  - webgpu: best-effort (headless WebGPU needs the full chromium build +
 *    flags; specs skip themselves when the capability probe falls back).
 *  One worker: drivers own real GPU contexts and all tests share one dev
 *  server + its dataset files.
 */
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "tests/e2e",
  outputDir: "tests/e2e/.artifacts",
  snapshotPathTemplate: "{testDir}/goldens/{projectName}/{arg}{ext}",
  timeout: 90_000,
  workers: 1,
  reporter: [["list"]],
  expect: { timeout: 20_000, toHaveScreenshot: { maxDiffPixelRatio: 0.02 } },
  use: {
    baseURL: "http://localhost:5173",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: true,
    timeout: 30_000,
  },
  projects: [
    {
      name: "webgl",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1280, height: 800 },
        // headless_shell renders on SwiftShader (~3fps at 50K points, input
        // starves); the full build + ANGLE Metal gives headless a real GPU
        channel: "chromium",
        launchOptions: { args: ["--use-angle=metal"] },
      },
    },
    {
      name: "webgpu",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1280, height: 800 },
        // headless_shell has no WebGPU; the full build does (with flags)
        channel: "chromium",
        launchOptions: {
          args: ["--enable-unsafe-webgpu", "--enable-gpu", "--use-angle=metal"],
        },
      },
    },
  ],
});
