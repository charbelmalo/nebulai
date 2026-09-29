/** Write docs/SETTINGS-OWNERSHIP.md from src/chrome/settingsScope.ts. */
import { writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { renderOwnershipMarkdown } from "../src/chrome/settingsScope.ts";

const out = resolve(dirname(fileURLToPath(import.meta.url)), "../../docs/SETTINGS-OWNERSHIP.md");
writeFileSync(out, renderOwnershipMarkdown());
console.log(`wrote ${out}`);
