/** Seer's entry — the instrument that maps what an agent did
 *  (Live · Transcripts · Topics). Nebulai's entry is `src/main.ts`.
 *
 *  This file is short, and that is the deliverable. Everything it does is the
 *  shared half of Nebulai's boot — `bootShell` probes the GPU tier, mounts the
 *  chrome and reads the permalink; `finishShellBoot` applies it and starts
 *  mirroring — with Seer's page set handed in instead of Nebulai's. There is
 *  no second phase, because Seer has no equivalent of `bootAtlas`: its three
 *  pages own their own canvases and fetch their own data on demand (the
 *  collector over HTTP, a dropped transcript, an IndexedDB rehydrate), so
 *  there is nothing this module could usefully load up front.
 *
 *  WHAT MUST NOT APPEAR IN THIS FILE, ever:
 *    · AtlasDriver / ChordDriver / CompareDriver / HierarchyDriver
 *    · data/loader, data/compare — anything that fetches `out/`
 *    · scene/interp/registry (and so InterpPage, and so 25 more drivers)
 *  Seer's production condition is a deploy with no baked artifacts at all. A
 *  single one of those imports would put a megabyte of atlas in its bundle and
 *  a 404 in its network log, and tests/e2e/seer-entry.spec.ts fails if the
 *  request ever happens.
 *
 *  The permalink's Internals-only keys (`feature`, `trace`) stay unvalidated
 *  here on purpose: `registerInterpUrlHooks` is never called, so urlState
 *  rejects both, which is the honest answer for a document that has no
 *  Internals page to show them on. */

import "@psychix/viz/tokens.css";
import "@psychix/viz/craft-tokens.css";
import "./styles/seer.css";

import { bootShell, finishShellBoot } from "./app/boot-shell";
import { appStore } from "./app/store";
import { SEER_APP } from "./chrome/apps/seer";

async function boot() {
  const shell = await bootShell(SEER_APP);

  // The status pill is the MetaLine on Nebulai — dataset provenance. Seer has
  // no dataset, and inventing one would be the exact dishonesty the pill
  // exists to prevent, so it states the two facts that ARE true at boot: the
  // rung we landed on, and where this document will look for a collector.
  // Whether that collector actually answers is the Live page's own live
  // readout ($link / $linkError), which updates; this line does not pretend to.
  const url = appStore.getState().seer.serverUrl;
  shell.say(
    `gpu: ${shell.caps.tier} · collector: ${url || "not configured — set one in Settings"}`,
  );

  finishShellBoot(shell.urlState);
}

boot().catch((e) => {
  console.error(e);
  const status =
    document.querySelector<HTMLElement>(".boot-status") ??
    document.getElementById("chrome")!.appendChild(
      Object.assign(document.createElement("div"), { className: "boot-status" }),
    );
  status.textContent = `boot failed: ${e instanceof Error ? e.message : e}`;
});
