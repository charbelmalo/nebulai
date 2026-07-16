/** Top-left brand mark, matching the video's logo + wordmark anatomy. The
 *  mark is a tiny nebula: three ramp-colored dots in orbit. A center nav
 *  switches between the semantic-cloud viewer and the Snapshot Map page.
 *  Top-right hosts share/export tools and the global gear that opens the
 *  full Settings page. */

import { useSignal } from "@preact/signals";
import { appStore, type Page } from "../app/store";
import { downloadStagePng } from "./exportPng";
import { $page } from "./state";
import { shareUrl } from "./urlState";

export function TopBar() {
  const page = $page.value;
  const copied = useSignal(false);
  const saving = useSignal(false);

  const copyLink = async () => {
    const url = shareUrl();
    try {
      await navigator.clipboard.writeText(url);
    } catch {
      // clipboard API needs a secure context — fall back to a transient textarea
      const ta = document.createElement("textarea");
      ta.value = url;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    copied.value = true;
    window.setTimeout(() => (copied.value = false), 1400);
  };

  const savePng = async () => {
    if (saving.value) return;
    saving.value = true;
    try {
      await downloadStagePng();
    } finally {
      saving.value = false;
    }
  };

  return (
    <>
      <header class="topbar">
        <svg class="topbar-mark" viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="9" cy="13" r="4.2" fill="#ea4f86" opacity="0.9" />
          <circle cx="15.5" cy="9" r="2.8" fill="#f5c33b" opacity="0.9" />
          <circle cx="16" cy="16" r="1.8" fill="#8b3bf0" opacity="0.95" />
        </svg>
        <span class="topbar-word">
          nebul<span class="topbar-word-dim">.ai</span>
        </span>
      </header>
      <nav class="topnav" aria-label="Primary">
        <NavPill label="Semantic map" pageId="map" active={page} />
        <NavPill label="Internals" pageId="interp" active={page} />
        <NavPill label="Guide" pageId="guide" active={page} />
        <NavPill label="Snapshot Map" pageId="snapshot" active={page} />
        <NavPill label="Sessions" pageId="sessions" active={page} />
      </nav>
      <div class="topbar-tools">
        <button
          type="button"
          class={`topbar-tool${copied.value ? " is-flash" : ""}`}
          aria-label="Copy link to this view"
          title={copied.value ? "Link copied" : "Copy link to this view"}
          onClick={copyLink}
        >
          {copied.value ? (
            <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
              <path
                d="M5 12.5l4.5 4.5L19 7.5"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
              <path
                d="M10 14a4.5 4.5 0 006.4 0l3.2-3.2a4.5 4.5 0 00-6.4-6.4l-1.7 1.7M14 10a4.5 4.5 0 00-6.4 0l-3.2 3.2a4.5 4.5 0 006.4 6.4l1.7-1.7"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
                stroke-linecap="round"
              />
            </svg>
          )}
        </button>
        {page !== "guide" && (
          <button
            type="button"
            class="topbar-tool"
            aria-label="Save view as PNG"
            title="Save view as PNG"
            disabled={saving.value}
            onClick={savePng}
          >
            <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
              <path
                d="M12 4v10m0 0l-4-4m4 4l4-4M5 18h14"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          </button>
        )}
      </div>
      <button
        type="button"
        class="topbar-settings"
        aria-label="Open settings"
        title="Settings"
        onClick={() => appStore.getState().setSettingsOpen(true)}
      >
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
          <path
            d="M12 8.5a3.5 3.5 0 100 7 3.5 3.5 0 000-7zm7.7 4.7l1.9 1.5-1.8 3.1-2.3-.6a7.7 7.7 0 01-1.5.9l-.4 2.4h-3.6l-.4-2.4a7.7 7.7 0 01-1.5-.9l-2.3.6-1.8-3.1 1.9-1.5a7.6 7.6 0 010-1.4l-1.9-1.5 1.8-3.1 2.3.6a7.7 7.7 0 011.5-.9l.4-2.4h3.6l.4 2.4c.5.2 1 .5 1.5.9l2.3-.6 1.8 3.1-1.9 1.5c.1.5.1.9 0 1.4z"
            fill="none"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linejoin="round"
          />
        </svg>
        <span>Settings</span>
      </button>
    </>
  );
}

function NavPill(props: { label: string; pageId: Page; active: Page }) {
  const isActive = props.active === props.pageId;
  return (
    <button
      type="button"
      class={`topnav-pill${isActive ? " is-active" : ""}`}
      aria-current={isActive}
      onClick={() => appStore.getState().setPage(props.pageId)}
    >
      {props.label}
    </button>
  );
}
