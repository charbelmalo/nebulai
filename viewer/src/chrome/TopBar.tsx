/** Top-left brand mark, matching the video's logo + wordmark anatomy. The
 *  mark is a tiny nebula: three ramp-colored dots in orbit. Top-right hosts
 *  the global gear that opens the full Settings page. */

import { appStore } from "../app/store";

export function TopBar() {
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
