/** Top-left brand mark + wordmark, matching the video's logo anatomy. A center
 *  nav switches between the three pages of whichever instrument is running.
 *  Top-right hosts share/export tools and the global gear that opens the
 *  full Settings page.
 *
 *  Everything instrument-specific arrives as the `app` prop (see
 *  chrome/apps/nav.ts): the mark, the wordmark, the three pills and their
 *  labels, and the single link across to the other instrument. Nothing about
 *  Nebulai or Seer is hardcoded here — this bar renders in both documents, and
 *  the branding in particular must not claim Seer is nebul.ai. Nebulai keeps
 *  the nebula mark; Seer flies its own.
 *
 *  The cross-instrument link is a plain anchor after a divider, NOT a fourth
 *  pill: it leaves this document for the other one. In a hub deploy a second
 *  such anchor points up at psychiX (`app.hub`, absent in the combined build
 *  where no hub document exists). Both stay subordinate: a person who has just
 *  learned that these are two instruments should not have to re-learn it every
 *  time their eye crosses the nav. */

import { useSignal } from "@preact/signals";
import { appStore, type Page } from "../app/store";
import type { MarkId, SiblingLink } from "./apps/nav";
import type { AppShell } from "./apps/types";
import { downloadStagePng } from "./exportPng";
import { $page } from "./state";
import { shareUrl } from "./urlState";

export function TopBar({ app }: { app: AppShell }) {
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
        <BrandMark mark={app.mark} />
        <span class="topbar-word" title={app.tagline}>
          {app.wordmark.head}
          {app.wordmark.tail && <span class="topbar-word-dim">{app.wordmark.tail}</span>}
        </span>
      </header>
      <nav class="topnav" aria-label="Primary">
        {app.nav.map((item) => (
          <NavPill key={item.page} label={item.label} pageId={item.page} active={page} />
        ))}
        <span class="topnav-sep" aria-hidden="true" />
        <CrossLink link={app.sibling} />
        {app.hub && <CrossLink link={app.hub} />}
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

/** A link that LEAVES this document — the other instrument, or the psychiX
 *  hub above both. Same treatment for both: subordinate to the pills, arrow
 *  included, `href` supplied by apps/nav.ts (relative in the combined build,
 *  an absolute sub-path in a per-app deploy — this component never decides). */
function CrossLink({ link }: { link: SiblingLink }) {
  return (
    <a class="topnav-cross" href={link.href} title={link.title}>
      {link.label}
      <span class="topnav-cross-arrow" aria-hidden="true">
        ↗
      </span>
    </a>
  );
}

/** The two brand marks, drawn here rather than in apps/nav.ts so that module
 *  stays data-only. Both live in the shared chunk; they are a few hundred
 *  bytes of path data and splitting them per entry would buy nothing.
 *
 *  `nebula` is three ramp-coloured dots in orbit — the semantic cloud in
 *  miniature. `eye` is a lens with a gold iris, matching seer.html's favicon:
 *  the instrument watches a run rather than mapping a space. */
function BrandMark({ mark }: { mark: MarkId }) {
  if (mark === "eye") {
    return (
      <svg class="topbar-mark" viewBox="0 0 24 24" aria-hidden="true">
        <path
          d="M2.4 12c3-5.2 6.7-7.8 9.6-7.8S18.6 6.8 21.6 12c-3 5.2-6.7 7.8-9.6 7.8S5.4 17.2 2.4 12z"
          fill="none"
          stroke="#46c8eb"
          stroke-width="1.7"
          stroke-linejoin="round"
        />
        <circle cx="12" cy="12" r="3.3" fill="#f5c33b" />
      </svg>
    );
  }
  return (
    <svg class="topbar-mark" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="9" cy="13" r="4.2" fill="#ea4f86" opacity="0.9" />
      <circle cx="15.5" cy="9" r="2.8" fill="#f5c33b" opacity="0.9" />
      <circle cx="16" cy="16" r="1.8" fill="#8b3bf0" opacity="0.95" />
    </svg>
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
