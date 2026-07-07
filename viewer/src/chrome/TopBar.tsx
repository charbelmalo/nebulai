/** Top-left brand mark, matching the video's logo + wordmark anatomy. The
 *  mark is a tiny nebula: three ramp-colored dots in orbit. */

export function TopBar() {
  return (
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
  );
}
