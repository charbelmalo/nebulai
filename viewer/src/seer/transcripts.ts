/** transcripts.ts — the sessions on this machine, without Finder.
 *
 *  Claude Code keeps one directory per project under `~/.claude/projects`, and
 *  one `.jsonl` per session inside it. The Keywords page has read that tree
 *  through the collector since it shipped; the Sessions plotter could only be
 *  fed by dragging a file out of Finder, which meant knowing that the directory
 *  exists, that its name is the working directory with every `/` and `_`
 *  replaced by `-`, and which of 47 identically-shaped hex filenames was the
 *  session you actually wanted. This module is the list.
 *
 *  It fetches metadata, not content: `bytes` and `modified` come from the stat,
 *  `title` from the sidecar the client writes beside the log. The transcript
 *  itself is fetched only when a session is chosen, and it arrives verbatim —
 *  the page parses it with `parseSessionTranscript`, the same function that
 *  parses a dropped file, because a session has one parser and it lives in the
 *  page.
 *
 *  Privacy, stated plainly: the bytes travel from the collector on loopback to
 *  the browser on the same machine, and the parse is still in memory. Nothing
 *  new leaves the machine, but "never transmitted" is no longer the right
 *  sentence for it, and the page says so.
 */

import { getJSON, seerBase } from "./client";

/** One project directory: a working directory that has sessions in it. */
export interface TranscriptProject {
  slug: string;
  path: string;
  sessions: number;
  bytes: number;
  /** How many of those sessions have a title the operator gave them. */
  titled: number;
  /** Newest transcript in the project, epoch seconds. 0 when unknown. */
  modified: number;
}

/** One transcript on disk. No line count: that would mean reading 125 MB to
 *  render a list row, and the size already answers "is this a big one". */
export interface TranscriptRow {
  session_id: string;
  title: string | null;
  path: string;
  bytes: number;
  /** `st_mtime`, epoch seconds — the filesystem's number, not a date string,
   *  because the only question the picker asks of it is which is newest. */
  modified: number;
}

export function fetchTranscriptProjects(
  root?: string,
): Promise<{ root: string; projects: TranscriptProject[] }> {
  const q = root ? `?root=${encodeURIComponent(root)}` : "";
  return getJSON(`/seer/projects${q}`);
}

export function fetchTranscripts(
  project: string,
  root?: string,
): Promise<{
  project: { slug: string; path: string; matched_how: string };
  transcripts: TranscriptRow[];
}> {
  const q = new URLSearchParams({ project });
  if (root) q.set("root", root);
  return getJSON(`/seer/transcripts?${q}`);
}

/** The raw `.jsonl`, as text.
 *
 *  Not `getJSON`: a transcript is newline-delimited JSON, so the document as a
 *  whole is not JSON and parsing it as such would fail on every real file. The
 *  error path is spelled out here rather than reused because the failure a
 *  reader needs to see is "the collector is not running", not "unexpected token".
 */
export async function fetchTranscriptText(
  project: string,
  session: string,
  root?: string,
): Promise<string> {
  const base = seerBase();
  if (!base) throw new Error("no seer server configured");
  const q = new URLSearchParams({ project, session });
  if (root) q.set("root", root);
  const res = await fetch(`${base}/seer/transcript?${q}`);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { error?: string };
      if (body.error) detail = body.error;
    } catch {
      /* the body was not JSON; the status is all we have */
    }
    throw new Error(detail);
  }
  return await res.text();
}

/** Newest first — the opposite of the order the collector returns.
 *
 *  The server sorts transcripts largest first because that is what a keyword
 *  scan wants from `--max-sessions`. A person looking for the session they just
 *  ran wants the most recent one, and a transcript with no usable mtime sorts
 *  last rather than first, so a missing number can never masquerade as "now".
 */
export function byNewest<T extends { modified: number }>(rows: T[]): T[] {
  return [...rows].sort((a, b) => (b.modified || 0) - (a.modified || 0));
}

/** "2h ago", "3d ago" — relative, because the useful comparison is with now.
 *  An absolute date goes in the `title` attribute, where it can be read when
 *  the relative one is ambiguous. */
export function ago(epochSeconds: number, now = Date.now()): string {
  if (!epochSeconds) return "date unknown";
  const s = Math.max(0, now / 1000 - epochSeconds);
  if (s < 90) return "just now";
  const m = s / 60;
  if (m < 60) return `${Math.round(m)}m ago`;
  const h = m / 60;
  if (h < 24) return `${Math.round(h)}h ago`;
  const d = h / 24;
  if (d < 30) return `${Math.round(d)}d ago`;
  return `${Math.round(d / 30)}mo ago`;
}

export function fmtBytes(n: number): string {
  if (n >= 1 << 30) return `${(n / (1 << 30)).toFixed(1)} GB`;
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(n >= 10 << 20 ? 0 : 1)} MB`;
  if (n >= 1 << 10) return `${Math.round(n / (1 << 10))} kB`;
  return `${n} B`;
}

/** A stable id for a transcript on disk, so picking it twice is picking one
 *  thing twice.
 *
 *  The pages that consume these mint a random id per parse, which is right for
 *  a dropped file and wrong for a file the picker can name: without a stable id
 *  a session re-read after it grew would appear twice, once at each size, with
 *  no way to tell which row was current. Project and session together, because
 *  a session id is only unique inside its project directory.
 */
export function localSessionId(project: string, session: string): string {
  return `local:${project}/${session}`;
}

/** A slug shortened to the part that distinguishes it.
 *
 *  `-Users-charbelmalo-Developer-gesture-sorcery-game-kit` is an encoded
 *  working directory, not a project name: every `/`, `_` and `.` became `-`, so
 *  the encoding cannot be inverted — `gesture-sorcery-game-kit` is either
 *  `gesture_sorcery_game_kit` or `gesture/sorcery/game/kit` and the slug no
 *  longer knows which. So this shortens rather than decodes: the home directory
 *  is the one prefix we can identify (the collector tells us where the projects
 *  root is), and dropping it leaves `Developer-gesture-sorcery-game-kit`, which
 *  is short enough for a list row and still says which project it is. Never
 *  just the last segment — that would label this project "kit". The full slug
 *  stays on the row as its tooltip.
 */
export function projectLabel(slug: string, projectsRoot = ""): string {
  // …/<home>/.claude/projects → the encoded form of <home>
  const home = projectsRoot.replace(/\/\.claude\/projects\/?$/, "");
  const encodedHome = home ? home.replace(/[/_.]/g, "-") : "";
  const short =
    encodedHome && slug.startsWith(encodedHome) ? slug.slice(encodedHome.length) : slug;
  return short.replace(/^-+/, "") || slug;
}
