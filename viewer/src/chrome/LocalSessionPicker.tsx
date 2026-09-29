/** LocalSessionPicker — the sessions already on this machine, listed.
 *
 *  Two pages here read a Claude Code transcript: the Sessions plotter and the
 *  Snapshot Map. Both could only be fed by hand — drag the file out of Finder,
 *  or paste it — even though the collector has been reading the same directory
 *  tree for the Keywords page all along. Finding a file by hand means knowing
 *  that `~/.claude/projects` exists, that a project's directory name is its
 *  working directory with every `/`, `_` and `.` replaced by `-`, and which of
 *  47 identically-shaped hex filenames was the session you meant. This is that
 *  list instead: project, then session, newest first, with the title you gave
 *  it if you gave it one.
 *
 *  Per-instance state, not module-level signals: two pages mount this, and a
 *  project chosen on one is not a project chosen on the other. (The page-level
 *  signals elsewhere in this directory are per-page state, which is a different
 *  thing.)
 *
 *  It hands the caller raw text, a label and a stable id, and does no parsing:
 *  the Sessions page reads a transcript with `parseSessionTranscript` and the
 *  Snapshot Map with `parseConversationText`, and neither should learn a second
 *  way in. Its own classes are all `lsp-*`, including the card, because the two
 *  host rails style their blocks under different names and a shared component
 *  cannot borrow one page's.
 */

import { useSignal } from "@preact/signals";
import { useEffect } from "preact/hooks";
import {
  ago,
  byNewest,
  fetchTranscriptProjects,
  fetchTranscripts,
  fetchTranscriptText,
  fmtBytes,
  localSessionId,
  projectLabel,
  type TranscriptProject,
  type TranscriptRow,
} from "../seer/transcripts";

/** Above this, a transcript takes long enough to read and parse that the row
 *  says so before you press it. Measured on this machine: the largest local
 *  transcript is 125 MB and parsing it blocks the page for several seconds,
 *  which is fine if you were told and alarming if you were not. */
const SLOW_BYTES = 24 << 20;

export function LocalSessionPicker(props: {
  /** `id` is `localSessionId(project, session)` — stable across picks, so the
   *  host page can replace what it loaded before instead of duplicating it. */
  onPick: (text: string, label: string, id: string) => void;
  /** Ids the caller already has loaded. A listed session that is already loaded
   *  says so, and picking it again re-reads the file — which is the point for a
   *  session that is still running and still growing. */
  loadedIds?: string[];
  /** One line under the heading, in the host page's own words. */
  hint?: string;
}) {
  const projects = useSignal<TranscriptProject[]>([]);
  const root = useSignal("");
  const slug = useSignal("");
  const rows = useSignal<TranscriptRow[] | null>(null);
  const loadError = useSignal<string | null>(null);
  const listError = useSignal<string | null>(null);
  const busyId = useSignal<string | null>(null);

  // ── the project list, once ────────────────────────────────────────────
  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const body = await fetchTranscriptProjects();
        if (!live) return;
        const byRecency = byNewest(body.projects);
        projects.value = byRecency;
        root.value = body.root;
        loadError.value = null;
        // The project worked in most recently, not the largest one: the reason
        // to open this panel is almost always the session just run. Nothing is
        // fetched until a project is chosen, and this is that choice.
        if (!slug.value && byRecency.length) slug.value = byRecency[0]!.slug;
      } catch (e) {
        if (live) loadError.value = e instanceof Error ? e.message : String(e);
      }
    })();
    return () => {
      live = false;
    };
  }, []);

  // ── one project's transcripts, on every change of project ─────────────
  const chosen = slug.value;
  useEffect(() => {
    if (!chosen) return;
    let live = true;
    rows.value = null;
    void (async () => {
      try {
        const body = await fetchTranscripts(chosen, root.value || undefined);
        if (!live) return;
        rows.value = byNewest(body.transcripts);
        listError.value = null;
      } catch (e) {
        if (!live) return;
        rows.value = [];
        listError.value = e instanceof Error ? e.message : String(e);
      }
    })();
    return () => {
      live = false;
    };
  }, [chosen, root.value]);

  const pick = async (row: TranscriptRow) => {
    busyId.value = row.session_id;
    listError.value = null;
    try {
      const text = await fetchTranscriptText(chosen, row.session_id, root.value || undefined);
      props.onPick(text, labelFor(row), localSessionId(chosen, row.session_id));
    } catch (e) {
      listError.value = e instanceof Error ? e.message : String(e);
    } finally {
      busyId.value = null;
    }
  };

  const loaded = new Set(props.loadedIds ?? []);

  return (
    <section class="lsp">
      <h3>
        On this machine
        {projects.value.length > 0 && <span class="lsp-count">{projects.value.length}</span>}
      </h3>

      {loadError.value !== null ? (
        // "The collector is not running" is something the reader can fix in one
        // command, so it is printed as that command rather than as a failure.
        <>
          <p class="lsp-hint">Can’t reach the local session reader: {loadError.value}</p>
          <p class="lsp-hint">
            Start it with <code>uv run seer serve</code> and reload. Dropping a file in still
            works without it.
          </p>
        </>
      ) : (
        <>
          <p class="lsp-hint">{props.hint ?? DEFAULT_HINT}</p>

          <label class="lsp-field">
            <span>Project</span>
            <select
              class="ctl-select"
              value={chosen}
              onChange={(e) => (slug.value = (e.currentTarget as HTMLSelectElement).value)}
            >
              {projects.value.length === 0 && <option value="">loading…</option>}
              {projects.value.map((p) => (
                <option key={p.slug} value={p.slug} title={p.slug}>
                  {projectLabel(p.slug, root.value)} · {p.sessions} · {ago(p.modified)}
                </option>
              ))}
            </select>
          </label>

          {rows.value === null && chosen && <p class="lsp-hint">reading the directory…</p>}

          {rows.value !== null && rows.value.length === 0 && !listError.value && (
            <p class="lsp-hint">No transcripts in that project.</p>
          )}

          {rows.value !== null && rows.value.length > 0 && (
            <ul class="lsp-list">
              {rows.value.map((row) => {
                const label = labelFor(row);
                const isLoaded = loaded.has(localSessionId(chosen, row.session_id));
                const slow = row.bytes >= SLOW_BYTES;
                const busy = busyId.value === row.session_id;
                return (
                  <li key={row.session_id}>
                    <button
                      type="button"
                      class={`lsp-item${isLoaded ? " is-loaded" : ""}`}
                      disabled={busyId.value !== null}
                      title={`${row.path}\n${
                        row.modified ? new Date(row.modified * 1000).toLocaleString() : "date unknown"
                      }`}
                      onClick={() => void pick(row)}
                    >
                      <span class="lsp-item-name">
                        {label}
                        {isLoaded && <span class="lsp-tag">loaded</span>}
                      </span>
                      <span class="lsp-item-meta">
                        {busy
                          ? slow
                            ? "reading — this one is large…"
                            : "reading…"
                          : `${ago(row.modified)} · ${fmtBytes(row.bytes)}${
                              slow ? " · slow to parse" : ""
                            }`}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}

          {listError.value && <p class="lsp-error">{listError.value}</p>}
        </>
      )}
    </section>
  );
}

const DEFAULT_HINT =
  "Your own Claude Code sessions, read from disk by the local collector. Nothing leaves this machine.";

/** A session's own title if it has one, else the front of its id.
 *
 *  Never the whole 36-character uuid: it is unreadable, and the first eight
 *  characters are what every other surface in Seer prints for a session. The
 *  untitled case is the common one — most sessions are never titled — so it has
 *  to be legible rather than merely correct.
 */
function labelFor(row: TranscriptRow): string {
  return row.title?.trim() || `untitled ${row.session_id.slice(0, 8)}`;
}
