/** KeywordPage — "who put this word here", over Claude Code's own transcripts.
 *
 *  The question this page exists for is not "does this word appear". It is:
 *  when an agent keeps doing something nobody asked for, *which party* keeps
 *  putting the idea in front of it. Five parties write into an agent's
 *  context — the operator, their standing instructions, the harness, the model
 *  itself, and the machine it reads — and a plain search answers for all five
 *  at once, which is to say it answers nothing.
 *
 *  Three rules follow, and they are the page:
 *
 *  1. **Every count is split by who wrote it.** The headline is not a total,
 *     it is the origin split, in a fixed order with the operator first. A
 *     total is available and it is deliberately the smaller number on screen.
 *
 *  2. **Nothing is dropped silently.** A duplicate the harness records twice
 *     is folded once and the fold is *counted* and shown; a channel switched
 *     off still reports how many hits it held; a compound the word boundary
 *     rejected (`interface` for `face`) is listed with the count a substring
 *     grep would have reported instead. "Not counted" is a section, not a
 *     silence.
 *
 *  3. **The scan runs on the server and the page says so.** Nearly a gigabyte
 *     of JSONL per project never enters the browser: the server reads it, this
 *     page polls a job and shows which transcript is under the head. Every
 *     number arrives from `sessionlog.py`; none is recomputed here, and the
 *     exact `seer keyword` command that reproduces the report is printed with
 *     it.
 *
 *  What this page cannot tell you is stated on it: a transcript records each
 *  injection as a line, not how long that text stayed in the window, so
 *  "context persistence" is reported `missing` rather than estimated.
 */

import { signal, useSignal } from "@preact/signals";
import type { JSX } from "preact";
import { useEffect } from "preact/hooks";
import { seerBase } from "../seer/client";
import {
  ORIGINS,
  ORIGIN_COLOR,
  ORIGIN_GLOSS,
  cliFor,
  fetchKeywordChannels,
  fetchKeywordJob,
  fetchKeywordProjects,
  fmtBytes,
  pct,
  startKeywordScan,
  type ChannelRow,
  type KeywordJob,
  type KeywordReport,
  type KeywordRequest,
  type KeywordSample,
  type ProjectRow,
} from "../seer/keyword";

// ── page state (module level: a five-minute scan outlives a nav click) ──────

const $projects = signal<ProjectRow[]>([]);
const $projectsRoot = signal<string>("");
const $channels = signal<ChannelRow[]>([]);
const $job = signal<KeywordJob | null>(null);
const $formError = signal<string | null>(null);
/** The request behind `$job`, kept so the page can print the command that
 *  reproduces it rather than reconstructing it from the report. */
const $lastRequest = signal<KeywordRequest | null>(null);

interface NamedPattern {
  name: string;
  pattern: string;
}

const $term = signal("");
const $project = signal("");
const $maxSessions = signal(0);
const $regex = signal(false);
const $caseSensitive = signal(false);
const $window = signal(120);
const $samples = signal(4);
/** `null` until the operator opens the channel panel: until then the server's
 *  own defaults apply and the page does not pretend to have an opinion. */
const $onlyChannels = signal<string[] | null>(null);
const $senses = signal<NamedPattern[]>([]);
const $excludes = signal<NamedPattern[]>([]);

function defaultChannelIds(rows: ChannelRow[]): string[] {
  return rows.filter((c) => c.on_by_default).map((c) => c.id);
}

// ── page ────────────────────────────────────────────────────────────────────

export function KeywordPage() {
  const loadError = useSignal<string | null>(null);

  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const [p, c] = await Promise.all([fetchKeywordProjects(), fetchKeywordChannels()]);
        if (!live) return;
        $projects.value = p.projects;
        $projectsRoot.value = p.root;
        $channels.value = c.channels;
        loadError.value = null;
      } catch (e) {
        if (live) loadError.value = e instanceof Error ? e.message : String(e);
      }
    })();
    return () => {
      live = false;
    };
  }, []);

  // Poll the running job. 1.2s rather than a stream: a scan reports once per
  // transcript, and a transcript takes tens of seconds — an SSE channel here
  // would carry one frame a minute.
  useEffect(() => {
    const id = window.setInterval(() => {
      const job = $job.value;
      if (!job || job.state !== "running") return;
      void fetchKeywordJob(job.job_id)
        .then((next) => {
          $job.value = next;
        })
        .catch((e) => {
          $job.value = {
            ...job,
            state: "error",
            error: e instanceof Error ? e.message : String(e),
          };
        });
    }, 1200);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div class="kw-page">
      <div class="kw-shell">
        <Rail loadError={loadError.value} />
        <div class="kw-stage">
          <Stage />
        </div>
      </div>
    </div>
  );
}

// ── the rail: the question ──────────────────────────────────────────────────

function Rail({ loadError }: { loadError: string | null }) {
  const showChannels = useSignal(false);
  const job = $job.value;
  const running = job?.state === "running";
  const base = seerBase();

  function run() {
    const term = $term.value.trim();
    if (!term) {
      $formError.value = "type a word to look for";
      return;
    }
    if (!$project.value) {
      $formError.value = "pick a project";
      return;
    }
    const req: KeywordRequest = {
      term,
      project: $project.value,
      regex: $regex.value,
      case_sensitive: $caseSensitive.value,
      window: $window.value,
      samples: $samples.value,
    };
    if ($maxSessions.value > 0) req.max_sessions = $maxSessions.value;
    if ($onlyChannels.value) req.only_channels = $onlyChannels.value;
    const senses = Object.fromEntries(
      $senses.value.filter((s) => s.name && s.pattern).map((s) => [s.name, s.pattern]),
    );
    if (Object.keys(senses).length) req.senses = senses;
    const ex = Object.fromEntries(
      $excludes.value.filter((s) => s.name && s.pattern).map((s) => [s.name, s.pattern]),
    );
    if (Object.keys(ex).length) req.excludes = ex;

    $formError.value = null;
    $lastRequest.value = req;
    void startKeywordScan(req)
      .then((j) => {
        $job.value = j;
      })
      .catch((e) => {
        $formError.value = e instanceof Error ? e.message : String(e);
      });
  }

  const project = $projects.value.find((p) => p.slug === $project.value);

  return (
    <aside class="kw-rail">
      <div class="kw-card">
        <div class="kw-eyebrow">the word</div>
        <input
          class="ctl-input kw-term"
          type="text"
          placeholder="face"
          value={$term.value}
          spellcheck={false}
          onInput={(e) => ($term.value = (e.target as HTMLInputElement).value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !running) run();
          }}
        />
        <div class="kw-note">
          Matched on word boundaries: <code>face</code> does not match{" "}
          <code>interface</code>. Every compound it rejects is counted and listed, so
          you can see the number a substring search would have shown you instead.
        </div>
        <label class="kw-check">
          <input
            type="checkbox"
            checked={$regex.value}
            onChange={(e) => ($regex.value = (e.target as HTMLInputElement).checked)}
          />
          <span>
            treat it as a regular expression
            <em>
              for a rule rather than a word — e.g. <code>{"exclud\\w*[^.]{0,40}\\bfaces?\\b"}</code>
            </em>
          </span>
        </label>
        <label class="kw-check">
          <input
            type="checkbox"
            checked={$caseSensitive.value}
            onChange={(e) =>
              ($caseSensitive.value = (e.target as HTMLInputElement).checked)
            }
          />
          <span>case sensitive</span>
        </label>
      </div>

      <div class="kw-card">
        <div class="kw-eyebrow">where to look</div>
        {loadError ? (
          <div class="kw-error">
            <strong>Cannot reach the capture server.</strong>
            <span class="kw-mono">{loadError}</span>
            <span>
              This page reads the transcripts through <code>seer serve</code> at{" "}
              <code>{base || "(no server configured)"}</code>, because a project's logs
              run to hundreds of megabytes and must not be loaded into a browser. Start
              it with <code>seer serve</code>, or set the address in Settings.
            </span>
          </div>
        ) : (
          <>
            <select
              class="ctl-select"
              value={$project.value}
              onChange={(e) => ($project.value = (e.target as HTMLSelectElement).value)}
            >
              <option value="">— pick a project —</option>
              {$projects.value.map((p) => (
                <option key={p.slug} value={p.slug}>
                  {shortSlug(p.slug)} · {p.sessions} sessions · {fmtBytes(p.bytes)}
                </option>
              ))}
            </select>
            {project && (
              <div class="kw-note">
                {project.sessions} transcripts, {fmtBytes(project.bytes)} on disk;{" "}
                {project.titled} of them titled. Read on the server — nothing is
                downloaded here.
              </div>
            )}
            <div class="kw-field">
              <label class="ctl-label" for="kw-max">
                largest transcripts only
              </label>
              <input
                id="kw-max"
                class="ctl-input kw-num"
                type="number"
                min={0}
                max={500}
                value={$maxSessions.value}
                onInput={(e) =>
                  ($maxSessions.value = Number((e.target as HTMLInputElement).value) || 0)
                }
              />
            </div>
            <div class="kw-note">
              0 scans every transcript in the project. A smaller number is a partial
              answer and the report says how many it read.
            </div>
            <div class="kw-field">
              <label class="ctl-label" for="kw-window">
                snippet width
              </label>
              <input
                id="kw-window"
                class="ctl-input kw-num"
                type="number"
                min={20}
                max={600}
                step={20}
                value={$window.value}
                onInput={(e) =>
                  ($window.value = Number((e.target as HTMLInputElement).value) || 120)
                }
              />
            </div>
            <div class="kw-field">
              <label class="ctl-label" for="kw-samples">
                examples per channel
              </label>
              <input
                id="kw-samples"
                class="ctl-input kw-num"
                type="number"
                min={1}
                max={40}
                value={$samples.value}
                onInput={(e) =>
                  ($samples.value = Number((e.target as HTMLInputElement).value) || 4)
                }
              />
            </div>
          </>
        )}
      </div>

      <div class="kw-card">
        <button
          class="kw-toggle"
          type="button"
          onClick={() => {
            if (!$onlyChannels.value) $onlyChannels.value = defaultChannelIds($channels.value);
            showChannels.value = !showChannels.value;
          }}
        >
          {showChannels.value ? "▾" : "▸"} which channels count
          <span class="kw-toggle-n">
            {$onlyChannels.value
              ? `${$onlyChannels.value.length} of ${$channels.value.length}`
              : "defaults"}
          </span>
        </button>
        {showChannels.value && <ChannelPicker />}
      </div>

      <div class="kw-card">
        <PatternRows
          title="senses"
          singular="sense"
          hint="Your own reading of the word, as a regular expression. Anything matching no sense stays visible as unclassified — a sense is your hypothesis, and every count from it is reported heuristic."
          placeholderName="performer"
          placeholderPattern="\bfaces?\b[^.]{0,40}\b(cam|detect)\b"
          rows={$senses}
        />
        <PatternRows
          title="exclusions"
          singular="exclusion"
          hint="Occurrences you have decided not to count. Each rule reports its own tally, so an exclusion can never quietly remove the evidence."
          placeholderName="huggingface"
          placeholderPattern="hugging\s*face"
          rows={$excludes}
        />
      </div>

      {$formError.value && <div class="kw-error kw-error-form">{$formError.value}</div>}

      <button class="btn-primary kw-run" type="button" disabled={running} onClick={run}>
        {running ? "scanning…" : "Scan the transcripts"}
      </button>
    </aside>
  );
}

function ChannelPicker() {
  const rows = $channels.value;
  const chosen = new Set($onlyChannels.value ?? defaultChannelIds(rows));

  function toggle(id: string) {
    const next = new Set(chosen);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    $onlyChannels.value = rows.filter((c) => next.has(c.id)).map((c) => c.id);
  }

  return (
    <div class="kw-channels">
      <div class="kw-note">
        A channel is a place text can come from. Switching one off does not hide it: the
        report lists every suppressed channel with the number of hits it held.
      </div>
      {ORIGINS.filter((o) => rows.some((c) => c.origin === o)).map((origin) => (
        <div class="kw-channel-group" key={origin}>
          <div class="kw-channel-origin">
            <span class="kw-swatch" style={{ background: ORIGIN_COLOR[origin] }} />
            {origin}
          </div>
          {rows
            .filter((c) => c.origin === origin)
            .map((c) => (
              <label class="kw-check kw-check-tight" key={c.id} title={c.why}>
                <input
                  type="checkbox"
                  checked={chosen.has(c.id)}
                  onChange={() => toggle(c.id)}
                />
                <span>
                  {c.label}
                  <em>{c.why}</em>
                </span>
              </label>
            ))}
        </div>
      ))}
      <button
        class="btn-ghost kw-reset"
        type="button"
        onClick={() => ($onlyChannels.value = defaultChannelIds(rows))}
      >
        back to the defaults
      </button>
    </div>
  );
}

function PatternRows({
  title,
  singular,
  hint,
  placeholderName,
  placeholderPattern,
  rows,
}: {
  title: string;
  singular: string;
  hint: string;
  placeholderName: string;
  placeholderPattern: string;
  rows: typeof $senses;
}) {
  const list = rows.value;
  return (
    <div class="kw-patterns">
      <div class="kw-eyebrow">{title}</div>
      <div class="kw-note">{hint}</div>
      {list.map((row, i) => (
        <div class="kw-pattern-row" key={i}>
          <input
            class="ctl-input kw-pattern-name"
            type="text"
            placeholder={placeholderName}
            value={row.name}
            spellcheck={false}
            onInput={(e) => {
              const next = list.slice();
              next[i] = { ...row, name: (e.target as HTMLInputElement).value };
              rows.value = next;
            }}
          />
          <input
            class="ctl-input kw-pattern-re"
            type="text"
            placeholder={placeholderPattern}
            value={row.pattern}
            spellcheck={false}
            onInput={(e) => {
              const next = list.slice();
              next[i] = { ...row, pattern: (e.target as HTMLInputElement).value };
              rows.value = next;
            }}
          />
          <button
            class="kw-drop"
            type="button"
            title="remove"
            onClick={() => (rows.value = list.filter((_, j) => j !== i))}
          >
            ×
          </button>
        </div>
      ))}
      <button
        class="btn-ghost kw-add"
        type="button"
        onClick={() => (rows.value = [...list, { name: "", pattern: "" }])}
      >
        + add a {singular}
      </button>
    </div>
  );
}

// ── the stage: the answer ───────────────────────────────────────────────────

function Stage() {
  const job = $job.value;
  if (!job) return <Explainer />;
  if (job.state === "error")
    return (
      <div class="kw-error kw-error-stage">
        <strong>The scan stopped.</strong>
        <span class="kw-mono">{job.error}</span>
      </div>
    );
  if (job.state === "running" || !job.report) return <Progress job={job} />;
  return <Report rep={job.report} job={job} />;
}

function Explainer() {
  return (
    <div class="kw-explainer">
      <h2>Who put this word in the agent's context?</h2>
      <p>
        Five different parties write into an agent's context, and only one of them is
        you. Searching a transcript answers for all five at once — which is why a word
        that “keeps coming up” tells you nothing about where the behaviour came from.
        This reads the transcripts on disk and splits every occurrence by who wrote it.
      </p>
      <ul class="kw-origin-legend">
        {ORIGINS.map((o) => (
          <li key={o}>
            <span class="kw-swatch" style={{ background: ORIGIN_COLOR[o] }} />
            <b>{o}</b>
            <span>{ORIGIN_GLOSS[o]}</span>
          </li>
        ))}
      </ul>
      <p class="kw-note">
        The same text is often recorded twice on one line — an attachment and the
        harness's rendered copy of it, a tool result and its echo under a second key.
        Those folds are counted and reported rather than quietly inflating a total, and
        duplicates that would have doubled an answer are the reason a number here can be
        lower than one from <code>grep</code>.
      </p>
    </div>
  );
}

function Progress({ job }: { job: KeywordJob }) {
  const frac = job.bytes_total > 0 ? job.bytes_done / job.bytes_total : 0;
  const secs = Math.max(0, (Date.now() - job.started * 1000) / 1000);
  return (
    <div class="kw-progress">
      <div class="kw-eyebrow">scanning · {job.project.replace(/^-/, "")}</div>
      <h2>
        {job.term} <span class="kw-dim">in {job.sessions_total} transcripts</span>
      </h2>
      <div class="kw-bar-track">
        <div class="kw-bar-fill" style={{ width: `${Math.round(frac * 100)}%` }} />
      </div>
      <div class="kw-progress-rows">
        <span>
          {job.sessions_done} of {job.sessions_total} transcripts
        </span>
        <span>
          {fmtBytes(job.bytes_done)} of {fmtBytes(job.bytes_total)}
        </span>
        <span>{secs.toFixed(0)}s elapsed</span>
      </div>
      <div class="kw-note">
        Reading <b>{job.reading}</b>. Progress moves once per transcript, and the largest
        here is over 100 MB, so a still bar is the scan working rather than stalling.
      </div>
    </div>
  );
}

function Report({ rep, job }: { rep: KeywordReport; job: KeywordJob }) {
  const total = rep.totals.counted;
  const notCounted =
    rep.scanned.duplicate_fields_folded +
    rep.totals.suppressed +
    rep.totals.excluded;
  const cli = $lastRequest.value
    ? cliFor($lastRequest.value, defaultChannelIds($channels.value))
    : null;

  return (
    <div class="kw-report">
      <header class="kw-report-head">
        <div class="kw-eyebrow">
          {shortSlug(rep.project?.slug ?? job.project)} ·{" "}
          <span title={rep.project?.matched_how ?? ""}>{rep.project?.matched_how}</span>
        </div>
        <h2>
          <span class="kw-term-echo">{rep.term}</span>
          <span class="kw-dim">
            {" "}
            matched as <code>{rep.pattern}</code>
            {rep.case_sensitive ? " (case-sensitive)" : " (case-insensitive)"}
          </span>
        </h2>
        <div class="kw-scanned">
          {rep.scanned.sessions} transcripts · {fmtBytes(rep.scanned.bytes)} ·{" "}
          {rep.scanned.lines.toLocaleString()} lines,{" "}
          {rep.scanned.lines_examined.toLocaleString()} examined ·{" "}
          {rep.scanned.elapsed_s.toFixed(1)}s
          {rep.scanned.lines_unparsable > 0 && (
            <span class="kw-warn">
              {" "}
              · {rep.scanned.lines_unparsable} lines were not valid JSON and were skipped
            </span>
          )}
        </div>
      </header>

      <section class="kw-origins">
        <div class="kw-section-head">
          <h3>Who put it there</h3>
          <div class="kw-total">
            <b>{total.toLocaleString()}</b> counted
          </div>
        </div>
        {/* All five parties are listed even at zero. A missing row would read as
            "not measured"; an explicit 0 is the finding — nobody in that role
            said the word. `metadata` is the exception: it is off unless the
            operator turns one of its channels on, so a 0 there means nothing. */}
        {ORIGINS.filter((o) => o !== "metadata" || (rep.by_origin[o] ?? 0) > 0).map((o) => {
          const n = rep.by_origin[o] ?? 0;
          return (
            <div class={n > 0 ? "kw-origin-row" : "kw-origin-row is-zero"} key={o}>
              <span class="kw-origin-name">
                <span class="kw-swatch" style={{ background: ORIGIN_COLOR[o] }} />
                {o}
              </span>
              <span class="kw-bar-track kw-bar-inline">
                <span
                  class="kw-bar-fill"
                  style={{
                    width: `${total > 0 ? (100 * n) / total : 0}%`,
                    background: ORIGIN_COLOR[o],
                  }}
                />
              </span>
              <span class="kw-origin-n">{n.toLocaleString()}</span>
              <span class="kw-origin-pct">{pct(n, total)}</span>
              <span class="kw-origin-gloss">{ORIGIN_GLOSS[o]}</span>
            </div>
          );
        })}
        {total === 0 && (
          <div class="kw-note">
            No occurrence of this word in any counted channel. That is an answer, not an
            empty page — check “not counted” below before concluding it is absent.
          </div>
        )}
      </section>

      <Fidelity rep={rep} />

      <section class="kw-block">
        <h3>By channel</h3>
        <table class="kw-table">
          <tbody>
            {Object.entries(rep.by_channel).map(([id, n]) => {
              const ch = $channels.value.find((c) => c.id === id);
              return (
                <tr key={id}>
                  <td class="kw-td-n">{n.toLocaleString()}</td>
                  <td class="kw-td-bar">
                    <span class="kw-bar-track kw-bar-inline">
                      <span
                        class="kw-bar-fill"
                        style={{
                          width: `${total > 0 ? (100 * n) / total : 0}%`,
                          background: ORIGIN_COLOR[ch?.origin ?? "metadata"],
                        }}
                      />
                    </span>
                  </td>
                  <td class="kw-td-label">{ch?.label ?? id}</td>
                  <td class="kw-td-why">{ch?.why ?? id}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      {Object.keys(rep.by_sense).length > 1 && (
        <section class="kw-block">
          <h3>
            By sense <span class="kw-chip kw-chip-heuristic">heuristic</span>
          </h3>
          <div class="kw-note">
            Your patterns, not a fact the transcript states. Anything matching none of
            them stays visible as <code>unclassified</code>.
          </div>
          <table class="kw-table">
            <tbody>
              {Object.entries(rep.by_sense)
                .sort((a, b) => b[1] - a[1])
                .map(([sense, n]) => (
                  <tr key={sense}>
                    <td class="kw-td-n">{n.toLocaleString()}</td>
                    <td class="kw-td-label">{sense}</td>
                    <td class="kw-td-why">{pct(n, total)} of counted</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </section>
      )}

      <section class="kw-block kw-notcounted">
        <div class="kw-section-head">
          <h3>Not counted</h3>
          <div class="kw-total kw-total-dim">{notCounted.toLocaleString()} occurrences</div>
        </div>
        {rep.scanned.duplicate_fields_folded > 0 && (
          <div class="kw-nc-row">
            <b>{rep.scanned.duplicate_fields_folded.toLocaleString()}</b>
            <span>
              the same text recorded twice on one line, under two different keys — folded
              to one. Counting both would inflate every total by the harness's own
              bookkeeping.
            </span>
          </div>
        )}
        {rep.suppressed_channels.map((s) => (
          <div class="kw-nc-row" key={s.channel}>
            <b>{s.hits.toLocaleString()}</b>
            <span>
              <code>{s.channel}</code> — {s.label}: {s.why}
            </span>
          </div>
        ))}
        {Object.entries(rep.excluded_by_rule).map(([name, n]) => (
          <div class="kw-nc-row" key={name}>
            <b>{n.toLocaleString()}</b>
            <span>
              excluded by your rule <code>{name}</code>
            </span>
          </div>
        ))}
        {notCounted === 0 && (
          <div class="kw-note">
            Nothing was folded, suppressed or excluded: every occurrence found is in the
            counts above.
          </div>
        )}
      </section>

      <section class="kw-block">
        <h3>What a substring search would have counted</h3>
        <div class="kw-note">{rep.compounds_note}</div>
        <div class="kw-compounds">
          {Object.entries(rep.compounds_rejected_by_word_boundary).map(([word, n]) => (
            <span
              class={
                rep.inflections_rejected[word] ? "kw-compound is-inflection" : "kw-compound"
              }
              key={word}
            >
              {word} <b>{n.toLocaleString()}</b>
            </span>
          ))}
          {rep.compounds_unwordlike.hits > 0 && (
            <span class="kw-compound is-lump" title={rep.compounds_unwordlike.note}>
              {rep.compounds_unwordlike.forms} strings too long to be words{" "}
              <b>{rep.compounds_unwordlike.hits.toLocaleString()}</b>
            </span>
          )}
        </div>
        {Object.keys(rep.inflections_rejected).length > 0 && (
          <div class="kw-inflections">
            <b>Highlighted above are inflections of your own word</b>, not other words:{" "}
            {Object.entries(rep.inflections_rejected)
              .map(([w, n]) => `${w} ${n.toLocaleString()}`)
              .join(" · ")}
            . To count them, search{" "}
            <code>
              {`\\b(${[rep.term, ...Object.keys(rep.inflections_rejected)].join("|")})\\b`}
            </code>{" "}
            as a regular expression. {rep.inflections_note}
          </div>
        )}
        {Object.keys(rep.surface_forms).length > 1 && (
          <div class="kw-note">
            Surface forms actually counted:{" "}
            {Object.entries(rep.surface_forms)
              .map(([f, n]) => `${f} ${n.toLocaleString()}`)
              .join(" · ")}
          </div>
        )}
      </section>

      <section class="kw-block">
        <div class="kw-section-head">
          <h3>Transcripts with hits</h3>
          <div class="kw-total kw-total-dim">
            {rep.totals.sessions_with_hits} of {rep.scanned.sessions}
          </div>
        </div>
        <table class="kw-table kw-table-sessions">
          <tbody>
            {rep.sessions
              .filter((s) => s.occurrences > 0)
              .map((s) => (
                <tr key={s.session_id}>
                  <td class="kw-td-n">{s.occurrences.toLocaleString()}</td>
                  <td class="kw-td-label">
                    {s.title ?? <span class="kw-dim">untitled</span>}
                    <span class="kw-mono kw-sid">{s.session_id.slice(0, 8)}</span>
                  </td>
                  <td class="kw-td-mix">
                    {ORIGINS.filter((o) => (s.by_origin[o] ?? 0) > 0).map((o) => (
                      <span
                        class="kw-mix"
                        key={o}
                        title={`${o}: ${s.by_origin[o]}`}
                        style={{
                          background: ORIGIN_COLOR[o],
                          width: `${(100 * (s.by_origin[o] ?? 0)) / s.occurrences}%`,
                        }}
                      />
                    ))}
                  </td>
                  <td class="kw-td-why">
                    {fmtBytes(s.bytes)} · lines {s.first_line}–{s.last_line}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </section>

      <section class="kw-block">
        <div class="kw-section-head">
          <h3>What it actually says</h3>
          <div class="kw-total kw-total-dim">
            {rep.samples.length} of {total.toLocaleString()}, spread across channels
          </div>
        </div>
        <div class="kw-note">
          Up to a few per channel and sense, so a rare but decisive channel is never
          crowded out by a noisy one.
        </div>
        {rep.samples.map((s, i) => (
          <Sample s={s} key={`${s.session_id}-${s.line}-${i}`} />
        ))}
      </section>

      {cli && (
        <section class="kw-block kw-cli">
          <h3>Reproduce this</h3>
          <code class="kw-mono kw-cli-line">{cli}</code>
          <div class="kw-note">
            The same scan from a terminal, with <code>--json</code> to write the full
            report. Schema <code>{rep.schema}</code>.
          </div>
        </section>
      )}
    </div>
  );
}

function Fidelity({ rep }: { rep: KeywordReport }) {
  const f = rep.fidelity;
  const rows: [string, string, string][] = [
    ["counts", f.counts, "every occurrence was read from the file, not sampled or estimated"],
    [
      "channel attribution",
      f.channel_attribution,
      "who wrote it comes from the line's own type and JSON path — not inferred from the text",
    ],
    ["sense classification", f.sense_classification, "your patterns, and your hypothesis"],
    ["context persistence", f.context_persistence, f.context_persistence_note],
  ];
  return (
    <section class="kw-block kw-fidelity">
      <h3>How much to trust each part</h3>
      {rows.map(([name, level, why]) => (
        <div class="kw-fid-row" key={name}>
          <span class={`kw-chip kw-chip-${level}`}>{level}</span>
          <b>{name}</b>
          <span>{why}</span>
        </div>
      ))}
    </section>
  );
}

function Sample({ s }: { s: KeywordSample }) {
  return (
    <div class="kw-sample">
      <div class="kw-sample-head">
        <span class="kw-swatch" style={{ background: ORIGIN_COLOR[s.origin] }} />
        <span class="kw-sample-channel">{s.channel}</span>
        <span class="kw-dim">
          {s.title ?? s.session_id.slice(0, 8)} · line {s.line}
        </span>
        <span class="kw-mono kw-sample-path">{s.json_path}</span>
        {s.sense !== "unclassified" && <span class="kw-chip kw-chip-sense">{s.sense}</span>}
        {s.timestamp && <span class="kw-dim">{s.timestamp.replace("T", " ").slice(0, 19)}</span>}
      </div>
      <div class="kw-snippet">{highlight(s.snippet, s.form)}</div>
    </div>
  );
}

// ── helpers ─────────────────────────────────────────────────────────────────

/** The project slug is an encoded absolute path; the tail is the part anyone
 *  recognises. The full slug stays in the title attribute — it is what the
 *  server resolved, and hiding it would make an ambiguous match invisible. */
function shortSlug(slug: string): string {
  const parts = slug.replace(/^-+/, "").split("-");
  return parts.slice(Math.max(0, parts.length - 4)).join("-");
}

/** Mark the matched form inside a snippet. Split on the literal text that was
 *  counted, never on a re-derived pattern: a second matcher on this side could
 *  disagree with the one that produced the count. */
function highlight(snippet: string, form: string) {
  if (!form) return snippet;
  const out: (string | JSX.Element)[] = [];
  const lower = snippet.toLowerCase();
  const needle = form.toLowerCase();
  let i = 0;
  let n = 0;
  for (;;) {
    const at = lower.indexOf(needle, i);
    if (at < 0 || n > 40) {
      out.push(snippet.slice(i));
      break;
    }
    if (at > i) out.push(snippet.slice(i, at));
    out.push(<mark key={`${at}-${n}`}>{snippet.slice(at, at + form.length)}</mark>);
    i = at + form.length;
    n += 1;
  }
  return out;
}

export default KeywordPage;
