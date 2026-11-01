'use strict';
// The Adapters tab's back end: one entry per adapter file with what its header says, what its card says (config.js
// readCard) and what the tests found, plus the changes to an adapter that have to reach more than the file: its card,
// and the recipes that export it (a re-export would otherwise bring an old name back).
//
// Test results are matched by what each test ran on, not by recipe name. The same weights are often tested several
// times (g26b4std at 0.5, 0.6 and 0.7; g26b4 with and without the carry-on bar) and only one of those is how the file
// runs, while a recipe's own results can be for another base (run12q has run12's weights, stamped for the stock Qwen).
const fs = require('fs');
const path = require('path');
const { ROOT, BASES, adapterInfo, RENAMED, cardPath, readCard, writeCard, formerNames } = require('./config');

const OUT = path.join(ROOT, 'tune', 'out');
const TESTS = path.join(ROOT, 'tests', 'out');
const RECIPES = path.join(ROOT, 'tune', 'recipes');
const LORA = path.join(ROOT, 'models', 'lora');

const fileOf = (p) => String(p || '').split(/[\\/]/).pop();
const baseName = (b) => (BASES[b] ? b : Object.keys(BASES).find((k) => (BASES[k].aliases || []).includes(b)) || b);
// a recipe's name in the app's list of bases, and the file it exports to, lower-cased (file systems here ignore case)
function recipes() {
  return fs.readdirSync(RECIPES).filter((f) => f.endsWith('.json')).flatMap((file) => {
    try {
      const text = fs.readFileSync(path.join(RECIPES, file), 'utf8'), r = JSON.parse(text);
      return [{ file, text, r, base: baseName(r.base || 'original'), exports: (r.export_as || `talkme-${r.name}.gguf`).toLowerCase() }];
    } catch { return []; }
  });
}
// One top-level text value of a recipe, leaving the rest of the file as it is written
function setRecipeKey(entry, key, value) {
  const field = new RegExp(`("${key}"\\s*:\\s*)"(?:[^"\\\\]|\\\\.)*"`);
  let text = field.test(entry.text) ? entry.text.replace(field, (_m, head) => head + JSON.stringify(value))
    : entry.text.replace(/("name"\s*:\s*"[^"]*",)(\r?\n)/, (_m, line, nl) => `${line}${nl}  "${key}": ${JSON.stringify(value)},${nl}`);
  let done = false;
  try { done = JSON.parse(text)[key] === value; } catch {}
  if (!done) text = JSON.stringify({ ...entry.r, [key]: value }, null, 2) + '\n';
  fs.writeFileSync(path.join(RECIPES, entry.file), text);
}

// ---- test results ----
// JSON outputs parsed once, by modified time: tens of MB in all, read a file at a time so a call keeps ticking
const parsed = new Map(); // path -> { mtime, rows }
async function rowsOf(file, parse) {
  const mtime = fs.statSync(file).mtimeMs, hit = parsed.get(file);
  if (hit && hit.mtime === mtime) return hit.rows;
  let rows = [];
  try { rows = parse(JSON.parse(await fs.promises.readFile(file, 'utf8')), mtime); } catch {}
  parsed.set(file, { mtime, rows });
  return rows;
}
const dir = (d) => { try { return fs.readdirSync(d); } catch { return []; } };

// tune/evaluate.py's output: what ran is in args. Which calls it replayed says which test it was (the published
// adapter's only run is on its author's held-out set, which is not ours).
async function evalRuns() {
  const out = [];
  for (const f of dir(OUT).filter((x) => /^eval_.*\.json$/.test(x))) {
    out.push(...await rowsOf(path.join(OUT, f), (j, date) => {
      const a = j.args || {}, set = String(a.calls || '');
      const kind = /heldout_yap/.test(set) ? 'yap' : /heldout_ours/.test(set) ? 'ours' : /calls_heldout/.test(set) ? 'ours' : null;
      if (!kind || !j.totals) return [];
      return [{ kind, run: f.slice(5, -5).replace(/^(ours|yap)_/, ''), adapter: fileOf(a.adapter), listen: a.listen_p, carryOn: a.carry_on_p || undefined,
        calls: (j.calls || []).length, totals: j.totals, date, ...(/calls_heldout/.test(set) ? { set: "its author's held-out set, not ours" } : {}) }];
    }));
  }
  return out;
}
// yap_metrics.py's tables: a column per run, in that run's table and in those of the runs compared with it
function yapTable(file) {
  const out = {};
  let names = null;
  for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    if (!line.startsWith('|')) { names = null; continue; }
    const cells = line.split('|').slice(1, -1).map((c) => c.trim());
    if (!names) { names = cells; continue; }
    if (/^-+$/.test(cells[0])) continue;
    cells.slice(1).forEach((cell, i) => { (out[names[i + 1]] = out[names[i + 1]] || {})[cells[0]] = cell; });
  }
  return out;
}
// each run's column: from its own table if it has one, else the newest that shows it
function yapTables() {
  const files = dir(OUT).filter((f) => /^yap_.*\.md$/.test(f)).map((f) => ({ f, t: fs.statSync(path.join(OUT, f)).mtimeMs })).sort((a, b) => a.t - b.t);
  const out = {}, own = {};
  for (const { f } of files) {
    for (const [run, rows] of Object.entries(yapTable(path.join(OUT, f)))) (f === `yap_${run}.md` ? own : out)[run] = rows;
  }
  return { ...out, ...own };
}
// the quiz (probe_<run>.json) and the 36-question one (wide_<run>.json): these name no adapter, only a run
async function quizRuns() {
  const out = [];
  for (const f of dir(OUT)) {
    const m = /^(probe|wide)_(.+)\.json$/.exec(f);
    if (!m) continue;
    out.push(...await rowsOf(path.join(OUT, f), (j, date) => (typeof j.score === 'number'
      ? [{ kind: m[1] === 'probe' ? 'quiz' : 'wide', run: m[2], score: j.score, of: j.of, silent: j.silent, date }] : [])));
  }
  return out;
}
// the live bench, a report and a scoreboard per caller (scripts/live_bench.sh): added up as scripts/live_bench_report.js does
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : undefined; };
async function liveRuns() {
  const runs = {};
  for (const f of dir(TESTS)) {
    const m = /^report-bench-(.+)-([a-z]+)\.json$/.exec(f);
    if (!m) continue;
    const [row] = await rowsOf(path.join(TESTS, f), (j, date) => {
      const checks = (j.results || []).flatMap((r) => r.checks || []);
      const num = (start, re) => checks.filter((c) => c.check.startsWith(start)).map((c) => (re.exec(c.detail) || [])[1]).filter((x) => x !== undefined).map(Number);
      return [{ adapter: fileOf(j.adapter), listen: j.threshold, date, passed: (j.results || []).reduce((n, r) => n + r.passed, 0), total: (j.results || []).reduce((n, r) => n + r.total, 0),
        reply: num('reply within', /first audio (-?[\d.]+) s/), stop: num('stops talking', /cut ([\d.]+) s/) }];
    });
    if (!row) continue;
    const score = path.join(TESTS, `scoreboard-bench-${m[1]}-${m[2]}.json`);
    const [board = {}] = fs.existsSync(score) ? await rowsOf(score, (j) => [j]) : [];
    const r = runs[m[1]] = runs[m[1]] || { kind: 'live', run: m[1], adapter: row.adapter, listen: row.listen, date: 0, passed: 0, total: 0, callers: {}, reply: [], stop: [], ticks: [] };
    r.date = Math.max(r.date, row.date); r.passed += row.passed; r.total += row.total; r.callers[m[2]] = [row.passed, row.total];
    r.reply.push(...row.reply); r.stop.push(...row.stop);
    if (board.tickP95ms) r.ticks.push(board.tickP95ms);
  }
  return Object.values(runs).map(({ reply, stop, ticks, ...r }) => ({ ...r, medianReplyS: median(reply), medianStopS: median(stop), tickP95ms: ticks.length ? Math.max(...ticks) : undefined }));
}

// Which runs tested these weights: the recipes exporting this file and the one its header (or card) names, unless that
// was on another base. A test belongs when it ran on one of their tune/out files, or on this file under any name.
function runsOf(file, info, card, all) {
  const name = fileOf(file).toLowerCase();
  const named = all.filter((x) => x.exports === name || x.r.name === info.recipe || x.r.name === card.recipe);
  const on = (x) => !info.base || x.base === info.base;
  return { own: named.filter(on), elsewhere: named.filter((x) => !on(x)) };
}
// Closest to how the file runs first: on its base, on the held-out set of ours, at its threshold and bar, then the
// exported file itself over a tune/out copy, the longest run, and the plainest run name (g26b4_bar9, not its variants)
function ranked(rows, info) {
  const gap = (a) => (a.listen == null || info.listen == null ? 1 : Math.abs(a.listen - info.listen));
  const bar = (a) => ((a.carryOn ?? null) === (info.carryOn ?? null) ? 0 : 1);
  const keys = (a) => [a.otherBase ? 1 : 0, a.set ? 1 : 0, Math.round(gap(a) * 1000), bar(a), a.exactFile ? 0 : 1, -(a.calls || 0), a.run.length, -(a.date || 0)];
  return [...rows].sort((a, b) => { const x = keys(a), y = keys(b), i = x.findIndex((v, n) => v !== y[n]); return i < 0 ? 0 : x[i] - y[i]; });
}
const KINDS = ['quiz', 'wide', 'ours', 'yap', 'live'];
function resultsFor(file, info, card, all, tests, tables) {
  const { own, elsewhere } = runsOf(file, info, card, all);
  const names = new Set([fileOf(file), ...formerNames(file)].map((n) => n.toLowerCase()));
  const runs = new Map([...own.map((x) => [x.r.name, x]), ...elsewhere.map((x) => [x.r.name, x])]);
  const runOf = (a) => {
    if (a.kind === 'quiz' || a.kind === 'wide') return [...runs.keys()].find((r) => a.run === r || a.run.startsWith(r + '_'));
    return [...runs.keys()].find((r) => (a.adapter || '').toLowerCase() === r.toLowerCase() + '.gguf');
  };
  const rows = [];
  for (const a of tests) {
    const exactFile = names.has((a.adapter || '').toLowerCase());
    const run = runOf(a);
    if (!exactFile && !run) continue;
    const x = run && runs.get(run);
    const elsewhereBase = x && !own.includes(x) ? x.base : undefined;
    const listen = a.listen ?? (a.run === run && a.kind === 'quiz' && x.r.tests ? x.r.tests.listen_p : undefined);
    rows.push({ ...a, listen, exactFile, otherBase: elsewhereBase });
  }
  // what a run recorded on the card (write_card), for results whose outputs are not on this machine
  for (const [kind, a] of Object.entries(card.results || {})) if (!rows.some((r) => r.kind === kind && r.run === a.run)) rows.push({ ...a, kind });
  return Object.fromEntries(KINDS.map((kind) => {
    const list = ranked(rows.filter((r) => r.kind === kind), info).map((r) => (kind === 'yap' && !r.table && tables[r.run] ? { ...r, table: tables[r.run] } : r));
    const best = list[0] && !list[0].otherBase ? list[0] : null;
    return [kind, { best, others: best ? list.slice(1) : list }];
  }));
}

// file: every adapter discoverAdapters found. promptOf: the app's own answer (index.js promptFor, older files included).
async function catalog(files, { saved, promptOf }) {
  const all = recipes();
  const tests = [...await quizRuns(), ...await evalRuns(), ...await liveRuns()], tables = yapTables();
  return files.map((file) => {
    const info = adapterInfo(file), card = readCard(file), name = fileOf(file), st = fs.statSync(file);
    // the file's own date moves whenever it is stamped: the run's export is the date it was trained
    let trained = card.trained ? Date.parse(card.trained) : 0;
    if (!trained && info.recipe) { try { trained = fs.statSync(path.join(OUT, `${info.recipe}.gguf`)).mtimeMs; } catch {} }
    const base = info.base && BASES[info.base];
    return {
      path: file, file: name, folder: path.relative(ROOT, path.dirname(file)).replace(/\\/g, '/'), label: info.label || '', suggestedLabel: card.suggestedLabel || '',
      base: info.base, baseLabel: base ? base.label : '', baseOnDisk: !!base && fs.existsSync(path.join(ROOT, base.gguf)), arch: info.arch,
      prompt: promptOf(file), recipe: info.recipe || card.recipe || '', size: st.size, trained: trained || null, fileDate: st.mtimeMs,
      listen: info.listen, carryOn: info.carryOn, savedListen: (saved.thresholds || {})[name], savedCarryOn: (saved.carryOns || {})[name],
      notes: card.notes || '', stamped: info.stamped, addon: /-addon-/i.test(name), results: resultsFor(file, info, card, all, tests, tables),
    };
  });
}

// ---- changes ----
// The name has to keep the adapter findable: discoverAdapters skips "test" in a name, and takes only talkme- files from
// tune/out; -addon- hides it from the dropdown. An adapter with no talkme.base is known by its file name, so it stays.
function checkName(file, wanted, all) {
  const old = fileOf(file), stem = String(wanted || '').trim().replace(/\.gguf$/i, ''), name = `${stem}.gguf`;
  if (!adapterInfo(file).stamped) throw new Error(`${old} carries no talkme metadata, so the app knows it by this file name: it keeps it`);
  if (!/^talkme-[A-Za-z0-9._-]+$/.test(stem)) throw new Error('a name starts with talkme- and has only letters, digits, dots, dashes and underscores (Git keeps only talkme- files here)');
  if (/test/i.test(stem)) throw new Error('a name cannot contain "test": the app skips such files when it looks for adapters');
  if (/-addon-/i.test(stem) !== /-addon-/i.test(old)) throw new Error('"-addon-" in a name marks an adapter that only plays in a blend: it cannot be added or dropped by renaming');
  const self = name.toLowerCase() === old.toLowerCase();
  if (!self && [LORA, OUT].some((d) => fs.existsSync(path.join(d, name)))) throw new Error(`there is already an adapter called ${name}`);
  if (!self && fs.existsSync(cardPath(path.join(path.dirname(file), name)))) throw new Error(`${stem}.json is already there; move it first`);
  if (RENAMED[name]) throw new Error(`${name} is an old name of ${RENAMED[name]}: saved settings would move to that one`);
  const taken = all.find((x) => x.exports === name.toLowerCase() && x.exports !== old.toLowerCase());
  if (taken) throw new Error(`recipe ${taken.r.name} exports to ${name}: its next run would overwrite this adapter`);
  return name;
}
// The file and its card move, every saved setting follows (config.js rewriteAdapter), and the recipes that export it
// get the new name, so a re-export does not bring the old one back. Returns the recipe files changed.
function rename(file, wanted, settings) {
  const all = recipes(), name = checkName(file, wanted, all), old = fileOf(file), target = path.join(path.dirname(file), name);
  if (name === old) return { path: file, name, recipes: [] };
  const hadCard = fs.existsSync(cardPath(file));
  fs.renameSync(file, target);
  if (hadCard) fs.renameSync(cardPath(file), cardPath(target));
  writeCard(target, { formerNames: [...new Set([old, ...(readCard(target).formerNames || [])])] });
  settings.rewriteAdapter(old, name);
  const changed = all.filter((x) => x.exports === old.toLowerCase());
  for (const x of changed) setRecipeKey(x, 'export_as', name);
  return { path: target, name, recipes: changed.map((x) => x.file) };
}
// Empty goes back to the name in the file. The recipes that export it take the name too, for the next export's stamp.
function setLabel(file, label) {
  writeCard(file, { label: String(label || '').trim() });
  const shown = adapterInfo(file).label, changed = recipes().filter((x) => x.exports === fileOf(file).toLowerCase() && shown && x.r.label !== shown);
  for (const x of changed) setRecipeKey(x, 'label', shown);
  return { label: shown, recipes: changed.map((x) => x.file) };
}
const setNotes = (file, notes) => { writeCard(file, { notes: String(notes || '').trimEnd() }); };
// the recipes whose next run would bring a removed adapter back
const exportedBy = (file) => recipes().filter((x) => x.exports === fileOf(file).toLowerCase()).map((x) => x.r.name);

module.exports = { catalog, rename, setLabel, setNotes, exportedBy };
