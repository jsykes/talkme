// Summarises live benchmark runs (scripts/live_bench.sh) side by side: pass rate per kind of check, per caller,
// median reply and barge-in stop times. Writes tests/out/live_bench.md.
//   node scripts/live_bench_report.js run2 run5
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, '..', 'tests', 'out');
const names = process.argv.slice(2);
const KINDS = [['reply within', 'Replies in time'], ['mentions', 'Says the right thing'], ['no answer during the pause', 'Waits through pauses'],
  ['stops talking', 'Stops when cut in on'], ['keeps talking through', 'Keeps going through back-channels'], ['speaks up within', 'Speaks up after a silence'], ['back-channels while', 'Back-channels while you talk'], ['keeps talking at least', 'Keeps a story going'],
  ['judged:', 'Judged right by the base model'], ['says at most', 'Answers, then stops']];
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };
const pct = (p, n) => (n ? `${Math.round(100 * p / n)}% (${p}/${n})` : '-');
const rows = {}; const callers = new Set(); const perScenario = {};
for (const name of names) {
  const files = fs.readdirSync(OUT).filter((f) => f.startsWith(`report-bench-${name}-`) && f.endsWith('.json'));
  const checks = []; const byCaller = {};
  for (const f of files) {
    const caller = f.slice(`report-bench-${name}-`.length, -5); callers.add(caller);
    const rep = JSON.parse(fs.readFileSync(path.join(OUT, f), 'utf8'));
    for (const r of rep.results) {
      for (const c of r.checks) { checks.push({ ...c, caller, scenario: r.scenario }); }
      const k = r.scenario; perScenario[k] = perScenario[k] || {}; perScenario[k][name] = perScenario[k][name] || [0, 0];
      perScenario[k][name][0] += r.passed; perScenario[k][name][1] += r.total;
      byCaller[caller] = byCaller[caller] || [0, 0]; byCaller[caller][0] += r.passed; byCaller[caller][1] += r.total;
    }
  }
  const num = (re, c) => { const m = c.detail.match(re); return m ? Number(m[1]) : null; };
  const reply = checks.filter((c) => c.check.startsWith('reply within')).map((c) => num(/first audio (-?[\d.]+) s/, c)).filter((x) => x !== null);
  const stop = checks.filter((c) => c.check.startsWith('stops talking')).map((c) => num(/cut ([\d.]+) s/, c)).filter((x) => x !== null);
  const unheard = checks.filter((c) => c.check.startsWith('keeps talking') && c.heardByApp === false).length;
  // every back-channel said while a caller talked, to see whether it is always the same sound
  const bcWords = checks.filter((c) => /back-channels while/.test(c.check)).flatMap((c) => [...c.detail.matchAll(/"([^"]+)"/g)].map((m) => m[1].toLowerCase().replace(/[.!?,]/g, '').trim()));
  rows[name] = { checks, byCaller, reply, stop, unheard, bcWords, files: files.length };
}
const L = [];
L.push(`| | ${names.join(' | ')} |`, `|---|${names.map(() => '---').join('|')}|`);
L.push(`| **All checks** | ${names.map((n) => pct(rows[n].checks.filter((c) => c.pass).length, rows[n].checks.length)).join(' | ')} |`);
for (const [k, label] of KINDS) L.push(`| ${label} | ${names.map((n) => { const c = rows[n].checks.filter((x) => x.check.startsWith(k) || x.check.replace(/^\d+\+ /, '').startsWith(k)); return pct(c.filter((x) => x.pass).length, c.length); }).join(' | ')} |`);
L.push(`| Different back-channels (of all said) | ${names.map((n) => `${new Set(rows[n].bcWords).size} (${rows[n].bcWords.length})${rows[n].bcWords.length ? ': ' + [...new Set(rows[n].bcWords)].slice(0, 6).join(', ') : ''}`).join(' | ')} |`);
L.push(`| Median reply start after the caller stops | ${names.map((n) => `${med(rows[n].reply).toFixed(2)} s`).join(' | ')} |`);
L.push(`| Median stop after a cut-in starts | ${names.map((n) => (rows[n].stop.length ? `${med(rows[n].stop).toFixed(2)} s` : '-')).join(' | ')} |`);
for (const c of [...callers].sort()) L.push(`| Caller ${c} | ${names.map((n) => (rows[n].byCaller[c] ? pct(...rows[n].byCaller[c]) : '-')).join(' | ')} |`);
L.push('', '| Scenario | ' + names.join(' | ') + ' |', `|---|${names.map(() => '---').join('|')}|`);
for (const [s, v] of Object.entries(perScenario)) L.push(`| ${s} | ${names.map((n) => (v[n] ? `${v[n][0]}/${v[n][1]}` : '-')).join(' | ')} |`);
L.push('', ...names.map((n) => `${n}: ${rows[n].files} caller passes; back-channels the recogniser never passed on: ${rows[n].unheard}`));
fs.writeFileSync(path.join(OUT, 'live_bench.md'), L.join('\n') + '\n');
console.log(L.join('\n'));
