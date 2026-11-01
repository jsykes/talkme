// Reviews demo takes (tests/out/demo/<tag>/<scene>.json from scripts/demo_record.js): for each, what the scene is meant
// to show, read from the ticks and the caller's line times, and the transcript. Prints one table per scene.
//   node scripts/demo_review.js [tag ...]
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const DEMO = path.join(ROOT, 'tests', 'out', 'demo');
const SCEN = JSON.parse(fs.readFileSync(path.join(ROOT, 'tests', 'demo_scenarios.json'), 'utf8')).scenarios;
const tags = process.argv.slice(2).length ? process.argv.slice(2) : fs.readdirSync(DEMO).filter((d) => fs.existsSync(path.join(DEMO, d, `${SCEN[0].name}.json`)));

const said = (t) => t.said && t.said !== '<listen>' ? t.said : '';
const sec = (ms) => (ms / 1000).toFixed(1) + ' s';
// ticks that spoke, with when they were decided (at = ms into the recording)
const spoken = (j) => j.ticks.filter((t) => said(t)).map((t) => ({ at: t.at, text: said(t) }));
const firstAfter = (sp, ms, maxMs = 1e9) => sp.find((s) => s.at >= ms && s.at < ms + maxMs);
const between = (sp, a, b) => sp.filter((s) => s.at >= a && s.at < b);
const has = (sp, re) => sp.some((s) => re.test(s.text));
const quote = (s) => (s ? `"${s.text.length > 50 ? s.text.slice(0, 48) + '…' : s.text}"` : 'nothing');
const stopAfter = (j, ms) => { const t = j.ticks.find((x) => x.at >= ms && x.action === 'stop'); return t ? t.at - ms : null; };

const checks = {
  backchannel(j, sp, c) {
    const end = c[2].end, during = between(sp, c[0].at, end).filter((s) => s.text.split(' ').length <= 3);
    const long = between(sp, c[0].at, end).filter((s) => s.text.split(' ').length > 6);
    const after = firstAfter(sp, end, 6000);
    return [`${during.length} short ones while the caller talked ${during.length ? '(' + during.map((s) => s.text).join(' / ') + ')' : ''}`,
      long.length ? `took over mid-story: ${quote(long[0])}` : 'did not take over',
      after ? `reacted ${sec(after.at - end)} after: ${quote(after)}` : 'no reaction after the story'].join(' | ');
  },
  'cut-in'(j, sp, c) {
    const stop = stopAfter(j, c[1].at);
    const answer = firstAfter(sp, c[1].at, 8000);
    const resumed = c[2] ? firstAfter(sp, c[2].end, 6000) : null;
    return [stop !== null ? `stopped ${sec(stop)} after "Wait"` : 'did not stop', answer ? `answered: ${quote(answer)}` : 'no answer',
      resumed ? `went on: ${quote(resumed)}` : 'did not go on'].join(' | ');
  },
  'thinking-pause'(j, sp, c) {
    const inPause = between(sp, c[0].end, c[1].at);
    const ans = firstAfter(sp, c[1].end - 500, 7000);
    return [inPause.length ? `jumped into the pause: ${quote(inPause[0])}` : 'waited through the pause',
      ans ? `answered ${sec(ans.at - c[1].end)} after: ${quote(ans)}${/everest/i.test(sp.map((s) => s.text).join(' ')) ? ' (Everest ✓)' : ''}` : 'no answer'].join(' | ');
  },
  warning(j, sp, c) {
    const warn = sp.filter((s) => s.at >= c[1].at && /don't|do not|never|toxic|dangerous|gas|stop|no\b|bad idea|not a good/i.test(s.text));
    const first = firstAfter(sp, c[1].at, 8000);
    return [warn.length ? `warned ${sec(warn[0].at - c[1].at)} after the plan: ${quote(warn[0])}` : `no warning (first said: ${quote(first)})`].join(' | ');
  },
  quiet(j, sp, c) {
    const reply = firstAfter(sp, c[0].end, 6000);
    const later = sp.filter((s) => s.at >= c[0].end + 4000);
    return [reply ? `replied: ${quote(reply)}` : 'no reply', later.length ? `spoke up ${sec(later[0].at - c[0].end)} into the silence: ${quote(later[0])}` : 'stayed silent'].join(' | ');
  },
  quick(j, sp, c) {
    const a1 = firstAfter(sp, c[0].end - 500, 6000), a2 = c[1] ? firstAfter(sp, c[1].end - 500, 6000) : null;
    return [a1 ? `${sec(a1.at - c[0].end)}: ${quote(a1)}${/144|forty[- ]?four/i.test(a1.text) ? ' ✓' : ' ✗'}` : 'no answer to 12×12',
      a2 ? `${sec(a2.at - c[1].end)}: ${quote(a2)}${/tokyo/i.test(a2.text) ? ' ✓' : ' ✗'}` : 'no answer to Japan'].join(' | ');
  },
};

for (const s of SCEN) {
  console.log(`\n## ${s.name}: ${s.title}`);
  for (const tag of tags) {
    const f = path.join(DEMO, tag, `${s.name}.json`);
    if (!fs.existsSync(f)) { console.log(`  ${tag}: no take`); continue; }
    const j = JSON.parse(fs.readFileSync(f, 'utf8'));
    const sp = spoken(j);
    const ms = j.ticks.map((t) => t.llmMs).filter(Number.isFinite).sort((a, b) => a - b);
    console.log(`  ${tag.padEnd(10)} ${checks[s.name] ? checks[s.name](j, sp, j.callerLines || j.caller) : ''}`);
    console.log(`  ${''.padEnd(10)} ticks ${j.ticks.length}, llm median ${ms[ms.length >> 1]} ms | ${j.transcript.join(' | ').replace(/\s+/g, ' ').slice(0, 260)}`);
  }
}
