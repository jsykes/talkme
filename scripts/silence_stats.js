// How long real people let a silence run before someone speaks again, from the transcribed otoSpeech calls
// (data/otospeech/asr). Used to set the partner's nudge timing from data instead of a guess.
//   node scripts/silence_stats.js [--min 2.5]
const fs = require('fs');
const path = require('path');
const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) => (a.startsWith('--') ? [a.slice(2), all[i + 1]] : [])).filter((x) => x.length));
const MIN = Number(args.min || 2.5);
const ASR = path.join(__dirname, '..', 'data', 'otospeech', 'asr');
const gaps = []; let same = 0, other = 0, minutes = 0;
for (const f of fs.readdirSync(ASR).filter((x) => x.endsWith('.json'))) {
  let j; try { j = JSON.parse(fs.readFileSync(path.join(ASR, f), 'utf8')); } catch { continue; }
  const red = j.redacted || [];
  const words = j.channels.flatMap((c, ch) => c.words.map((w) => ({ ...w, ch }))).sort((a, b) => a.start - b.start);
  minutes += j.duration / 60;
  let end = -1, lastCh = -1;
  for (const w of words) {
    if (end >= 0 && w.start - end >= MIN && !red.some((r) => r.end_sec > end && r.start_sec < w.start)) {
      gaps.push(w.start - end);
      if (w.ch === lastCh) same++; else other++;
    }
    if (w.end > end) { end = w.end; lastCh = w.ch; }
  }
}
gaps.sort((a, b) => a - b);
const q = (p) => gaps[Math.min(gaps.length - 1, Math.floor(p * gaps.length))].toFixed(1);
console.log(`${gaps.length} silences of ${MIN}+ s in ${(minutes / 60).toFixed(0)} h (${(gaps.length / minutes).toFixed(2)} a minute)`);
console.log(`length: p10 ${q(0.1)} s, p25 ${q(0.25)} s, median ${q(0.5)} s, p75 ${q(0.75)} s, p90 ${q(0.9)} s`);
console.log(`broken by the person who spoke last: ${(100 * same / gaps.length).toFixed(0)}%, by the other person: ${(100 * other / gaps.length).toFixed(0)}%`);
