// Unit check of the final-punctuation rule in src/main/stt.js.
//   node scripts/check_final_punct.js
const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.join(__dirname, '..', 'src', 'main', 'stt.js'), 'utf8');
const code = src.slice(src.indexOf('const FINAL_SILENCE_MS'), src.indexOf('class STT'));
const finalMark = new Function('TRAILING', `${code}; return finalMark;`)(/[.,!?;:]+$/);

const CASES = [
  ["What's the capital of France", '?'],
  ['And of Japan', '.'],
  ["So here's the thing, I was thinking about maybe learning to play the piano this year, but", ''],
  ["I'm not so sure if I'm too old to start", '.'],
  ['Do you think I can just ignore that, or should I get it checked', '?'],
  ['So last weekend I finally took my old car up the coast road, and about an hour in the engine light started', '.'],
  ['I was going to', ''],
  ['Okay, thanks for the warning', '.'],
  ['Can you explain how a rainbow forms', '?'],
  ['So, how are you', '?'],
  ['The capital of France?', ''],
  ['It is cold. Should I bring a coat', '?'],
  ['um', ''],
  // a question asking for a preposition's object ends on it; a yes/no question or an infinitive there is open (2026-10-28)
  ['What are you up to', '?'],
  ['What are these new patterns about', '?'],
  ['Where are you from', '?'],
  ['Do you want to', ''],
  ['What do you want to', ''],
  ['and I went biking before that', ''],
];
let bad = 0;
for (const [text, want] of CASES) {
  const got = finalMark(text.split(' '));
  if (got !== want) bad++;
  console.log(`${got === want ? 'ok  ' : 'DIFF'} ${JSON.stringify(text).padEnd(110)} -> ${JSON.stringify(got)}${got === want ? '' : ` (expected ${JSON.stringify(want)})`}`);
}
console.log(bad ? `${bad} differ` : 'all as expected');
