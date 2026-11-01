// The voice anchor (loop.js VOICE_LEAD_MS) with a stand-in model and voice, no GPU, at real speed: the model writes
// 3 words a tick, the voice says a word in 400 ms (Orpheus's pace, 1.5 words a 0.6 s tick), the caller cuts in once.
// Checks that a tick waits while the voice is behind (held words or not: the model's chunks straddle sentence ends),
// that every tick keeps the model's own words, and that a cut-in gets a tick at once and stops it.
//   node scripts/check_anchor.js [--off]
const { Loop, VOICE_LEAD_MS } = require('../src/main/loop');
const { LISTEN, NOVOICE } = require('../src/main/llama');

const MS_PER_WORD = 400, CUT_AT = 9000; // late enough for a hold to run its course first
const leadAtStart = [], heldAtStart = []; // the voice's backlog, and words held for a sentence end, as each tick starts
const STORY = 'Once there was a dragon who lived in a quiet village and baked sweet buns for everyone every morning. He wore a little hat and a tiny apron. Everyone loved him a lot.'.split(' ');
let written = 0, stopped = false;
const llama = {
  async tick(history, heard) {
    leadAtStart.push(tts.backlogMs()); heldAtStart.push(tts.buffer.length);
    if (/wait/.test(heard)) stopped = true;
    if (stopped || written >= STORY.length) return { said: LISTEN, pListen: 0.95, ms: 1 };
    const w = STORY.slice(written, written + 3); written += w.length;
    return { said: w.join(' '), pListen: 0.1, ms: 1 };
  },
};
// the stand-in voice holds words for a sentence end (14 words; the first piece 6; 4 or a comma when about to run dry) as the real one does, and uses its backlog formula
const { TTS } = require('../src/main/tts');
const tts = {
  buffer: [], inflightWords: 0, inflight: {}, playEnd: 0, msPerWord: MS_PER_WORD, cancelled: 0,
  say(text) {
    this.buffer.push(...text.split(' '));
    const first = this.playEnd === 0, dry = this.playEnd - Date.now() < 700;
    if ((first || dry ? /[.!?,;:]$/ : /[.!?]$/).test(this.buffer[this.buffer.length - 1]) || this.buffer.length >= (first ? 6 : dry ? 4 : 14)) return this.flush();
    return false;
  },
  flush() {
    if (!this.buffer.length) return false;
    this.playEnd = Math.max(Date.now(), this.playEnd) + this.buffer.length * MS_PER_WORD; this.buffer = [];
    return true;
  },
  cancel() { this.playEnd = 0; this.buffer = []; this.cancelled++; },
  backlogMs: TTS.prototype.backlogMs,
};
const injected = [];
let injectedAt = 0;
const stt = { hasWords: () => injected.length > 0, drain: () => injected.splice(0).join(' ') || NOVOICE };
const t0 = Date.now(), ticks = [];
const loop = new Loop({ llama, stt, tts, log: () => {}, emit: (ev, p) => {
  if (ev === 'tick') ticks.push({ ...p, at: Date.now() - t0, lead: leadAtStart[ticks.length], held: heldAtStart[ticks.length] });
} });
loop.start({ tickMs: 600, listenThreshold: 0.6, ttsFlushWords: 6, ttsSpeed: 1, llamaCtx: 32768, anchorToVoice: !process.argv.includes('--off') });
setTimeout(() => { injected.push('wait'); injectedAt = Date.now() - t0; }, CUT_AT);
setTimeout(() => {
  loop.stop();
  for (const t of ticks.slice(0, 18)) console.log(`${String(t.at).padStart(6)} ms  ${t.heard.padEnd(10)} ${t.said.padEnd(22)} ${t.action.padEnd(6)} voice behind ${(t.lead / 1000).toFixed(1)} s${t.held ? ` (${t.held} held)` : ''}${t.waitedMs > 20 ? `, waited ${t.waitedMs} ms` : ''}`);
  const speak = ticks.filter((t) => t.action === 'speak');
  const cut = ticks.find((t) => t.heard.includes('wait'));
  const fail = [];
  const slow = ticks.filter((t) => t.waitedMs >= 5500);
  if (slow.length) fail.push(`${slow.length} ticks held to the 6 s limit (a hold waiting on its own held words)`);
  // a quiet tick starts only once the voice is within the lead (a tick the caller started runs whatever the lead)
  const quiet = speak.filter((t) => t.heard === NOVOICE);
  const maxLead = Math.max(...quiet.map((t) => t.lead));
  const over = quiet.filter((t) => t.lead > VOICE_LEAD_MS + 100);
  if (over.length) fail.push(`a tick started with the voice ${(over[0].lead / 1000).toFixed(1)} s behind`);
  if (maxLead > VOICE_LEAD_MS + 14 * MS_PER_WORD) fail.push(`the voice fell ${(maxLead / 1000).toFixed(1)} s behind`);
  if (speak.some((t) => t.said.split(' ').length !== 3)) fail.push('a tick lost its words');
  if (!cut || cut.at - injectedAt > 150) fail.push(`the cut-in waited ${cut ? cut.at - injectedAt : '?'} ms for a tick`);
  if (!cut || cut.action !== 'stop' || tts.cancelled < 1) fail.push('the cut-in did not stop it');
  console.log(`\nvoice at most ${(maxLead / 1000).toFixed(1)} s behind as a quiet tick starts; cut-in answered ${cut ? cut.at - injectedAt : '?'} ms after it was heard`);
  console.log(fail.length ? 'FAIL: ' + fail.join('; ') : 'ok');
  process.exit(fail.length ? 1 : 0);
}, CUT_AT + 2500);
