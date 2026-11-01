// Feeds a voice the way a call does: three words every 600 ms through the app's own TTS class (its piece rules, the
// voice's server for GPU voices), plays the pieces back to back as the renderer would, and reports the silence
// between pieces (trimmed edges plus any wait for the next piece). Writes out/stream/<voice>.wav to listen to.
//   node tts_bench/stream_test.js kokoro:7 chatterbox:turbo orpheus:tara
const fs = require('fs');
const path = require('path');
const { TTS } = require('../src/main/tts');
const { voiceConfig } = require('../src/main/voices');
const { DEFAULTS } = require('../src/main/config');

const LINES = [
  'So anyway, the fox finally made it across the river, soaking wet but pretty proud of himself, and he just sat there on the bank grinning.',
  "Whoa, wait, don't mix those! Bleach and ammonia make a toxic gas, so open a window and use one at a time.",
  "Oh, I'm so sorry. That's really hard. How are you holding up?",
];
const OUT = path.join(__dirname, 'out', 'stream');
fs.mkdirSync(OUT, { recursive: true });

function writeWav(file, samples, sr) {
  const buf = Buffer.alloc(44 + samples.length * 2);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + samples.length * 2, 4); buf.write('WAVEfmt ', 8);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22); buf.writeUInt32LE(sr, 24);
  buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34); buf.write('data', 36); buf.writeUInt32LE(samples.length * 2, 40);
  for (let i = 0; i < samples.length; i++) buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), 44 + i * 2);
  fs.writeFileSync(file, buf);
}

async function one(id) {
  const tts = new TTS((l) => { if (/error|exited|streamed/i.test(l)) console.log('   ', l.slice(0, 150)); });
  const t0 = Date.now();
  // "<voice>+old": the pieces as before (fixed 8 words, no continuing tone), to compare
  const old = id.endsWith('+old');
  await tts.start({ ...voiceConfig(id.replace('+old', '')), ...(old ? { justInTime: false, contTone: false } : {}) });
  console.log(`${id}: ready in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  const sr = tts.sampleRate;
  const timeline = []; // { at: ms when it can start playing, samples }
  let start = 0;
  // streamed pieces come in several parts: each part is queued as it arrives, a piece is done at its final part
  let finals = 0;
  let pieceText = '', partNo = 0;
  tts.onAudio = (a) => {
    const fresh = !!a.text; // a piece's first part carries its text
    if (fresh) { pieceText = a.text; partNo = 0; }
    timeline.push({ arrived: Date.now() - start, samples: a.samples, first: fresh, text: pieceText, part: partNo++ });
  };
  tts.onPieceDone = () => { finals++; };
  const gaps = [], stalls = [], events = []; let firstSound = null; const track = [];
  let trackLen = 0; // samples written so far (where in the WAV each event lands)
  for (const line of LINES) {
    timeline.length = 0; finals = 0;
    start = Date.now();
    const words = line.split(' ');
    for (let i = 0; i < words.length; i += 3) {
      tts.say(words.slice(i, i + 3).join(' '), { flushWords: DEFAULTS.ttsFlushWords, speed: 1.0, first: i === 0 });
      await new Promise((r) => setTimeout(r, 600));
    }
    tts.flush({ speed: 1.0 });
    const pieces = (tts.seq - (tts._lastSeq || 0)); tts._lastSeq = tts.seq;
    while (finals < pieces) await new Promise((r) => setTimeout(r, 50));
    // playback: each piece starts when it has arrived and the previous one has finished
    let t = 0; const out = [];
    timeline.forEach((p, k) => {
      const begin = Math.max(t, p.arrived);
      // a seam is where a new piece starts (the parts of one streamed piece join seamlessly by design)
      if (k === 0) firstSound = p.arrived; else if (p.first) gaps.push(begin - t + leadMs(p.samples, sr) + tailMs(timeline[k - 1].samples, sr));
      else if (begin > t + 5) { // a streamed piece ran dry in the middle: a hitch, and a pop
        stalls.push(begin - t);
        const prev = timeline[k - 1].samples;
        events.push({ at_s: +((trackLen + out.reduce((n, a) => n + a.length, 0)) / sr).toFixed(3), stall_ms: Math.round(begin - t), piece: p.text, words: p.text.split(' ').length,
          part: p.part, into_piece_s: +(timeline.slice(0, k).filter((x) => x.text === p.text).reduce((n, x) => n + x.samples.length, 0) / sr).toFixed(2),
          last_sample: +Math.abs(prev[prev.length - 1]).toFixed(3), next_sample: +Math.abs(p.samples[0]).toFixed(3) });
      }
      if (begin > t) out.push(new Float32Array(Math.round((begin - t) / 1000 * sr)));
      out.push(p.samples); t = begin + p.samples.length / sr * 1000;
    });
    track.push(...out, new Float32Array(Math.round(0.8 * sr)));
    trackLen += out.reduce((n, a) => n + a.length, 0) + Math.round(0.8 * sr);
    console.log(`  "${line.slice(0, 40)}..." ${pieces} pieces, first sound ${(firstSound / 1000).toFixed(2)} s`);
  }
  const all = new Float32Array(track.reduce((n, a) => n + a.length, 0)); let o = 0; for (const a of track) { all.set(a, o); o += a.length; }
  writeWav(path.join(OUT, id.replace(':', '-').replace('+', '-') + '.wav'), all, sr);
  fs.writeFileSync(path.join(OUT, id.replace(':', '-').replace('+', '-') + '.stalls.json'), JSON.stringify(events, null, 1));
  gaps.sort((a, b) => a - b);
  console.log(`  stalls inside pieces: ${stalls.length}${stalls.length ? ` (longest ${(Math.max(...stalls) / 1000).toFixed(2)} s)` : ''}`);
  console.log(`  ${tts.seq} pieces in all; silence between pieces: median ${(gaps[Math.floor(gaps.length / 2)] / 1000).toFixed(2)} s, worst ${(gaps[gaps.length - 1] / 1000).toFixed(2)} s`);
  await tts.stop();
}
const edge = (s, sr, fromEnd) => { const thr = 0.01; const n = s.length; for (let i = 0; i < n; i++) { const v = s[fromEnd ? n - 1 - i : i]; if (Math.abs(v) > thr) return i / sr * 1000; } return n / sr * 1000; };
const leadMs = (s, sr) => edge(s, sr, false);
const tailMs = (s, sr) => edge(s, sr, true);

(async () => { for (const id of process.argv.slice(2)) await one(id); process.exit(0); })().catch((e) => { console.error(e); process.exit(1); });
