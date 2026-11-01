// Kokoro, the app's voice today (speaker 7, CPU), through the app's own TTS worker: the baseline for the bake-off.
// Same lines and modes as the Python runners; tags are dropped (Kokoro has none).   node tts_bench/run_kokoro.js
const fs = require('fs');
const path = require('path');
const { TTS } = require('../src/main/tts');
const { voiceConfig } = require('../src/main/voices');

const ROOT = __dirname;
const OUT = path.join(ROOT, 'out');
const DIR = path.join(OUT, 'kokoro');
fs.mkdirSync(DIR, { recursive: true });
const LINES = JSON.parse(fs.readFileSync(path.join(ROOT, 'lines.json'), 'utf8'));
const plain = (l) => l.text.replace(/\{\w+\}/g, '').replace(/\s+/g, ' ').trim();
const chunks = (t) => { const w = t.split(' '); const out = []; for (let i = 0; i < w.length; i += 3) out.push(w.slice(i, i + 3).join(' ')); return out; };

function writeWav(file, samples, sr) {
  const buf = Buffer.alloc(44 + samples.length * 2);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + samples.length * 2, 4); buf.write('WAVEfmt ', 8);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22); buf.writeUInt32LE(sr, 24);
  buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34); buf.write('data', 36); buf.writeUInt32LE(samples.length * 2, 40);
  for (let i = 0; i < samples.length; i++) buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), 44 + i * 2);
  fs.writeFileSync(file, buf);
}

(async () => {
  const tts = new TTS(() => {});
  const t0 = Date.now();
  await tts.start(voiceConfig('kokoro:7'));
  const sr = tts.sampleRate;
  // pieces: the texts to speak, one utterance each; resolves when all have come back
  const speak = (pieces) => new Promise((resolve) => {
    const got = []; const start = Date.now(); let first = null;
    tts.onAudio = (a) => {
      if (first === null) first = (Date.now() - start) / 1000;
      got[a.seq - base] = a.samples;
      if (got.filter(Boolean).length === pieces.length) resolve({ first, total: (Date.now() - start) / 1000, audio: Float32Array.from(got.flatMap((g) => Array.from(g))) });
    };
    const base = tts.seq;
    for (const p of pieces) { tts.say(p, { flushWords: 99, speed: 1.0 }); tts.flush({ speed: 1.0 }); }
  });
  await speak(['Warming up.']);
  const load = (Date.now() - t0) / 1000;
  const rows = [];
  for (const line of LINES) {
    for (const [mode, pieces] of [['whole', [plain(line)]], ['chunked', chunks(plain(line))]]) {
      const r = await speak(pieces);
      writeWav(path.join(DIR, `${line.id}.${mode}.wav`), r.audio, sr);
      const secs = r.audio.length / sr;
      rows.push({ id: line.id, mode, text: plain(line), first_audio_s: +r.first.toFixed(3), total_s: +r.total.toFixed(3), audio_s: +secs.toFixed(2), realtime_x: +(secs / r.total).toFixed(2) });
      console.log(`[kokoro] ${line.id.padEnd(12)} ${mode.padEnd(7)} first ${r.first.toFixed(2)} s, ${secs.toFixed(1)} s of audio in ${r.total.toFixed(2)} s`);
    }
  }
  fs.writeFileSync(path.join(OUT, 'kokoro.json'), JSON.stringify({ model: 'kokoro', note: "the app's voice today: speaker 7, CPU, no tags", load_s: +load.toFixed(1), gpu_mb: 0, sample_rate: sr, rows }, null, 1));
  await tts.stop();
})();
