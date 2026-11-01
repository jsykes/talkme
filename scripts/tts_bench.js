// Compares voices: time to synthesise the short phrases the app sends, and writes a comparison clip per voice.
//   node scripts/tts_bench.js
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const sherpa = require('sherpa-onnx-node');

const ROOT = path.resolve(__dirname, '..');
const T = (p) => path.join(ROOT, 'models', 'tts', p);
const OUT = path.join(ROOT, 'tests', 'listen', 'voices');
fs.mkdirSync(OUT, { recursive: true });
const PHRASES = ['Mm-hm.', 'Stop! Do not mix those together!', 'That creates toxic gas.', 'Oh, absolutely not. It is never too late to pick up a new skill.', 'Yeah, definitely get it checked.'];
const KOKORO = T(process.env.KOKORO || 'kokoro-multi-lang-v1_0');
// Kokoro v1.0 speaker ids (sherpa-onnx table): 3 af_heart, 2 af_bella, 9 af_sarah, 16 am_michael, 15 am_liam, 26 bm_george
const VOICES = [
  { name: 'piper-lessac (today)', engine: 'piper', dir: 'vits-piper-en_US-lessac-medium', onnx: 'en_US-lessac-medium.onnx', sid: 0 },
  { name: 'kokoro-af_heart', engine: 'kokoro', sid: 3 },
  { name: 'kokoro-af_bella', engine: 'kokoro', sid: 2 },
  { name: 'kokoro-am_michael', engine: 'kokoro', sid: 16 },
  { name: 'kokoro-bm_george', engine: 'kokoro', sid: 26 },
];
const engines = {};
function engine(v, threads) {
  const key = v.engine + (v.dir || '') + threads;
  if (engines[key]) return engines[key];
  const model = v.engine === 'piper'
    ? { vits: { model: T(path.join(v.dir, v.onnx)), tokens: T(path.join(v.dir, 'tokens.txt')), dataDir: T(path.join(v.dir, 'espeak-ng-data')) } }
    : { kokoro: { model: path.join(KOKORO, fs.readdirSync(KOKORO).find((f) => /^model.*.onnx$/.test(f))), voices: path.join(KOKORO, 'voices.bin'), tokens: path.join(KOKORO, 'tokens.txt'),
        dataDir: path.join(KOKORO, 'espeak-ng-data'), lexicon: path.join(KOKORO, 'lexicon-us-en.txt'), dictDir: path.join(KOKORO, 'dict') } };
  return (engines[key] = new sherpa.OfflineTts({ model: { ...model, numThreads: threads, provider: 'cpu', debug: 0 }, maxNumSentences: 1 }));
}
const threads = Number(process.argv[2] || 4);
console.log(`${threads} threads; median synthesis time per phrase and speech length\n`);
for (const v of VOICES) {
  const tts = engine(v, threads);
  const parts = []; const times = []; let sr = 24000;
  tts.generate({ text: 'Warm up.', sid: v.sid, speed: 1.0, enableExternalBuffer: false });
  for (const p of PHRASES) {
    const t0 = process.hrtime.bigint();
    const a = tts.generate({ text: p, sid: v.sid, speed: 1.0, enableExternalBuffer: false });
    times.push(Number(process.hrtime.bigint() - t0) / 1e6);
    sr = a.sampleRate; parts.push(Float32Array.from(a.samples), new Float32Array(Math.round(0.6 * sr)));
  }
  const n = parts.reduce((s, x) => s + x.length, 0); const all = new Float32Array(n); let o = 0; for (const x of parts) { all.set(x, o); o += x.length; }
  const file = path.join(OUT, v.name.replace(/[^a-z0-9_-]+/gi, '-') + '.wav');
  sherpa.writeWave(file, { samples: all, sampleRate: sr });
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', file, '-ac', '1', '-b:a', '64k', file.replace(/\.wav$/, '.mp3')]);
  const s = [...times].sort((a, b) => a - b);
  console.log(`${v.name.padEnd(22)} median ${s[2].toFixed(0).padStart(4)} ms, slowest ${s[4].toFixed(0).padStart(4)} ms  (${(n / sr).toFixed(1)} s of audio)`);
}
