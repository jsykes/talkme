// Checks that each test caller voice is understood by the app's recognizer: synthesises every user line in
// tests/scenarios.json with the voice and reports the word error rate. CPU only, no LLM needed.
//   node scripts/caller_voice_check.js piper:vits-piper-en_US-amy-medium kokoro:16 kokoro:21 kokoro:26
const fs = require('fs');
const path = require('path');
const sherpa = require('sherpa-onnx-node');
const { voiceConfig } = require('../src/main/voices');
const { DEFAULTS } = require('../src/main/config');
const ROOT = path.resolve(__dirname, '..'); const SR = 16000;
const LEAD = Number(process.env.LEAD || 0.5); // seconds of silence before each line, as on a live mic
const TAIL = Number(process.env.TAIL || 2.5); // seconds of silence after it (the app keeps feeding silence; short words need ~2 s to come out)
const norm = (s) => s.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
function wer(ref, hyp) {
  const r = norm(ref), h = norm(hyp);
  const d = Array.from({ length: r.length + 1 }, (_, i) => [i, ...Array(h.length).fill(0)]);
  for (let j = 1; j <= h.length; j++) d[0][j] = j;
  for (let i = 1; i <= r.length; i++) for (let j = 1; j <= h.length; j++) d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] === h[j - 1] ? 0 : 1));
  return { errs: d[r.length][h.length], n: r.length };
}
function resample(x, from) {
  if (from === SR) return x;
  const n = Math.floor(x.length * SR / from), y = new Float32Array(n), k = from / SR;
  for (let i = 0; i < n; i++) { const p = i * k, a = Math.floor(p), f = p - a; y[i] = (x[a] || 0) * (1 - f) + (x[a + 1] || 0) * f; }
  return y;
}
const dir = DEFAULTS.stt.dir;
const pick = (p) => path.join(dir, fs.readdirSync(dir).find((f) => f.startsWith(p) && f.endsWith('.onnx')));
const rec = new sherpa.OnlineRecognizer({ featConfig: { sampleRate: SR, featureDim: 128 }, modelConfig: { transducer: { encoder: pick('encoder'), decoder: pick('decoder'), joiner: pick('joiner') }, tokens: path.join(dir, 'tokens.txt'), numThreads: 4, provider: 'cpu', debug: 0 }, decodingMethod: 'greedy_search' });
const lines = [...new Set(JSON.parse(fs.readFileSync(path.join(ROOT, 'tests/scenarios.json'), 'utf8')).flatMap((s) => s.turns.map((t) => t.say)))];
for (const id of process.argv.slice(2)) {
  const c = voiceConfig(id);
  const tts = c.engine === 'kokoro'
    ? new sherpa.OfflineTts({ model: { kokoro: { model: path.join(c.dir, 'model.onnx'), voices: path.join(c.dir, 'voices.bin'), tokens: path.join(c.dir, 'tokens.txt'), dataDir: path.join(c.dir, 'espeak-ng-data'), lexicon: path.join(c.dir, 'lexicon-us-en.txt'), dictDir: path.join(c.dir, 'dict') }, numThreads: 4, provider: 'cpu', debug: 0 }, maxNumSentences: 1 })
    : new sherpa.OfflineTts({ model: { vits: { model: path.join(c.dir, c.model), tokens: path.join(c.dir, 'tokens.txt'), dataDir: path.join(c.dir, 'espeak-ng-data') }, numThreads: 2, provider: 'cpu', debug: 0 }, maxNumSentences: 1 });
  let errs = 0, n = 0; const bad = [];
  for (const line of lines) {
    const a = tts.generate({ text: line, sid: c.sid || 0, speed: 1.0, enableExternalBuffer: false });
    const s = resample(Float32Array.from(a.samples), a.sampleRate);
    const st = rec.createStream(); const lead = Math.round(LEAD * SR); const pad = new Float32Array(lead + s.length + Math.round(TAIL * SR)); pad.set(s, lead);
    st.acceptWaveform({ sampleRate: SR, samples: pad }); while (rec.isReady(st)) rec.decode(st);
    const heard = rec.getResult(st).text.trim(); const w = wer(line, heard); errs += w.errs; n += w.n;
    if (w.errs) bad.push(`   "${line}" -> "${heard}"`);
  }
  console.log(`${id.padEnd(36)} WER ${(100 * errs / n).toFixed(1)}% over ${lines.length} lines`);
  for (const b of bad) console.log(b);
}
