// Transcribes the bake-off's audio with the app's recogniser (Nemotron, CPU) and adds a word error rate per file to
// out/<model>.json: a voice the recogniser can't follow is one a listener will struggle with too, and a garbled or
// silent output shows up at once.   node tts_bench/hear.js [out dir]
const fs = require('fs');
const path = require('path');
const sherpa = require('sherpa-onnx-node');
const { DEFAULTS } = require('../src/main/config');

const OUT = path.resolve(process.argv[2] || path.join(__dirname, 'out'));
const SR = 16000;
const earDir = DEFAULTS.stt.dir;
const pick = (p) => { const all = fs.readdirSync(earDir).filter((f) => f.startsWith(p) && f.endsWith('.onnx')); return path.join(earDir, all.find((x) => !x.includes('int8')) || all[0]); };
const ear = new sherpa.OnlineRecognizer({ featConfig: { sampleRate: SR, featureDim: /nemotron/i.test(earDir) ? 128 : 80 }, modelConfig: { transducer: { encoder: pick('encoder'), decoder: pick('decoder'), joiner: pick('joiner') }, tokens: path.join(earDir, 'tokens.txt'), numThreads: 4, provider: 'cpu', debug: 0 }, decodingMethod: 'greedy_search' });
const norm = (s) => s.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim();
function wer(ref, hyp) {
  const r = norm(ref).split(' ').filter(Boolean), h = norm(hyp).split(' ').filter(Boolean);
  const d = Array.from({ length: r.length + 1 }, (_, i) => [i, ...Array(h.length).fill(0)]);
  for (let j = 1; j <= h.length; j++) d[0][j] = j;
  for (let i = 1; i <= r.length; i++) for (let j = 1; j <= h.length; j++) d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] === h[j - 1] ? 0 : 1));
  return r.length ? d[r.length][h.length] / r.length : 0;
}
for (const f of fs.readdirSync(OUT).filter((x) => x.endsWith('.json'))) {
  const res = JSON.parse(fs.readFileSync(path.join(OUT, f), 'utf8'));
  for (const row of res.rows) {
    const w = sherpa.readWave(path.join(OUT, res.model, `${row.id}.${row.mode}.wav`));
    const s16 = w.sampleRate === SR ? w.samples : new sherpa.LinearResampler(w.sampleRate, SR).resample(w.samples);
    const st = ear.createStream();
    const padded = new Float32Array(s16.length + SR); padded.set(s16);
    st.acceptWaveform({ sampleRate: SR, samples: padded });
    while (ear.isReady(st)) ear.decode(st);
    row.heard = ear.getResult(st).text.trim();
    row.wer = +wer(row.text, row.heard).toFixed(2);
    console.log(`${res.model.padEnd(17)} ${row.id.padEnd(12)} ${row.mode.padEnd(7)} wer ${row.wer.toFixed(2)}  "${row.heard}"`);
  }
  fs.writeFileSync(path.join(OUT, f), JSON.stringify(res, null, 1));
}
