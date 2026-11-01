// Benchmarks streaming recognizers on the scenario lines spoken by the test voice (Amy), in simulated time:
// accuracy (word error rate), how long after the speech starts the first word appears, how long after the speech ends
// the last word appears (what decides when the LLM can answer), and compute cost (real-time factor).
//   node scripts/stt_bench.js
const fs = require('fs');
const path = require('path');
const sherpa = require('sherpa-onnx-node');

const ROOT = path.resolve(__dirname, '..');
const SR = 16000, FRAME = 1600;
const A = (p) => path.join(ROOT, 'models', 'asr', p);
const MODELS = {
  'zipformer-2023-06-26': { dir: A('sherpa-onnx-streaming-zipformer-en-2023-06-26'), enc: 'encoder-epoch-99-avg-1-chunk-16-left-128.onnx', dec: 'decoder-epoch-99-avg-1-chunk-16-left-128.onnx', join: 'joiner-epoch-99-avg-1-chunk-16-left-128.onnx' },
  'kroko-2025-08-06': { dir: A('sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06'), enc: 'encoder.onnx', dec: 'decoder.onnx', join: 'joiner.onnx' },
  'nemotron-160ms': { dir: A('sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25'), enc: 'encoder.int8.onnx', dec: 'decoder.int8.onnx', join: 'joiner.int8.onnx', featDim: 128 },
  'nemotron-560ms': { dir: A('sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25'), enc: 'encoder.int8.onnx', dec: 'decoder.int8.onnx', join: 'joiner.int8.onnx', featDim: 128 },
};

const norm = (s) => s.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
function wer(ref, hyp) {
  const r = norm(ref), h = norm(hyp);
  const d = Array.from({ length: r.length + 1 }, (_, i) => [i, ...Array(h.length).fill(0)]);
  for (let j = 1; j <= h.length; j++) d[0][j] = j;
  for (let i = 1; i <= r.length; i++) for (let j = 1; j <= h.length; j++) d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] === h[j - 1] ? 0 : 1));
  return { errs: d[r.length][h.length], n: r.length };
}

function makeRecognizer(m, threads) {
  return new sherpa.OnlineRecognizer({
    featConfig: { sampleRate: SR, featureDim: m.featDim || 80 },
    modelConfig: { transducer: { encoder: path.join(m.dir, m.enc), decoder: path.join(m.dir, m.dec), joiner: path.join(m.dir, m.join) }, tokens: path.join(m.dir, 'tokens.txt'), numThreads: threads, provider: 'cpu', debug: 0 },
    decodingMethod: 'greedy_search', enableEndpoint: false,
  });
}

(async () => {
  const sc = JSON.parse(fs.readFileSync(path.join(ROOT, 'tests', 'scenarios.json'), 'utf8'));
  const lines = sc.flatMap((s) => s.turns.map((t) => t.say));
  const amy = path.join(ROOT, 'models/tts/vits-piper-en_US-amy-medium');
  const tts = new sherpa.OfflineTts({ model: { vits: { model: path.join(amy, 'en_US-amy-medium.onnx'), tokens: path.join(amy, 'tokens.txt'), dataDir: path.join(amy, 'espeak-ng-data') }, numThreads: 2, provider: 'cpu', debug: 0 }, maxNumSentences: 1 });
  const clips = lines.map((text) => {
    const a = tts.generate({ text, sid: 0, speed: 1.0, enableExternalBuffer: false });
    const s = new sherpa.LinearResampler(a.sampleRate, SR).resample(Float32Array.from(a.samples));
    let on = 0, off = s.length - 1;
    while (on < s.length && Math.abs(s[on]) < 0.02) on++;
    while (off > 0 && Math.abs(s[off]) < 0.02) off--;
    const pad = new Float32Array(SR / 2 + s.length + SR * 2); pad.set(s, SR / 2);
    return { text, samples: pad, onset: (SR / 2 + on) / SR, offset: (SR / 2 + off) / SR };
  });
  const threads = Number(process.argv[2] || 2);
  console.log(`${lines.length} lines, ${threads} threads\n`);
  console.log('model'.padEnd(22), 'WER'.padStart(6), 'first word'.padStart(11), 'last word'.padStart(10), 'RTF'.padStart(6), ' sample');
  for (const [name, m] of Object.entries(MODELS).filter(([k]) => !process.env.ONLY || k.includes(process.env.ONLY))) {
    if (!fs.existsSync(m.dir)) continue;
    let rec;
    try { rec = makeRecognizer(m, threads); } catch (e) { console.log(name.padEnd(22), 'failed to load:', e.message); continue; }
    let errs = 0, n = 0, firstLat = [], lastLat = [], compute = 0, audio = 0; let sample = '';
    for (const c of clips) {
      const st = rec.createStream();
      let words = 0, tFirst = null, tLast = null, text = '';
      for (let i = 0; i < c.samples.length; i += FRAME) {
        const t0 = process.hrtime.bigint();
        st.acceptWaveform({ sampleRate: SR, samples: c.samples.subarray(i, i + FRAME) });
        while (rec.isReady(st)) rec.decode(st);
        text = rec.getResult(st).text.trim();
        compute += Number(process.hrtime.bigint() - t0) / 1e9;
        const tNow = (i + FRAME) / SR;
        const w = norm(text).length;
        if (w > 0 && tFirst === null) tFirst = tNow;
        if (w > words) { words = w; tLast = tNow; }
      }
      audio += c.samples.length / SR;
      const e = wer(c.text, text); errs += e.errs; n += e.n;
      if (tFirst !== null) firstLat.push(tFirst - c.onset);
      if (tLast !== null) lastLat.push(tLast - c.offset);
      if (/ammonia/.test(c.text)) sample = text;
    }
    const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
    console.log(name.padEnd(22), (errs / n * 100).toFixed(1).padStart(5) + '%', (med(firstLat).toFixed(2) + ' s').padStart(11), (med(lastLat).toFixed(2) + ' s').padStart(10), (compute / audio).toFixed(3).padStart(6), ' ', JSON.stringify(sample.slice(0, 110)));
  }
})();
