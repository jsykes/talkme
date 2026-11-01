// Turns real two-person recordings (Open Yap 1K sample: one track per speaker, word timings for both) into held-out
// calls for tune/evaluate.py. One speaker plays the caller: their track goes through the app's own recogniser and
// fragment logic (src/main/stt.js) tick by tick in simulated time, so `heard` is exactly what the model would see live.
// The other speaker is the reference for WHEN to talk: their words fill `said`, and the tags mark where they took the
// turn (start), back-channelled (continuer), stayed quiet through the caller's pauses (listen_pause), and so on.
// Each conversation gives two perspectives (a calls b, b calls a) and a few 3-minute windows of each.
//   node scripts/yap_to_calls.js [--windows 0.25,0.6] [--seconds 180] [--out tune/data/heldout_yap.jsonl] [--limit N]
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const sherpa = require('sherpa-onnx-node');
const { STT } = require('../src/main/stt');
const { DEFAULTS } = require('../src/main/config');
const { buildCall } = require('./lib/turntags');

const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) => (a.startsWith('--') ? [a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true] : [])).filter((x) => x.length));
const ROOT = path.resolve(__dirname, '..');
const RAW = path.join(ROOT, 'data', 'openyap', 'raw');
const OUT = path.resolve(ROOT, args.out || 'tune/data/heldout_yap.jsonl');
const WINDOWS = String(args.windows || '0.25,0.6').split(',').map(Number);
const SECONDS = Number(args.seconds || 180);
const SR = 16000, FRAME = 1600, TICK = 0.6, NOVOICE = '<no voice>';

const dir = DEFAULTS.stt.dir;
const pick = (p) => { const all = fs.readdirSync(dir).filter((f) => f.startsWith(p) && f.endsWith('.onnx')); return path.join(dir, all.find((x) => !x.includes('int8')) || all[0]); };
const rec = new sherpa.OnlineRecognizer({
  featConfig: { sampleRate: SR, featureDim: 128 },
  modelConfig: { transducer: { encoder: pick('encoder'), decoder: pick('decoder'), joiner: pick('joiner') }, tokens: path.join(dir, 'tokens.txt'), numThreads: Number(args.threads || 4), provider: 'cpu', debug: 0 },
  decodingMethod: 'greedy_search',
  enableEndpoint: true, rule1MinTrailingSilence: 2.4, rule2MinTrailingSilence: 1.2, rule3MinUtteranceLength: 20, // as in stt-worker.js
});

function decode(file, t0, dur) { // 16 kHz mono float32 of [t0, t0+dur]
  const buf = execFileSync('ffmpeg', ['-v', 'error', '-ss', String(t0), '-t', String(dur), '-i', file, '-ac', '1', '-ar', String(SR), '-f', 'f32le', '-'], { maxBuffer: 1 << 30 });
  const out = new Float32Array(Math.floor(buf.length / 4));
  for (let i = 0; i < out.length; i++) out[i] = buf.readFloatLE(i * 4);
  return out;
}

// The caller's track through the app's recogniser + fragment logic: one `heard` per 600 ms tick.
function heardTicks(samples, nTicks) {
  const stt = new STT(() => {}); let sim = 0; stt.now = () => sim;
  const stream = rec.createStream(); let last = '';
  const out = [];
  const perTick = Math.round(TICK * SR / FRAME);
  for (let k = 0; k < nTicks; k++) {
    for (let f = 0; f < perTick; f++) {
      const a = (k * perTick + f) * FRAME;
      const frame = samples.subarray(a, a + FRAME);
      sim += 100;
      if (frame.length) {
        stt.feed(frame); // no worker: this only updates the voice-activity clock
        stream.acceptWaveform({ sampleRate: SR, samples: frame });
        while (rec.isReady(stream)) rec.decode(stream);
      }
      const text = rec.getResult(stream).text.trim(); const endpoint = rec.isEndpoint(stream);
      if (text !== last || endpoint) { stt.onResult({ text, punctuated: null, endpoint }); last = text; }
      if (endpoint) { rec.reset(stream); last = ''; }
    }
    out.push(stt.drain() || NOVOICE);
  }
  return out;
}

const words = (tr) => tr.words.filter((w) => w.type === 'word' || w.type === 'filler');

const files = fs.readdirSync(RAW).filter((f) => f.endsWith('.json')).sort().slice(0, Number(args.limit || 1e9));
const calls = [];
const t00 = Date.now();
for (const f of files) {
  const conv = JSON.parse(fs.readFileSync(path.join(RAW, f), 'utf8'));
  const key = f.replace(/\.json$/, '');
  for (const [user, ref] of [['a', 'b'], ['b', 'a']]) {
    for (const frac of WINDOWS) {
      const t0 = Math.round(conv.duration_seconds * frac);
      if (t0 + SECONDS > conv.duration_seconds) continue;
      const n = Math.floor(SECONDS / TICK);
      const heard = heardTicks(decode(path.join(RAW, `${key}.${user}.flac`), t0, SECONDS + 1), n);
      const { tags, said } = buildCall(words(conv.transcripts[user]), words(conv.transcripts[ref]), t0, SECONDS);
      const ticks = heard.map((h, k) => ({ heard: h, said: said[k], w: 1, tag: tags[k] }));
      calls.push({ split: 'heldout_yap', source: `open-yap-1k-sample/${key}`, caller: user, t0, timing: 'nemotron-real', dialogues: [`${key}:${user}:${t0}`], ticks });
      const c = (t) => tags.filter((x) => x === t).length;
      console.log(`${key} caller ${user} @${t0}s: start ${c('start')}, own-resume ${c('self_resume')}, overlap ${c('overlap_start')}, continuer ${c('continuer')}, pause ${c('listen_pause')}, barge ${c('barge_stop')} (${((Date.now() - t00) / 1000).toFixed(0)} s)`);
    }
  }
}
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, calls.map((c) => JSON.stringify(c)).join('\n') + '\n');
const count = {};
for (const c of calls) for (const t of c.ticks) count[t.tag] = (count[t.tag] || 0) + 1;
console.log(`wrote ${calls.length} calls, ${calls.reduce((s, c) => s + c.ticks.length, 0)} ticks to ${path.relative(ROOT, OUT)}`, JSON.stringify(count));
