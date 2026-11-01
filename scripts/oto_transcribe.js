// Transcribes the otoSpeech full-duplex corpus (stereo FLAC, one speaker per channel, no transcripts) with the app's
// own recogniser. Each channel is streamed once in simulated time, which gives both:
//   heard  what the app would hand the model every 600 ms if this speaker were the caller (src/main/stt.js logic)
//   words  this speaker's word timings (from the recogniser's token timestamps), the reference when they are the
//          other side of the call
// Output: data/otospeech/asr/<session>.json. Resumable (finished sessions are skipped); CPU only.
//   node scripts/oto_transcribe.js [--worker 0 --workers 3] [--threads 4] [--limit N]
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const sherpa = require('sherpa-onnx-node');
const { STT } = require('../src/main/stt');
const { DEFAULTS } = require('../src/main/config');

const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) => (a.startsWith('--') ? [a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true] : [])).filter((x) => x.length));
const ROOT = path.resolve(__dirname, '..');
const OTO = path.join(ROOT, 'data', 'otospeech');
const OUT = path.join(OTO, 'asr');
const W = Number(args.worker || 0), NW = Number(args.workers || 1);
const WORK = path.join(OTO, 'work', String(W));
const SR = 16000, FRAME = 1600, TICK_FRAMES = 6, NOVOICE = '<no voice>';
// Windows' own tar (bsdtar): GNU tar from Git Bash reads 'C:' in a path as a remote host
const TAR = process.platform === 'win32' ? path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe') : 'tar';
fs.mkdirSync(OUT, { recursive: true });
// a background batch job: lowest priority, so the app or a game always gets the CPU first (children inherit it)
try { require('os').setPriority(0, require('os').constants.priority.PRIORITY_LOW); } catch {}

const dir = DEFAULTS.stt.dir;
const pick = (p) => { const all = fs.readdirSync(dir).filter((f) => f.startsWith(p) && f.endsWith('.onnx')); return path.join(dir, all.find((x) => !x.includes('int8')) || all[0]); };
const rec = new sherpa.OnlineRecognizer({
  featConfig: { sampleRate: SR, featureDim: 128 },
  modelConfig: { transducer: { encoder: pick('encoder'), decoder: pick('decoder'), joiner: pick('joiner') }, tokens: path.join(dir, 'tokens.txt'), numThreads: Number(args.threads || 4), provider: 'cpu', debug: 0 },
  decodingMethod: 'greedy_search',
  enableEndpoint: true, rule1MinTrailingSilence: 2.4, rule2MinTrailingSilence: 1.2, rule3MinUtteranceLength: 20, // as in stt-worker.js
});

function channel(file, ch) {
  const buf = execFileSync('ffmpeg', ['-v', 'error', '-i', file, '-af', `pan=mono|c0=c${ch}`, '-ar', String(SR), '-f', 'f32le', '-'], { maxBuffer: 2 * 1024 * 1024 * 1024 });
  const out = new Float32Array(Math.floor(buf.length / 4));
  for (let i = 0; i < out.length; i++) out[i] = buf.readFloatLE(i * 4);
  return out;
}

// tokens + timestamps of one recogniser segment -> words with absolute times
function segWords(r, offset, into) {
  const toks = r.tokens || [], ts = r.timestamps || [];
  let cur = null;
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i];
    if (t.startsWith(' ') || !cur) { if (cur) into.push(cur); cur = { word: t.trim(), start: +(offset + ts[i]).toFixed(2), end: +(offset + ts[i] + 0.12).toFixed(2) }; }
    else { cur.word += t; cur.end = +(offset + ts[i] + 0.12).toFixed(2); }
  }
  if (cur && cur.word) into.push(cur);
}

function transcribe(samples) {
  const stt = new STT(() => {}); let sim = 0; stt.now = () => sim;
  const stream = rec.createStream(); let last = '', segStart = 0;
  const heard = [], words = [];
  const frames = Math.ceil(samples.length / FRAME);
  for (let f = 0; f < frames + 20; f++) { // 2 s of silence at the end flushes the last words
    const frame = f < frames ? samples.subarray(f * FRAME, (f + 1) * FRAME) : new Float32Array(FRAME);
    sim += 100;
    stt.feed(frame);
    stream.acceptWaveform({ sampleRate: SR, samples: frame });
    while (rec.isReady(stream)) rec.decode(stream);
    const r = rec.getResult(stream); const text = r.text.trim(); const endpoint = rec.isEndpoint(stream);
    if (text !== last || endpoint) { stt.onResult({ text, punctuated: null, endpoint }); last = text; }
    if (endpoint) { segWords(r, segStart, words); rec.reset(stream); last = ''; segStart = (f + 1) * 0.1; }
    if ((f + 1) % TICK_FRAMES === 0 && f < frames) heard.push(stt.drain() || NOVOICE);
  }
  segWords(rec.getResult(stream), segStart, words);
  return { heard, words };
}

const shards = fs.readdirSync(path.join(OTO, 'data', 'train')).filter((f) => f.endsWith('.tar')).sort().filter((_, i) => i % NW === W);
let done = 0; const t00 = Date.now(); const limit = Number(args.limit || 1e9);
for (const shard of shards) {
  if (done >= limit) break;
  const list = execFileSync(TAR, ['-tf', path.join(OTO, 'data', 'train', shard)]).toString().split(/\r?\n/).filter((x) => x.endsWith('.json'));
  // finished = a complete output file (one cut off by a stopped run is redone)
  const finished = (id) => { try { return !!JSON.parse(fs.readFileSync(path.join(OUT, id + '.json'), 'utf8')).channels; } catch { return false; } };
  const todo = list.map((x) => x.replace(/\.json$/, '')).filter((id) => !finished(id));
  if (!todo.length) continue;
  fs.rmSync(WORK, { recursive: true, force: true }); fs.mkdirSync(WORK, { recursive: true });
  execFileSync(TAR, ['-xf', path.join(OTO, 'data', 'train', shard), '-C', WORK]);
  for (const id of todo) {
    if (done >= limit) break;
    const meta = JSON.parse(fs.readFileSync(path.join(WORK, id + '.json'), 'utf8'));
    const t0 = Date.now();
    const chans = [0, 1].map((ch) => transcribe(channel(path.join(WORK, id + '.flac'), ch)));
    const out = {
      session_id: id, duration: meta.duration, session_type: meta.session_type, topics: (meta.session_instruction_metadata?.topics || []).map((t) => t.topic),
      redacted: meta.redacted_segments || [], speakers: (meta.speakers || []).map((s) => ({ channel: s.channel, gender: s.gender, age: s.age, first_language: s.first_language })),
      channels: chans,
    };
    fs.writeFileSync(path.join(OUT, id + '.json'), JSON.stringify(out));
    done++;
    console.log(`[w${W}] ${shard} ${id} ${(meta.duration / 60).toFixed(1)} min in ${((Date.now() - t0) / 1000).toFixed(0)} s, words ${chans[0].words.length}/${chans[1].words.length} (${done} done, ${((Date.now() - t00) / 3.6e6).toFixed(2)} h)`);
  }
  fs.rmSync(WORK, { recursive: true, force: true });
}
console.log(`[w${W}] finished: ${done} sessions`);
