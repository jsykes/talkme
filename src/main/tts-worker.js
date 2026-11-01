'use strict';
// Speech synthesis off the main thread. workerData: { engine: 'piper' | 'kokoro', dir, model (piper onnx), sid, threads },
// or { engine: 'remote', url } for a GPU voice served over HTTP (voice-server.js).
const { parentPort, workerData } = require('worker_threads');
const path = require('path');
const fs = require('fs');

if (workerData.engine === 'remote') {
  let closed = false;
  fetch(workerData.url + '/health').then((r) => r.json())
    .then(async (h) => {
      // the server may be speaking with another voice from the last call: set this one (Chatterbox prepares a new
      // reference clip once, a few seconds; Orpheus just switches speaker)
      const v = await fetch(workerData.url + '/voice', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ ref: workerData.ref || null, speaker: workerData.speaker || null }) });
      if (!v.ok) throw new Error(await v.text());
      return h;
    })
    .then((h) => parentPort.postMessage({ type: 'ready', sampleRate: h.sample_rate }))
    .catch((e) => parentPort.postMessage({ type: 'error', error: 'voice server: ' + e.message }));
  // Pieces are generated up to `parallel` at a time (Orpheus: 2, so the next piece's setup overlaps the current one;
  // Chatterbox: 1) but always handed over in the order they were sent: a later piece's parts wait for the earlier one.
  const parallel = workerData.parallel || 1;
  const jobs = []; // in order: { m, parts: [], done, sent, ctl, t0 }
  let running = 0;
  // TALKME_DUMP_TTS=<folder>: every part as it came from the voice server (<seq>-<part>.f32, float32 at its rate), to
  // tell a glitch in the voice's own audio from one in playback
  const dump = process.env.TALKME_DUMP_TTS;
  if (dump) fs.mkdirSync(dump, { recursive: true });
  const post = (j, samples) => {
    if (dump) fs.writeFileSync(path.join(dump, `${String(j.m.seq).padStart(4, '0')}-${String(j.sent).padStart(3, '0')}-${Date.now()}.f32`), Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength));
    parentPort.postMessage({ type: 'audio', seq: j.m.seq, generation: j.m.generation, text: j.sent++ === 0 ? j.m.text : '', words: j.m.words, final: false, samples, sampleRate: j.sr, ms: Date.now() - j.t0 });
  };
  // every part goes on the moment it is this piece's turn; the end of a piece is a separate 'done' message (holding
  // a part back to mark it "last" made playback run dry right after a piece's first 0.15 s: the stalls and pops)
  const deliver = () => {
    while (jobs.length) {
      const j = jobs[0];
      while (j.parts.length) post(j, j.parts.shift());
      if (!j.done) return;
      parentPort.postMessage({ type: 'done', seq: j.m.seq, generation: j.m.generation, words: j.m.words, text: j.sent ? '' : j.m.text });
      jobs.shift();
    }
  };
  const pump = () => {
    for (const j of jobs) {
      if (running >= parallel) return;
      if (!j.started) { j.started = true; running++; run(j).finally(() => { running--; j.done = true; deliver(); pump(); }); }
    }
  };
  const run = async (j) => {
    j.t0 = Date.now();
    try {
      const r = await fetch(workerData.url + '/tts', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text: j.m.text, cont: !!j.m.cont, stream: !!workerData.stream }), signal: j.ctl.signal });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
      j.sr = Number(r.headers.get('x-sample-rate'));
      if (r.headers.get('x-stream') !== '1') { j.parts.push(new Float32Array(await r.arrayBuffer())); return; }
      // streamed: whole floats only, each part handed on as it comes (if it is this piece's turn)
      const reader = r.body.getReader();
      let rest = new Uint8Array(0);
      for (;;) {
        const { value, done } = await reader.read();
        if (value && value.length) {
          const bytes = new Uint8Array(rest.length + value.length); bytes.set(rest); bytes.set(value, rest.length);
          const whole = bytes.length - (bytes.length % 4);
          rest = bytes.slice(whole);
          if (whole) { j.parts.push(new Float32Array(bytes.slice(0, whole).buffer)); deliver(); }
        }
        if (done) break;
      }
    } catch (e) {
      if (!j.ctl.signal.aborted) parentPort.postMessage({ type: 'error', error: 'voice server: ' + e.message });
    }
  };
  parentPort.on('message', (m) => {
    if (m.type === 'close') { closed = true; for (const j of jobs) j.ctl.abort(); parentPort.close(); return; }
    // a cut-in: stop generating what will never be played, so the GPU is free for the next reply
    if (m.type === 'cancel') { for (const j of jobs) j.ctl.abort(); jobs.length = 0; return; }
    if (m.type !== 'generate' || closed) return;
    const ctl = new AbortController();
    setTimeout(() => ctl.abort(), 30000);
    jobs.push({ m, parts: [], done: false, sent: 0, ctl, sr: 24000 });
    pump();
  });
  return;
}

let tts = null;
const sid = workerData.sid || 0;
try {
  const sherpa = require('sherpa-onnx-node');
  const dir = workerData.dir;
  const model = workerData.engine === 'kokoro'
    ? {
      kokoro: {
        model: path.join(dir, fs.readdirSync(dir).find((f) => /^model.*\.onnx$/.test(f))),
        voices: path.join(dir, 'voices.bin'),
        tokens: path.join(dir, 'tokens.txt'),
        dataDir: path.join(dir, 'espeak-ng-data'),
        // loaded but not consulted for English in this build: English goes through espeak (see tts.js MMHM)
        lexicon: path.join(dir, 'lexicon-us-en.txt'),
        dictDir: path.join(dir, 'dict'),
      },
    }
    : { vits: { model: path.join(dir, workerData.model), tokens: path.join(dir, 'tokens.txt'), dataDir: path.join(dir, 'espeak-ng-data') } };
  tts = new sherpa.OfflineTts({ model: { ...model, numThreads: workerData.threads || 2, provider: 'cpu', debug: 0 }, maxNumSentences: 1 });
  parentPort.postMessage({ type: 'ready', sampleRate: tts.sampleRate });
} catch (e) {
  parentPort.postMessage({ type: 'error', error: e.message });
}

parentPort.on('message', (m) => {
  if (m.type === 'close') { tts = null; parentPort.close(); return; }
  if (m.type !== 'generate' || !tts) return;
  const t0 = Date.now();
  try {
    // enableExternalBuffer=false and no transfer list: Electron forbids passing native (external) buffers between threads
    const audio = tts.generate({ text: m.text, sid, speed: m.speed || 1.0, enableExternalBuffer: false });
    const samples = Float32Array.from(audio.samples);
    parentPort.postMessage({ type: 'audio', seq: m.seq, generation: m.generation, text: m.text, words: m.words, samples, sampleRate: audio.sampleRate, ms: Date.now() - t0 });
  } catch (e) {
    parentPort.postMessage({ type: 'error', error: e.message });
  }
});
