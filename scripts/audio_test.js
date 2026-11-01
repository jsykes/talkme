// Audio round-trip test of the whole call path, in real time, with no microphone or speakers:
//   a second Piper voice (Amy) speaks the user's lines -> 16 kHz frames every 100 ms into the real STT
//   -> the real 600 ms loop against llama-server -> the assistant's Piper audio is captured, "played" on a
//   simulated speaker queue, transcribed back through the STT, and checked against each scenario's expectations
//   (reply latency, silence during pauses, barge-in stop time, keywords). A stereo WAV (left: user, right: assistant)
//   and a JSON timeline go to tests/out/.
//
//   node scripts/audio_test.js [--scenario name] [--adapter <path|none>] [--echo 0.3] [--threshold 0.3]
//
// --echo mixes the assistant's playback into the "microphone" at that gain, to see what happens on speakers.
const fs = require('fs');
const path = require('path');
const sherpa = require('sherpa-onnx-node');
const { DEFAULTS, baseRow, baseKey } = require('../src/main/config');
const { Llama, LISTEN, NOVOICE } = require('../src/main/llama');
const { STT } = require('../src/main/stt');
const { TTS } = require('../src/main/tts');
const { Loop } = require('../src/main/loop');

const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) => (a.startsWith('--') ? [a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true] : [])).filter((x) => x.length));
const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'tests', 'out');
fs.mkdirSync(OUT, { recursive: true });
const SR = 16000, FRAME = 1600; // 100 ms mic frames
const BENCH_VOICE = 'piper:vits-piper-en_US-lessac-medium';
const scenarios = JSON.parse(fs.readFileSync(path.join(ROOT, 'tests', 'scenarios.json'), 'utf8')).filter((s) => !args.scenario || String(args.scenario).split(',').includes(s.name));
const echoGain = args.echo ? Number(args.echo) : 0;
const threshold = args.threshold ? Number(args.threshold) : DEFAULTS.listenThreshold;

// A yes/no question about one moment of the call, answered by the base model with every adapter off. For checks a
// keyword list scores too strictly (run 9 warned about bleach on all three passes and passed the keyword check once).
async function judge(llama, question, userSaid, aiSaid) {
  const body = {
    messages: [{ role: 'system', content: 'You grade one moment of a phone call. Answer with one word: yes or no.' },
      { role: 'user', content: `The caller said: "${userSaid}"\nThe other person replied: "${aiSaid}"\n${question}` }],
    temperature: 0, max_tokens: 3, chat_template_kwargs: { enable_thinking: false },
    lora: (llama.loraScales || []).map((s) => ({ id: s.id, scale: 0 })),
  };
  let r;
  for (let attempt = 0; ; attempt++) { // the first try can hit a kept-alive connection the server already closed
    try { r = await fetch(llama.url + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body), signal: AbortSignal.timeout(30000) }); break; }
    catch (e) { if (attempt >= 2) return { pass: false, answer: `judge failed: ${e.cause?.code || e.message}` }; }
  }
  let answer = (await r.json()).choices[0].message.content.trim();
  // gpt-oss thinks first, in its analysis channel, whatever it is asked: 3 tokens are only that channel's header. It
  // gets room to think, briefly, and its answer is the final channel (the server returns only that once the reply is
  // whole, the header and all when it is not).
  if (answer.startsWith('<|channel|>')) {
    body.max_tokens = 400; body.chat_template_kwargs = { reasoning_effort: 'low' };
    try {
      const text = (await (await fetch(llama.url + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body), signal: AbortSignal.timeout(60000) })).json()).choices[0].message.content;
      const final = text.match(/<\|channel\|>final<\|message\|>([\s\S]*)/);
      answer = (final ? final[1] : text).replace(/<\|[a-z]+\|>/g, '').trim();
      if (/<\|channel\|>analysis/.test(text) && !final) answer = `no final answer: ${text.slice(-80)}`;
    } catch (e) { return { pass: false, answer: `judge failed: ${e.message}` }; }
  }
  return { pass: /^yes/i.test(answer), answer };
}

const resampleTo16k = (samples, sr) => sr === SR ? samples : new sherpa.LinearResampler(sr, SR).resample(samples);
const norm = (s) => s.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim();
function wer(ref, hyp) {
  const r = norm(ref).split(' ').filter(Boolean), h = norm(hyp).split(' ').filter(Boolean);
  const d = Array.from({ length: r.length + 1 }, (_, i) => [i, ...Array(h.length).fill(0)]);
  for (let j = 1; j <= h.length; j++) d[0][j] = j;
  for (let i = 1; i <= r.length; i++) for (let j = 1; j <= h.length; j++) d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] === h[j - 1] ? 0 : 1));
  return r.length ? d[r.length][h.length] / r.length : 0;
}
function writeStereoWav(file, left, right, sr) {
  const n = Math.max(left.length, right.length);
  const buf = Buffer.alloc(44 + n * 4);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 4, 4); buf.write('WAVE', 8); buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(2, 22); buf.writeUInt32LE(sr, 24);
  buf.writeUInt32LE(sr * 4, 28); buf.writeUInt16LE(4, 32); buf.writeUInt16LE(16, 34); buf.write('data', 36); buf.writeUInt32LE(n * 4, 40);
  for (let i = 0; i < n; i++) {
    buf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round((left[i] || 0) * 32767))), 44 + i * 4);
    buf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round((right[i] || 0) * 32767))), 46 + i * 4);
  }
  fs.writeFileSync(file, buf);
}

(async () => {
  const log = (l) => { if (!/^llama: \d/.test(l)) console.log('    · ' + l); };
  const llama = new Llama(log);
  // --adapter blend:talkme-partner-v0-gemma4a4b.gguf=0.5 plays several adapters at once (files from models/lora)
  const isBlend = !!args.adapter && String(args.adapter).startsWith('blend:');
  const blendFiles = isBlend ? String(args.adapter).slice(6).split(',').map((p) => path.join(ROOT, 'models', 'lora', p.split('=')[0])) : [];
  // --port / --ncpumoe: run against (or spawn) a separate server, e.g. while another job uses the default one
  // --model <gguf>: the base model (default: the app's), e.g. the abliterated one an adapter was trained on
  // only adapters made for the model under test can load beside it (a Qwen adapter on Gemma stops the server), and
  // a base with a shorter context gets it here too (config.js BASES)
  const model = args.model ? path.resolve(args.model) : DEFAULTS.llama.model;
  const sameFamily = (f) => f && baseKey(f) === baseKey(model);
  const extra = [...(args.loras || '').split(','), args.adapter && args.adapter !== 'none' && !isBlend ? path.resolve(args.adapter) : '', ...blendFiles].filter(Boolean);
  await llama.start({ ...DEFAULTS.llama, model, lora: sameFamily(DEFAULTS.llama.lora) ? DEFAULTS.llama.lora : '', ctx: baseRow(model).ctx || DEFAULTS.llama.ctx,
    port: Number(args.port || DEFAULTS.llama.port), nCpuMoe: Number(args.ncpumoe || DEFAULTS.llama.nCpuMoe), extraLoras: extra });
  // (the adapter under test is always loaded, next to the app's default one)
  llama.proc.on('exit', (code, sig) => console.log(`llama-server exited: code ${code} signal ${sig}`));
  const adapters = await (await fetch(llama.url + '/lora-adapters')).json();
  if (args.adapter) await llama.useAdapter(args.adapter === 'none' ? null : isBlend ? args.adapter : path.resolve(args.adapter));
  else if (adapters.length) await llama.useAdapter(adapters[0].path);
  llama.promptKind = args.prompt === 'partner' ? 'partner' : 'assistant'; // --prompt partner: the system prompt of run 9 on
  console.log(`server adapters: ${adapters.map((a) => path.basename(a.path) + (llama.loraScales ? `=${llama.loraScales.find((s) => s.id === a.id).scale}` : '')).join(', ') || '(none)'}; listen threshold ${threshold}; echo ${echoGain}`);
  // --user-voice piper:<folder> | kokoro:<speaker id>: who plays the caller (default Piper Amy); --user-speed 0.8-1.3
  const { voiceConfig: callerVoiceConfig } = require('../src/main/voices');
  const uc = callerVoiceConfig(args['user-voice'] || 'piper:vits-piper-en_US-amy-medium');
  const userTts = uc.engine === 'kokoro'
    ? new sherpa.OfflineTts({ model: { kokoro: { model: path.join(uc.dir, 'model.onnx'), voices: path.join(uc.dir, 'voices.bin'), tokens: path.join(uc.dir, 'tokens.txt'),
        dataDir: path.join(uc.dir, 'espeak-ng-data'), lexicon: path.join(uc.dir, 'lexicon-us-en.txt'), dictDir: path.join(uc.dir, 'dict') }, numThreads: 4, provider: 'cpu', debug: 0 }, maxNumSentences: 1 })
    : new sherpa.OfflineTts({ model: { vits: { model: path.join(uc.dir, uc.model), tokens: path.join(uc.dir, 'tokens.txt'), dataDir: path.join(uc.dir, 'espeak-ng-data') }, numThreads: 2, provider: 'cpu', debug: 0 }, maxNumSentences: 1 });
  const userSid = uc.sid || 0, userSpeed = Number(args['user-speed'] || 1.0);
  // a second recogniser to transcribe what the assistant said (what a listener would hear)
  const earDir = DEFAULTS.stt.dir;
  const pickEar = (p) => { const all = fs.readdirSync(earDir).filter((f) => f.startsWith(p) && f.endsWith('.onnx')); return path.join(earDir, all.find((x) => !x.includes('int8')) || all[0]); };
  const ear = new sherpa.OnlineRecognizer({ featConfig: { sampleRate: SR, featureDim: /nemotron/i.test(earDir) ? 128 : 80 }, modelConfig: { transducer: { encoder: pickEar('encoder'), decoder: pickEar('decoder'), joiner: pickEar('joiner') }, tokens: path.join(earDir, 'tokens.txt'), numThreads: 4, provider: 'cpu', debug: 0 }, decodingMethod: 'greedy_search' });
  const hear = (samples22k, sr) => {
    const st = ear.createStream();
    const s16 = resampleTo16k(samples22k, sr);
    const padded = new Float32Array(s16.length + SR); padded.set(s16); // a second of silence to flush
    st.acceptWaveform({ sampleRate: SR, samples: padded });
    while (ear.isReady(st)) ear.decode(st);
    return ear.getResult(st).text.trim();
  };

  const results = [];
  for (const sc of scenarios) {
    console.log(`\n=== ${sc.name}: ${sc.about}`);
    const stt = new STT(log); await stt.start({ ...DEFAULTS.stt, finalPunct: args.finalpunct !== 'off' });
    // --voice piper:<dir> | kokoro:<speaker id> (default: the app's default voice)
    // the assistant speaks with a fixed voice, not the app's default, so runs stay comparable (the voice's speed and
    // delay are part of every timing check); Piper Lessac is what every benchmark before 2026-10-21 used
    const tts = new TTS(log); await tts.start(require('../src/main/voices').voiceConfig(args.voice || BENCH_VOICE));
    const timeline = []; const t00 = Date.now(); const T = () => (Date.now() - t00) / 1000;
    const ev = (who, text, extra = {}) => { timeline.push({ t: +T().toFixed(2), who, text, ...extra }); };
    // mic timeline: pre-synthesise every user line. A line starts `pause` ms after the previous one ends; a line with
    // "whenAssistantTalkingMs" instead waits until the assistant has been audible that long (so a cut-in or a
    // back-channel really lands over the answer), or 15 s at most.
    let micTrack = []; const utter = [];
    for (const turn of sc.turns) {
      const a = userTts.generate({ text: turn.say, sid: userSid, speed: userSpeed, enableExternalBuffer: false });
      const s16 = resampleTo16k(Float32Array.from(a.samples), a.sampleRate);
      utter.push({ turn, startSample: null, endSample: null, samples: s16, started: false, ended: false });
    }
    utter[0].startSample = Math.round(0.3 * SR); utter[0].endSample = utter[0].startSample + utter[0].samples.length;
    let totalSamples = Infinity;
    const schedule = (pos, nowMs) => {
      const k = utter.findIndex((u) => u.startSample === null);
      if (k < 0) {
        const last = utter[utter.length - 1];
        totalSamples = last.endSample + Math.round((last.turn.pause || 1000) / 1000 * SR) + SR;
        return;
      }
      const prev = utter[k - 1];
      const earliest = prev.endSample + Math.round((prev.turn.pause || 1000) / 1000 * SR);
      const u = utter[k];
      const need = u.turn.whenAssistantTalkingMs;
      let go = false;
      if (!need) go = pos >= earliest;
      else {
        const prevEndMs = prev.tEnd || Infinity;
        const talking = spk.find((s) => s.start >= prevEndMs - 500 && s.start <= nowMs && s.end > nowMs);
        const audibleFor = talking ? nowMs - Math.min(...spk.filter((s) => s.start >= prevEndMs - 500 && s.start <= nowMs).map((s) => s.start)) : 0;
        go = (talking && audibleFor >= need && pos >= prev.endSample) || pos >= prev.endSample + 15 * SR;
      }
      if (go) { u.startSample = pos; u.endSample = pos + u.samples.length; }
    };
    // simulated speaker: queue of assistant chunks with play times; "spk" is what the listener hears (22.05 kHz)
    const spk = []; let queueEnd = 0; const playing = () => Date.now() < queueEnd;
    const rightTrack = []; // assistant audio at 22050 for the WAV, indexed by time
    let outSr = tts.sampleRate;
    tts.onAudio = (a) => {
      const start = Math.max(Date.now(), queueEnd); const dur = a.samples.length / a.sampleRate * 1000;
      queueEnd = start + dur;
      const item = { text: a.text, start, end: queueEnd, samples: a.samples, sr: a.sampleRate, synthMs: a.ms };
      spk.push(item);
      item.ev = { t: +T().toFixed(2), who: 'ai-audio', text: a.text, startsIn: +((start - Date.now()) / 1000).toFixed(2), dur: +(dur / 1000).toFixed(2) };
      timeline.push(item.ev);
    };
    const loop = new Loop({ llama, stt, tts, log, emit: (name, p) => {
      if (name === 'tick') { ev('tick', `${p.heard} -> ${p.said}`, { p: +p.pListen.toFixed(2), ms: p.totalMs, action: p.action }); }
      else if (name === 'audio-stop') { const now = Date.now(); for (const s of spk) if (s.end > now) s.end = Math.max(s.start, now); queueEnd = now; ev('ai-stop', 'playback cut (barge-in)'); }
      else if (name === 'error') ev('error', p);
    } });
    const origSpeaking = Object.getOwnPropertyDescriptor(loop, 'speaking');
    Object.defineProperty(loop, 'speaking', { get: () => playing(), set: () => {} });
    // --old-flush: the original behaviour, where an answer's first words wait for punctuation or ttsFlushWords
    loop.start({ tickMs: DEFAULTS.tickMs, listenThreshold: threshold, listenSample: !!args.sample, ttsFlushWords: DEFAULTS.ttsFlushWords, ttsSpeed: 1.0, llamaCtx: DEFAULTS.llama.ctx, speakFirstAtOnce: !args['old-flush'],
      historyTicks: baseRow(llama.model).historyTicks }); // as the app (config.js BASES)
    // feed the mic in real time
    let pos = 0; const t0 = Date.now();
    await new Promise((resolve) => {
      const iv = setInterval(() => {
        const target = Math.floor((Date.now() - t0) / 1000 * SR);
        while (pos < target && pos < totalSamples) {
          schedule(pos, t0 + pos / SR * 1000);
          const frame = new Float32Array(FRAME);
          for (const u of utter) {
            if (u.startSample === null) continue;
            if (pos + FRAME > u.startSample && pos < u.endSample) {
              for (let i = 0; i < FRAME; i++) { const k = pos + i - u.startSample; if (k >= 0 && k < u.samples.length) frame[i] += u.samples[k]; }
              if (!u.started) { u.started = true; u.tStart = Date.now(); ev('user', u.turn.say); if (u.turn.bargeIn) u.wasPlaying = playing(); }
            }
            if (!u.ended && u.started && pos >= u.endSample) { u.ended = true; u.tEnd = Date.now(); ev('user-end', '(finished speaking)'); }
          }
          if (echoGain) { // speaker bleed: whatever is "playing" right now, resampled to 16 kHz, into the mic
            const now = t0 + pos / SR * 1000;
            for (const s of spk) if (now >= s.start && now < s.end) {
              const off = Math.floor((now - s.start) / 1000 * s.sr); const chunk = s.samples.subarray(off, off + Math.ceil(FRAME * s.sr / SR));
              const r = resampleTo16k(chunk, s.sr); for (let i = 0; i < FRAME && i < r.length; i++) frame[i] += echoGain * r[i];
            }
          }
          stt.feed(frame); micTrack.push(frame); pos += FRAME;
        }
        if (pos >= totalSamples) { clearInterval(iv); resolve(); }
      }, 50);
    });
    await new Promise((r) => setTimeout(r, 800));
    loop.stop(); await tts.stop(); await stt.stop();

    for (const s of spk) { s.heard = hear(s.samples, s.sr); s.wer = +wer(s.text, s.heard).toFixed(2); Object.assign(s.ev, { heard: s.heard, wer: s.wer, playedS: +((s.end - s.start) / 1000).toFixed(2) }); }
    // ---- evaluate ----
    const checks = [];
    const firstAudioAfter = (t) => spk.filter((s) => s.start >= t).sort((a, b) => a.start - b.start)[0];
    for (const u of utter) {
      const e = u.turn.expect || {};
      const next = utter[utter.indexOf(u) + 1];
      const windowEnd = next ? next.tStart : u.tEnd + (u.turn.pause || 1000) + 500;
      // what the app's recogniser passed to the model for this line (a back-channel it never heard can't be "kept talking through")
      const appHeard = timeline.filter((x) => x.who === 'tick' && x.t * 1000 + t00 >= u.tStart && x.t * 1000 + t00 <= u.tEnd + 2000)
        .map((x) => x.text.split(' -> ')[0]).filter((h) => h && !/^<no voice>$/.test(h)).join(' ');
      if (e.noReplyDuringPause) {
        const spoke = spk.find((s) => s.start >= u.tEnd - 300 && s.start < windowEnd && !/^(mm-?hm|uh-?huh|right|yeah|okay|i see|sure)[.!,]?$/i.test(s.text.trim()));
        checks.push({ turn: u.turn.say.slice(0, 40), check: 'no answer during the pause', pass: !spoke, detail: spoke ? `spoke "${spoke.text}" ${((spoke.start - u.tEnd) / 1000).toFixed(1)} s after the pause began` : 'stayed quiet (fillers allowed)' });
      }
      if (e.replyWithinMs) {
        const s = firstAudioAfter(u.tStart); // a reply may start over the user (a warning), never before they spoke
        const lat = s ? (s.start - u.tEnd) / 1000 : null;
        checks.push({ turn: u.turn.say.slice(0, 40), check: `reply within ${e.replyWithinMs / 1000} s`, pass: s && s.start < windowEnd && lat * 1000 <= e.replyWithinMs, detail: s ? `first audio ${lat.toFixed(2)} s after the user finished: "${s.text}"` : 'no reply' });
      }
      if (e.keywords) {
        const said = spk.filter((s) => s.start >= u.tStart - 1500 && s.start < windowEnd).map((s) => s.text).join(' ');
        const heard = spk.filter((s) => s.start >= u.tStart - 1500 && s.start < windowEnd).map((s) => s.heard).join(' ');
        const plain = said.replace(/’/g, "'"); // some models write a curly apostrophe
        const missing = e.keywords.filter((k) => !new RegExp(k, 'i').test(plain));
        checks.push({ turn: u.turn.say.slice(0, 40), check: `mentions ${e.keywords.join(' + ')}`, pass: !missing.length, detail: `said: "${said}" | heard back: "${heard}"${missing.length ? ` | missing ${missing.join(', ')}` : ''}` });
      }
      if (e.judge) {
        const said = spk.filter((s) => s.start >= u.tStart - 1500 && s.start < windowEnd).map((s) => s.text).join(' ');
        for (const q of [].concat(e.judge)) {
          const j = said ? await judge(llama, q, u.turn.say, said) : { pass: false, answer: 'no reply' };
          checks.push({ turn: u.turn.say.slice(0, 40), check: `judged: ${q}`, pass: j.pass, detail: `${j.answer} — said: "${said}"` });
        }
      }
      if (e.maxWords) { // answers, then stops: no drifting on into its own plans
        const said = spk.filter((s) => s.start >= u.tStart - 1500 && s.start < windowEnd).map((s) => s.text).join(' ');
        const n = said.split(/\s+/).filter(Boolean).length;
        checks.push({ turn: u.turn.say.slice(0, 40), check: `says at most ${e.maxWords} words`, pass: n > 0 && n <= e.maxWords, detail: `${n} words: "${said}"` });
      }
      if (e.keepsTalkingMs) {
        const stop = timeline.find((x) => x.who === 'ai-stop' && x.t * 1000 + t00 >= u.tStart && x.t * 1000 + t00 <= u.tEnd + e.keepsTalkingMs);
        const playingAtStart = spk.some((s) => s.start <= u.tStart && s.end > u.tStart);
        const continued = spk.some((s) => s.start > u.tEnd && s.start < u.tEnd + e.keepsTalkingMs);
        checks.push({ turn: u.turn.say.slice(0, 40), check: `keeps talking through a back-channel`, pass: playingAtStart && !stop && continued, detail: `${!playingAtStart ? 'assistant was not talking when the user said it' : stop ? `stopped ${((stop.t * 1000 + t00 - u.tStart) / 1000).toFixed(2)} s after it` : continued ? 'kept going' : 'went quiet after it'}${appHeard ? ` | app heard "${appHeard}"` : ' | the app never heard it'}`, heardByApp: !!appHeard });
      }
      // partner checks: speaks up again after a silence, back-channels while the caller talks, keeps a story going
      if (e.nudgeWithinMs) {
        const before = spk.filter((s) => s.start < windowEnd).sort((a, b) => a.start - b.start);
        const reply = before.find((s) => s.start >= u.tStart);
        const replyEnd = reply ? Math.max(...before.filter((s) => s.start >= u.tStart && s.start < reply.start + 6000).map((s) => s.end)) : u.tEnd;
        const nudge = spk.find((s) => s.start >= replyEnd + 2500 && s.start <= replyEnd + e.nudgeWithinMs && s.start < windowEnd);
        checks.push({ turn: u.turn.say.slice(0, 40), check: `speaks up within ${e.nudgeWithinMs / 1000} s of silence`, pass: !!nudge, detail: nudge ? `after ${((nudge.start - replyEnd) / 1000).toFixed(1)} s of quiet: "${nudge.text}"` : 'stayed quiet' });
      }
      if (e.backchannelsDuring) {
        const bcs = spk.filter((s) => s.start >= u.tStart + 1000 && s.start <= u.tEnd && s.text.trim().split(/\s+/).length <= 3);
        checks.push({ turn: u.turn.say.slice(0, 40), check: `${e.backchannelsDuring}+ back-channels while the caller talks`, pass: bcs.length >= e.backchannelsDuring, detail: `${bcs.length} during ${((u.tEnd - u.tStart) / 1000).toFixed(0)} s: ${bcs.map((s) => '"' + s.text + '"').join(' ') || 'none'}` });
      }
      if (e.talksAtLeastMs) {
        const ms = spk.filter((s) => s.start >= u.tEnd - 500 && s.start < windowEnd).reduce((a, s) => a + (s.end - s.start), 0);
        checks.push({ turn: u.turn.say.slice(0, 40), check: `keeps talking at least ${e.talksAtLeastMs / 1000} s`, pass: ms >= e.talksAtLeastMs, detail: `${(ms / 1000).toFixed(1)} s of speech after the request` });
      }
      if (e.stopWithinMs) {
        const stop = timeline.find((x) => x.who === 'ai-stop' && x.t * 1000 + t00 >= u.tStart);
        const lat = stop ? stop.t * 1000 + t00 - u.tStart : null;
        checks.push({ turn: u.turn.say.slice(0, 40), check: `stops talking within ${e.stopWithinMs / 1000} s of the cut-in`, pass: u.wasPlaying && stop && lat <= e.stopWithinMs, detail: !u.wasPlaying ? 'assistant was not talking when the user cut in (nothing to stop)' : stop ? `cut ${(lat / 1000).toFixed(2)} s after the user started` : 'playback was never cut' });
      }
    }
    // overlap: assistant audio playing while the user speaks (excluding fillers and hazard cut-ins is the model's call; just report it)
    let overlapMs = 0;
    for (const s of spk) for (const u of utter) overlapMs += Math.max(0, Math.min(s.end, u.tEnd) - Math.max(s.start, u.tStart));
    const avgWer = spk.length ? spk.reduce((a, s) => a + s.wer, 0) / spk.length : 0;
    const ticks = timeline.filter((x) => x.who === 'tick');
    const summary = { scenario: sc.name, passed: checks.filter((c) => c.pass).length, total: checks.length, assistantChunks: spk.length, overlapS: +(overlapMs / 1000).toFixed(1), avgWerOfAssistantAudio: +avgWer.toFixed(2), avgTickMs: Math.round(ticks.reduce((a, x) => a + x.ms, 0) / Math.max(1, ticks.length)), maxTickMs: Math.max(0, ...ticks.map((x) => x.ms)) };
    results.push({ ...summary, checks, timeline });
    // ---- print ----
    for (const x of timeline) {
      if (x.who === 'tick' && !/->\s(?!<listen>)/.test(x.text) && !x.text.startsWith('<no voice>')) continue; // hide silent-listen ticks
      if (x.who === 'tick' && x.text === '<no voice> -> <listen>') continue;
      console.log(`${x.t.toFixed(1).padStart(6)}s ${x.who.padEnd(9)} ${x.text}${x.who === 'tick' ? ` (P=${x.p}, ${x.ms} ms${x.action === 'stop' ? ', STOP' : ''})` : ''}${x.who === 'ai-audio' ? ` [${x.dur}s, heard back: "${x.heard}", wer ${x.wer}]` : ''}`);
    }
    for (const c of checks) console.log(`  ${c.pass ? 'PASS' : 'FAIL'}  ${c.check} — ${c.detail}`);
    console.log(`  ${summary.passed}/${summary.total} checks; ${summary.assistantChunks} assistant chunks; overlap with user ${summary.overlapS} s; avg WER of assistant audio ${summary.avgWerOfAssistantAudio}; ticks avg ${summary.avgTickMs} ms max ${summary.maxTickMs} ms`);
    // ---- wav ----
    const left = new Float32Array(Math.ceil(micTrack.length * FRAME / SR * outSr));
    const mic = new Float32Array(micTrack.length * FRAME); micTrack.forEach((f, i) => mic.set(f, i * FRAME));
    const up = new sherpa.LinearResampler(SR, outSr).resample(mic); left.set(up.subarray(0, left.length));
    const right = new Float32Array(left.length);
    for (const s of spk) { const o = Math.floor((s.start - t0) / 1000 * s.sr); const n = Math.min(s.samples.length, Math.floor((s.end - s.start) / 1000 * s.sr), right.length - o); for (let i = 0; i < n; i++) right[o + i] += s.samples[i]; }
    const wav = path.join(OUT, `${sc.name}${args.tag ? '-' + args.tag : ''}${echoGain ? '-echo' : ''}.wav`);
    writeStereoWav(wav, left, right, outSr);
    console.log(`  wrote ${path.relative(ROOT, wav)} (left: user, right: assistant)`);
  }
  fs.writeFileSync(path.join(OUT, `report${args.tag ? '-' + args.tag : ''}${echoGain ? '-echo' : ''}.json`), JSON.stringify({ adapter: args.adapter || 'default', threshold, echoGain, results }, null, 1));
  // ---- scoreboard across scenarios ----
  const allChecks = results.flatMap((r) => r.checks);
  const num = (re, c) => { const m = c.detail.match(re); return m ? Number(m[1]) : null; };
  const replyLat = allChecks.filter((c) => /^reply within/.test(c.check)).map((c) => num(/first audio (-?[\d.]+) s/, c)).filter((x) => x !== null);
  const stopLat = allChecks.filter((c) => /^stops talking/.test(c.check)).map((c) => num(/cut ([\d.]+) s/, c)).filter((x) => x !== null);
  const tickMs = results.flatMap((r) => r.timeline.filter((x) => x.who === 'tick').map((x) => x.ms)).sort((x, y) => x - y);
  const pct = (arr, p) => (arr.length ? arr[Math.min(arr.length - 1, Math.floor(arr.length * p))] : NaN);
  const med = (arr) => pct([...arr].sort((x, y) => x - y), 0.5);
  const board = {
    checksPassed: allChecks.filter((c) => c.pass).length, checks: allChecks.length,
    medianReplyS: +med(replyLat).toFixed(2), medianBargeStopS: stopLat.length ? +med(stopLat).toFixed(2) : null,
    tickP50ms: pct(tickMs, 0.5), tickP95ms: pct(tickMs, 0.95),
    ticksOverBudgetPct: +(100 * tickMs.filter((x) => x > DEFAULTS.tickMs).length / Math.max(1, tickMs.length)).toFixed(1),
  };
  fs.writeFileSync(path.join(OUT, `scoreboard${args.tag ? '-' + args.tag : ''}.json`), JSON.stringify({ adapter: args.adapter || 'default', ...board }, null, 1));
  console.log('\n=== summary ===');
  for (const r of results) console.log(`${r.scenario.padEnd(10)} ${r.passed}/${r.total} checks, overlap ${r.overlapS} s, WER ${r.avgWerOfAssistantAudio}, tick avg ${r.avgTickMs} ms`);
  console.log(`\n=== scoreboard ===\nchecks ${board.checksPassed}/${board.checks} | median reply ${board.medianReplyS} s after the user stops | median barge-in stop ${board.medianBargeStopS} s | tick p50 ${board.tickP50ms} ms, p95 ${board.tickP95ms} ms, ${board.ticksOverBudgetPct}% over ${DEFAULTS.tickMs} ms`);
  await llama.stop();
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
