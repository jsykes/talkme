// Records demo calls of the app window for a video: a caller voiced by Orpheus (tests/out/demo/caller, rendered by
// scripts/demo_callers.js) talks to the app through a stand-in microphone, so the whole call path runs as in a real
// call (echo canceller, recogniser, model, voice, transcript, tick tape, log). Each scenario in
// tests/demo_scenarios.json is one video with a caption bar at the top: tests/out/demo/<tag>/<scenario>.webm (1080p),
// plus <scenario>.json with every tick for checking. The call runs live in the window; --audible also plays both voices
// on the speakers (the stand-in mic does not hear them), otherwise only the video carries them.
//   npx electron scripts/demo_record.js --adapter <file> --model "Gemma 4 26B-A4B" --tag gemma [--only cut-in,quick] [--threshold 0.6] [--no-aec] [--audible]
//     [--voice=orpheus:zac] [--caller dan] [--scenarios tests/demo_scenarios_long.json]
// --voice: the app's voice (a voices.js id; the saved one otherwise), written with "=": Electron exits at launch, silently,
// when an argument on its own looks like a URL ("orpheus:zac") and another option follows it. --caller: the caller's Orpheus speaker, rendered by
// scripts/demo_callers.js --voice <speaker> (the scenarios' default caller otherwise). The caption bar names both.
const { app, session } = require('electron');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const arg = (k, d) => { const eq = process.argv.find((a) => a.startsWith(`--${k}=`)); if (eq) return eq.slice(k.length + 3); const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const ADAPTER = path.resolve(ROOT, arg('adapter', 'models/lora/talkme-partner-v1-gemma4a4bstd.gguf'));
const MODEL = arg('model', '');
const TAG = arg('tag', path.basename(ADAPTER, '.gguf'));
const ONLY = (arg('only', '') || '').split(',').filter(Boolean);
const THRESHOLD = arg('threshold', null); // the adapter's own listen threshold unless given
const NO_AEC = process.argv.includes('--no-aec');
const VOICE = arg('voice', null);
const CALLER_VOICE = arg('caller', null);
const SCEN = JSON.parse(fs.readFileSync(path.resolve(ROOT, arg('scenarios', 'tests/demo_scenarios.json')), 'utf8'));
const CALLER = path.join(ROOT, 'tests', 'out', 'demo', 'caller', ...(CALLER_VOICE ? [CALLER_VOICE] : []));
// a voice's name as the app's list shows it, without the notes in brackets ("Kokoro Fable (UK, male)": "Kokoro Fable")
const nice = (id) => { const v = require(path.join(ROOT, 'src', 'main', 'voices')).listVoices().find((x) => x.id === id); return v ? v.label.replace(/\s*\(.*$/, '') : id; };
// the caption's model line, with the voices when they are not the usual pair
const LABEL = [MODEL, VOICE && `voice ${nice(VOICE)}`, CALLER_VOICE && `caller ${nice('orpheus:' + CALLER_VOICE)}`].filter(Boolean).join(' · ');
// A scene with "persona" sends the model that persona with the persona prompt (config.js personaPrompt), whatever
// prompt the adapter was trained with: a demo of what the prompt alone does. The app gives it only to persona adapters,
// so the recorder holds the prompt kind at "persona" for that scene (the app's own assignments are kept underneath).
let personaScene = false;
Object.defineProperty(require(path.join(ROOT, 'src', 'main', 'llama')).Llama.prototype, 'promptKind', {
  configurable: true, get() { return personaScene ? 'persona' : this._promptKind; }, set(v) { this._promptKind = v; },
});
const OUT = path.join(ROOT, 'tests', 'out', 'demo', TAG);
fs.mkdirSync(OUT, { recursive: true });
app.setPath('userData', path.join(ROOT, 'tests', 'out', 'demo', 'userdata'));

// The browser writes WebM without a duration (it streams), so players cannot seek and editors guess. This adds a
// Duration element to the Segment's Info. EBML: each element is an ID, a variable-length size, then its content.
function fixWebmDuration(buf, ms) {
  const vint = (o) => { // { len, value, unknown } of the variable-length integer at o
    const len = Math.clz32(buf[o]) - 23; // leading zeros of the first byte, + 1
    let v = BigInt(buf[o] & (0xff >> len));
    for (let i = 1; i < len; i++) v = (v << 8n) | BigInt(buf[o + i]);
    return { len, value: v, unknown: v === (1n << BigInt(7 * len)) - 1n };
  };
  const enc = (v, len) => { const b = Buffer.alloc(len); let x = v | (1n << BigInt(7 * len)); for (let i = len - 1; i >= 0; i--) { b[i] = Number(x & 0xffn); x >>= 8n; } return b; };
  const idLen = (o) => Math.clz32(buf[o]) - 23;
  let o = 4; const ebmlSize = vint(o); o += ebmlSize.len + Number(ebmlSize.value); // past the EBML header
  if (buf.readUInt32BE(o) !== 0x18538067) return buf; // not a Segment
  const segSizeAt = o + 4, seg = vint(segSizeAt); let c = segSizeAt + seg.len;
  while (c < buf.length) {
    const il = idLen(c), id = buf.readUIntBE(c, il), sz = vint(c + il), body = c + il + sz.len;
    if (id === 0x1549a966) { // Info
      let scale = 1000000;
      for (let k = body; k < body + Number(sz.value);) {
        const kl = idLen(k), kid = buf.readUIntBE(k, kl), ks = vint(k + kl), kb = k + kl + ks.len;
        if (kid === 0x4489) return buf; // has one already
        if (kid === 0x2ad7b1) scale = buf.readUIntBE(kb, Number(ks.value));
        k = kb + Number(ks.value);
      }
      const dur = Buffer.alloc(11); dur[0] = 0x44; dur[1] = 0x89; dur[2] = 0x88; dur.writeDoubleBE(ms * 1e6 / scale, 3);
      const end = body + Number(sz.value);
      const parts = [buf.subarray(0, c + il), enc(sz.value + 11n, sz.len), buf.subarray(body, end), dur, buf.subarray(end)];
      if (!seg.unknown) parts.splice(0, 1, buf.subarray(0, segSizeAt), enc(seg.value + 11n, seg.len), buf.subarray(segSizeAt + seg.len, c + il));
      return Buffer.concat(parts);
    }
    if (sz.unknown) break;
    c = body + Number(sz.value);
  }
  return buf;
}

// In the page: a stand-in microphone fed by the caller's lines, a recorder of the window with both voices mixed, the
// caption bar, and the caller's turn-taking (wait for the assistant to talk or to go quiet). Classic-script globals of
// app.js (audioCtx, voiceOut, playing, status) are in scope.
const AUDIBLE = process.argv.includes('--audible') || process.argv.includes('--speakers');
// --speakers: also record what reached the speakers (Windows loopback: everything the PC plays), to find the beep
const SPEAKERS = process.argv.includes('--speakers');
const PAGE = String.raw`
window.demo = ((AUDIBLE, SPEAKERS) => {
  const ctx = audioCtx();
  // silent in the room unless --audible (the recording has both voices either way)
  if (!AUDIBLE && ctx.setSinkId) ctx.setSinkId({ type: 'none' }).catch(() => {});
  const recDest = ctx.createMediaStreamDestination();
  voiceOut().connect(recDest);
  const callerGain = ctx.createGain();
  callerGain.connect(recDest);
  if (AUDIBLE) callerGain.connect(ctx.destination);
  let micDest = null;
  const realGUM = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (c) => {
    if (!c || !c.audio || c.video) return realGUM(c);
    if (micDest) { try { callerGain.disconnect(micDest); } catch {} }
    micDest = ctx.createMediaStreamDestination(); // a fresh one per call: ending a call stops the stream's track
    callerGain.connect(micDest);
    return micDest.stream;
  };
  let talkingSince = 0, quietSince = performance.now(), spokeSinceLine = false;
  setInterval(() => {
    const now = performance.now();
    if (playing.size > 0) { if (!talkingSince) talkingSince = now; quietSince = 0; spokeSinceLine = true; }
    else { talkingSince = 0; if (!quietSince) quietSince = now; }
  }, 25);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const b64of = async (blob) => {
    const buf = new Uint8Array(await blob.arrayBuffer());
    let s = '';
    for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    return btoa(s);
  };
  const ticks = [];
  api.onTick((t) => ticks.push({ ...t, at: Math.round(performance.now()) }));
  let rec = null, chunks = [], t0 = 0, spk = null, spkChunks = [];
  const events = [];
  return {
    async caption(title, text, model) {
      document.getElementById('demo-bar')?.remove();
      const bar = document.createElement('div');
      bar.id = 'demo-bar';
      bar.innerHTML = '<div class="t"></div><div class="c"></div><div class="m"></div>';
      bar.querySelector('.t').textContent = title;
      bar.querySelector('.c').textContent = text;
      bar.querySelector('.m').textContent = model;
      Object.assign(bar.style, { display: 'grid', gridTemplateColumns: '1fr auto', gridTemplateRows: 'auto auto', columnGap: '24px',
        padding: '14px 20px 12px', background: '#1d2733', borderBottom: '2px solid #4aa3ff', color: '#e8eef5', flex: '0 0 auto' });
      Object.assign(bar.querySelector('.t').style, { fontSize: '26px', fontWeight: '700', letterSpacing: '0.01em' });
      Object.assign(bar.querySelector('.c').style, { fontSize: '17px', color: '#b8c6d6', gridColumn: '1', marginTop: '4px' });
      Object.assign(bar.querySelector('.m').style, { gridColumn: '2', gridRow: '1 / span 2', alignSelf: 'center', fontSize: '15px',
        color: '#8fc3ff', border: '1px solid #3b5675', borderRadius: '6px', padding: '6px 12px' });
      document.body.prepend(bar);
      document.getElementById('log').textContent = '';
      return true;
    },
    async load(b64) {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      return ctx.decodeAudioData(bytes.buffer);
    },
    async startRecording() {
      await ctx.resume();
      const screen = await navigator.mediaDevices.getDisplayMedia({ video: { width: 1920, height: 1080, frameRate: 30 }, audio: SPEAKERS });
      if (SPEAKERS) {
        spk = new MediaRecorder(new MediaStream(screen.getAudioTracks()), { mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 256e3 });
        spkChunks = [];
        spk.ondataavailable = (e) => { if (e.data.size) spkChunks.push(e.data); };
        spk.start(1000);
      }
      const stream = new MediaStream([...screen.getVideoTracks(), ...recDest.stream.getAudioTracks()]);
      // VP9 in WebM, encoded on the CPU: H.264 went to the GPU's encoder and came out as garbage with the models loaded
      const types = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/mp4'];
      const mimeType = types.find((t) => MediaRecorder.isTypeSupported(t));
      rec = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 10e6, audioBitsPerSecond: 192e3 });
      chunks = []; ticks.length = 0; events.length = 0;
      rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      rec.start(1000);
      t0 = performance.now();
      return mimeType;
    },
    async stopRecording() {
      const done = new Promise((r) => { rec.onstop = r; });
      rec.stop();
      await done;
      let speakers = null;
      if (spk) {
        const d2 = new Promise((r) => { spk.onstop = r; });
        spk.stop(); await d2;
        speakers = await b64of(new Blob(spkChunks, { type: spk.mimeType }));
      }
      rec.stream.getVideoTracks().forEach((t) => t.stop());
      return { b64: await b64of(new Blob(chunks, { type: rec.mimeType })), ms: Math.round(performance.now() - t0), speakers, ticks: ticks.map((t) => ({ ...t, at: t.at - Math.round(t0) })), events,
        transcript: [...document.querySelectorAll('#transcript p')].map((p) => p.className + ': ' + p.textContent) };
    },
    async play(buffer, turn) {
      const now = () => performance.now();
      if (turn.whenAssistantTalkingMs) {
        const until = now() + 20000;
        while (!(talkingSince && now() - talkingSince >= turn.whenAssistantTalkingMs) && now() < until) await sleep(20);
      }
      if (turn.afterAssistantQuietMs) {
        const until = now() + (turn.maxWaitMs || 10000);
        while (!(spokeSinceLine && quietSince && now() - quietSince >= turn.afterAssistantQuietMs) && now() < until) await sleep(20);
      }
      events.push({ say: turn.say, at: Math.round(now() - t0) });
      spokeSinceLine = false;
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(callerGain);
      await new Promise((r) => { src.onended = r; src.start(); });
      events[events.length - 1].end = Math.round(now() - t0);
      await sleep(turn.pause || 0);
    },
    callLive: () => status.call === 'live',
  };
})(AUDIBLE_FLAG, SPEAKERS_FLAG);
true`.replace('AUDIBLE_FLAG', String(AUDIBLE)).replace('SPEAKERS_FLAG', String(SPEAKERS));

app.whenReady().then(() => {
  // the window records itself (its own page, whatever covers it on screen)
  session.defaultSession.setDisplayMediaRequestHandler((req, cb) => cb({ video: req.frame, ...(SPEAKERS ? { audio: 'loopback' } : {}) }));
});

app.on('browser-window-created', (_e, win) => {
  win.webContents.setBackgroundThrottling(false);
  win.webContents.once('did-finish-load', () => setTimeout(async () => {
    if (!/index.html/.test(win.webContents.getURL())) return;
    win.setContentSize(1760, 990); // 16:9
    const js = (s) => win.webContents.executeJavaScript(s, true);
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    try {
      await js(`api.useAdapter(${JSON.stringify(ADAPTER)})`);
      if (THRESHOLD) await js(`api.rememberThreshold(${Number(THRESHOLD)})`);
      await js(`api.updateSettings({ aec: { enabled: ${!NO_AEC} }${VOICE ? ', tts: { voice: ' + JSON.stringify(VOICE) + ' }' : ''} })`);
      await js('api.getVoices().then(renderVoices)'); // the voice list was drawn with the saved voice
      const t0 = Date.now();
      await js('api.startServices()');
      console.log(`models ready in ${((Date.now() - t0) / 1000).toFixed(0)} s (${path.basename(ADAPTER)})`);
      await js(PAGE);
      for (const s of SCEN.scenarios.filter((x) => !ONLY.length || ONLY.includes(x.name))) {
        personaScene = !!s.persona;
        await js(`api.updateSettings({ persona: { text: ${JSON.stringify(s.persona || '')}, memories: '' } })`);
        await js(`demo.caption(${JSON.stringify(s.title)}, ${JSON.stringify(s.caption)}, ${JSON.stringify(LABEL)})`);
        const bufs = [];
        for (let i = 0; i < s.turns.length; i++) {
          const b64 = fs.readFileSync(path.join(CALLER, `${s.name}-${i}.wav`)).toString('base64');
          await js(`demo.load("${b64}").then((b) => { (window.demoBufs = window.demoBufs || [])[${i}] = b; true; })`);
          bufs.push(i);
        }
        const mime = await js('demo.startRecording()');
        await wait(1500);
        await js(`document.getElementById('btn-call').click(); true`);
        for (let k = 0; k < 50 && !(await js('demo.callLive()')); k++) await wait(100);
        await wait(1500);
        for (let i = 0; i < s.turns.length; i++) await js(`demo.play(window.demoBufs[${i}], ${JSON.stringify(s.turns[i])})`);
        await js(`document.getElementById('btn-call').click(); true`);
        await wait(1500);
        const r = await js('demo.stopRecording()');
        const ext = /mp4/.test(mime) ? 'mp4' : 'webm';
        const video = Buffer.from(r.b64, 'base64');
        fs.writeFileSync(path.join(OUT, `${s.name}.${ext}`), ext === 'webm' ? fixWebmDuration(video, r.ms) : video);
        if (r.speakers) fs.writeFileSync(path.join(OUT, `${s.name}.speakers.webm`), Buffer.from(r.speakers, 'base64'));
        fs.writeFileSync(path.join(OUT, `${s.name}.json`), JSON.stringify({ adapter: path.basename(ADAPTER), model: MODEL, voice: VOICE, caller: CALLER_VOICE, scenario: s.name, persona: s.persona || '', callerLines: r.events, transcript: r.transcript, ticks: r.ticks }, null, 1));
        console.log(`${s.name}: ${ext}, ${(r.b64.length * 0.75 / 1e6).toFixed(1)} MB | ${r.transcript.join(' | ').slice(0, 300)}`);
        await js('window.demoBufs = []; true');
        await wait(1000);
      }
      await js('api.stopServices()');
    } catch (e) { console.log('ERROR', e.message); }
    app.quit();
  }, 2500));
});

require(path.join(ROOT, 'src', 'main', 'index.js'));
