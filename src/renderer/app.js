/* global talkme */
'use strict';
const $ = (s) => document.querySelector(s);
const api = window.talkme;

// ---------- tabs ----------
document.querySelectorAll('nav button').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('nav button').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === 'tab-' + b.dataset.tab));
  if (b.dataset.tab === 'adapters') loadCatalog();
}));

// ---------- status / log ----------
let status = {};
function renderStatus(s) {
  // a call that ended without the button (the main process ends it, e.g. switching to an adapter on another base)
  // releases the mic too
  if (status.call === 'live' && s.call !== 'live' && typeof stopMic === 'function') stopMic();
  status = s;
  document.querySelectorAll('.pill').forEach((p) => { p.className = 'pill ' + (s[p.dataset.svc] || ''); p.textContent = `${p.dataset.svc}: ${s[p.dataset.svc]}`; });
  const ready = s.llama === 'ready' && s.stt === 'ready' && s.tts === 'ready';
  $('#btn-call').disabled = !ready && s.call !== 'live';
  $('#btn-call').textContent = s.call === 'live' ? 'End call' : 'Start call';
  $('#btn-services').textContent = (s.llama === 'stopped' && s.stt === 'stopped') ? 'Load models' : 'Stop services';
}
api.onStatus(renderStatus);
api.getStatus().then(renderStatus);
const logEl = $('#log');
api.onLog((line) => { logEl.textContent += line + '\n'; if (logEl.textContent.length > 60000) logEl.textContent = logEl.textContent.slice(-40000); logEl.scrollTop = logEl.scrollHeight; });

$('#btn-services').addEventListener('click', async () => {
  try {
    if (status.llama === 'stopped' && status.stt === 'stopped') await api.startServices(); else { stopMic(); await api.stopServices(); }
  } catch (e) { alert(e.message); }
});

// ---------- settings ----------
let settings = null;
function getPath(o, p) { return p.split('.').reduce((a, k) => (a == null ? a : a[k]), o); }
function setPath(o, p, v) { const ks = p.split('.'); let cur = o; for (const k of ks.slice(0, -1)) cur = cur[k] = cur[k] || {}; cur[ks[ks.length - 1]] = v; }
async function loadSettings() {
  settings = await api.getSettings();
  document.querySelectorAll('#settings-form [name]').forEach((i) => { if (i.type === 'checkbox') i.checked = !!getPath(settings, i.name); else i.value = getPath(settings, i.name) ?? ''; });
  $('#thr').value = settings.listenThreshold; $('#thr-v').textContent = settings.listenThreshold;
  showBar(settings.carryOnP);
  showSampling(!!settings.listenSample);
}
loadSettings();
$('#settings-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const patch = {};
  document.querySelectorAll('#settings-form [name]').forEach((i) => { setPath(patch, i.name, i.type === 'checkbox' ? i.checked : i.type === 'number' ? Number(i.value) : i.value); });
  settings = await api.updateSettings(patch);
  api.callSettings({ tickMs: settings.tickMs, listenThreshold: settings.listenThreshold, ttsFlushWords: settings.ttsFlushWords, ttsSpeed: settings.ttsSpeed });
  $('#thr').value = settings.listenThreshold; $('#thr-v').textContent = settings.listenThreshold;
});
document.querySelectorAll('[data-pick]').forEach((b) => b.addEventListener('click', async () => {
  const f = await api.pickFile(b.dataset.dir ? { properties: ['openDirectory', 'createDirectory'] } : {});
  if (f) document.querySelector(`[name="${b.dataset.pick}"]`).value = f;
}));
// "human timing": the threshold and the bar are unused while the decision is sampled, so the sliders dim
function showSampling(on) {
  $('#sample').checked = on;
  for (const s of ['#thr', '#bar']) { $(s).disabled = on; $(s).title = on ? 'not used while human timing is on' : ''; }
}
function showBar(v) { $('#bar').value = v; $('#bar-v').textContent = v; }
$('#bar').addEventListener('input', () => { const v = Number($('#bar').value); $('#bar-v').textContent = v; api.setCarryOn(v); });
api.onCarryOn(showBar);
$('#sample').addEventListener('change', async () => { showSampling($('#sample').checked); await api.setSampling($('#sample').checked); });
api.onSampling((on) => showSampling(!!on));
$('#thr').addEventListener('input', async () => {
  const v = Number($('#thr').value); $('#thr-v').textContent = v; if (thrField()) thrField().value = v;
  await api.updateSettings({ listenThreshold: v }); api.callSettings({ listenThreshold: v }); api.rememberThreshold(v);
});

// ---------- adapters ----------
const normPath = (p) => (p || '').toLowerCase().split('\\').join('/');
// the Model dropdown lists the bases on disk; the Adapter dropdown only the adapters (and blends) trained on the picked one
function renderAdapters({ models = [], model, adapters = [], active, blends = [] }) {
  const fill = (sel, opts, selected) => {
    sel.innerHTML = '';
    for (const [v, label, title, off] of opts) {
      const o = document.createElement('option');
      o.value = v; o.textContent = label; o.selected = selected(v);
      if (title) o.title = title;
      o.disabled = !!off;
      sel.appendChild(o);
    }
  };
  fill($('#model'), models.map((m) => [m.name, m.label]), (v) => v === model);
  const file = (a) => a.path.split(/[\\/]/).pop();
  const own = adapters.filter((a) => a.base === model);
  // its name from the file itself (adapter_meta.py), else the file name; one that names no model is shown, greyed out,
  // under the models of its architecture
  const arch = (models.find((m) => m.name === model) || {}).arch;
  const unknown = adapters.filter((a) => !a.base && a.arch && a.arch === arch);
  fill($('#adapter'), [['', 'none (the model alone)'], ...own.map((a) => [a.path, a.label || file(a).replace(/\.gguf$/, ''), file(a)]),
    ...blends.filter((b) => b.base === model).map((b) => [b.spec, 'blend: ' + b.label]),
    ...unknown.map((a) => [a.path, `${file(a).replace(/\.gguf$/, '')} (which model? not recorded)`,
      `${file(a)} does not say which model it was trained on. Write it in with tune/adapter_meta.py (--base, --prompt), or end its name with the model's key, e.g. -gemma4a4b.gguf`, true])],
  (v) => normPath(v) === normPath(active || ''));
  $('#model').disabled = models.length < 2;
}
api.onAdapters(renderAdapters);
// the Settings tab's field follows too: left stale, its Save sent the old value back over the adapter's own threshold
const thrField = () => document.querySelector('#settings-form [name="listenThreshold"]');
api.onThreshold((v) => { $('#thr').value = v; $('#thr-v').textContent = v; if (thrField()) thrField().value = v; });
api.getAdapters().then(renderAdapters);
$('#adapter').addEventListener('change', async () => { try { await api.useAdapter($('#adapter').value || null); } catch (e) { alert(e.message); } });
$('#model').addEventListener('change', async () => {
  $('#model').disabled = $('#adapter').disabled = true; // the server restarts on the new model (30-60 s)
  try { renderAdapters(await api.useModel($('#model').value)); } catch (e) { alert(e.message); api.getAdapters().then(renderAdapters); }
  finally { $('#adapter').disabled = false; }
});

// ---------- voices ----------
function renderVoices({ voices, active }) {
  const sel = $('#voice');
  sel.innerHTML = '';
  for (const v of voices) {
    const o = document.createElement('option');
    o.value = v.id; o.textContent = v.label; o.selected = v.id === active;
    sel.appendChild(o);
  }
}
api.getVoices().then(renderVoices);
// ---------- new Chatterbox voice: upload a clip or record one ----------
const vmsg = (t) => { $('#voice-msg').textContent = t; };
$('#voice-new').addEventListener('click', () => { $('#voice-panel').hidden = !$('#voice-panel').hidden; });
async function voiceAdded(r) {
  renderVoices(r);
  vmsg(`added ${r.added}; switching to it (the first time takes a few seconds)`);
  try { renderVoices(await api.useVoice(r.added)); vmsg(`now speaking as ${r.added}`); } catch (e) { vmsg(e.message); }
}
$('#voice-upload').addEventListener('click', async () => {
  const name = $('#voice-name').value.trim();
  if (!name) return vmsg('give the voice a name first');
  const file = await api.pickFile({ filters: [{ name: 'Audio', extensions: ['wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'] }] });
  if (!file) return;
  vmsg('reading the clip...');
  try { await voiceAdded(await api.addVoiceFile(name, file)); } catch (e) { vmsg(e.message.replace(/^.*Error: /, '')); }
});
let voiceRec = null;
$('#voice-record').addEventListener('click', async () => {
  if (voiceRec) { voiceRec.stop(); return; }
  const name = $('#voice-name').value.trim();
  if (!name) return vmsg('give the voice a name first');
  try {
    const dev = $('#mic-device').value;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: dev ? { exact: dev } : undefined, echoCancellation: false, noiseSuppression: true } });
    const chunks = [];
    voiceRec = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    voiceRec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    voiceRec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(tick); voiceRec = null; $('#voice-record').textContent = 'Record 10 s';
      vmsg('saving...');
      try { await voiceAdded(await api.addVoiceBytes(name, await new Blob(chunks).arrayBuffer(), 'webm')); } catch (e) { vmsg(e.message.replace(/^.*Error: /, '')); }
    };
    voiceRec.start();
    let left = 10;
    $('#voice-record').textContent = 'Stop';
    vmsg(`recording... talk naturally for ${left} s`);
    const tick = setInterval(() => { left--; vmsg(`recording... ${left} s`); if (left <= 0 && voiceRec) voiceRec.stop(); }, 1000);
  } catch (e) { voiceRec = null; vmsg('could not record: ' + e.message); }
});
$('#voice-remove').addEventListener('click', async () => {
  const id = $('#voice').value;
  try { renderVoices(await api.removeVoice(id)); vmsg(`removed ${id}`); } catch (e) { vmsg(e.message.replace(/^.*Error: /, '')); }
});
$('#voice').addEventListener('change', async () => {
  const id = $('#voice').value;
  // a GPU voice that is not running yet loads its model: up to a minute (switching within one engine is instant)
  // (the panel opens for the message and closes again once the voice is up, unless it was open already)
  const wasHidden = $('#voice-panel').hidden;
  if (/^(orpheus|chatterbox):/.test(id)) { $('#voice-panel').hidden = false; $('#voice-msg').textContent = 'starting the voice (a GPU voice can take up to a minute the first time)...'; }
  $('#voice').disabled = true;
  try { renderVoices(await api.useVoice(id)); $('#voice-msg').textContent = ''; $('#voice-panel').hidden = wasHidden; } catch (e) { $('#voice-msg').textContent = e.message; alert(e.message); }
  $('#voice').disabled = false;
});

// ---------- microphone ----------
// Input device picker. The choice is remembered in this window's storage; '' means the Windows default device.
const MIC_KEY = 'talkme.mic';
const savedMic = () => { try { return localStorage.getItem(MIC_KEY) || ''; } catch { return ''; } };
async function listMics() {
  const sel = $('#mic-device');
  let devices = [];
  try { devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput' && d.deviceId !== 'default' && d.deviceId !== 'communications'); } catch {}
  const want = sel.value || savedMic();
  sel.innerHTML = '';
  const def = document.createElement('option'); def.value = ''; def.textContent = 'Windows default'; sel.appendChild(def);
  devices.forEach((d, i) => {
    const o = document.createElement('option');
    o.value = d.deviceId; o.textContent = d.label || `Microphone ${i + 1}`; o.selected = d.deviceId === want;
    sel.appendChild(o);
  });
}
listMics();
navigator.mediaDevices.addEventListener?.('devicechange', listMics);
$('#mic-device').addEventListener('change', async () => {
  try { localStorage.setItem(MIC_KEY, $('#mic-device').value); } catch {}
  if (status.call === 'live' && $('#mic-on').checked) { stopMic(); await startMic(); } // switch live, mid-call
});
let ctx = null, micNode = null, micStream = null, micSrc = null;
// Output device picker: the assistant's voice plays on the chosen device ('' = Windows default), remembered here.
const SPK_KEY = 'talkme.speaker';
const savedSpk = () => { try { return localStorage.getItem(SPK_KEY) || ''; } catch { return ''; } };
function audioCtx() {
  if (!ctx) {
    ctx = new AudioContext();
    const id = $('#spk-device').value || savedSpk();
    if (id && ctx.setSinkId) ctx.setSinkId(id).catch((e) => console.warn('output device', e));
  }
  return ctx;
}
async function listSpeakers() {
  const sel = $('#spk-device');
  let devices = [];
  try { devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audiooutput' && d.deviceId !== 'default' && d.deviceId !== 'communications'); } catch {}
  const want = sel.value || savedSpk();
  sel.innerHTML = '';
  const def = document.createElement('option'); def.value = ''; def.textContent = 'Windows default'; sel.appendChild(def);
  devices.forEach((d, i) => {
    const o = document.createElement('option');
    o.value = d.deviceId; o.textContent = d.label || `Output ${i + 1}`; o.selected = d.deviceId === want;
    sel.appendChild(o);
  });
}
listSpeakers();
navigator.mediaDevices.addEventListener?.('devicechange', listSpeakers);
$('#spk-device').addEventListener('change', async () => {
  const id = $('#spk-device').value;
  try { localStorage.setItem(SPK_KEY, id); } catch {}
  if (ctx && ctx.setSinkId) { try { await ctx.setSinkId(id); } catch (e) { alert('Could not switch output: ' + e.message); } }
});
async function startMic() {
  if (micNode || !$('#mic-on').checked) return;
  audioCtx();
  await ctx.resume();
  const deviceId = $('#mic-device').value;
  // With our own echo canceller running (main/aec.js), the mic comes in untouched: Chromium's canceller does not see
  // the voice we play through Web Audio, and its gain control would keep changing the echo ours is learning. Ours
  // does the noise suppression and gain control after cancelling, as WebRTC orders them.
  const ownAec = status.aec === 'ready';
  micStream = await navigator.mediaDevices.getUserMedia({ audio: {
    ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
    channelCount: 1, echoCancellation: !ownAec, noiseSuppression: !ownAec, autoGainControl: !ownAec } });
  listMics(); listSpeakers(); // device names are only visible once the mic has been opened
  await ctx.audioWorklet.addModule('worklet.js');
  const src = micSrc = ctx.createMediaStreamSource(micStream);
  micNode = new AudioWorkletNode(ctx, 'mic-processor', { numberOfInputs: 2 });
  micNode.port.onmessage = (e) => {
    const { mic, ref } = e.data; api.sendAudio(mic, ref);
    let peak = 0, sq = 0;
    for (let i = 0; i < mic.length; i += 4) { const v = mic[i]; peak = Math.max(peak, Math.abs(v)); sq += v * v; }
    $('#mic-level').style.setProperty('--lvl', Math.min(100, peak * 300) + '%');
    if (!ownAec) duckOnVoice(Math.sqrt(sq / (mic.length / 4)));
  };
  src.connect(micNode, 0, 0); // not connected to destination: no monitoring
  voiceOut().connect(micNode, 0, 1); // what the speaker plays, after ducking: the echo canceller's reference
  // the part of the speaker-to-mic delay the browser knows about; the canceller finds the rest itself
  api.aecDelay(Math.round(1000 * ((ctx.baseLatency || 0) + (ctx.outputLatency || 0))) + 10);
}
// The capture node must be unplugged on both sides: wired to the speaker tap (the echo canceller's reference) it stayed
// alive after a call and kept posting frames of silence, interleaved with the next call's mic (no words recognised).
function stopMic() {
  if (micNode) {
    micNode.port.onmessage = null;
    if (micSrc) { try { micSrc.disconnect(); } catch {} micSrc = null; }
    if (voiceGain) { try { voiceGain.disconnect(micNode); } catch {} }
    micNode.disconnect();
    micNode = null;
  }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  $('#mic-level').style.setProperty('--lvl', '0%');
}
$('#mic-on').addEventListener('change', () => { if (status.call === 'live') ($('#mic-on').checked ? startMic() : stopMic()); });
// Echo test: with the models loaded and no call running, play one sentence on the speaker with the mic open and see how
// many of its words the recogniser picks up. Through headphones or with the echo canceller working: none.
$('#echo-test').addEventListener('click', async () => {
  if (status.call === 'live' || status.stt !== 'ready' || status.tts !== 'ready') { $('#echo-result').textContent = 'load the models, outside a call'; return; }
  $('#echo-test').disabled = true; $('#echo-result').textContent = 'playing a sentence; stay quiet...';
  const hadMic = !!micNode;
  try {
    if (!hadMic) { const on = $('#mic-on').checked; $('#mic-on').checked = true; await startMic(); $('#mic-on').checked = on; }
    const r = await api.echoTest();
    $('#echo-result').textContent = r.words === 0 ? `heard none of it (echo canceller ${status.aec})`
      : `heard ${r.words} words of ${r.of}: "${r.heard}" (echo canceller ${status.aec})`;
  } catch (e) { $('#echo-result').textContent = 'echo test failed: ' + e.message; }
  finally { if (!hadMic) stopMic(); $('#echo-test').disabled = false; }
});

// ---------- ducking ----------
// The moment the mic hears you while the assistant is talking, its voice drops to a quarter of its volume, so a cut-in
// feels immediate. Whether it actually stops is still the model's decision on its next tick (then playback is cut as
// before); if it keeps going, the volume comes back once you have been quiet for 0.4 s. Speech is "louder than the
// room": a slowly tracked noise floor times four, so fans and hum don't trigger it.
let voiceGain = null, noiseFloor = 0.005, lastVoiceAt = 0, ducked = false;
const DUCK_KEY = 'talkme.duck';
try { const v = localStorage.getItem(DUCK_KEY); if (v !== null) $('#duck').checked = v === '1'; } catch {}
$('#duck').addEventListener('change', () => { try { localStorage.setItem(DUCK_KEY, $('#duck').checked ? '1' : '0'); } catch {} if (!$('#duck').checked) setDuck(false); });
function voiceOut() {
  audioCtx();
  if (!voiceGain) { voiceGain = ctx.createGain(); voiceGain.connect(ctx.destination); }
  return voiceGain;
}
function setDuck(on) {
  if (!voiceGain || on === ducked) return;
  ducked = on;
  const t = ctx.currentTime;
  voiceGain.gain.cancelScheduledValues(t);
  voiceGain.gain.setTargetAtTime(on ? 0.25 : 1.0, t, on ? 0.02 : 0.08);
}
// with the echo canceller running, duck on the mic it has cleaned: the raw mic hears the assistant's own voice on
// speakers, which would duck it every time it talks
api.onAecLevel((rms) => { if (micNode) duckOnVoice(rms); });
function duckOnVoice(rms) {
  const speaking = rms > Math.max(0.015, noiseFloor * 4);
  if (!speaking) noiseFloor = noiseFloor * 0.98 + Math.min(rms, 0.05) * 0.02; // adapt to the room while you're quiet
  const now = performance.now();
  if (speaking) lastVoiceAt = now;
  if (!$('#duck').checked || playing.size === 0) { setDuck(false); return; }
  if (speaking) setDuck(true);
  else if (now - lastVoiceAt > 400) setDuck(false);
}

// ---------- playback queue ----------
let playHead = 0; const playing = new Set();
function reportPlaying() { api.setPlaying(playing.size > 0); if (playing.size === 0) setDuck(false); }
// The beep heard between pieces (2026-10-28, found by replaying saved voice parts): a buffer at another sample rate
// than the context's goes through Chromium's own resampler, and when it ends part-way through a render quantum
// (Electron 44) the graph keeps repeating that last quantum for ~0.25 s instead of going silent. A piece's last part
// ends on a fade, so that repeat was silent; a streamed part ends mid-word, loud. So every part is brought to the
// context's rate here (linear interpolation, what Chromium's resampler does) and Chromium never resamples.
function atContextRate(f32, rate) {
  const sr = ctx.sampleRate;
  if (rate === sr || !f32.length) return f32;
  const n = Math.round(f32.length * sr / rate), out = new Float32Array(n), step = rate / sr, last = f32.length - 1;
  for (let i = 0; i < n; i++) {
    const p = Math.min(i * step, last), j = Math.floor(p), f = p - j;
    out[i] = f32[j] * (1 - f) + f32[Math.min(j + 1, last)] * f;
  }
  return out;
}
api.onTtsAudio(({ samples, sampleRate, text }) => {
  audioCtx();
  const f32 = atContextRate(samples instanceof Float32Array ? samples : new Float32Array(samples.buffer || samples), sampleRate);
  const buf = ctx.createBuffer(1, f32.length, ctx.sampleRate);
  buf.copyToChannel(f32, 0);
  const src = ctx.createBufferSource(); src.buffer = buf; src.connect(voiceOut());
  const at = Math.max(ctx.currentTime + 0.02, playHead);
  src.start(at); playHead = at + buf.duration;
  playing.add(src); reportPlaying();
  src.onended = () => { playing.delete(src); reportPlaying(); };
  if (text) addTranscript('a', text); // a streamed piece carries its text on its first part only
});
// "[stopped]" only when some of it was heard: a stop before the first sound (it began a reply, then thought better of
// it) left the marker on a line of its own
api.onAudioStop(() => {
  const heard = playing.size > 0;
  for (const s of playing) { try { s.stop(); } catch {} }
  playing.clear(); playHead = 0; reportPlaying();
  if (heard) addTranscript('a', '[stopped]');
});

// ---------- call ----------
$('#btn-call').addEventListener('click', async () => {
  try {
    if (status.call === 'live') { await api.stopCall(); stopMic(); }
    else { $('#tape tbody').innerHTML = ''; $('#transcript').innerHTML = ''; await api.startCall(); await startMic(); }
  } catch (e) { alert(e.message); }
});
$('#inject').addEventListener('keydown', (e) => { if (e.key === 'Enter' && $('#inject').value.trim()) { api.inject($('#inject').value); $('#inject').value = ''; } });

// ---------- transcript + tape ----------
const tr = $('#transcript');
let lastRole = null, lastP = null, lastUserP = null, partialEl = null;
const bare = (s) => s.toLowerCase().replace(/[^a-z0-9' ]/g, '').replace(/\s+/g, ' ').trim();
function addTranscript(role, text, joins = false) {
  if (partialEl) { partialEl.remove(); partialEl = null; }
  // the recogniser closes a quiet utterance with "." or "?" a moment later, often while the ai is already talking:
  // it belongs at the end of the user's last line, not on a line of its own
  if (role === 'u' && /^[.?!,]+$/.test(text.trim()) && lastUserP) {
    lastUserP.textContent = lastUserP.textContent.replace(/[.?!,]*$/, '') + text.trim();
    return;
  }
  // punctuation that arrived in a later tick (". So", "?") belongs to the previous word on screen
  // a gap of more than 2.5 s starts a new line, so pauses show in the transcript
  const gap = Date.now() - (addTranscript.lastAt || 0) > 2500;
  addTranscript.lastAt = Date.now();
  // joins: the rest of a word whose start came last tick ("dis" + "aster")
  if (role === lastRole && lastP && !gap) { lastP.textContent = (lastP.textContent + (joins ? '' : ' ') + text).replace(/\s+([.,!?;:])/g, '$1'); }
  else { lastP = document.createElement('p'); lastP.className = role; lastP.textContent = (role === 'u' ? 'you: ' : 'ai: ') + text; tr.appendChild(lastP); lastRole = role; }
  if (role === 'u') lastUserP = lastP;
  tr.scrollTop = tr.scrollHeight;
}
api.onPartial((text) => {
  // a partial repeating words already on the user's last line is the recogniser catching up, not new speech
  if (!text.trim() || (lastUserP && bare(lastUserP.textContent).endsWith(bare(text)))) {
    if (partialEl) { partialEl.remove(); partialEl = null; }
    return;
  }
  if (!partialEl) { partialEl = document.createElement('p'); partialEl.className = 'partial'; tr.appendChild(partialEl); }
  partialEl.textContent = '… ' + text.split(' ').slice(-12).join(' ');
  tr.scrollTop = tr.scrollHeight;
});
api.onTick((t) => {
  if (t.heard !== '<no voice>') addTranscript('u', t.heard, t.joins);
  const tb = $('#tape tbody');
  const row = document.createElement('tr'); row.className = t.action;
  for (const [text, cls] of [[String(t.n)], [t.heard], [t.said + (t.rerun ? ' *' : '')], [t.pListen == null ? '-' : t.pListen.toFixed(2)], [String(t.llmMs)], [String(t.totalMs), t.totalMs > 600 ? 'over' : '']]) {
    const td = document.createElement('td'); td.textContent = text; td.title = text; if (cls) td.className = cls; row.appendChild(td);
  }
  tb.appendChild(row); if (tb.children.length > 400) tb.removeChild(tb.firstChild);
  $('#tape').scrollTop = $('#tape').scrollHeight;
  $('#tick-stats').textContent = `avg ${t.avgMs} ms · max ${t.maxMs} ms · over budget ${t.over} · history ${t.historyTicks} ticks · cached ${t.cached} tok`;
});
api.onLoopError((m) => addTranscript('a', '[error] ' + m));

// ---------- tune ----------
// recipes in tune/recipes; each stage runs tune/pipeline.py in the main process and streams its output here
const tlog = $('#tune-log');
const tres = $('#tune-results');
const recipeSel = $('#recipe');
let recipeFile = null;
let savedRecipe = null; // the recipe as it is on disk, to tell unsaved changes
api.onTuneLog((line) => { tlog.textContent += line + '\n'; if (tlog.textContent.length > 120000) tlog.textContent = tlog.textContent.slice(-80000); tlog.scrollTop = tlog.scrollHeight; });
api.onTuneDone(({ stage, recipe, code }) => {
  tlog.textContent += `\n[${stage} of ${recipe} finished with code ${code}]\n`;
  if (stage === 'download') { tuneBases = []; renderRecipeForm(); fillBaseDownloads(recipe); } // a base came onto the disk
  tuneState(); showResults();
  if ($('#tab-adapters').classList.contains('active')) loadCatalog(); // a run's new results or adapter
});
async function loadRecipes(select) {
  const list = await api.tuneRecipes();
  recipeSel.innerHTML = '';
  for (const r of list) { const o = document.createElement('option'); o.value = r.file; o.textContent = `${r.name}  (${r.file})`; o.title = r.about; recipeSel.appendChild(o); }
  recipeFile = select || recipeFile || (list[0] && list[0].file);
  if (recipeFile) { recipeSel.value = recipeFile; await loadRecipe(); }
}
async function loadRecipe() {
  recipeFile = recipeSel.value;
  $('#recipe-text').value = savedRecipe = await api.tuneRecipe(recipeFile);
  $('#recipe-msg').textContent = ''; // its notes are in the form
  renderRecipeForm();
  showResults();
}

// The form over the recipe: the settings worth changing by hand, written straight back into the JSON (which holds
// everything else: the data it trains on and how the calls are built). An empty number means the default.
// The loss weights, one per kind of tick in the training calls: how much getting that moment right counts. In the
// order people tend to reach for them, each with what it does and where the tests show it.
const WEIGHT_INFO = [
  ['start', 'Answering when you finish', 'its first words once you are done; higher answers sooner and more surely', 'held-out start; takes the turn'],
  ['listen_pause', 'Waiting through your pauses', 'quiet in a pause in the middle of what you say; higher cuts in less but may answer later', 'mid-turn pause starts; waits through pauses'],
  ['listen_speech', 'Quiet while you talk', 'saying nothing while you are talking; higher talks over you less', 'talks over'],
  ['barge_stop', 'Stopping when you cut in', 'going quiet when you talk over it; higher stops sooner', 'barge; stops when cut in on'],
  ['continuer', 'Back-channels', '"mm-hm", "yeah" while you talk; higher gives more of them', 'filler; back-channels while you talk'],
  ['keep_talking', 'Talking through your back-channels', 'carrying on when you say "mm-hm" over it; higher is less thrown by them', 'keeps going through back-channels'],
  ['speak_on', 'Finishing its thought', 'carrying on after its own beat at a sentence end; higher stalls less', 'stalls; unfinished; carry'],
  ['speak_next', 'Going straight on', 'opening its next sentence with no pause', 'after its own sentence: straight on'],
  ['speak_pause', 'Taking a beat', 'a short pause after its own sentence', 'after its own sentence: a beat'],
  ['listen_done', 'Leaving room after answering', 'waiting for you once it has answered; higher rambles on less', 'answers, then stops'],
  ['nudge', 'Speaking up after a silence', 'saying something when the call goes quiet', 'speaks up after a silence; quiet nudge'],
  ['hazard_start', 'Cutting in with a warning', 'interrupting when you describe something dangerous (mixing bleach)', 'hazard; bleach'],
  ['hazard', 'The rest of the warning', 'the words of that warning', 'hazard'],
  ['speak', 'Its own wording', 'how much it learns the words of its replies; lower keeps more of the base model\'s own voice', 'says the right thing'],
];
const TRAIN_FIELDS = [['train.steps', 'training steps', 600], ['train.lr', 'learning rate', 0.0002], ['train.rank', 'LoRA rank (its size)', 32],
  ['train.batch', 'calls per step', 4], ['train.chunk', 'calls per cache chunk (lower uses less RAM)', 160], ['train.max_tokens', 'longest call (tokens)', 8000],
  ['train.top', 'layers it learns in (the last N)', 4, 'more layers can learn more (what it says, not only when); another number builds its own training cache (about an hour) and takes more GPU memory while training']];
const TEST_FIELDS = [['tests.listen_p', 'listen threshold', 0.6], ['tests.carry_on_p', 'carry-on bar (empty: none)', ''], ['tests.yap', 'recorded calls to replay', 24]];
let recipeObj = null;
let tuneBases = [];
let tuneDefaults = {};
function deletePath(o, p) {
  const ks = p.split('.'); const parent = getPath(o, ks.slice(0, -1).join('.')) ?? (ks.length === 1 ? o : null);
  if (parent) delete parent[ks[ks.length - 1]];
}
function el(tag, props = {}, ...kids) { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; }
async function renderRecipeForm() {
  const form = $('#recipe-form');
  form.innerHTML = '';
  try { recipeObj = JSON.parse($('#recipe-text').value); } catch { form.append(el('p', { className: 'note', textContent: 'the JSON below does not parse; fix it to see the form' })); return; }
  if (!tuneBases.length) [tuneBases, tuneDefaults] = await Promise.all([api.tuneBases(), api.tuneWeights()]);
  // every input writes its value into the recipe at its key (empty: the key goes, so the default applies)
  const bind = (input, key, kind) => {
    const cur = getPath(recipeObj, key);
    if (kind === 'check') input.checked = cur !== false; else input.value = cur == null ? '' : Array.isArray(cur) ? cur.join(', ') : cur;
    input.addEventListener('change', () => {
      const v = kind === 'check' ? input.checked : input.value.trim();
      if (v === '') deletePath(recipeObj, key);
      else setPath(recipeObj, key, kind === 'number' ? Number(v) : kind === 'list' ? v.split(/[\s,]+/).filter(Boolean) : v);
      $('#recipe-text').value = JSON.stringify(recipeObj, null, 2) + '\n';
    });
    return input;
  };
  const row = (label, input, title = '') => el('label', { title }, label, input);
  const num = (key, ph) => bind(el('input', { type: 'number', step: 'any', placeholder: ph === '' ? '' : String(ph) }), key, 'number');

  const base = el('select');
  if (!recipeObj.base) recipeObj.base = 'original';
  // a recipe may name a base by an alias ("original", the first Qwen): its option keeps that name, so nothing changes
  for (const b of tuneBases) {
    base.append(el('option', { value: b.aliases.includes(recipeObj.base) ? recipeObj.base : b.name, disabled: !b.trainable,
      textContent: b.label + (b.trainable ? '' : b.download ? ' (not downloaded: Add a base model, below)' : ' (no training weights on disk)') }));
  }
  const prompt = el('select');
  for (const p of ['partner', 'assistant']) prompt.append(el('option', { value: p, textContent: p === 'partner' ? 'partner (a friend on the phone)' : 'assistant' }));
  if (!recipeObj.prompt) recipeObj.prompt = 'assistant';
  form.append(el('fieldset', {}, el('legend', { textContent: `Run ${recipeObj.name}` }),
    row('model', bind(base, 'base'), 'the base model it learns on; another model builds its own training cache (about an hour)'),
    row('system prompt', bind(prompt, 'prompt')),
    row('name in the app', bind(el('input', { placeholder: recipeObj.name }), 'label'), "what the Call tab's adapter list calls it (written into the adapter file with its model, prompt, threshold and bar); empty: the run's name"),
    el('label', { textContent: 'notes' }), bind(el('textarea', { rows: 3 }), '_about')));

  const weights = el('fieldset', { className: 'wide' }, el('legend', { textContent: 'What it learns (loss weights: how much each moment counts)' }));
  const wrow = (tag, name, what, shows) => el('div', { className: 'w' }, el('span', { className: 'name', textContent: name }),
    num('weights.' + tag, tuneDefaults[tag] ?? ''), el('span', { className: 'what' }, what, shows ? el('em', { textContent: ` (tests: ${shows})` }) : ''));
  for (const [tag, name, what, shows] of WEIGHT_INFO) weights.append(wrow(tag, name, what, shows));
  // the rarer ticks (listening after it stops, the memory moments of persona calls, ...) by their tag names
  const known = new Set(WEIGHT_INFO.map((w) => w[0]));
  const more = el('details', {}, el('summary', { textContent: 'other kinds of tick' }));
  for (const tag of Object.keys(tuneDefaults).filter((t) => !known.has(t))) more.append(wrow(tag, tag, '', ''));
  weights.append(more);
  form.append(weights);

  const training = el('fieldset', {}, el('legend', { textContent: 'Training' }), ...TRAIN_FIELDS.map(([k, label, d, title]) => row(label, num(k, d), title)));
  // gpt-oss keeps its experts in fused tensors PEFT cannot adapt: its expert adapters are a choice of their own (off: attention only)
  if (tuneBases.find((b) => b.name === recipeObj.base || b.aliases.includes(recipeObj.base))?.format === 'harmony') {
    const experts = el('input', { type: 'checkbox', checked: getPath(recipeObj, 'train.experts') === true });
    experts.addEventListener('change', () => {
      if (experts.checked) setPath(recipeObj, 'train.experts', true); else deletePath(recipeObj, 'train.experts');
      $('#recipe-text').value = JSON.stringify(recipeObj, null, 2) + '\n';
    });
    training.append(row('learns in the experts too', experts, 'LoRA on the mixture-of-experts layers as well as attention (one adapter a layer, shared by its 32 experts); off: attention only. The adapter file grows to about 600 MB'));
  }
  form.append(training);
  form.append(el('fieldset', {}, el('legend', { textContent: 'Tests' }), ...TEST_FIELDS.map(([k, label, d]) => row(label, num(k, d))),
    row('live calls with three callers', bind(el('input', { type: 'checkbox' }), 'tests.live', 'check'), 'about 40 minutes'),
    row('compare with (runs)', bind(el('input', { placeholder: 'g26b4' }), 'tests.compare', 'list'), 'other runs whose results go in the same tables')));
}
$('#recipe-text').addEventListener('change', renderRecipeForm);

// Add a base model (src/main/bases.js): look the GGUF repo up for its files, Check the pair without changing anything,
// Add it to models/bases.json, Download it (its output goes to Progress, like a stage's)
const baseMsg = (t) => { $('#base-msg').textContent = t; };
const baseSpec = () => ({ name: $('#base-name').value.trim(), label: $('#base-label').value.trim(), ggufRepo: $('#base-gguf-repo').value.trim(),
  ggufFile: $('#base-gguf-file').value, trainRepo: $('#base-train-repo').value.trim() });
const gbs = (n) => `${(n / 1e9).toFixed(1)} GB`;
async function fillBaseDownloads(select) {
  const list = await api.tuneBases();
  const s = $('#base-dl');
  s.innerHTML = '';
  for (const b of list.filter((x) => x.download)) {
    s.append(el('option', { value: b.name, textContent: `${b.label}${b.gguf && b.trainable ? ' (downloaded)' : ''}` }));
  }
  if (select) s.value = select;
}
for (const id of ['#base-name', '#base-gguf-repo', '#base-gguf-file', '#base-train-repo']) $(id).addEventListener('change', () => { $('#base-add').disabled = true; });
$('#base-lookup').addEventListener('click', async () => {
  baseMsg('looking up the GGUF files ...');
  try {
    const files = await api.baseGgufs($('#base-gguf-repo').value.trim());
    const s = $('#base-gguf-file');
    s.innerHTML = '';
    for (const f of files) s.append(el('option', { value: f.file, textContent: `${f.file}  (${gbs(f.size)})` }));
    baseMsg(`${files.length} GGUF files; the first is the usual pick (Q4_K_M).`);
  } catch (e) { baseMsg(String(e.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, '')); }
});
$('#base-check').addEventListener('click', async () => {
  baseMsg('checking ...');
  $('#base-add').disabled = true;
  try {
    const r = await api.baseInspect(baseSpec());
    baseMsg([`${r.name}: ${r.row.label}, the ${r.family} chat format.`,
      r.sameAs ? `The same layer shapes as ${r.sameAs}: it shares that base's adapters and call settings.` : 'A new shape: no adapter here fits it yet, and its call settings are defaults until measured.',
      `The GGUF (${gbs(r.sizes.gguf)}): ${r.have.gguf ? 'already on disk' : 'to download'}, ${r.row.gguf}.`,
      `The checkpoint (${gbs(r.sizes.checkpoint)}): ${r.have.checkpoint ? 'already on disk' : 'to download'}, ${r.row.hf}.`,
      'Add puts it in models/bases.json; nothing is downloaded until you click Download.'].join('\n'));
    $('#base-add').disabled = false;
  } catch (e) { baseMsg(String(e.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, '')); }
});
$('#base-add').addEventListener('click', async () => {
  try {
    const { name, row } = await api.baseAdd(baseSpec());
    $('#base-add').disabled = true;
    tuneBases = []; // the form's model list reads it again
    await fillBaseDownloads(name);
    renderRecipeForm();
    baseMsg(`Added ${name} (${row.label}). Download it next; once it is on disk it can be picked as a run's model, and in the Call tab once an adapter is trained on it.`);
  } catch (e) { baseMsg(String(e.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, '')); }
});
$('#base-download').addEventListener('click', async () => {
  const name = $('#base-dl').value;
  if (!name) return;
  try { baseMsg((await api.baseDownload(name)).message + ': see Progress below.'); } catch (e) { baseMsg(String(e.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, '')); }
});
$('#add-base').addEventListener('toggle', () => { if ($('#add-base').open) fillBaseDownloads(); });

async function showResults() {
  if (!recipeFile) return;
  const r = await api.tuneResults(recipeFile);
  tres.textContent = r.text || '(nothing yet: run a stage)';
  $('#tune-use').disabled = !r.adapter;
  $('#tune-use').dataset.adapter = r.adapter || '';
  $('#recipe-trained').hidden = !r.trained;
}
async function tuneState() {
  const s = await api.tuneStatus();
  const where = s.running ? await api.tuneProgress(s.recipe) : '';
  $('#tune-state').textContent = s.running ? `running: ${s.stage} of ${s.recipe}${where ? ' · ' + where : ''}` : '';
  document.querySelectorAll('[data-stage]').forEach((b) => { b.disabled = s.running; });
}
recipeSel.addEventListener('change', loadRecipe);
$('#recipe-reload').addEventListener('click', loadRecipe);
async function saveRecipe(saveAs) {
  try {
    const r = await api.tuneSaveRecipe(recipeFile, $('#recipe-text').value, saveAs);
    $('#recipe-saveas').value = '';
    await loadRecipes(r.file);
    $('#recipe-msg').textContent = `saved ${r.file}${r.note ? ': ' + r.note : ''}`;
  } catch (e) { $('#recipe-msg').textContent = e.message; }
}
$('#recipe-save').addEventListener('click', () => saveRecipe());
$('#recipe-new').addEventListener('click', () => {
  const name = $('#recipe-saveas').value.trim();
  if (!name) { $('#recipe-msg').textContent = 'type a name for the new run first'; $('#recipe-saveas').focus(); return; }
  saveRecipe(name);
});
// a stage runs the saved recipe: one typed but not saved yet would run the run it was copied from (and its name, the
// new run, does not exist yet)
function notSaved() {
  if ($('#recipe-saveas').value.trim()) return `"${$('#recipe-saveas').value.trim()}" is not a run yet: click New run from this first (or clear the name to run ${recipeFile})`;
  if ($('#recipe-text').value !== savedRecipe) return `${recipeFile} has changes that are not saved: Save changes, or New run from this, or Revert, first`;
  return '';
}
document.querySelectorAll('[data-stage]').forEach((b) => b.addEventListener('click', async () => {
  const why = notSaved();
  if (why) { $('#recipe-msg').textContent = why; return; }
  tlog.textContent += `\n$ ${b.dataset.stage} (${recipeFile})\n`;
  try { const r = await api.tuneRun(b.dataset.stage, recipeFile); if (r && r.message) tlog.textContent += r.message + '\n'; }
  catch (e) { tlog.textContent += 'error: ' + e.message + '\n'; }
  tuneState();
}));
$('#tune-stop').addEventListener('click', async () => { await api.tuneStop(); tlog.textContent += '[paused; running the same stage again resumes where it was]\n'; tuneState(); });
$('#tune-use').addEventListener('click', async () => {
  const adapter = $('#tune-use').dataset.adapter;
  if (!adapter) return;
  tlog.textContent += `loading models with ${adapter}\n`;
  try { await api.startServices(); await api.useAdapter(adapter); tlog.textContent += 'ready: it is the active adapter (Call tab)\n'; }
  catch (e) { tlog.textContent += 'error: ' + e.message + '\n'; }
});
loadRecipes().then(tuneState);
setInterval(async () => { if ((await api.tuneStatus()).running) { showResults(); tuneState(); } }, 15000);

// ---------- adapter browser ----------
// Every adapter with what its file, its card and the tests say (main/adapters.js). Notes and the label save when you
// leave their box; a rename or remove also changes saved settings and recipes, so the tabs showing those follow.
let catalog = { adapters: [] };
let pickedAdapter = null; // its path
const plainError = (e) => String(e.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, '');
const theAdapter = () => catalog.adapters.find((a) => a.path === pickedAdapter);
const dayOf = (ms) => new Date(ms).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
const sizeOf = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.round(n / 1e6)} MB`);
const pctOf = (p) => (p && p[1] ? `${Math.round(100 * p[0] / p[1])}% (${p[0]}/${p[1]})` : '-');
const adMsg = (t, bad = false) => { const m = $('#ad-msg'); if (m) { m.textContent = t; m.classList.toggle('bad', bad); } };
const isActive = (a) => normPath(catalog.active || '').includes(normPath(a.file));
const differs = (x, y) => x != null && y != null && Math.abs(x - y) > 0.001;
async function loadCatalog() {
  if (!catalog.adapters.length) $('#ad-detail').textContent = 'reading the test results...';
  catalog = await api.adapterCatalog();
  if (!theAdapter()) pickedAdapter = (catalog.adapters.find(isActive) || catalog.adapters[0] || {}).path || null;
  renderAdapterList();
  // not from under someone typing: their box saves when they leave it
  if (!$('#ad-detail').contains(document.activeElement)) renderAdapterDetail();
}
$('#ad-filter').addEventListener('input', renderAdapterList);
// the list and the entry show the active adapter, wherever it was switched from
api.onAdapters((s) => { if ($('#tab-adapters').classList.contains('active')) { catalog.active = s.active; catalog.serving = { ...catalog.serving, base: s.model }; renderAdapterList(); } });
const TEST_NAMES = [['quiz', 'quiz'], ['ours', 'held-out'], ['yap', 'recorded'], ['live', 'live']];
function renderAdapterList() {
  const q = $('#ad-filter').value.trim().toLowerCase();
  const rows = catalog.adapters.filter((a) => [a.label, a.file, a.base, a.baseLabel, a.recipe].join(' ').toLowerCase().includes(q))
    .sort((a, b) => (b.trained || 0) - (a.trained || 0) || a.file.localeCompare(b.file));
  $('#ad-list').replaceChildren(...rows.map((a) => {
    const row = el('button', { className: 'ad-row' + (a.path === pickedAdapter ? ' picked' : '') },
      el('span', { className: 'name', textContent: (a.label || a.file.replace(/\.gguf$/, '')) + (isActive(a) ? ' (active)' : '') }),
      el('span', { className: 'file', textContent: a.file }),
      el('span', { className: 'meta', textContent: [a.base || 'base not recorded', a.trained ? dayOf(a.trained) : ''].filter(Boolean).join(' · ') }),
      el('span', { className: 'ad-tests' }, ...TEST_NAMES.map(([k, name]) => el('span', { className: a.results[k].best && !a.results[k].best.set ? 'on' : '', textContent: name }))));
    row.addEventListener('click', () => { if (document.activeElement) document.activeElement.blur(); pickedAdapter = a.path; renderAdapterList(); renderAdapterDetail(); });
    return row;
  }));
  if (!rows.length) $('#ad-list').append(el('p', { className: 'note', textContent: catalog.adapters.length ? 'none match' : 'no adapters found in models/lora' }));
}
// how a test ran, and where that is not how this file runs
function testedHow(a, r) {
  const said = [r.run, r.listen != null ? `listen ${r.listen}` : '', r.carryOn != null ? `bar ${r.carryOn}` : '', r.calls ? `${r.calls} calls` : '', r.date ? dayOf(r.date) : ''].filter(Boolean).join(' · ');
  const flags = [];
  if (r.set) flags.push(`on ${r.set}`);
  if (differs(r.listen, a.listen)) flags.push(`tested at ${r.listen}; the file runs at ${a.listen}`);
  if (r.kind !== 'live' && r.kind !== 'quiz' && r.kind !== 'wide' && (r.carryOn ?? null) !== (a.carryOn ?? null)) flags.push(r.carryOn == null ? `tested without the carry-on bar; the file has ${a.carryOn}` : `tested with bar ${r.carryOn}; the file has ${a.carryOn ?? 'none'}`);
  return { said, flags };
}
const table = (rows) => el('table', {}, ...rows.map(([k, v]) => el('tr', {}, el('td', { textContent: k }), el('td', { textContent: v }))));
const OURS_ROWS = [['start', 'Answers when you finish'], ['hazard', 'Cuts in with a warning'], ['barge', 'Stops when cut in on'], ['filler', 'Back-channels where a person did'],
  ['over', 'Talks over the caller (lower is better)'], ['unfinished', 'Leaves a reply unfinished (lower is better)']];
const ADAPTER_TESTS = [
  ['quiz', 'Quiz', (r) => table([['Answered', `${r.score}/${r.of}`], ...(r.silent != null ? [['Silent', String(r.silent)]] : [])])],
  ['wide', 'Wide quiz', (r) => table([['Answered', `${r.score}/${r.of}`], ...(r.silent != null ? [['Silent', String(r.silent)]] : [])])],
  ['ours', 'Held-out set', (r) => table(OURS_ROWS.filter(([k]) => r.totals[k]).map(([k, name]) => [name, pctOf(r.totals[k])]))],
  ['yap', 'Recorded conversations', (r) => (r.table ? table(Object.entries(r.table).filter(([k]) => !/^Minutes/.test(k)))
    : table([['Takes the turn where the person did', pctOf(r.totals.start)], ['Quiet when the person gave way to a cut-in', pctOf(r.totals.barge)]]))],
  ['live', 'Live bench', (r) => table([['All checks', pctOf([r.passed, r.total])], ...Object.entries(r.callers).sort().map(([c, p]) => [`Caller ${c}`, pctOf(p)]),
    ...(r.medianReplyS != null ? [['Median reply start after the caller stops', `${r.medianReplyS.toFixed(2)} s`]] : []),
    ...(r.medianStopS != null ? [['Median stop after a cut-in starts', `${r.medianStopS.toFixed(2)} s`]] : []),
    ...(r.tickP95ms != null ? [['Slowest tick p95', `${r.tickP95ms} ms`]] : [])])],
];
function testBlock(a, [kind, title, body]) {
  const { best, others } = a.results[kind];
  const box = el('div', { className: 'ad-test' }, el('h3', { textContent: title }));
  if (best) {
    const { said, flags } = testedHow(a, best);
    box.append(el('p', { className: 'how', textContent: said }), ...flags.map((t) => el('p', { className: 'flag', textContent: t })), body(best));
  } else if (others.some((r) => r.otherBase)) {
    box.append(el('p', { className: 'none', textContent: `not tested on this base (${a.base})` }));
  } else if (kind === 'wide') {
    return null; // only some runs have it; nothing to say when one did not
  } else box.append(el('p', { className: 'none', textContent: 'not tested' }));
  if (others.length) {
    box.append(el('details', {}, el('summary', { textContent: `other test runs of these weights (${others.length})` }),
      el('ul', {}, ...others.map((r) => {
        const score = r.score != null ? `${r.score}/${r.of}` : r.passed != null ? pctOf([r.passed, r.total]) : `start ${pctOf(r.totals.start)}`;
        const how = testedHow(a, r);
        return el('li', { textContent: `${how.said}: ${score}${r.otherBase ? ` (on ${r.otherBase})` : ''}${how.flags.length ? ` (${how.flags.join('; ')})` : ''}` });
      }))));
  }
  return box;
}
function renderAdapterDetail() {
  const a = theAdapter(), d = $('#ad-detail');
  if (!a) { d.textContent = catalog.adapters.length ? '' : 'No adapters found in models/lora.'; return; }
  const file = a.path, serving = catalog.serving || {};
  const label = el('input', { value: a.label, placeholder: a.suggestedLabel || a.file.replace(/\.gguf$/, ''), title: 'its name in the list and the Call tab; kept beside the file, and in the recipes that export it. Empty: the name in the file' });
  label.addEventListener('change', async () => {
    try { await afterAdapterChange((await api.setAdapterLabel(file, label.value)).recipes, 'label saved'); } catch (e) { adMsg(plainError(e), true); }
  });
  const head = el('div', { className: 'ad-head' }, label);
  if (isActive(a)) head.append(el('span', { className: 'ad-tag', textContent: 'active' }));
  if (a.file === catalog.defaultFile) head.append(el('span', { className: 'ad-tag', textContent: 'app default' }));
  if (a.addon) head.append(el('span', { className: 'ad-tag', textContent: 'add-on: plays only in a blend' }));
  const saved = (v, own) => el('span', {}, v != null ? String(v) : 'none, it uses the file\'s', differs(v, own) ? el('span', { className: 'flag', textContent: ` differs from the file's ${own}` }) : '');
  const facts = [['Base', a.baseLabel ? `${a.baseLabel}${a.baseOnDisk ? '' : ' (not downloaded)'}` : 'not recorded'], ['Prompt', a.prompt], ['Recipe', a.recipe || 'none (not trained here)'],
    ['Size', sizeOf(a.size)], ['Trained', a.trained ? dayOf(a.trained) : `not recorded (file dated ${dayOf(a.fileDate)})`],
    [a.stamped ? 'Listen threshold in the file' : 'Listen threshold (known for this file)', a.listen ?? 'not recorded'], ['Carry-on bar in the file', a.carryOn ?? 'not recorded'],
    ['Your saved threshold', saved(a.savedListen, a.listen)], ['Your saved bar', saved(a.savedCarryOn, a.carryOn)]];
  const notes = el('textarea', { id: 'ad-notes', rows: 5, value: a.notes, placeholder: 'what this one is for, what it does well or badly' });
  notes.addEventListener('change', async () => {
    try { await api.setAdapterNotes(file, notes.value); a.notes = notes.value.trimEnd(); adMsg('notes saved'); } catch (e) { adMsg(plainError(e), true); }
  });
  const use = el('button', { className: 'primary', textContent: isActive(a) ? 'In use' : 'Use this adapter', disabled: isActive(a) || !a.base || a.addon });
  if (!a.base) use.title = 'this file does not say which model it was trained on (tune/adapter_meta.py --base writes it in)';
  use.addEventListener('click', async () => {
    use.disabled = true;
    adMsg(serving.llama === 'ready' && a.base !== serving.base ? `restarting the model server on ${a.baseLabel} (30-60 s)...` : 'switching...');
    try { renderAdapters(await api.useAdapter(file)); await afterAdapterChange([], `now using ${a.label || a.file}`); }
    catch (e) { adMsg(plainError(e), true); use.disabled = false; }
  });
  const rename = el('button', { textContent: 'Rename', disabled: !a.stamped, title: a.stamped ? '' : 'it carries no talkme metadata, so the app knows it by this file name' });
  const remove = el('button', { className: 'danger', textContent: 'Remove' });
  const extra = el('div', {}); // the rename box or the confirmation, one at a time
  rename.addEventListener('click', () => {
    const name = el('input', { value: a.file.replace(/\.gguf$/, '') });
    const go = el('button', { className: 'primary', textContent: 'Rename the file' });
    go.addEventListener('click', async () => {
      go.disabled = true;
      try {
        const r = await api.renameAdapter(file, name.value);
        pickedAdapter = r.path;
        await afterAdapterChange(r.recipes, `renamed to ${r.name}${r.recipes.length ? `; updated ${r.recipes.join(', ')}` : ''}${a.file === catalog.defaultFile ? '. This was the app default: a fresh install looks for the old name until DEFAULTS.llama.lora in src/main/config.js changes too' : ''}`);
      } catch (e) { adMsg(plainError(e), true); go.disabled = false; }
    });
    const cancel = el('button', { textContent: 'Cancel' });
    cancel.addEventListener('click', () => extra.replaceChildren());
    extra.replaceChildren(el('div', { className: 'ad-actions' }, name, go, cancel), el('p', { className: 'note', textContent: 'Saved settings and the recipes that export it follow the new name. Other mentions (README, scripts) do not.' }));
    name.focus();
  });
  remove.addEventListener('click', () => {
    const lines = [`Move ${a.file} (${sizeOf(a.size)}) and its notes to the Recycle Bin?`, 'Its saved threshold and other settings are cleared.'];
    if (isActive(a)) lines.push('It is the active adapter: another one on this base takes over, or none if it is the last.');
    if (a.file === catalog.defaultFile) lines.push('It is the app default: a fresh install starts with no adapter until DEFAULTS.llama.lora in src/main/config.js changes.');
    const yes = el('button', { className: 'danger', textContent: 'Yes, remove it' });
    yes.addEventListener('click', async () => {
      yes.disabled = true;
      try {
        const r = await api.removeAdapter(file);
        pickedAdapter = r.next;
        await afterAdapterChange([], `${a.file} is in the Recycle Bin${r.recipes.length ? `. Re-exporting ${r.recipes.join(', ')} would bring it back` : ''}`);
      } catch (e) { adMsg(plainError(e), true); yes.disabled = false; }
    });
    const keep = el('button', { textContent: 'Keep it' });
    keep.addEventListener('click', () => extra.replaceChildren());
    extra.replaceChildren(el('div', { className: 'ad-confirm' }, ...lines.map((t) => el('p', { textContent: t })), el('div', { className: 'ad-actions' }, yes, keep)));
  });
  d.replaceChildren(head, el('p', { className: 'ad-file', textContent: `${a.folder}/${a.file}` }),
    el('dl', { className: 'ad-facts' }, ...facts.flatMap(([k, v]) => [el('div', {}, el('dt', { textContent: k }), el('dd', {}, typeof v === 'object' ? v : String(v)))])),
    el('div', { className: 'ad-actions' }, use, rename, remove), extra, el('span', { id: 'ad-msg', className: 'note' }),
    el('h2', { textContent: 'Notes' }), notes,
    el('h2', { textContent: 'Test results' }), ...ADAPTER_TESTS.map((t) => testBlock(a, t)).filter(Boolean));
}
// A change that reached saved settings or recipes reaches what is on screen too: a stale Settings field or recipe text
// would otherwise write the old name straight back on its next Save.
async function afterAdapterChange(recipes, message) {
  settings = await api.getSettings();
  document.querySelector('#settings-form [name="llama.lora"]').value = settings.llama.lora || '';
  if (recipes.includes(recipeFile)) {
    if ($('#recipe-text').value === savedRecipe) await loadRecipe();
    else $('#recipe-msg').textContent = `${recipeFile} changed on disk (its adapter's name): Revert to pick that up before saving`;
  } else showResults();
  await loadCatalog();
  renderAdapterDetail();
  adMsg(message);
}
