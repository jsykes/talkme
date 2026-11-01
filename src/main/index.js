'use strict';
const { app, BrowserWindow, ipcMain, dialog, screen, shell } = require('electron');
const path = require('path');
const { Settings, DEFAULTS, BASES, baseOf, baseRow, baseKey, adapterInfo, ggufArch, cardPath } = require('./config');
const { Llama } = require('./llama');
const { STT } = require('./stt');
const { TTS } = require('./tts');
const { Loop } = require('./loop');
const { Aec } = require('./aec');
const { CallRecorder } = require('./callrec');
const { Tune } = require('./tune');
const adapters = require('./adapters');
const { listVoices, voiceConfig, addChatterboxVoice, removeChatterboxVoice } = require('./voices');
const fs = require('fs');
const { ROOT } = require('./config');

// Shipped adapters (models/lora/*.gguf) and your own exports (tune/out/talkme-*.gguf) are loaded unapplied, so the
// active one can be swapped per request. Experiment files in tune/out (run1.gguf, warmv2.gguf, ...) stay out of the list.
function discoverAdapters(family) {
  const out = [];
  for (const dir of [path.join(ROOT, 'models', 'lora'), path.join(ROOT, 'tune', 'out')]) {
    try { for (const f of fs.readdirSync(dir)) if (f.endsWith('.gguf') && (dir.endsWith('lora') || /^talkme-/.test(f)) && !/test/i.test(f)) out.push(path.join(dir, f)); } catch {}
  }
  // a model server can only load adapters made for its own base (Qwen adapters on Gemma, or 31B ones on the 26B, fail);
  // one that does not say which base is loaded by none
  return family ? out.filter((f) => { const b = baseOf(f); return b && BASES[b].arch === family; }) : out;
}

let win = null;
const settings = new Settings(path.join(app.getPath('userData'), 'settings.json'));

function send(ch, payload) { if (win && !win.isDestroyed()) win.webContents.send(ch, payload); }
// the log also goes to talkme.log in the app's data folder (the last ~2 MB), so a problem can be read after the fact
const LOG_FILE = path.join(app.getPath('userData'), 'talkme.log');
try { if (fs.statSync(LOG_FILE).size > 4e6) fs.renameSync(LOG_FILE, LOG_FILE + '.old'); } catch {}
function log(line) {
  console.log(line); send('log', line);
  try { fs.appendFileSync(LOG_FILE, `${new Date().toISOString().slice(0, 19)} ${line}
`); } catch {}
}

const llama = new Llama(log);
const stt = new STT(log);
const tts = new TTS(log);
const aec = new Aec(log);
const callrec = new CallRecorder(path.join(app.getPath('userData'), 'calls'), log);
const loop = new Loop({ llama, stt, tts, log, emit: (ev, p) => send('loop:' + ev, p) });
// the pipeline runs its own llama-server on the same port, so the app's goes first (Load models brings it back)
const tune = new Tune(log, (ev, p) => send('tune:' + ev, p), { beforeRun: async () => {
  if (services.call === 'live') { loop.stop(); setStatus('call', 'idle'); }
  if (services.llama !== 'stopped') { log('tune: stopping the model server for the pipeline'); await llama.stop(); setStatus('llama', 'stopped'); }
}, storage: () => settings.get().storage });

stt.onPartial = (text) => send('stt:partial', text);
tts.onAudio = (a) => send('tts:audio', a);

const services = { llama: 'stopped', stt: 'stopped', tts: 'stopped', aec: 'off', call: 'idle' };
function setStatus(k, v) { services[k] = v; send('status', services); }

// Adapters trained with the partner system prompt: "partner" file names from run 9 up, and every Gemma one (talkme-g12-,
// talkme-g26-; these got the assistant prompt in the app until g26m). Those trained with a persona in the prompt too
// (g26m on) are listed in PERSONA_ADAPTERS; they get the Settings persona when one is set.
// "-addon-" adapters (g26p on) are trained on top of another adapter and only play blended with it (BLENDS)
const PERSONA_ADAPTERS = /^talkme-g26(m\d*|p)-/i;
function promptFor(adapter) {
  if (adapter && adapter.startsWith('blend:')) {
    // a blend speaks with the prompt of its most specific part: persona over partner over assistant
    const kinds = adapter.slice(6).split(',').map((p) => promptFor(p.split('=')[0]));
    return kinds.includes('persona') ? 'persona' : kinds.includes('partner') ? 'partner' : 'assistant';
  }
  if (adapter && adapterInfo(adapter).prompt) return adapterInfo(adapter).prompt; // what the file says (adapter_meta.py)
  const name = adapter ? path.basename(adapter) : '';
  if (PERSONA_ADAPTERS.test(name)) return 'persona';
  const m = /talkme-run(\d+)-partner/i.exec(name);
  return (m && Number(m[1]) >= 9) || /^talkme-g\d+\w*-partner/i.test(name) ? 'partner' : 'assistant';
}
// The base an adapter runs on (config.js BASES, from the end of its file name), if that base is on disk; otherwise
// `fallback`, the model picked in the Call tab (Settings llama.model), which is also the base with no adapter
function baseFor(adapter, fallback) {
  // a blend runs on the base of its first adapter (they all share one)
  if (adapter && adapter.startsWith('blend:')) return baseFor(adapter.slice(6).split(',')[0].split('=')[0], fallback);
  if (!adapter) return fallback;
  const name = baseOf(adapter);
  if (!name) return fallback; // says nothing about its base (useAdapter refuses it)
  // one with no base in its name (the author's, run 2-7) runs on the picked model when that is a Qwen too
  if (name === 'qwen' && baseKey(fallback) === 'qwen3.6') return fallback;
  const gguf = path.join(ROOT, BASES[name].gguf);
  return fs.existsSync(gguf) ? gguf : fallback;
}

// An adapter whose base model is not on disk: which file it needs and where it comes from (baseFor would otherwise
// fall back to another model, and the adapter would be dropped for not fitting it)
function missingBase(adapter, fallback) {
  const file = adapter && adapter.startsWith('blend:') ? adapter.slice(6).split(',')[0].split('=')[0] : adapter;
  const key = file && baseOf(file);
  const b = key && BASES[key];
  if (!b || fs.existsSync(path.join(ROOT, b.gguf))) return null;
  if (key === 'qwen' && baseKey(fallback) === 'qwen3.6' && fs.existsSync(fallback)) return null; // any Qwen3.6 serves it
  return `${labelOf(adapter)} runs on ${b.label}, which is not downloaded: npm run setup -- --base ${key} (or put ${path.basename(b.gguf)} from huggingface.co/${b.from} in ${path.dirname(b.gguf)}/)`;
}

// If the recogniser's thread dies, say so and restart it (at most three times a minute) instead of going deaf.
let sttRestarts = [];
stt.onDied = async (code) => {
  log(`stt: the recogniser stopped unexpectedly (exit code ${code})`);
  sttRestarts = sttRestarts.filter((t) => Date.now() - t < 60000);
  if (sttRestarts.length >= 3) { setStatus('stt', 'stopped'); log('stt: not restarting again (three failures in a minute); use Load models'); return; }
  sttRestarts.push(Date.now());
  setStatus('stt', 'starting');
  try { await stt.start(settings.get().stt); setStatus('stt', 'ready'); log('stt: restarted'); } catch (e) { setStatus('stt', 'stopped'); log('stt: restart failed: ' + e.message); }
};

// The chat model's settings with room made for a GPU voice: ~3.5 GB, so more of its experts stay on the CPU
function llamaCfg(s, extra = {}) {
  const model = extra.model || s.llama.model;
  const base = baseRow(model);
  const key = base.arch;
  const gpuVoice = !!voiceConfig(s.tts.voice).moeExtra;
  // a GPU voice gets room by parking more experts in system RAM (not where base.expertsOnGpu: see config.js BASES)
  const moe = (s.llama.nCpuMoe ?? 8) +(gpuVoice && !base.expertsOnGpu ? voiceConfig(s.tts.voice).moeExtra : 0);
  const cfg = { ...s.llama, nCpuMoe: moe, extraLoras: discoverAdapters(key), ...extra };
  if (cfg.lora && BASES[baseOf(cfg.lora)]?.arch !== key) cfg.lora = ''; // the saved adapter is for another base (or none it names)
  if (base.ctx) cfg.ctx = Math.min(cfg.ctx || 32768, base.ctx);
  // a dense model has no experts to park on the CPU: with a GPU voice (~3.5 GB) some of its layers go instead
  if (gpuVoice && base.gpuLayersWithVoice) cfg.ngl = base.gpuLayersWithVoice;
  return cfg;
}

async function startServices() {
  const s = settings.get();
  if (services.stt === 'stopped') {
    setStatus('stt', 'starting');
    try { await stt.start(s.stt); setStatus('stt', 'ready'); } catch (e) { setStatus('stt', 'stopped'); throw e; }
  }
  if (services.aec !== 'ready' && s.aec && s.aec.enabled !== false) {
    if (!aec.installed()) log('aec: not installed (uv sync --project aec); the mic goes to the recogniser as it is');
    else {
      setStatus('aec', 'starting');
      try { await aec.start(s.aec); setStatus('aec', 'ready'); }
      catch (e) { setStatus('aec', 'off'); log('aec: ' + e.message + '; the mic goes to the recogniser as it is'); }
    }
  }
  if (services.tts === 'stopped') {
    setStatus('tts', 'starting');
    try { await tts.start(voiceConfig(s.tts.voice)); setStatus('tts', 'ready'); } catch (e) { setStatus('tts', 'stopped'); throw e; }
  }
  if (services.llama === 'stopped') {
    setStatus('llama', 'starting');
    try {
      const missing = missingBase(s.activeBlend || s.llama.lora, s.llama.model);
      if (missing) throw new Error(missing);
      await llama.start(llamaCfg(s, { model: baseFor(s.activeBlend || s.llama.lora, s.llama.model) }));
      // a saved blend first; if its adapters are gone, the saved single adapter
      try { await llama.useAdapter(s.activeBlend || s.llama.lora || null); } catch { await llama.useAdapter(s.llama.lora || null); settings.update({ activeBlend: '' }); }
      llama.promptKind = promptFor(llama.activeAdapter);
      llama.persona = s.persona;
      // the loaded adapter's own threshold, whatever was saved last
      const thr = thresholdFor(llama.activeAdapter);
      if (thr !== s.listenThreshold) { settings.update({ listenThreshold: thr }); send('threshold', thr); }
      log(`llama: active adapter ${labelOf(llama.activeAdapter)}, listen threshold ${thr}`);
      setStatus('llama', 'ready');
      send('adapters', await adapterState());
    }
    catch (e) { setStatus('llama', 'stopped'); throw e; }
  }
}

ipcMain.handle('settings:get', () => settings.get());
ipcMain.handle('settings:update', (_e, patch) => {
  const out = settings.update(patch);
  llama.persona = settings.get().persona; // a persona applies from the next tick
  return out;
});
ipcMain.handle('status:get', () => services);
ipcMain.handle('services:start', async () => { await startServices(); return services; });
ipcMain.handle('services:stop', async () => {
  if (services.call === 'live') { loop.stop(); setStatus('call', 'idle'); }
  await llama.stop(); setStatus('llama', 'stopped');
  await tts.stop(); setStatus('tts', 'stopped');
  await stt.stop(); setStatus('stt', 'stopped');
  await aec.stop(); setStatus('aec', 'off');
  return services;
});
ipcMain.handle('call:start', async () => {
  await startServices();
  stt.reset(); // nothing half-heard from the previous call
  const s = settings.get();
  if (s.saveCallAudio) callrec.start();
  loop.start({ tickMs: s.tickMs, listenThreshold: s.listenThreshold, listenSample: !!s.listenSample, ttsFlushWords: s.ttsFlushWords, ttsSpeed: s.ttsSpeed, llamaCtx: s.llama.ctx,
    ownWordsFilter: s.ownWordsFilter !== false, wordSampling: s.wordSampling || null, carryOnP: s.carryOnP,
    historyTicks: baseRow(llama.model).historyTicks });
  setStatus('call', 'live');
  return services;
});
ipcMain.handle('call:stop', () => { loop.stop(); stt.reset(); callrec.stop(); setStatus('call', 'idle'); return services; });
ipcMain.handle('call:settings', (_e, patch) => { Object.assign(loop.settings || {}, patch); });
// What the Call tab's Model and Adapter dropdowns show: the bases on disk, the one serving (or the one Load models will
// start), and every adapter with its base; the renderer lists those of the picked model
async function adapterState() {
  const s = settings.get();
  const running = services.llama === 'ready';
  const active = running ? llama.activeAdapter : s.activeBlend || s.llama.lora || null;
  const model = baseOf(running ? llama.model : baseFor(active, s.llama.model));
  const models = Object.entries(BASES).filter(([name, b]) => name === model || fs.existsSync(path.join(ROOT, b.gguf)))
    .map(([name, b]) => ({ name, label: b.label, arch: ggufArch(b.gguf) }));
  // an add-on is loaded (for its blends) but not offered on its own: alone it has no timing
  const adapters = discoverAdapters().filter((f) => !/-addon-/i.test(path.basename(f))).map((f) => { const i = adapterInfo(f); return { path: f, base: baseOf(f), label: i.label, arch: i.arch }; });
  let blends = [];
  if (running) {
    const list = await (await fetch(llama.url + '/lora-adapters')).json();
    const have = new Set(list.map((a) => llama.fileName(a.path)));
    blends = BLENDS.filter((b) => b.spec.slice(6).split(',').every((p) => have.has(p.split('=')[0].toLowerCase())))
      .map((b) => ({ ...b, base: baseOf(b.spec.slice(6).split(',')[0].split('=')[0]) }));
  }
  return { models, model, adapters, blends, active };
}
ipcMain.handle('adapters:get', () => adapterState());
ipcMain.handle('voices:get', () => ({ voices: listVoices(), active: settings.get().tts.voice }));
// A new Chatterbox voice from an uploaded file or a recording (bytes from the renderer); it becomes the active voice.
ipcMain.handle('voices:add', async (_e, name, { file, bytes, ext }) => {
  let src = file;
  if (bytes) {
    src = path.join(require('os').tmpdir(), `talkme-voice-${Date.now()}.${String(ext || 'webm').replace(/\W/g, '')}`);
    fs.writeFileSync(src, Buffer.from(bytes));
  }
  try {
    const id = addChatterboxVoice(name, src);
    log(`tts: new Chatterbox voice ${id}`);
    return { voices: listVoices(), active: settings.get().tts.voice, added: id };
  } finally { if (bytes) { try { fs.unlinkSync(src); } catch {} } }
});
ipcMain.handle('voices:remove', async (_e, id) => {
  removeChatterboxVoice(id);
  if (settings.get().tts.voice === id) settings.update({ tts: { voice: 'chatterbox:turbo' } });
  return { voices: listVoices(), active: settings.get().tts.voice };
});
// Switch the voice: restarts only the speech worker (a second or two); a call can keep running meanwhile.
ipcMain.handle('voices:use', async (_e, id) => {
  settings.update({ tts: { voice: id } });
  if (services.tts !== 'stopped') {
    setStatus('tts', 'starting');
    // another voice of the same GPU engine: keep its server (switching is then a few seconds, not a reload); anything
    // else stops it, which frees its GPU memory before the chat model may need it back
    const same = voiceConfig(id).kind && voiceConfig(id).kind === tts.kind;
    await tts.stop({ keepServer: same });
    // the chat model makes room on the GPU for a GPU voice, or takes it back: restart it with the new split
    const want = llamaCfg(settings.get()).nCpuMoe;
    if (services.llama === 'ready' && llama.cfg && llama.cfg.nCpuMoe !== want) {
      log(`llama: restarting with ${want} MoE layers on the CPU (${voiceConfig(id).moeExtra ? 'room for a GPU voice' : 'GPU voice gone'})`);
      if (services.call === 'live') { loop.stop(); setStatus('call', 'idle'); }
      const active = llama.activeAdapter, kind = llama.promptKind;
      setStatus('llama', 'starting');
      await llama.stop();
      try { await llama.start(llamaCfg(settings.get(), { model: llama.model || settings.get().llama.model })); await llama.useAdapter(active); llama.promptKind = kind; setStatus('llama', 'ready'); }
      catch (e) { setStatus('llama', 'stopped'); throw e; }
    }
    try { await tts.start(voiceConfig(id)); setStatus('tts', 'ready'); }
    catch (e) { setStatus('tts', 'stopped'); throw e; }
  }
  log(`tts: voice ${id}`);
  return { voices: listVoices(), active: id };
});
// Swap the active adapter (null = base model). Takes effect on the next tick; that tick re-reads the call history
// once (~1-2 s) because the prompt cache belongs to the previous adapter.
// Each adapter has its own best listen threshold; switching adapters switches it too, and moving the slider afterwards
// is remembered for that adapter. An adapter's own (tune/adapter_meta.py, or config.js KNOWN_ADAPTERS) comes before the
// default: 0.6 for partners (trained on real people), 0.3 for assistants.
// (Older adapters and their thresholds and blends were removed 2026-10-28; they are in git history.)
// Blends: several adapters at once, each at its own strength (llama-server mixes them per request). Listed in the
// adapter dropdown when all their files are loaded; e.g. { label: 'g26 at half strength', spec:
// 'blend:talkme-partner-v0-gemma4a4b.gguf=0.5' }.
const BLENDS = [];
const keyOf = (sel) => (!sel ? '' : sel.startsWith('blend:') ? sel : path.basename(sel));
const labelOf = (sel) => (!sel ? '(none, base model)' : sel.startsWith('blend:') ? (BLENDS.find((b) => b.spec === sel) || { label: sel }).label
  : adapterInfo(sel).label ? `${adapterInfo(sel).label} (${path.basename(sel)})` : path.basename(sel));
function thresholdFor(file) {
  const name = keyOf(file);
  const saved = (settings.get().thresholds || {})[name];
  const partnerDefault = promptFor(file) !== 'assistant' ? 0.6 : undefined; // every partner adapter since run 7
  return saved ?? (file && !file.startsWith('blend:') ? adapterInfo(file).listen : undefined) ?? partnerDefault ?? 0.3;
}
// The carry-on bar (how sure it must be to stop mid-sentence into a silence) is remembered per adapter the same way;
// every adapter starts at its own (adapter_meta.py), else the app's default (0.9, v1's)
function carryOnFor(file) {
  const own = file && !file.startsWith('blend:') ? adapterInfo(file).carryOn : undefined;
  return (settings.get().carryOns || {})[keyOf(file)] ?? own ?? DEFAULTS.carryOnP;
}
// "Human timing" (sample the listen decision) suits adapters trained on real conversations; remembered per adapter.
// off by default: on recorded calls it over-talked (run 6: 9.9 utterances a minute vs a person's 7.4, 3.9 interruptions a minute)
const ADAPTER_SAMPLING = {};
function samplingFor(file) {
  const name = keyOf(file);
  const saved = (settings.get().sampling || {})[name];
  return saved ?? ADAPTER_SAMPLING[name] ?? false;
}
ipcMain.handle('adapters:use', (_e, file) => useAdapter(file));
// The Adapters tab (adapters.js). The renderer names a file by its path: only one of the adapters the app itself found.
function adapterFile(file) {
  const f = discoverAdapters().find((a) => path.resolve(a).toLowerCase() === path.resolve(String(file)).toLowerCase());
  if (!f) throw new Error(`${path.basename(String(file))} is not in the adapter list any more`);
  return f;
}
// A rename or remove takes the file away from under anything about to read it: the server while it starts up, and the
// pipeline, which works from the recipe it was started with.
function changeAllowed(file) {
  if (services.llama === 'starting') throw new Error('the model server is starting: wait until it is up');
  const t = tune.status();
  if (t.running && adapters.exportedBy(file).includes(t.recipe.replace(/\.json$/, ''))) throw new Error(`${t.stage} of ${t.recipe} is running on this adapter`);
}
ipcMain.handle('adapters:catalog', async () => {
  const s = settings.get(), state = await adapterState();
  return { adapters: await adapters.catalog(discoverAdapters(), { saved: s, promptOf: promptFor }), active: state.active, defaultFile: path.basename(DEFAULTS.llama.lora),
    serving: { llama: services.llama, call: services.call, base: state.model } };
});
ipcMain.handle('adapters:notes', (_e, file, notes) => adapters.setNotes(adapterFile(file), notes));
ipcMain.handle('adapters:label', async (_e, file, label) => {
  const r = adapters.setLabel(adapterFile(file), label);
  send('adapters', await adapterState());
  return r;
});
ipcMain.handle('adapters:rename', async (_e, file, name) => {
  const f = adapterFile(file);
  changeAllowed(f);
  const r = adapters.rename(f, name, settings);
  llama.renamed(f, r.path);
  log(`adapters: ${path.basename(f)} is now ${r.name}${r.recipes.length ? ` (recipes: ${r.recipes.join(', ')})` : ''}`);
  send('adapters', await adapterState());
  return r;
});
// To the Recycle Bin with its card, not deleted: a file that was never committed has no other copy once its tune/out
// run is cleared. An active adapter gives way to another on its base, so nothing restarts; the next Load models would
// otherwise find it gone.
ipcMain.handle('adapters:remove', async (_e, file) => {
  const f = adapterFile(file), name = path.basename(f), base = baseOf(f);
  changeAllowed(f);
  const active = services.llama === 'ready' ? llama.activeAdapter : settings.get().activeBlend || settings.get().llama.lora;
  const inUse = !!active && active.toLowerCase().includes(name.toLowerCase());
  if (inUse && services.call === 'live') throw new Error('this adapter is on a live call: end the call first');
  await shell.trashItem(f);
  if (fs.existsSync(cardPath(f))) await shell.trashItem(cardPath(f));
  settings.rewriteAdapter(name, null);
  log(`adapters: ${name} moved to the Recycle Bin`);
  let next = null;
  if (inUse) {
    const own = discoverAdapters().filter((a) => baseOf(a) === base && !/-addon-/i.test(path.basename(a)));
    next = own.find((a) => path.basename(a) === path.basename(DEFAULTS.llama.lora)) || own[0] || null;
    try { await useAdapter(next); } catch (e) { log(`adapters: could not switch to another adapter: ${e.message}`); }
  }
  send('adapters', await adapterState());
  return { next, recipes: adapters.exportedBy(f) };
});
// The Model dropdown: that base with the adapter last used on it (else the app's default if it is one of its own, else
// its first; none if it has none)
ipcMain.handle('model:use', async (_e, name) => {
  const b = BASES[name];
  if (!b) throw new Error(`no model ${name}`);
  const own = discoverAdapters().filter((f) => baseOf(f) === name && !/-addon-/i.test(path.basename(f)));
  const same = (x, y) => path.basename(x || '').toLowerCase() === path.basename(y || '').toLowerCase();
  const last = (settings.get().modelAdapters || {})[name];
  const pick = last === '' || (last && (last.startsWith('blend:') || own.some((f) => same(f, last)))) ? last
    : own.find((f) => same(f, DEFAULTS.llama.lora)) || own[0] || '';
  settings.update({ llama: { model: path.join(ROOT, b.gguf) } });
  return useAdapter(pick || null);
});
async function useAdapter(file) {
  if (file && !file.startsWith('blend:') && !baseOf(file)) {
    throw new Error(`${path.basename(file)} does not say which model it was trained on, and running it on the wrong one stops the model server or quietly misbehaves. Adapters trained in the Tune tab say so themselves; for another one, write it in with tune/adapter_meta.py (--base, --prompt), or end its name with the model's key, like -gemma4a4b.gguf.`);
  }
  const missing = missingBase(file, settings.get().llama.model);
  if (missing) throw new Error(missing);
  if (services.llama !== 'ready' && services.llama !== 'starting') {
    // no model server (services stopped): keep the choice for the next Load models instead of failing on a dead server
    const blend = !!file && file.startsWith('blend:');
    // its own threshold too (only the adapter was kept, and g26m6 then ran at 0.6 instead of its 0.75)
    const thr = thresholdFor(file), smp = samplingFor(file), bar = carryOnFor(file);
    settings.update(blend ? { activeBlend: file, listenThreshold: thr, listenSample: smp, carryOnP: bar } : { llama: { lora: file || '' }, activeBlend: '', listenThreshold: thr, listenSample: smp, carryOnP: bar });
    send('threshold', thr); send('sampling', smp); send('carryOn', bar);
    rememberFor(baseFor(file, settings.get().llama.model), file);
    log(`llama: ${labelOf(file)} will be loaded with the models (Load models), listen threshold ${thr}`);
    return adapterState();
  }
  // an adapter trained on another base model needs that base: restart the model server on it (30-60 s); no adapter
  // stays on the model serving
  const want = file ? baseFor(file, settings.get().llama.model) : llama.model || settings.get().llama.model;
  if (services.llama === 'ready' && want !== llama.model) {
    log(`llama: ${path.basename(file || '')} needs the base ${path.basename(want)}; restarting the model server`);
    if (services.call === 'live') {
      // no model answers for the minute the restart takes: end the call cleanly rather than leave it ticking into errors
      loop.stop(); stt.reset(); callrec.stop(); setStatus('call', 'idle');
      log('loop: call ended for the switch to another base model; start a new call when it is loaded');
    }
    setStatus('llama', 'starting');
    await llama.stop();
    try { await llama.start(llamaCfg(settings.get(), { model: want })); setStatus('llama', 'ready'); }
    catch (e) { setStatus('llama', 'stopped'); throw e; }
  }
  await llama.useAdapter(file || null);
  llama.promptKind = promptFor(file); // the system prompt this adapter was trained with
  const thr = thresholdFor(file);
  const smp = samplingFor(file);
  const bar = carryOnFor(file);
  const blend = !!file && file.startsWith('blend:');
  // a blend is not a file the server loads at start, so it is kept apart from llama.lora
  settings.update(blend ? { activeBlend: file, listenThreshold: thr, listenSample: smp, carryOnP: bar } : { llama: { lora: file || '' }, activeBlend: '', listenThreshold: thr, listenSample: smp, carryOnP: bar });
  if (loop.settings) { loop.settings.listenThreshold = thr; loop.settings.listenSample = smp; loop.settings.carryOnP = bar; }
  send('sampling', smp); send('carryOn', bar);
  rememberFor(want, file);
  log(`llama: active adapter ${labelOf(file)}, listen threshold ${thr}`);
  const st = await adapterState(); send('adapters', st); send('threshold', thr); return st;
}
// the model picked is the one serving, and its adapter is the one the Model dropdown goes back to
function rememberFor(model, file) {
  settings.update({ llama: { model }, modelAdapters: { [baseOf(model)]: file || '' } });
}
ipcMain.handle('sampling:set', (_e, on) => {
  const name = keyOf(settings.get().activeBlend || settings.get().llama.lora);
  settings.update({ listenSample: !!on, sampling: { [name]: !!on } });
  if (loop.settings) loop.settings.listenSample = !!on;
  log(`llama: human timing ${on ? 'on (listen decision sampled)' : 'off (threshold rule)'} for ${name || 'the base model'}`);
});
ipcMain.handle('carryon:set', (_e, p) => {
  const name = keyOf(settings.get().activeBlend || settings.get().llama.lora);
  settings.update({ carryOnP: p, carryOns: { [name]: p } });
  if (loop.settings) loop.settings.carryOnP = p;
  log(`llama: carry-on bar ${p} for ${name || 'the base model'}`);
});
ipcMain.handle('threshold:remember', (_e, thr) => {
  const name = keyOf(settings.get().activeBlend || settings.get().llama.lora);
  settings.update({ thresholds: { [name]: thr } });
});
// mic frames, each with what the speaker played over the same stretch: through the echo canceller to the recogniser
const f32 = (b) => (b ? new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)) : null);
ipcMain.on('audio:in', (_e, mic, ref) => {
  const m = f32(mic), r = f32(ref);
  aec.process(m, r, (clean, level) => {
    stt.feed(clean);
    callrec.add({ mic: m, speaker: r, clean });
    // the cleaned mic's level before gain control, for ducking: the raw mic hears the assistant's own voice on
    // speakers, and gain control lifts the leftover echo; either would duck it every time it talks
    if (level !== undefined) send('aec:level', level);
  });
});
ipcMain.on('aec:delay', (_e, ms) => aec.setDelay(ms));
ipcMain.on('audio:playing', (_e, playing) => loop.setSpeaking(!!playing));
// Echo test (Call tab): one sentence on the speaker with the mic open and nobody talking, outside a call. Every word
// the recogniser picks up is echo the call would have heard as you.
const ECHO_SENTENCE = 'This is the echo test. If you can read these words below, the microphone is hearing my own voice from the speaker.';
ipcMain.handle('echo:test', async () => {
  if (services.call === 'live') throw new Error('end the call first');
  if (services.stt !== 'ready' || services.tts !== 'ready') throw new Error('load the models first');
  const s = settings.get();
  stt.reset();
  let heard = '';
  const collect = () => { const d = stt.drain(); if (d) heard += ' ' + d; };
  const poll = setInterval(collect, 300);
  tts.say(ECHO_SENTENCE, { flushWords: 999, speed: s.ttsSpeed, first: true });
  tts.flush({ speed: s.ttsSpeed });
  // until it has played and 1.5 s more (the recogniser's last chunk), at most 30 s
  await new Promise((resolve) => {
    const t0 = Date.now(); let started = false, endedAt = 0;
    const iv = setInterval(() => {
      if (loop.speaking) { started = true; endedAt = 0; } else if (started && !endedAt) endedAt = Date.now();
      if ((endedAt && Date.now() - endedAt > 1500) || Date.now() - t0 > 30000) { clearInterval(iv); resolve(); }
    }, 100);
  });
  clearInterval(poll); collect(); stt.reset();
  const words = heard.split(/\s+/).filter((w) => /[a-z0-9]/i.test(w));
  log(`echo test: heard ${words.length} words of ${ECHO_SENTENCE.split(' ').length} (echo canceller ${services.aec})${words.length ? ': ' + heard.trim() : ''}`);
  return { heard: heard.trim(), words: words.length, of: ECHO_SENTENCE.split(' ').length };
});
// Text-only call: type what the mic "heard" (used for testing without a microphone and by the tune tab)
ipcMain.handle('call:inject', (_e, text) => { stt.inject(text); });
ipcMain.handle('dialog:pickFile', async (_e, opts) => {
  const r = await dialog.showOpenDialog(win, { properties: ['openFile'], ...opts });
  return r.canceled ? null : r.filePaths[0];
});
ipcMain.handle('tune:run', (_e, stage, recipe) => tune.run(stage, recipe));
ipcMain.handle('tune:stop', () => tune.stop());
ipcMain.handle('tune:status', () => tune.status());
ipcMain.handle('tune:recipes', () => tune.recipes());
ipcMain.handle('tune:recipe', (_e, file) => tune.readRecipe(file));
ipcMain.handle('tune:saveRecipe', (_e, file, text, saveAs) => tune.saveRecipe(file, text, saveAs));
ipcMain.handle('tune:results', (_e, file) => tune.results(file));
ipcMain.handle('tune:bases', () => tune.bases());
ipcMain.handle('tune:weights', () => tune.weights());
ipcMain.handle('tune:progress', (_e, file) => tune.progress(file));
// Add a base model (src/main/bases.js): look it up, add it to models/bases.json, download it; the download's output goes
// to the Tune tab's log like a stage's
ipcMain.handle('bases:ggufs', (_e, repo) => require('./bases').ggufFiles(repo));
ipcMain.handle('bases:inspect', (_e, spec) => require('./bases').inspect(spec));
ipcMain.handle('bases:add', (_e, spec) => require('./bases').add(spec));
ipcMain.handle('bases:download', (_e, name) => require('./bases').download(name, (ev, p) => send('tune:' + ev, p), settings.get().storage));

function createWindow() {
  // 1760 x 1080 fits the Call tab's toolbar on two rows and the whole tab without scrolling (at 1280 x 860 it wrapped to
  // three and the log ran off the bottom); a smaller screen gets its work area
  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;
  win = new BrowserWindow({
    width: Math.min(1760, sw), height: Math.min(1080, sh), title: 'talkme',
    backgroundColor: '#101418',
    webPreferences: { preload: path.join(__dirname, '..', 'preload.js'), contextIsolation: true, nodeIntegration: false },
  });
  win.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
}

app.whenReady().then(createWindow);
app.on('window-all-closed', async () => { loop.stop(); await llama.stop(); await tts.stop(); tune.stop(); app.quit(); });
process.on('uncaughtException', (e) => log('main: uncaught ' + (e.stack || e.message)));
