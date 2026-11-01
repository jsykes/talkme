'use strict';
// Paths and settings. Settings persist to <userData>/settings.json; paths default to the repo layout.
const path = require('path');
const fs = require('fs');
const { readMeta } = require('./gguf');

const ROOT = path.resolve(__dirname, '..', '..');
const WIN = process.platform === 'win32';

// llama.cpp's server: the release unpacked into vendor/llama (scripts/setup.js), else on Linux one on the PATH
function llamaServer() {
  const exe = path.join(ROOT, 'vendor', 'llama', WIN ? 'llama-server.exe' : 'llama-server');
  return fs.existsSync(exe) || WIN ? exe : 'llama-server';
}
// the python of a uv environment (aec/, tts_bench/<voice>/)
const venvPython = (dir) => path.join(dir, '.venv', ...(WIN ? ['Scripts', 'python.exe'] : ['bin', 'python']));

const DEFAULTS = {
  tickMs: 600,
  listenThreshold: 0.3, // for assistant adapters that do not carry their own (adapterInfo); partner adapters use 0.6
  carryOnP: 0.9, // P(listen) needed to stop mid-sentence into a silence (llama.js carry-on bar); 0 = the threshold
  ttsFlushWords: 6,
  ttsSpeed: 1.0,
  llama: {
    exe: llamaServer(),
    model: path.join(ROOT, 'models', 'base', 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'),
    // default adapter: v1, the g26b4 partner (2026-10-28: picked from live calls; the app serves it on the
    // Gemma 4 26B-A4B heretic base and sends its own system prompt, see config.js BASES, index.js promptFor; listen threshold
    // 0.6, the partner default, with the carry-on bar below). talkme-partner-v0-hauhau.gguf was the default before.
    lora: path.join(ROOT, 'models', 'lora', 'talkme-partner-v1-gemma4a4b.gguf'),
    port: 8080,
    ctx: 32768,
    ubatch: 512,
    ngl: 99,
    nCpuMoe: 8, // raise this when the GPU is shared (e.g. a game is running); 30+ needs only a few GB of VRAM
  },
  stt: {
    // Nemotron 0.6B, 560 ms chunks: best accuracy, native punctuation and casing, last word ~0.2 s after speech ends.
    // Alternatives in models/asr: the 160 ms Nemotron (faster but needs a quiet CPU) and the older zipformer.
    dir: path.join(ROOT, 'models', 'asr', 'sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25'),
    threads: 4,
    punctDir: path.join(ROOT, 'models', 'punct', 'sherpa-onnx-online-punct-en-2024-08-06'),
    finalPunct: true, // close a quiet utterance with ? or . (see stt.js finalMark); false = leave it as the recogniser has it
  },
  tts: {
    // see voices.js: Orpheus Tara (GPU, the most natural; the default, picked by ear in tts_bench), Chatterbox (GPU,
    // custom voices from a clip), Kokoro (CPU, ~0.3 s a phrase) or Piper (CPU, fastest). A GPU voice that is not
    // installed falls back to Kokoro 7.
    voice: 'orpheus:tara',
  },
  // who the friend is and what it remembers about you, for adapters trained with a persona (g26m on). Empty: the plain
  // partner prompt, which those adapters also know (and then invent no shared past).
  persona: { text: '', memories: '' },
  // For calls on speakers. aec: cancel the assistant's own voice out of the mic before the recogniser hears it (WebRTC
  // AEC3 in aec/aec_server.py; needs the aec/ environment). ownWordsFilter: while it is speaking, and briefly after,
  // drop heard words that are the ones it just said: the last bit of echo either one lets through.
  aec: { enabled: true, noiseSuppression: true, gainControl: true },
  ownWordsFilter: true,
  saveCallAudio: false, // each call's mic, speaker and cleaned-mic audio to the data folder's calls/ (last 5 calls)
  // a folder (on another drive) for the large files: training checkpoints, new bases' GGUFs and runs' caches go there,
  // linked into the project (storageLink); '' = inside the project
  storage: '',
  // once it speaks, its words are drawn (llama.js tick wordSampling); null for always the likeliest words
  wordSampling: { temperature: 0.7, top_p: 0.9 },
  // the next tick waits while the voice is more than ~1.2 s behind the model's words (loop.js VOICE_LEAD_MS)
  anchorToVoice: true,
};

// The adapter was trained on this text. Keep it byte for byte (only the tick length is interpolated).
// kind: 'assistant' (every adapter up to run 8), 'partner' (run 9 on) or 'persona' (g26m on: the partner text plus who
// the friend is and what it remembers, when a persona is set). Each adapter must get the exact text it was trained with
// (tune/common.py SYSTEM / PARTNER_SYSTEM / persona_system).
function systemPrompt(tickMs, kind = 'assistant', persona = null) {
  const tick = (tickMs / 1000).toString();
  if (kind === 'persona' && persona && String(persona.text || '').trim()) return personaPrompt(systemPrompt(tickMs, 'partner'), persona);
  if (kind === 'partner' || kind === 'persona') return `You are a friend on a live phone call; both of you can talk at the same time. Every ${tick} seconds you get what the microphone heard in that moment, as a streaming transcript fragment, or <no voice>. You reply with up to three new words to say next, or <listen> to add nothing.`;
  return `You are a voice assistant in a live phone call; both sides can talk at the same time. Every ${tick} seconds you get what the microphone heard in that moment, as a streaming transcript fragment, or <no voice>. You reply with up to three new words to say next, or <listen> to add nothing.`;
}

// tune/common.py persona_system, byte for byte. memories: one per line (a leading "- " is dropped).
function personaPrompt(partner, persona) {
  const mems = String(persona.memories || '').split(/\r?\n/).map((m) => m.replace(/^\s*[-*•]\s*/, '').trim()).filter(Boolean);
  const s = `${partner}\n\nYou are ${String(persona.text).trim()}`;
  if (mems.length) return `${s}\nWhat you remember about the caller from earlier calls:\n${mems.map((m) => `- ${m}`).join('\n')}\nOnly bring up a shared past that is in this list.`;
  return `${s}\nYou know the caller as a friend, but nothing about their life from earlier calls.`;
}

// Adapters were renamed to carry their training run (2026-10-18), then to say what they are (2026-10-31; the run is in
// the file, talkme.recipe). Saved settings follow these names, and the tests that ran under an old one still count.
const RENAMED = { 'talkme-polite.gguf': 'talkme-run4-polite.gguf', 'talkme-attentive.gguf': 'talkme-run5-attentive.gguf',
  'talkme-brisk.gguf': 'talkme-run2-brisk.gguf', 'talkme-warm.gguf': 'talkme-run2-warm.gguf', 'talkme-playful.gguf': 'talkme-run2-playful.gguf',
  'talkme-g26b4-partner-gemma4a4b.gguf': 'talkme-partner-v1-gemma4a4b.gguf',
  'talkme-g26b4std-partner-gemma4a4bstd.gguf': 'talkme-partner-v1-gemma4a4bstd.gguf',
  'talkme-m3b4-partner-ministral3.gguf': 'talkme-partner-v1-ministral3.gguf',
  'talkme-gptoss12-partner-gptoss.gguf': 'talkme-partner-light-gptoss.gguf',
  'talkme-gptoss13-partner-gptoss.gguf': 'talkme-partner-full-gptoss.gguf',
  'talkme-g26-partner-gemma4a4b.gguf': 'talkme-partner-v0-gemma4a4b.gguf',
  'talkme-run12-partner-hauhau.gguf': 'talkme-partner-v0-hauhau.gguf' };

function deepMerge(a, b) {
  const out = { ...a };
  for (const k of Object.keys(b || {})) {
    out[k] = b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) ? deepMerge(a[k] || {}, b[k]) : b[k];
  }
  return out;
}

class Settings {
  constructor(file) {
    this.file = file;
    this.data = DEFAULTS;
    try {
      if (fs.existsSync(file)) this.data = deepMerge(DEFAULTS, JSON.parse(fs.readFileSync(file, 'utf8')));
    } catch (e) {
      console.error('settings: could not read', file, e.message);
    }
    this.migrateAdapters();
  }
  // Saved choices and per-adapter settings follow RENAMED; a saved adapter that no longer exists falls back to the
  // default (none, if that has gone too) instead of stopping the server.
  migrateAdapters() {
    const moved = (file) => (file && RENAMED[path.basename(file)] ? path.join(path.dirname(file), RENAMED[path.basename(file)]) : file);
    const byName = (saved) => Object.fromEntries(Object.entries(saved).map(([k, v]) => [RENAMED[k] || k, v]));
    const d = this.data;
    let lora = moved(d.llama && d.llama.lora);
    if (lora && !fs.existsSync(lora)) lora = fs.existsSync(DEFAULTS.llama.lora) ? DEFAULTS.llama.lora : '';
    // a saved server path from another checkout or system: this one's
    const exe = d.llama && d.llama.exe && path.isAbsolute(d.llama.exe) && fs.existsSync(d.llama.exe) ? d.llama.exe : llamaServer();
    this.data = { ...d, llama: { ...d.llama, lora, exe }, ...(d.thresholds ? { thresholds: byName(d.thresholds) } : {}),
      ...(d.sampling ? { sampling: byName(d.sampling) } : {}), ...(d.carryOns ? { carryOns: byName(d.carryOns) } : {}),
      ...(d.modelAdapters ? { modelAdapters: Object.fromEntries(Object.entries(d.modelAdapters).map(([k, v]) => [k, moved(v)])) } : {}) };
  }
  // An adapter file was renamed (`to`, a file name) or removed (`to` null): every saved setting that names it follows.
  // Done here and not through update(), which merges and so can only add keys. A blend using a removed adapter goes too.
  rewriteAdapter(name, to) {
    const same = (f) => path.basename(String(f || '')).toLowerCase() === name.toLowerCase();
    const file = (f) => (same(f) ? (to ? path.join(path.dirname(f), to) : '') : f);
    const spec = (s) => {
      if (same(s)) return to || '';
      if (!String(s).startsWith('blend:')) return s;
      const parts = s.slice(6).split(',').map((p) => p.split('='));
      if (!parts.some(([f]) => same(f))) return s;
      return to ? 'blend:' + parts.map(([f, sc]) => `${same(f) ? to : f}=${sc}`).join(',') : '';
    };
    const keys = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [spec(k), v]).filter(([k]) => k));
    // a value of '' there means "no adapter on this base"; with a removed adapter's entry gone instead, the Model
    // dropdown goes back to that base's usual one
    const bases = (o) => Object.fromEntries(Object.entries(o).filter(([, v]) => to || !v || spec(v)).map(([k, v]) => [k, same(v) ? file(v) : spec(v)]));
    const d = this.data;
    this.data = { ...d, llama: { ...d.llama, lora: file(d.llama.lora) }, activeBlend: spec(d.activeBlend || ''),
      ...(d.thresholds ? { thresholds: keys(d.thresholds) } : {}), ...(d.sampling ? { sampling: keys(d.sampling) } : {}),
      ...(d.carryOns ? { carryOns: keys(d.carryOns) } : {}), ...(d.modelAdapters ? { modelAdapters: bases(d.modelAdapters) } : {}) };
    this.save();
    return this.data;
  }
  get() { return this.data; }
  update(patch) {
    this.data = deepMerge(this.data, patch);
    this.save();
    return this.data;
  }
  save() {
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    fs.writeFileSync(this.file, JSON.stringify(this.data, null, 2));
  }
}

// Puts a folder of the project (rel, e.g. models/hf/gpt-oss-20b-BF16) in the storage folder, at the same path there,
// and links it into the project (a directory junction on Windows, a symlink elsewhere), so everything that reads the
// project path follows it. Only for a folder that does not exist yet: one already here stays where it is. Returns the
// folder it made, or null.
function storageLink(rel, storage) {
  if (!storage) return null;
  const here = path.join(ROOT, rel);
  try { fs.lstatSync(here); return null; } catch {}
  if (!path.isAbsolute(storage)) throw new Error(`the folder for large files (Settings) has to be a full path: ${storage}`);
  if (!fs.existsSync(path.parse(storage).root)) throw new Error(`the folder for large files is on a drive that is not here: ${storage}`);
  const there = path.join(storage, rel);
  fs.mkdirSync(there, { recursive: true });
  fs.mkdirSync(path.dirname(here), { recursive: true });
  fs.symlinkSync(there, here, 'junction');
  return there;
}

// The base model's chat format, byte for byte as training renders it (tune/profiles.py FORMATS). Qwen3.6 (original
// and HauhauCS) speaks ChatML; Gemma 4 its own <|turn> format, without the empty thought block its template adds;
// Ministral 3 Mistral's [INST] format, replies closed by </s> with nothing between turns; gpt-oss OpenAI's harmony.
const FORMATS = {
  'qwen3.6': { bos: '', system: (s) => `<|im_start|>system\n${s}<|im_end|>\n`, user: (h) => `<|im_start|>user\n${h}<|im_end|>\n<|im_start|>assistant\n`, end: '<|im_end|>', sep: '\n' },
  gemma4: { bos: '<bos>', system: (s) => `<|turn>system\n${s}<turn|>\n`, user: (h) => `<|turn>user\n${h}<turn|>\n<|turn>model\n`, end: '<turn|>', sep: '\n' },
  mistral: { bos: '<s>', system: (s) => `[SYSTEM_PROMPT]${s}[/SYSTEM_PROMPT]`, user: (h) => `[INST]${h}[/INST]`, end: '</s>', sep: '' },
  // gpt-oss: harmony, replies straight in the final channel and closed by <|return|> (tune/profiles.py says why)
  harmony: { bos: '', system: (s) => `<|start|>system<|message|>Reasoning: low\n\n# Valid channels: final.<|end|><|start|>developer<|message|>${s}<|end|>`,
    user: (h) => `<|start|>user<|message|>${h}<|end|><|start|>assistant<|channel|>final<|message|>`, end: '<|return|>', sep: '' },
};

// The base models the app can serve and train on: models/bases.json, which the training pipeline and setup read too
// (its _about says what each field means). An adapter runs on the base it records in its own file (adapterInfo below),
// else the one its file name ends with, talkme-<prompt>-<version>-<base>.gguf (talkme-partner-v1-gemma4a4b.gguf: the
// 26B-A4B heretic). Bases with the same `arch` have the same weight shapes, so one server loads all their adapters (a
// Qwen adapter on Gemma stops the server). addBase (the Tune tab) writes a new one into the file and into BASES.
const BASES_FILE = path.join(ROOT, 'models', 'bases.json');
const BASES = Object.fromEntries(Object.entries(JSON.parse(fs.readFileSync(BASES_FILE, 'utf8'))).filter(([k]) => !k.startsWith('_')));
function addBase(name, row) {
  if (!/^[a-z0-9]+$/.test(name)) throw new Error('a base name is lower-case letters and digits (it ends adapter file names)');
  if (BASES[name]) throw new Error(`there is already a base called ${name}`);
  const all = JSON.parse(fs.readFileSync(BASES_FILE, 'utf8'));
  all[name] = row;
  fs.writeFileSync(BASES_FILE, JSON.stringify(all, null, 2) + '\n');
  BASES[name] = row;
  return row;
}
// Adapters from elsewhere that carry no talkme metadata, by file name
const KNOWN_ADAPTERS = {
  // the author's; tuned for 0.3 on calls with the "?" on the last word, but our recogniser sends it a tick or two later,
  // after quiet ticks mid-question, and at 0.3 it often never answered (P(listen) 0.4-0.9 for seconds); 0.65 works live
  'microturn-duplex-lora.gguf': { base: 'qwen', prompt: 'assistant', listen: 0.65, label: 'Published micro-turn adapter (codacus)' },
};
const TALKME_KEYS = ['general.type', 'general.architecture', 'talkme.base', 'talkme.base_model', 'talkme.prompt',
  'talkme.listen_threshold', 'talkme.carry_on', 'talkme.label', 'talkme.recipe'];
// a base model's header runs to megabytes of tokenizer; these come first in it, so reading stops early
const BASE_KEYS = ['general.architecture', 'general.name'];
const headers = new Map(); // path and keys -> { mtime, meta }
function header(file, keys = TALKME_KEYS) {
  try {
    const p = path.resolve(ROOT, String(file)), mtime = fs.statSync(p).mtimeMs, id = p + '|' + keys.join(), hit = headers.get(id);
    if (hit && hit.mtime === mtime) return hit.meta;
    const meta = readMeta(p, { keys });
    headers.set(id, { mtime, meta });
    return meta;
  } catch { return null; }
}
// the base on disk that calls itself `name` (its GGUF's general.name, e.g. "Gemma 4 26B A4B It Heretic")
function baseNamed(name) {
  return Object.keys(BASES).find((k) => { const h = header(BASES[k].gguf, BASE_KEYS); return h && h['general.name'] === name; }) || null;
}
const round2 = (x) => (typeof x === 'number' ? Math.round(x * 100) / 100 : undefined);
// An adapter's card: a small JSON file beside it (talkme-x.gguf, talkme-x.json) with the notes and name given in the
// Adapters tab and what the pipeline last recorded about it (tune/pipeline.py write_card). Not in the GGUF: changing a
// key there rewrites the whole file, up to 587 MB, and adds a new Git LFS copy each time.
const cardPath = (file) => path.resolve(ROOT, String(file)).replace(/\.gguf$/i, '.json');
const cards = new Map(); // path -> { mtime, card }
function readCard(file) {
  try {
    const p = cardPath(file), mtime = fs.statSync(p).mtimeMs, hit = cards.get(p);
    if (hit && hit.mtime === mtime) return hit.card;
    const card = JSON.parse(fs.readFileSync(p, 'utf8'));
    cards.set(p, { mtime, card });
    return card;
  } catch { return {}; }
}
// patch: the keys to set; an empty value removes the key, and a card with nothing left goes
function writeCard(file, patch) {
  const p = cardPath(file), card = { ...readCard(file), ...patch };
  for (const k of Object.keys(card)) if (card[k] === '' || card[k] == null) delete card[k];
  cards.delete(p);
  if (!Object.keys(card).length) { fs.rmSync(p, { force: true }); return card; }
  fs.writeFileSync(p + '.tmp', JSON.stringify(card, null, 2) + '\n');
  fs.renameSync(p + '.tmp', p);
  return card;
}
// every other name this file has gone by: renames in the Adapters tab (its card remembers them), then RENAMED
function formerNames(file) {
  const now = path.basename(String(file)), names = new Set([now, ...(readCard(file).formerNames || [])]);
  for (let grew = true; grew;) {
    grew = false;
    for (const [old, to] of Object.entries(RENAMED)) if (names.has(to) && !names.has(old)) { names.add(old); grew = true; }
  }
  names.delete(now);
  return [...names];
}
// What an adapter file says about itself (tune/adapter_meta.py writes it at export): the base it was trained on, its
// system prompt, listen threshold, carry-on bar and name in the list. Older files and other people's: KNOWN_ADAPTERS,
// then the base the file name ends with. base null: an adapter that says nothing (the app will not guess: a Qwen
// adapter on Gemma stops the server, and v1's on the standard Gemma went deaf to short questions). A name given in the
// Adapters tab is on its card and comes first.
function adapterInfo(file) {
  const name = path.basename(String(file || '')).toLowerCase();
  const m = header(file) || {};
  const card = readCard(file);
  const known = KNOWN_ADAPTERS[name] || {};
  let base = (m['talkme.base_model'] && baseNamed(m['talkme.base_model'])) || (BASES[m['talkme.base']] ? m['talkme.base'] : null) || known.base || null;
  if (!base) { const e = /-([a-z0-9]+).gguf$/.exec(name); if (e && BASES[e[1]]) base = e[1]; }
  return {
    base, prompt: m['talkme.prompt'] || known.prompt, listen: round2(m['talkme.listen_threshold']) ?? known.listen,
    carryOn: round2(m['talkme.carry_on']), label: card.label || m['talkme.label'] || card.suggestedLabel || known.label, recipe: m['talkme.recipe'],
    arch: m['general.architecture'], stamped: !!m['talkme.base'],
  };
}
const isAdapter = (file) => { const h = header(file); return h ? h['general.type'] === 'adapter' : /[\\/]lora[\\/]|(^|[\\/])talkme-/i.test(String(file)); };
// a GGUF (a base model) or an adapter -> the name of its base (a BASES key; null for an adapter that does not say)
function baseOf(file) {
  const b = path.basename(String(file || '')).toLowerCase();
  const named = Object.keys(BASES).find((k) => path.basename(BASES[k].gguf).toLowerCase() === b);
  if (named) return named;
  if (file && isAdapter(file)) return adapterInfo(file).base;
  // a base GGUF not in the table: guess from its name
  if (/gemma-?4/.test(b)) return /a4b/.test(b) ? 'gemma4a4b' : 'gemma4';
  if (/ministral-?3/.test(b)) return 'ministral3';
  return 'qwen';
}
// a base model's own architecture name (general.architecture: qwen35moe, gemma4, mistral3), to tell which models an
// adapter that does not name its base could belong to
const ggufArch = (file) => (header(file, BASE_KEYS) || {})['general.architecture'] || null;
const baseRow = (file) => BASES[baseOf(file)];
const familyOf = (file) => baseRow(file).format; // its chat format
const baseKey = (file) => baseRow(file).arch; // which adapters can load beside it

module.exports = { ROOT, DEFAULTS, Settings, storageLink, llamaServer, venvPython, systemPrompt, personaPrompt, FORMATS, BASES, BASES_FILE, addBase, baseOf, baseRow, familyOf, baseKey, adapterInfo, ggufArch, RENAMED, cardPath, readCard, writeCard, formerNames };
