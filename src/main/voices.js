'use strict';
// The voices the app can speak with. A voice id is "piper:<dir>", "kokoro:<speaker id>", "chatterbox:turbo" or
// "orpheus:<speaker>" (the last two are GPU voices served by voice-server.js).
// Kokoro v1.0 speaker ids follow the sherpa-onnx voice table (English voices only listed here).
const path = require('path');
const fs = require('fs');
const { ROOT, llamaServer } = require('./config');
const { available } = require('./voice-server');

const TTS_DIR = path.join(ROOT, 'models', 'tts');
// Chatterbox clones a voice from a clip: every .wav here is a voice (tts_bench/make_refs.js makes some from Kokoro's
// synthetic voices; the app's "new voice" panel adds uploads and recordings)
const CB_VOICES = path.join(ROOT, 'models', 'tts_bench', 'chatterbox-voices');
const FALLBACK = 'kokoro:7';
const KOKORO_DIR = path.join(TTS_DIR, 'kokoro-multi-lang-v1_0');
const KOKORO_EN = [
  [0, 'Alloy', 'US, female'], [1, 'Aoede', 'US, female'], [2, 'Bella', 'US, female'], [3, 'Heart', 'US, female'],
  [4, 'Jessica', 'US, female'], [5, 'Kore', 'US, female'], [6, 'Nicole', 'US, female'], [7, 'Nova', 'US, female'],
  [8, 'River', 'US, female'], [9, 'Sarah', 'US, female'], [10, 'Sky', 'US, female'],
  [11, 'Adam', 'US, male'], [12, 'Echo', 'US, male'], [13, 'Eric', 'US, male'], [14, 'Fenrir', 'US, male'],
  [15, 'Liam', 'US, male'], [16, 'Michael', 'US, male'], [17, 'Onyx', 'US, male'], [18, 'Puck', 'US, male'],
  [20, 'Alice', 'UK, female'], [21, 'Emma', 'UK, female'], [22, 'Isabella', 'UK, female'], [23, 'Lily', 'UK, female'],
  [24, 'Daniel', 'UK, male'], [25, 'Fable', 'UK, male'], [26, 'George', 'UK, male'], [27, 'Lewis', 'UK, male'],
];

function listVoices() {
  const out = [];
  // the GPU voices first: Orpheus is the default engine, Chatterbox the one for custom voices
  if (available('orpheus')) for (const [sp, kind] of ORPHEUS) out.push({ id: `orpheus:${sp}`, label: `Orpheus ${sp[0].toUpperCase() + sp.slice(1)} (GPU, ${kind})` });
  if (available('chatterbox')) {
    out.push({ id: 'chatterbox:turbo', label: 'Chatterbox built-in (GPU)' });
    for (const v of chatterboxVoices()) out.push({ id: `chatterbox:${v}`, label: `Chatterbox ${cbLabel(v)} (GPU, cloned)` });
  }
  try {
    for (const d of fs.readdirSync(TTS_DIR)) {
      if (!d.startsWith('vits-piper-')) continue;
      const name = d.replace('vits-piper-', '').replace(/-/g, ' ');
      out.push({ id: `piper:${d}`, label: `Piper ${name} (fastest)` });
    }
  } catch {}
  if (fs.existsSync(path.join(KOKORO_DIR, 'model.onnx'))) {
    for (const [sid, name, kind] of KOKORO_EN) out.push({ id: `kokoro:${sid}`, label: `Kokoro ${name} (${kind})` });
  }
  return out;
}

function chatterboxVoices() {
  try { return fs.readdirSync(CB_VOICES).filter((f) => f.endsWith('.wav')).map((f) => f.slice(0, -4)).sort(); } catch { return []; }
}
const KOKORO_NAME = Object.fromEntries(KOKORO_EN.map(([sid, name]) => [String(sid), name]));
const cbLabel = (v) => (/^kokoro-\d+$/.test(v) ? `as ${KOKORO_NAME[v.split('-')[1]] || v} (from Kokoro)` : v);
const ORPHEUS = [['tara', 'female'], ['leah', 'female'], ['jess', 'female'], ['mia', 'female'], ['zoe', 'female'], ['leo', 'male'], ['dan', 'male'], ['zac', 'male']];
// a GPU voice takes ~3.5 GB, so the chat model keeps this many more MoE layers' experts on the CPU (~0.6 GB each)
const GPU_VOICE_MOE = 7;

// Worker config for a voice id. Piper's dictionary spells "mm-hm"; sayAs rewrites such words before speaking.
function voiceConfig(id) {
  let [engine, rest] = String(id || '').split(':');
  // a GPU voice that is not installed (or a deleted clip) falls back to Kokoro
  if ((engine === 'orpheus' && !available('orpheus')) || (engine === 'chatterbox' && (!available('chatterbox') || (rest !== 'turbo' && !chatterboxVoices().includes(rest))))) {
    [engine, rest] = FALLBACK.split(':');
  }
  if (engine === 'kokoro') {
    return { engine: 'kokoro', dir: KOKORO_DIR, sid: Number(rest) || 0, threads: 8, sayAs: true };
  }
  // GPU voices: synthesised by a local server (voice-server.js); longer pieces (8 words or a sentence end) because
  // every piece restarts the voice's intonation; they say "mm-hm" themselves, so no rewriting
  if (engine === 'chatterbox') {
    return { engine: 'remote', kind: 'chatterbox', port: 8091, url: 'http://127.0.0.1:8091', flushWords: 8, justInTime: true, sayAs: false, moeExtra: GPU_VOICE_MOE,
      ref: rest && rest !== 'turbo' ? path.join(CB_VOICES, rest + '.wav') : null };
  }
  if (engine === 'orpheus') {
    // streams: each piece plays while it is generated, after a 0.3 s pre-roll (serve_common.py)
    // two pieces at once (its llama-server runs 2 slots): the next piece's setup overlaps the current one
    // Q4_K_M: preferred by ear over Q8_0 on the tag lines (tts_bench/orpheus_tags.py, 2026-10-27), and ~40% faster to
    // synthesise, which with the GPU shared is a shorter first sound
    return { engine: 'remote', kind: 'orpheus', speaker: rest || 'tara', port: 8092, url: 'http://127.0.0.1:8092', flushWords: 8, justInTime: true, stream: true, parallel: 2, sayAs: false, moeExtra: GPU_VOICE_MOE,
      llamaPort: 8090, llamaExe: llamaServer(), llamaModel: path.join(ROOT, 'models', 'tts_bench', 'orpheus', 'orpheus-3b-0.1-ft-Q4_K_M.gguf') };
  }
  const dir = path.join(TTS_DIR, rest || 'vits-piper-en_US-lessac-medium');
  const onnx = fs.existsSync(dir) ? fs.readdirSync(dir).find((f) => f.endsWith('.onnx')) : 'model.onnx';
  return { engine: 'piper', dir, model: onnx, sid: 0, threads: 2, sayAs: true };
}

// Saves a new Chatterbox voice from a clip (any format ffmpeg reads: an upload, or a recording from the app):
// mono 24 kHz WAV, at most 15 s. Returns the voice id.
function addChatterboxVoice(name, srcFile) {
  const { spawnSync } = require('child_process');
  const safe = String(name || '').trim().replace(/[^\w -]+/g, '').replace(/\s+/g, '-').slice(0, 40);
  if (!safe) throw new Error('give the voice a name');
  if (/^kokoro-\d+$/.test(safe) || safe === 'turbo') throw new Error('that name is taken');
  fs.mkdirSync(CB_VOICES, { recursive: true });
  const out = path.join(CB_VOICES, safe + '.wav');
  const r = spawnSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', srcFile, '-ac', '1', '-ar', '24000', '-t', '15', out], { windowsHide: true });
  if (r.error) throw new Error('ffmpeg is needed to read the clip (' + r.error.message + ')');
  if (r.status !== 0 || !fs.existsSync(out)) throw new Error('could not read the clip: ' + String(r.stderr || '').trim());
  const secs = (fs.statSync(out).size - 44) / 2 / 24000;
  if (secs < 3) { fs.unlinkSync(out); throw new Error(`the clip is only ${secs.toFixed(1)} s; 5-15 s of clear speech works best`); }
  return `chatterbox:${safe}`;
}

function removeChatterboxVoice(id) {
  const [engine, name] = String(id).split(':');
  if (engine !== 'chatterbox' || !name || name === 'turbo') throw new Error('only your own Chatterbox voices can be removed');
  fs.unlinkSync(path.join(CB_VOICES, name + '.wav'));
}

module.exports = { listVoices, voiceConfig, addChatterboxVoice, removeChatterboxVoice };
