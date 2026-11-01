// Standalone check of STT (a test wav fed in 100 ms frames, drained every 600 ms like the loop) and TTS. No GPU needed.
//   node scripts/check_speech.js
const path = require('path');
const fs = require('fs');
const sherpa = require('sherpa-onnx-node');
const { DEFAULTS } = require('../src/main/config');
const { STT } = require('../src/main/stt');
const { TTS } = require('../src/main/tts');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const stt = new STT(console.log); await stt.start(DEFAULTS.stt);
  const dir = path.join(DEFAULTS.stt.dir, 'test_wavs');
  const wav = path.join(dir, fs.readdirSync(dir).filter((f) => f.endsWith('.wav'))[0]);
  const w = sherpa.readWave(wav);
  const s = w.sampleRate === 16000 ? w.samples : new sherpa.LinearResampler(w.sampleRate, 16000).resample(w.samples);
  console.log('wav', path.basename(wav), (s.length / 16000).toFixed(1), 's');
  const ticks = [];
  const audio = new Float32Array(s.length + 16000 * 2); audio.set(s);
  for (let i = 0, t = 0; i < audio.length; i += 1600, t += 100) {
    stt.feed(audio.slice(i, i + 1600));
    await sleep(100); // real time, so the worker keeps up as it would live
    if ((t + 100) % 600 === 0) ticks.push(stt.drain() || '<no voice>');
  }
  console.log('ticks:', JSON.stringify(ticks));
  await stt.stop();
  const tts = new TTS(console.log); await tts.start(require('../src/main/voices').voiceConfig(DEFAULTS.tts.voice));
  tts.onAudio = (a) => { console.log(`tts: "${a.text}" -> ${(a.samples.length / a.sampleRate).toFixed(2)} s of audio in ${a.ms} ms`); tts.stop(); };
  tts.say('Do not mix those chemicals.', { flushWords: 6 });
})();
