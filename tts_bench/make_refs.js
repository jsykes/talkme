// Reference clips for Chatterbox's voice cloning, spoken by Kokoro's synthetic voices (no real person is cloned):
// ~10 s of plain, friendly speech per voice, written to models/tts_bench/chatterbox-voices/kokoro-<id>.wav.
//   node tts_bench/make_refs.js 3 2 7 11 16 21 26
const fs = require('fs');
const path = require('path');
const sherpa = require('sherpa-onnx-node');
const { voiceConfig } = require('../src/main/voices');

const OUT = path.join(__dirname, '..', 'models', 'tts_bench', 'chatterbox-voices');
fs.mkdirSync(OUT, { recursive: true });
const TEXT = "Hey, it's good to hear from you. I was just making some coffee and thinking about the weekend. " +
  "Honestly, I could use a slow morning. So, tell me, what have you been up to lately?";
const cfg = voiceConfig('kokoro:0');
const dir = cfg.dir;
const tts = new sherpa.OfflineTts({ model: { kokoro: { model: path.join(dir, fs.readdirSync(dir).find((f) => /^model.*\.onnx$/.test(f))), voices: path.join(dir, 'voices.bin'),
  tokens: path.join(dir, 'tokens.txt'), dataDir: path.join(dir, 'espeak-ng-data'), lexicon: path.join(dir, 'lexicon-us-en.txt'), dictDir: path.join(dir, 'dict') }, numThreads: 8, provider: 'cpu', debug: 0 }, maxNumSentences: 1 });
for (const sid of process.argv.slice(2).map(Number)) {
  const a = tts.generate({ text: TEXT, sid, speed: 1.0, enableExternalBuffer: false });
  const file = path.join(OUT, `kokoro-${sid}.wav`);
  sherpa.writeWave(file, { samples: a.samples, sampleRate: a.sampleRate });
  console.log(`${file}: ${(a.samples.length / a.sampleRate).toFixed(1)} s`);
}
