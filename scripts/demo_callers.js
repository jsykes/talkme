// Renders the demo caller's lines (tests/demo_scenarios.json) with an Orpheus voice, once, so every model hears the
// same caller: tests/out/demo/caller/<scenario>-<turn>.wav (24 kHz mono). Starts the Orpheus servers (GPU, ~3 GB)
// and stops them after. Lines already rendered are kept; --force renders them again. --voice <speaker> renders them
// with another Orpheus speaker into caller/<speaker>/ (for demo_record.js --caller <speaker>).
//   node scripts/demo_callers.js [--force] [--voice dan] [--scenarios tests/demo_scenarios_long.json]
const fs = require('fs');
const path = require('path');
const { ROOT } = require('../src/main/config');
const { voiceConfig } = require('../src/main/voices');
const { VoiceServer } = require('../src/main/voice-server');

const si = process.argv.indexOf('--scenarios');
const SCEN = require(path.resolve(ROOT, si > 0 ? process.argv[si + 1] : 'tests/demo_scenarios.json'));
const force = process.argv.includes('--force');
const vi = process.argv.indexOf('--voice');
const SPEAKER = vi > 0 ? process.argv[vi + 1] : null;
const OUT = path.join(ROOT, 'tests', 'out', 'demo', 'caller', ...(SPEAKER ? [SPEAKER] : []));

function wav(file, f32, sr) {
  const pcm = Buffer.alloc(f32.length * 2);
  for (let i = 0; i < f32.length; i++) pcm.writeInt16LE(Math.round(Math.max(-1, Math.min(1, f32[i])) * 32767), i * 2);
  const h = Buffer.alloc(44);
  h.write('RIFF', 0); h.writeUInt32LE(36 + pcm.length, 4); h.write('WAVEfmt ', 8); h.writeUInt32LE(16, 16);
  h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22); h.writeUInt32LE(sr, 24); h.writeUInt32LE(sr * 2, 28);
  h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34); h.write('data', 36); h.writeUInt32LE(pcm.length, 40);
  fs.writeFileSync(file, Buffer.concat([h, pcm]));
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const todo = SCEN.scenarios.flatMap((s) => s.turns.map((t, i) => ({ file: path.join(OUT, `${s.name}-${i}.wav`), text: t.say })))
    .filter((x) => force || !fs.existsSync(x.file));
  if (!todo.length) { console.log('all caller lines are rendered'); return; }
  const cfg = voiceConfig(SPEAKER ? 'orpheus:' + SPEAKER : SCEN.callerVoice);
  const server = new VoiceServer((s) => { if (/ready|error|stopped/i.test(s)) console.log(s); });
  try {
    await server.start(cfg);
    for (const x of todo) {
      const r = await fetch(cfg.url + '/tts', { method: 'POST', body: JSON.stringify({ text: x.text, stream: false }) });
      if (!r.ok) throw new Error(`tts ${r.status} for "${x.text}"`);
      const sr = Number(r.headers.get('x-sample-rate')) || 24000;
      const f32 = new Float32Array(await r.arrayBuffer());
      wav(x.file, f32, sr);
      console.log(`${path.basename(x.file)}  ${(f32.length / sr).toFixed(1)} s  "${x.text}"`);
    }
  } finally { await server.stop(); }
})().catch((e) => { console.error(e.message); process.exit(1); });
