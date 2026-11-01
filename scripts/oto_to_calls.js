// Builds calls from the transcribed otoSpeech corpus (data/otospeech/asr/*.json, from oto_transcribe.js). Each
// conversation gives two perspectives (channel 0 calls channel 1 and the reverse), cut into consecutive windows of
// --ticks ticks (460 = the training window). `heard` is the caller's recogniser output; `said` and the tags come
// from the other person's word timings (scripts/lib/turntags.js, the same rules as the real-conversation test).
// By default the calls are marked `timing_only`: training learns WHEN to talk from them (listen vs speak, back-channel
// placement) but not WHAT the person said, which suits an assistant. --words keeps the person's words as targets too,
// for the conversation-partner adapter (casual phrasing, asking back, volunteering things).
// About 5% of conversations (by session id) go to a development file and never to training.
//   node scripts/oto_to_calls.js [--ticks 460] [--words] [--out tune/out/calls_oto.jsonl] [--dev-out tune/out/calls_oto_dev.jsonl]
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { buildCall, TICK } = require('./lib/turntags');

const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) => (a.startsWith('--') ? [a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true] : [])).filter((x) => x.length));
const ROOT = path.resolve(__dirname, '..');
const ASR = path.join(ROOT, 'data', 'otospeech', 'asr');
const OUT = path.resolve(ROOT, args.out || 'tune/out/calls_oto.jsonl');
const DEV = path.resolve(ROOT, args['dev-out'] || 'tune/out/calls_oto_dev.jsonl');
const N = Number(args.ticks || 460);
// per-tag loss weights, as tune/common.py WEIGHTS (the values double as tag ids for train_lora --weight-map);
// overlap_start (talking over a caller who keeps going) and self_resume (carrying on one's own turn) are new here
const W = { listen_speech: 0.3, speak: 2.0, listen_idle: 0.2, listen_pause: 2.0, listen_last: 0.5, continuer: 1.5, start: 6.0, barge_stop: 3.0, overlap_start: 0.4, self_resume: 1.2 };
const isDev = (id) => crypto.createHash('md5').update(id).digest()[0] < 13; // ~5%, stable

const train = [], dev = [], count = {};
let sessions = 0, ticks = 0;
for (const f of fs.readdirSync(ASR).filter((x) => x.endsWith('.json')).sort()) {
  let j;
  try { j = JSON.parse(fs.readFileSync(path.join(ASR, f), 'utf8')); } catch { continue; } // still being written
  sessions++;
  for (const c of [0, 1]) {
    const caller = j.channels[c], ref = j.channels[1 - c];
    for (let k0 = 0; k0 + N <= caller.heard.length; k0 += N) {
      const t0 = k0 * TICK;
      // skip windows that touch a redacted (muted) stretch: half a conversation would be missing
      if (j.redacted.some((r) => r.end_sec > t0 && r.start_sec < t0 + N * TICK)) continue;
      const { tags, said } = buildCall(caller.words, ref.words, t0, N * TICK);
      // nobody talking for most of the window (one side muted or away): not useful
      if (tags.filter((t) => t === 'listen_idle').length > N * 0.5) continue;
      const call = {
        split: 'oto', source: `otoSpeech-full-duplex-processed-141h/${j.session_id}`, caller: c, t0: +t0.toFixed(1), timing: 'nemotron-real',
        timing_only: !args.words, dialogues: [`${j.session_id}:${c}:${k0}`],
        ticks: caller.heard.slice(k0, k0 + N).map((h, k) => ({ heard: h, said: said[k], w: W[tags[k]] ?? 1, tag: tags[k] })),
      };
      (isDev(j.session_id) ? dev : train).push(call);
      for (const t of tags) count[t] = (count[t] || 0) + 1;
      ticks += N;
    }
  }
}
for (const [file, calls] of [[OUT, train], [DEV, dev]]) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, calls.map((x) => JSON.stringify(x)).join('\n') + (calls.length ? '\n' : ''));
}
console.log(`${sessions} sessions -> ${train.length} training calls, ${dev.length} dev calls, ${ticks} ticks (${(ticks * TICK / 3600).toFixed(1)} h)`);
console.log(JSON.stringify(count));
