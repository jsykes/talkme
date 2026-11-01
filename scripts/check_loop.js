// Headless check of the micro-turn loop: real llama-server + real TTS, with typed "heard" fragments instead of a
// microphone. Run with llama-server already listening on the configured port.
//   node scripts/check_loop.js
const path = require('path');
const { DEFAULTS } = require('../src/main/config');
const { Llama } = require('../src/main/llama');
const { STT } = require('../src/main/stt');
const { TTS } = require('../src/main/tts');
const { Loop } = require('../src/main/loop');

const SCRIPT = ['Hey I\'m cleaning', 'the bathroom', 'right now and', 'I got some', 'bleach here', '', 'and I\'m going', 'to mix it',
  'with the ammonia', 'cleaner so it', 'gets extra strong', '', '', '', '', '', '', '', 'okay thanks', '', '', '', ''];

(async () => {
  const log = (l) => console.log('  ' + l);
  const llama = new Llama(log);
  await llama.start(DEFAULTS.llama);
  const stt = new STT(log); // not started: we push fragments straight into pending
  const tts = new TTS(log); await tts.start(require('../src/main/voices').voiceConfig(DEFAULTS.tts.voice));
  let audioMs = 0;
  tts.onAudio = (a) => { audioMs += a.samples.length / a.sampleRate * 1000; console.log(`      ♪ "${a.text}" ${(a.samples.length / a.sampleRate).toFixed(2)} s (synth ${a.ms} ms)`); };
  const loop = new Loop({ llama, stt, tts, log, emit: (ev, p) => {
    if (ev === 'tick') console.log(`${String(p.n).padStart(3)} ${p.heard.padEnd(20)} ${p.said.padEnd(26)} P=${p.pListen.toFixed(2)} llm=${p.llmMs}ms tick=${p.totalMs}ms ${p.action}${p.rerun ? ' (rerun)' : ''}`);
    else console.log('event', ev, p);
  } });
  let i = 0;
  const feeder = setInterval(() => {
    if (i >= SCRIPT.length) { clearInterval(feeder); setTimeout(async () => { loop.stop(); await tts.stop(); await llama.stop(); console.log(`done; ${(audioMs / 1000).toFixed(1)} s of speech synthesised`); process.exit(0); }, 1500); return; }
    const frag = SCRIPT[i++];
    if (frag) stt.inject(frag);
  }, DEFAULTS.tickMs);
  loop.start({ tickMs: DEFAULTS.tickMs, listenThreshold: DEFAULTS.listenThreshold, ttsFlushWords: DEFAULTS.ttsFlushWords, ttsSpeed: 1.0, llamaCtx: DEFAULTS.llama.ctx });
})();
