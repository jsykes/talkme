'use strict';
// Text-to-speech. Synthesis runs in a worker thread so the 600 ms loop is never blocked. Text is buffered and
// flushed at punctuation, at ttsFlushWords words, or when the model goes to <listen>, because Piper sounds choppy
// when fed three words at a time.
const { Worker } = require('worker_threads');
const path = require('path');
const { VoiceServer } = require('./voice-server');

// Native addon code must never be interrupted: ask the worker to exit after its current call, terminate as a fallback.
function closeWorker(w, ms = 3000) {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    w.once('exit', finish);
    try { w.postMessage({ type: 'close' }); } catch { finish(); }
    setTimeout(() => { if (!done) w.terminate().then(finish, finish); }, ms);
  });
}


// Both voices turn English into sounds with espeak, which spells any all-consonant word it doesn't know: "Mm-hm"
// comes out "em em aych em". (Kokoro's lexicon files are not consulted for English in this build.) The model writes
// back-channels the way its training data spells them, so they are rewritten here into forms espeak says properly:
// "[[m'hm]]" is raw espeak phonemes (a hummed "m-HM"), "Umhum" and "Uh-huh" are words it knows. One is picked at
// random each time, never the same twice in a row, so back-channels don't sound canned.
const MMHM = /\b(mm-?hm+|m+hm+|mm hmm)\b/gi;
const KOKORO_MMHM = [["[[m'hm]]", 0.5], ['Umhum', 0.25], ['Uh-huh', 0.25]];
let lastMmhm = '';
function pickMmhm() {
  const choices = KOKORO_MMHM.filter(([w]) => w !== lastMmhm);
  const total = choices.reduce((s, [, p]) => s + p, 0);
  let r = Math.random() * total;
  for (const [w, p] of choices) { if ((r -= p) <= 0) return (lastMmhm = w); }
  return (lastMmhm = choices[0][0]);
}
function speakable(text, engine = 'piper') {
  const keepCase = (m, rep) => (m[0] === m[0].toUpperCase() ? rep[0].toUpperCase() + rep.slice(1) : rep);
  return text.replace(MMHM, (m) => (engine === 'kokoro' ? pickMmhm() : keepCase(m, 'uh-huh')));
}
const SAY_AS = [[MMHM, 'uh-huh']]; // kept for reference by older scripts

// sentence pieces (GPU voices): a sentence that runs on is cut at these many words (the first piece sooner); when the
// queued audio is about to run out, a comma seam or a few words will do
const JIT_FIRST_WORDS = 6;
const JIT_MAX_WORDS = 14;
const JIT_DRY_MS = 700;
const JIT_DRY_WORDS = 4; // with nothing left to play, this much goes as it is (continuing tone) rather than a gap

class TTS {
  constructor(log) {
    this.log = log || (() => {});
    this.worker = null;
    this.buffer = [];
    this.seq = 0;
    this.onAudio = null; // callback({ seq, samples: Float32Array, sampleRate, text })
    this.sampleRate = 22050;
    this.generation = 0; // bumped on cancel so stale results are dropped
    this.server = new VoiceServer(this.log);
  }

  async start(cfg) {
    this.sayAs = cfg.sayAs;
    this.engine = cfg.engine || 'piper';
    this.kind = cfg.kind || cfg.engine; // which engine, so a switch within one GPU engine can keep its server
    this.flushWords = cfg.flushWords || null; // a voice's own piece length (GPU voices: longer, see voices.js)
    this.justInTime = !!cfg.justInTime; // GPU voices: pieces as long as the audio queue allows (see say())
    this.contTone = cfg.contTone !== false; // pieces that stop mid-sentence ask for a continuing tone
    this.playEnd = 0; this.inflightWords = 0;
    // how long this voice takes per word, measured on its own pieces (loop.js paces the model's words by it)
    this.msPerWord = this.msPerWord || 380; this.pieceMs = {}; this.inflight = {}; // seq -> words of a piece being synthesised
    if (cfg.engine === 'remote') await this.server.start(cfg);
    return new Promise((resolve, reject) => {
      this.worker = new Worker(path.join(__dirname, 'tts-worker.js'), { workerData: cfg });
      this.worker.on('message', (m) => {
        if (m.type === 'ready') { this.sampleRate = m.sampleRate; this.log(`tts: ready (${m.sampleRate} Hz)`); resolve(); }
        else if (m.type === 'done') { // a streamed piece has nothing more to come
          if (m.generation !== this.generation) return;
          this.inflightWords = Math.max(0, this.inflightWords - (m.words || 0)); delete this.inflight[m.seq];
          this.measure(m.seq, m.words);
          if (this.onPieceDone) this.onPieceDone({ seq: m.seq, text: m.text });
        } else if (m.type === 'audio') {
          if (m.generation !== this.generation) return; // cancelled while synthesising
          // what is queued to play: this piece goes after whatever is still playing
          const ms = (m.samples.length / m.sampleRate) * 1000;
          this.pieceMs[m.seq] = (this.pieceMs[m.seq] || 0) + ms;
          if (m.final !== false) { this.inflightWords = Math.max(0, this.inflightWords - (m.words || 0)); delete this.inflight[m.seq]; this.measure(m.seq, m.words); if (this.onPieceDone) this.onPieceDone({ seq: m.seq, text: '' }); }
          this.playEnd = Math.max(Date.now(), this.playEnd) + ms;
          if (this.onAudio) this.onAudio({ seq: m.seq, samples: m.samples, sampleRate: m.sampleRate, text: m.text, ms: m.ms, final: m.final !== false });
        } else if (m.type === 'error') { this.log('tts: ' + m.error); if (!this.sampleRate) reject(new Error(m.error)); }
      });
      this.worker.on('error', (e) => { this.log('tts worker: ' + e.message); reject(e); });
      this.worker.on('exit', () => { this.worker = null; });
    });
  }

  // keepServer: leave a GPU voice's server running (switching to another voice of the same engine)
  async stop({ keepServer = false } = {}) {
    if (this.worker) { const w = this.worker; this.worker = null; await closeWorker(w); }
    if (!keepServer) await this.server.stop();
    this.buffer = [];
  }

  // Called with each tick's words. Returns true if a chunk was flushed.
  // sayAs: rewrite words the voice cannot pronounce (see SAY_AS); false leaves the text as the model wrote it
  // first: these are the opening words of an utterance. They are spoken at once instead of waiting for punctuation or
  // flushWords, which held an answer's first audio back by up to two ticks (~1 s); later chunks still batch so the
  // voice doesn't sound choppy.
  say(text, { flushWords, speed, first = false }) {
    const words = text.trim().split(/\s+/).filter(Boolean);
    if (!words.length) return false;
    if (first) this.seqAtStart = this.seq; // the next piece is this utterance's first: it may be cut sooner
    this.buffer.push(...words);
    const last = this.buffer[this.buffer.length - 1];
    if (this.justInTime) {
      // Every piece is spoken as its own utterance, so each seam is heard, and the voice starts its tone afresh at each
      // ("Who lived in a huge," / "dark forest." came out as two moods). So a piece is a whole sentence, except: the
      // first piece of a reply is cut at its first comma or JIT_FIRST_WORDS, for a quick start ("Oh no," / "Whoa,
      // wait,"); when the queued audio is about to run out and nothing is being synthesised, the next comma will do;
      // a sentence that runs on is cut at JIT_MAX_WORDS. (loop.js holds the model only between pieces, so a sentence is
      // written at full speed while the one before it plays.)
      // the model hands over three words at a time, so the boundary can be anywhere in the buffer: cut at the last one
      const first = this.seqAtStart === this.seq;
      const dry = this.inflightWords === 0 && this.playEnd - Date.now() < JIT_DRY_MS;
      const seam = first || dry ? /[.!?,;:]["')]*$/ : /[.!?]["')]*$/;
      let cut = -1;
      for (let i = this.buffer.length - 1; i >= 0; i--) if (seam.test(this.buffer[i])) { cut = i; break; }
      if (cut >= 0) return this.flush({ speed, upTo: cut + 1 });
      if (this.buffer.length >= (first ? JIT_FIRST_WORDS : dry ? JIT_DRY_WORDS : JIT_MAX_WORDS)) return this.flush({ speed });
      return false;
    }
    if (first || /[.!?;:,]["')]*$/.test(last) || this.buffer.length >= (this.flushWords || flushWords)) return this.flush({ speed });
    return false;
  }


  // upTo: flush only the first upTo words (a clause), keeping the rest for the next piece
  flush({ speed = 1.0, upTo } = {}) {
    if (!this.buffer.length || !this.worker) return false;
    const take = this.buffer.slice(0, upTo || this.buffer.length);
    const text = this.sayAs === false ? take.join(' ') : speakable(take.join(' '), this.engine);
    const words = take.length;
    this.buffer = this.buffer.slice(words);
    this.inflightWords += words; this.inflight[this.seq] = words;
    // a piece that stops mid-sentence: the voice should sound like it is going on (see serve_common.py "cont")
    const cont = this.contTone && !/[.!?]["')]*$/.test(text);
    this.worker.postMessage({ type: 'generate', text, words, cont, seq: this.seq++, speed, generation: this.generation });
    return true;
  }

  // A finished piece: its audio length per word, averaged over the last pieces (a one- or two-word piece is mostly the
  // voice's lead-in and tail, so it is left out)
  measure(seq, words) {
    const ms = this.pieceMs[seq];
    delete this.pieceMs[seq];
    if (!ms || !words || words < 3) return;
    this.msPerWord = Math.round(0.8 * this.msPerWord + 0.2 * Math.min(800, Math.max(150, ms / words)));
  }

  // Milliseconds of speech still to come: audio queued to play, plus what is left of each piece being synthesised (its
  // words at this voice's pace, less the audio of it that has arrived, which playEnd already holds). Words held in
  // `buffer` for a sentence end are not counted: they go out only once the model's next words arrive, and counting
  // them held the loop (loop.js waits while this is long) until its 6 s limit, 10-16 s pauses mid-sentence.
  backlogMs() {
    let left = 0;
    for (const [seq, words] of Object.entries(this.inflight || {})) left += Math.max(0, words * this.msPerWord - (this.pieceMs[seq] || 0));
    return Math.max(0, this.playEnd - Date.now()) + left;
  }

  // Drop anything queued or in flight (barge-in).
  cancel() {
    this.buffer = [];
    this.generation++;
    if (this.worker && this.engine === 'remote') this.worker.postMessage({ type: 'cancel' }); // stop generating it too
    this.playEnd = 0; this.inflightWords = 0; this.pieceMs = {}; this.inflight = {};
  }
}

module.exports = { TTS, speakable };
