'use strict';
// Streaming speech-to-text. The recogniser runs in stt-worker.js; this side turns its running hypothesis into the
// per-tick fragments the LLM sees: drain() returns the words (or pieces of words) that became visible since the
// previous tick, which is "what the microphone heard" this tick.
//
// The adapter was trained on fragments from a Kyutai-style recogniser: cased, punctuated, long words sometimes split
// across ticks ("a cli" / "cking sound."), punctuation sometimes arriving a tick late (", like"). Nemotron produces
// cased, punctuated text natively and the growth logic below reproduces the splits and the late punctuation. Upper-case
// recognisers (the older zipformer) are lower-cased and punctuated by a punctuation model in the worker; the last word
// of an open utterance keeps no punctuation until the user goes quiet, because the punctuation model always ends what
// it sees with a full stop.
const { Worker } = require('worker_threads');
const path = require('path');

const TRAILING = /[.,!?;:]+$/;
// Final punctuation. Nemotron punctuates a word only once the next word gives it context, so the last word of an
// utterance never gets its "?" or "." - and the published adapter answers on exactly that cue. After the mic has been
// quiet for FINAL_SILENCE_MS with no new words, close the utterance: "?" if its sentence opens like a question, "."
// otherwise, nothing after a word that leaves the thought hanging. The model still decides whether to speak.
const FINAL_SILENCE_MS = 450;
const QUESTION_OPENERS = new Set(("what whats what's where when why who whose whom how which is are am was were do does did " +
  "can could would should will shall may might must have has had isn't aren't don't doesn't didn't can't won't " +
  "wouldn't shouldn't couldn't").split(' '));
const HANGING = new Set(('and but or so because if then to of the a an with for from in on at by my your their our his her ' +
  'its um uh like that which who when while about as than into very really just maybe').split(' '));
// A question that asks for the object of a preposition ends on it ("What are you up to", "Where are you from"); a
// yes/no question there is still open ("Do you want to"), and so is an infinitive ("What do you want to").
const PREPOSITIONS = new Set('to of with for from in on at by about into'.split(' '));
const WH_OBJECT = new Set(['what', 'whats', "what's", 'who', 'whom', 'where', 'which']);
const INFINITIVE_VERBS = new Set('want wanted wants going gonna have has had need needs needed got trying try able used ought'.split(' '));
function finalMark(words) {
  const last = (words[words.length - 1] || '').toLowerCase();
  if (!last || TRAILING.test(last)) return '';
  let i = words.length - 1;
  while (i > 0 && !/[.!?]$/.test(words[i - 1])) i--;
  const sentence = words.slice(i).map((w) => w.toLowerCase().replace(/[^a-z']/g, ''));
  const opener = sentence.find((w) => !['so', 'and', 'but', 'okay', 'ok', 'well', 'um', 'uh', 'hey', 'oh', 'like', 'just', 'then'].includes(w));
  if (HANGING.has(last)) {
    const before = sentence[sentence.length - 2];
    return PREPOSITIONS.has(last) && WH_OBJECT.has(opener) && !(last === 'to' && INFINITIVE_VERBS.has(before)) ? '?' : '';
  }
  return QUESTION_OPENERS.has(opener) || sentence.includes('or') && sentence.some((w) => QUESTION_OPENERS.has(w)) ? '?' : '.';
}


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


class STT {
  constructor(log) {
    this.log = log || (() => {});
    this.worker = null;
    this.segment = [];     // words of the current utterance, as last shown
    this.raw = [];         // the same words without trailing punctuation (for growth matching)
    this.emitted = [];     // per word: the text already handed over
    this.pending = [];     // pieces not yet drained
    this.injected = [];    // typed fragments (text-only calls)
    this.carry = '';       // pieces kept from a segment that ended (endpoint) before the drain
    this.upper = false;    // recogniser is upper-case: punctuation from the model, last word held back
    this.onPartial = null;
    this.finalPunct = true; // see finalMark
    this.lastVoiceAt = 0;
    this.now = () => Date.now(); // replaced by offline scripts that run the recogniser in simulated time
  }

  start(cfg) {
    this.finalPunct = cfg.finalPunct !== false;
    return new Promise((resolve, reject) => {
      this.worker = new Worker(path.join(__dirname, 'stt-worker.js'), { workerData: cfg });
      this.worker.on('message', (m) => {
        if (m.type === 'ready') { this.log(`stt: ready (${path.basename(cfg.dir)}, ${m.featureDim} mel${m.punct ? ', punctuation model loaded' : ''})`); resolve(); }
        else if (m.type === 'result') this.onResult(m);
        else if (m.type === 'error') { this.log('stt: ' + m.error); reject(new Error(m.error)); }
      });
      this.worker.on('error', (e) => { this.log('stt worker: ' + e.message); reject(e); });
      // An unexpected exit used to be silent: audio kept going into nothing while the status still said ready.
      this.worker.on('exit', (code) => { this.worker = null; if (!this.stopping && this.onDied) this.onDied(code); });
    });
  }

  // Forget a half-heard utterance (between calls): the recogniser's stream and the fragment state start fresh.
  reset() {
    this.segment = []; this.raw = []; this.emitted = []; this.pending = []; this.injected = []; this.carry = ''; this.lastTaken = -1;
    if (this.worker) this.worker.postMessage({ type: 'reset' });
  }

  async stop() {
    this.stopping = true;
    if (this.worker) { const w = this.worker; this.worker = null; await closeWorker(w); }
    this.stopping = false;
    this.segment = []; this.raw = []; this.emitted = []; this.pending = []; this.injected = []; this.carry = ''; this.lastTaken = -1;
  }

  feed(samples) {
    let e = 0; for (let i = 0; i < samples.length; i += 4) e += samples[i] * samples[i];
    if (Math.sqrt(e / (samples.length / 4)) > 0.01) this.lastVoiceAt = this.now();
    if (this.worker) this.worker.postMessage({ type: 'audio', samples });
  }

  inject(text) { if (text && text.trim()) this.injected.push(text.trim()); }

  // whether the caller has said something not drained yet (loop.js: a tick waiting on the voice runs at once)
  hasWords() { return this.pending.length > 0 || this.injected.length > 0 || !!this.carry; }

  onResult({ text, punctuated, endpoint }) {
    this.upper = punctuated !== null && punctuated !== undefined;
    let words = (this.upper ? punctuated : text).split(/\s+/).filter(Boolean);
    if (this.upper && punctuated.split(/\s+/).length !== text.split(/\s+/).length) words = text.toLowerCase().split(/\s+/).filter(Boolean);
    const raw = words.map((w) => w.replace(TRAILING, '').toLowerCase());
    // grow: same words so far, possibly with the last one longer or punctuated now
    const n = this.segment.length;
    for (let i = 0; i < Math.min(n, words.length); i++) {
      if (raw[i] !== this.raw[i]) {
        if (raw[i].startsWith(this.raw[i])) { this.raw[i] = raw[i]; this.segment[i] = words[i]; }
        // a revised earlier word is ignored: the adapter was trained with recogniser noise
      } else this.segment[i] = words[i];
    }
    for (let i = n; i < words.length; i++) { this.segment.push(words[i]); this.raw.push(raw[i]); this.emitted.push(''); }
    this.collect(false);
    if (this.onPartial && text) this.onPartial(this.segment.join(' '));
    if (endpoint) {
      // the recogniser ended the utterance (~1.2 s of silence): close it now, or the reset below loses the mark when
      // the last words were drained too late for a quiet drain to add it (seen live: "What would be more serious")
      if (this.finalPunct && !this.upper) this.closeUtterance();
      this.collect(true);
      const was = this.carry, taken = this.take();
      if (!was && taken) this.carryJoins = this.joined;
      this.carry = [this.carry, taken].filter(Boolean).join(' ');
      this.segment = []; this.raw = []; this.emitted = []; this.lastTaken = -1;
    }
  }

  // Move whatever part of each word has not been handed over yet into `pending`.
  // final=true: the utterance is over, so the last word's punctuation may go too.
  collect(final) {
    for (let i = 0; i < this.segment.length; i++) {
      let shown = this.segment[i];
      const isLast = i === this.segment.length - 1;
      if (this.upper && isLast && !final) shown = shown.replace(TRAILING, ''); // the punct model's guess, not real yet
      const done = this.emitted[i];
      if (shown.length > done.length && shown.toLowerCase().startsWith(done.toLowerCase())) {
        this.pending.push({ idx: i, piece: shown.slice(done.length) });
        this.emitted[i] = shown;
      }
    }
  }

  take() {
    this.joined = false;
    if (!this.pending.length) return '';
    // the first piece finishes the word handed over last time ("dis" then "aster"): the transcript joins them
    this.joined = this.pending[0].idx === this.lastTaken;
    // pieces of the same word go together; a piece that only adds punctuation sticks to the next word (", like")
    const out = [];
    let lastIdx = -1;
    for (const p of this.pending) {
      if (p.idx === lastIdx && out.length) out[out.length - 1] += p.piece;
      else out.push(p.piece);
      lastIdx = p.idx;
    }
    this.lastTaken = lastIdx;
    this.pending = [];
    return out.join(' ').trim();
  }

  // Give the open utterance its final mark (finalMark), once: after FINAL_SILENCE_MS of quiet at a drain, or at the
  // recogniser's endpoint, whichever comes first.
  closeUtterance() {
    const i = this.segment.length - 1;
    if (i < 0 || this.emitted[i] !== this.segment[i]) return;
    const mark = finalMark(this.segment);
    if (mark) { this.segment[i] += mark; this.emitted[i] += mark; this.pending.push({ idx: i, piece: mark }); }
  }

  // Words heard since the last drain, as one fragment, or '' if none.
  drain() {
    // the user went quiet this tick: release the held-back punctuation of the last word (late punctuation)
    if (this.upper && !this.pending.length && this.segment.length) this.collect(true);
    else if (this.finalPunct && !this.upper && !this.pending.length && this.now() - this.lastVoiceAt >= FINAL_SILENCE_MS) this.closeUtterance();
    const fromCarry = !!this.carry, taken = this.take();
    // whether these words go straight on from the last ones heard (part of one word), for the transcript
    this.heardJoins = fromCarry ? !!this.carryJoins : !!taken && this.joined;
    const parts = [this.carry, taken, this.injected.join(' ')].filter(Boolean);
    this.carry = ''; this.injected = [];
    return parts.join(' ');
  }
}

module.exports = { STT, finalMark, FINAL_SILENCE_MS };
