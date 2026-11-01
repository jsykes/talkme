'use strict';
// The micro-turn scheduler. Every tickMs: drain STT -> ask the LLM -> speak or listen. Ticks never overlap; if one
// runs long the next starts right away but no catch-up burst happens (next = max(now, next + tick)).
const { LISTEN, NOVOICE } = require('./llama');

// The own-words filter: on speakers, whatever echo gets past the echo canceller reaches the recogniser as words the
// assistant itself just said, and the model takes them for the caller (it stops itself, or answers itself). While its
// voice is playing and for ECHO_TAIL_MS after, a stretch of heard words is dropped when it repeats its own last
// OWN_WORDS_MS of speech in order and is long enough not to be chance (ECHO_MIN_LETTERS): "that sounds", "exhausting"
// go, "I was", "at a", "you" stay. Dropping every word it had also said shredded the caller's own sentences into gaps
// it then apologised for. Also dropped: a lone piece of one of its long words (the recogniser splits long words).
const ECHO_TAIL_MS = 1500;
const OWN_WORDS_MS = 8000;
// The voice anchor: the model writes a tick's words at its training pace (about 2.9 words a 0.6 s tick), faster than a
// voice says them (Orpheus about 1.6), so a long reply ran up to 15 s ahead of what the caller had heard: it took its
// own still-playing speech for silence and spoke up over it ("Anyway,"), and the caller's questions landed far behind.
// So while its voice has more than VOICE_LEAD_MS of speech queued, the next tick waits (each tick keeps its usual
// words; a tick just lasts as long as the voice takes to say them), and runs at once when the caller says something.
// The lead keeps the voice from running dry while the next words are synthesised (first sound ~0.6-0.9 s). The hold
// applies whether or not the voice holds words for a sentence end: the model's 3-word chunks straddle sentence ends
// ("fine. Why you're"), so an exemption for a held sentence start let it write eight sentences unheld (2026-10-27).
// When the voice is about to run dry, tts.js sends what it holds at a comma or after a few words.
// Thinning the words out instead (a tick's history holding only what the voice said in it) was tried first: one word
// a tick, never seen in training, made it hesitate, stop and restart ("so I" ... "so I don't really").
const VOICE_LEAD_MS = 1800; // a clause can take two ticks to complete, then Orpheus ~0.6 s to its first sound
const VOICE_WAIT_MAX_MS = 6000; // a voice that stopped reporting (an error) holds the loop no longer than this
const VOICE_POLL_MS = 40;
const ECHO_MIN_LETTERS = 7;
const norm = (w) => w.toLowerCase().replace(/[^a-z0-9']/g, '');
// own: its recent words in the order said (normalised)
function dropOwnWords(heard, own) {
  if (heard === NOVOICE || !own.length) return { heard, dropped: 0 };
  const parts = heard.split(/\s+/).filter(Boolean);
  const idx = parts.map((w, i) => (norm(w) ? i : -1)).filter((i) => i >= 0); // the words, without lone punctuation
  const ownText = ` ${own.join(' ')} `;
  const said = (a, b) => ownText.includes(` ${idx.slice(a, b + 1).map((i) => norm(parts[i])).join(' ')} `);
  const drop = new Set();
  for (let a = 0; a < idx.length; a++) {
    if (!said(a, a)) continue;
    let b = a;
    while (b + 1 < idx.length && said(a, b + 1)) b++;
    if (idx.slice(a, b + 1).reduce((n, i) => n + norm(parts[i]).length, 0) >= ECHO_MIN_LETTERS) for (let k = a; k <= b; k++) drop.add(idx[k]);
    a = b;
  }
  if (idx.length === 1 && !drop.size) {
    const n = norm(parts[idx[0]]);
    if (n.length >= 3 && own.some((o) => o.length >= ECHO_MIN_LETTERS && o.length > n.length && (o.startsWith(n) || o.endsWith(n)))) drop.add(idx[0]);
  }
  const keep = parts.filter((_, i) => !drop.has(i));
  return { heard: keep.some((w) => norm(w)) ? keep.join(' ') : NOVOICE, dropped: drop.size };
}

// Where to cut a history that has grown past its cap: from the wanted cut on, the first tick where the caller starts
// talking into a silence, so what the model sees still opens like a call (a cut at an arbitrary tick left it starting
// mid-reply; in a 7-minute call, after two such cuts, it spoke in fragments and stalled mid-sentence, 2026-10-27).
// history: [role, text] entries, two per tick. Returns the number of entries to drop.
function trimPoint(history, want) {
  for (let i = want; i < Math.min(history.length, want + 120); i += 2) {
    if (history[i][1] !== NOVOICE && history[i - 2][1] === NOVOICE && history[i - 1][1] === LISTEN) return i;
  }
  return want;
}

class Loop {
  constructor({ llama, stt, tts, log, emit }) {
    this.llama = llama; this.stt = stt; this.tts = tts;
    this.log = log || (() => {});
    this.emit = emit || (() => {}); // emit(event, payload) to the renderer
    this.history = [];
    this.running = false;
    this.timer = null;
    this.tickNo = 0;
    this.speaking = false;   // true while the renderer reports audio in the playback queue
    this.lastSpeakingAt = 0;  // when it last stopped (the echo's tail)
    this.ownWords = [];       // [time, word] of what it said lately, for the own-words filter
    this.settings = null;
    this.stats = { ticks: 0, latSum: 0, latMax: 0, over: 0 };
  }

  start(settings) {
    this.settings = settings;
    this.history = [];
    this.tickNo = 0;
    this.prevSaid = undefined;
    this.lastPhraseIsFiller = false;
    this.ownWords = [];
    this.midStop = null;
    this.stats = { ticks: 0, latSum: 0, latMax: 0, over: 0 };
    this.running = true;
    this.next = Date.now();
    this.schedule();
    this.log('loop: call started');
  }

  stop() {
    this.running = false;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.tts.cancel();
    this.log('loop: call ended');
  }

  schedule() {
    if (!this.running) return;
    const delay = Math.max(0, this.next - Date.now());
    const due = this.next;
    const run = () => this.runTick().catch((e) => {
      this.log('loop: tick failed: ' + e.message);
      this.emit('error', e.message);
      this.next = Date.now() + this.settings.tickMs;
      this.schedule();
    });
    // the voice anchor: wait while the voice is behind, unless the caller speaks
    const wait = () => {
      if (!this.running) return;
      const behind = this.settings.anchorToVoice !== false && this.tts.backlogMs && this.tts.backlogMs() > VOICE_LEAD_MS;
      if (behind && !this.stt.hasWords() && Date.now() - due < VOICE_WAIT_MAX_MS) { this.timer = setTimeout(wait, VOICE_POLL_MS); return; }
      this.waitedMs = Math.max(0, Date.now() - due);
      if (this.waitedMs) this.next = Date.now(); // a tick that waited: the next one comes a full tick after it
      run();
    };
    this.timer = setTimeout(wait, delay);
  }

  setSpeaking(on) {
    if (this.speaking && !on) this.lastSpeakingAt = Date.now();
    this.speaking = on;
  }

  async runTick() {
    if (!this.running) return;
    const s = this.settings;
    const t0 = Date.now();
    let heard = this.stt.drain() || NOVOICE;
    let echoDropped = 0;
    this.ownWords = this.ownWords.filter(([t]) => t0 - t < OWN_WORDS_MS);
    if (s.ownWordsFilter && (this.speaking || t0 - this.lastSpeakingAt < ECHO_TAIL_MS)) {
      const r = dropOwnWords(heard, this.ownWords.map(([, w]) => w));
      if (r.dropped) this.log(`loop: dropped ${r.dropped} heard words that were its own (echo): "${heard}"`);
      heard = r.heard; echoDropped = r.dropped;
    }
    if (heard !== NOVOICE) this.log(`heard: ${heard}`);
    this.trimIfNeeded();
    const r = await this.llama.tick(this.history, heard, { tickMs: s.tickMs, listenThreshold: s.listenThreshold, listenSample: !!s.listenSample,
      wordSampling: s.wordSampling || null, ...(s.carryOnP != null ? { carryOnP: s.carryOnP } : {}) });
    if (!this.running) return;
    this.history.push(['user', heard], ['assistant', r.said]);
    if (r.repeat) this.log('loop: repeat guard: listened instead of repeating itself');
    if (r.capped) this.log('loop: back-channel cap: listened instead of a second back-channel');
    if (this.waitedMs >= 1000) this.log(`loop: tick waited ${this.waitedMs} ms for the voice`);
    this.noteStop(r, heard);
    const userSpoke = heard !== NOVOICE;
    let action = 'listen';
    if (r.said === LISTEN) {
      // still talking: the renderer says audio is playing, or audio or words are still on their way to it
      const talking = this.speaking || this.tts.playEnd > Date.now() || this.tts.buffer.length;
      if (userSpoke && talking && !this.lastPhraseIsFiller) {
        // barge_stop: the user cut in and the model chose to stop. A back-channel ("Mm-hm.") is never cut: the user
        // talking on over it is the point, and clipping it mid-sound is jarring.
        this.tts.cancel();
        this.emit('audio-stop');
        action = 'stop';
      } else {
        // natural end of an answer (or still listening): let what is queued play out
        this.tts.flush({ speed: s.ttsSpeed });
      }
    } else {
      // a reply that starts after a listen tick and is at most two words is a back-channel until more words follow
      const continuing = this.prevSaid !== undefined && this.prevSaid !== LISTEN;
      this.lastPhraseIsFiller = !continuing && r.said.trim().split(/\s+/).length <= 2;
      this.tts.say(r.said, { flushWords: s.ttsFlushWords, speed: s.ttsSpeed, first: !continuing && s.speakFirstAtOnce !== false });
      for (const w of r.said.split(/\s+/)) if (norm(w)) this.ownWords.push([t0, norm(w)]);
      action = 'speak';
    }
    this.prevSaid = r.said;
    const total = Date.now() - t0;
    this.stats.ticks++; this.stats.latSum += total; this.stats.latMax = Math.max(this.stats.latMax, total);
    if (total > s.tickMs) this.stats.over++;
    this.emit('tick', {
      n: this.tickNo++, heard, joins: heard !== NOVOICE && !echoDropped && !!this.stt.heardJoins, said: r.said, pListen: r.pListen, llmMs: r.ms, totalMs: total, action,
      rerun: r.rerun, cached: r.tokensCached, historyTicks: this.history.length / 2, echoDropped, waitedMs: this.waitedMs || 0,
      avgMs: Math.round(this.stats.latSum / this.stats.ticks), maxMs: this.stats.latMax, over: this.stats.over,
    });
    this.next = Math.max(Date.now(), this.next + s.tickMs);
    this.schedule();
  }

  // Mid-sentence stops into a silence: why it stopped (P(listen) over the bar, or the repeat guard), then how the quiet
  // after it ended, so a stall in a live call can be read from the log. One line at the stop, one when the quiet ends.
  noteStop(r, heard) {
    const p = (x) => x.toFixed(2);
    if (this.midStop) {
      if (heard === NOVOICE && r.said === LISTEN) { this.midStop.ps.push(r.pListen); return; }
      const ps = this.midStop.ps;
      if (ps.length) {
        const how = heard !== NOVOICE ? `the caller spoke ("${heard}")` : `it went on: "${r.said}"`;
        this.log(`loop: after the mid-sentence stop, ${ps.length} quiet ticks (P(listen) ${p(Math.min(...ps))}-${p(Math.max(...ps))}), then ${how}`);
      }
      this.midStop = null;
    }
    if (!r.midSentence || r.said !== LISTEN) return;
    const words = this.history[this.history.length - 3]?.[1] || '';
    const why = r.repeat ? `P(listen) ${p(r.pListen)} under the bar ${p(r.bar)}, but the repeat guard blocked "${r.blocked}"`
      : `P(listen) ${p(r.pListen)} over the bar ${p(r.bar)}`;
    this.log(`loop: stopped mid-sentence after "${words}": ${why}`);
    this.midStop = { ps: [] };
  }

  // ~17 tokens per tick; keep well under the context size by dropping the oldest 40 % of ticks once.
  // historyTicks caps it sooner for a model whose tick time grows with the call (Gemma 4: see config.js BASES)
  trimIfNeeded() {
    const maxTicks = this.settings.historyTicks || Math.floor((this.settings.llamaCtx || 32768) / 17 * 0.85);
    if (this.history.length / 2 > maxTicks) {
      const drop = trimPoint(this.history, Math.floor(this.history.length * 0.4 / 2) * 2);
      this.history.splice(0, drop);
      this.log(`loop: trimmed ${drop / 2} ticks of history to stay inside the context`);
    }
  }
}

module.exports = { Loop, dropOwnWords, trimPoint, VOICE_LEAD_MS };
