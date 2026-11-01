'use strict';
// Spawns llama-server and runs one micro-turn per call to tick().
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { systemPrompt, FORMATS, familyOf } = require('./config');

const LISTEN = '<listen>';
const NOVOICE = '<no voice>';
const SENTENCE_END = /[.!?]["')]*$/;
// A tick's words as spoken: cut at the model's own tag if it wrote one mid-reply ("How would you define '<listen>",
// 2026-10-28: the voice read it out), with an opening quote or bracket left hanging dropped; nothing left is a listen.
// Mirrors tune/common.py spoken().
const spoken = (out) => out.split('<')[0].trimEnd().replace(/\s+[‘“"'(]+$/, '').trim() || LISTEN;
const CARRY_ON_P = 0.9; // P(listen) needed to stop mid-sentence into a silence (tick: carry-on bar); 0 = the threshold

// a reply that is only a back-channel (the partner adapters' training sounds plus the usual ones)
const BACKCHANNEL = /^(mm-?hm|mhm|uh-?huh|yeah|yep|right|okay|ok|i see|got it|sure|huh|totally|really\??|oh( (yeah\??|wow|nice|no|really\??))?|ha,? yeah|wow)[.!?,]*$/i;
const BACKCHANNEL_GAP_TICKS = 17; // ~10 s at 600 ms ticks

class Llama {
  constructor(log) {
    this.log = log || (() => {});
    this.proc = null;
    this.url = null;
    this.cfg = null;
    this.served = new Map(); // adapters renamed while loaded: the name the server knows -> the file's name now
  }

  async start(cfg, { withLora = true } = {}) {
    await this.stop();
    this.served.clear();
    this.cfg = cfg;
    this.url = `http://127.0.0.1:${cfg.port}`;
    this.model = cfg.model;
    this.format = familyOf(cfg.model); // the chat format follows the base model (also when reusing a running server)
    if (await this.healthy()) {
      this.log(`llama: a server is already listening on ${this.url}; using it`);
      return { reused: true };
    }
    if (path.isAbsolute(cfg.exe) && !fs.existsSync(cfg.exe)) throw new Error(`llama.cpp's server is not set up (no ${cfg.exe}): run npm run setup, see the README`);
    if (!fs.existsSync(cfg.model)) throw new Error(`llama: missing ${cfg.model}`);
    const args = ['-m', cfg.model, '-ngl', String(cfg.ngl), '--n-cpu-moe', String(cfg.nCpuMoe),
      '-c', String(cfg.ctx), '-ub', String(cfg.ubatch), '--port', String(cfg.port), '--host', '127.0.0.1',
      '--no-reasoning-preserve', '-np', '1', ...(cfg.threads ? ['-t', String(cfg.threads)] : [])];
    this.adapters = [];
    if (withLora) {
      for (const f of [cfg.lora, ...(cfg.extraLoras || [])].filter(Boolean)) {
        if (!fs.existsSync(f)) throw new Error(`llama: missing adapter ${f}`);
        // one file under two spellings (C:/x and C:\x) would load twice and then match ambiguously
        if (!this.adapters.some((a) => path.resolve(a).toLowerCase() === path.resolve(f).toLowerCase())) this.adapters.push(f);
      }
      // all adapters load with scale 0; every request names the active one, so they can be swapped mid-session
      // one --lora with a comma-separated list: this llama.cpp build deprecates repeated --lora flags
      if (this.adapters.length) args.push('--lora', this.adapters.join(','));
      if (this.adapters.length) args.push('--lora-init-without-apply');
    }
    this.log(`llama: starting ${path.basename(cfg.exe)} ${args.join(' ')}`);
    // on Linux the release's CUDA libraries sit beside the server (setup.js unpacks both archives into vendor/llama)
    const dir = path.isAbsolute(cfg.exe) ? path.dirname(cfg.exe) : process.cwd();
    const env = process.platform === 'linux' ? { ...process.env, LD_LIBRARY_PATH: [dir, process.env.LD_LIBRARY_PATH].filter(Boolean).join(':') } : process.env;
    const p = spawn(cfg.exe, args, { cwd: dir, env, windowsHide: true });
    this.proc = p;
    // the server reports every request (slot picked, timings, released): about 15 lines a tick, which buried the rest
    const ROUTINE = /\b(slot +(print_timing|release|get_availabl|launch_slot_)|log_server_r)\b|^\s*[\d.]+ I slot +print_timing|^\s*[\d.]+ I +(slot|srv) .*\|\s+(eval time|total time|graphs reused|graph reallocs|prompt eval time)|need to evaluate at least 1 token|n_past was set to|^\s*[\d.]+( [IWE])?\s*$/;
    const out = (d) => d.toString().split(/\r?\n/).filter((l) => l.trim() && !ROUTINE.test(l)).forEach((l) => this.log('llama: ' + l));
    p.stdout.on('data', out);
    p.stderr.on('data', out);
    // only this server's own exit clears it: the previous one's exit can arrive after this one has started (switching
    // to an adapter on another base), and was taken for this one dying during startup
    p.on('exit', (code) => { this.log(`llama: exited ${code}`); if (this.proc === p) this.proc = null; });
    const t0 = Date.now();
    while (Date.now() - t0 < 600000) {
      if (this.proc !== p) throw new Error('llama-server exited during startup (see log)');
      if (await this.healthy()) {
        this.log(`llama: ready in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
        return { reused: false };
      }
      await new Promise((r) => setTimeout(r, 500));
    }
    throw new Error('llama-server did not become healthy in 10 minutes');
  }

  async healthy() {
    if (!this.url) return false;
    try {
      const r = await fetch(this.url + '/health', { signal: AbortSignal.timeout(1500) });
      return r.ok && (await r.json()).status === 'ok';
    } catch { return false; }
  }

  // Waits until the server has really gone (up to 10 s), so its port and GPU memory are free for the next one.
  async stop() {
    const p = this.proc;
    if (!p) return;
    this.proc = null;
    const gone = p.exitCode !== null || p.signalCode !== null ? Promise.resolve() : new Promise((r) => p.once('exit', r));
    p.kill();
    await Promise.race([gone, new Promise((r) => setTimeout(r, 10000))]);
  }

  // The server reports paths as given on its command line (possibly relative to its folder): compare file names, and
  // one renamed since it started (the server holds its weights in memory) under the name it has now
  fileName(serverPath) {
    const n = (serverPath || '').toLowerCase().split(/[\\/]/).pop();
    return this.served.get(n) || n;
  }

  // An adapter file was renamed (the Adapters tab): keep answering for it until the server next starts
  renamed(from, to) {
    const old = path.basename(from), now = path.basename(to).toLowerCase();
    const key = [...this.served.keys()].find((k) => this.served.get(k) === old.toLowerCase()) || old.toLowerCase();
    this.served.set(key, now);
    if (this.activeAdapter) this.activeAdapter = this.activeAdapter.replace(old, path.basename(to));
  }

  // Active adapter by path (null = base model). Server adapter ids come from /lora-adapters.
  // file: an adapter path, null for the base model, or a blend "blend:a.gguf=0.5,b.gguf=0.5" (scales per file name).
  async useAdapter(file) {
    const list = await (await fetch(this.url + '/lora-adapters')).json();
    const name = (p) => (p || '').toLowerCase().split(/[\\/]/).pop();
    if (file && file.startsWith('blend:')) {
      const want = Object.fromEntries(file.slice(6).split(',').map((p) => { const [f, sc] = p.split('='); return [name(f), Number(sc)]; }));
      const missing = Object.keys(want).filter((f) => !list.some((a) => this.fileName(a.path) === f));
      if (missing.length) throw new Error(`blend adapters not loaded: ${missing.join(', ')}`);
      this.loraScales = list.map((a) => ({ id: a.id, scale: want[this.fileName(a.path)] ?? 0 }));
      this.activeAdapter = file;
      return list;
    }
    const hit = file ? list.filter((a) => this.fileName(a.path) === name(file)) : [];
    if (file && hit.length !== 1) throw new Error(`adapter not loaded (or ambiguous): ${file}`);
    this.loraScales = list.map((a) => ({ id: a.id, scale: hit.length && a.id === hit[0].id ? 1 : 0 }));
    this.activeAdapter = file || null;
    return list;
  }

  async complete(body) {
    if (this.loraScales) body = { ...body, lora: this.loraScales };
    const r = await fetch(this.url + '/completion', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
      signal: AbortSignal.timeout(20000),
    });
    if (!r.ok) throw new Error(`llama /completion ${r.status}: ${await r.text()}`);
    return r.json();
  }

  renderPrompt(history, heard, tickMs) {
    // in the base model's own chat format (config.js FORMATS; set from the model file in start())
    const f = FORMATS[this.format || 'qwen3.6'];
    // no f.bos: llama-server puts the BOS token in front itself for models that want one (Gemma), and with ours too
    // the model saw two at the start of every request where training had one (every Gemma result up to 2026-10-24)
    let p = f.system(systemPrompt(tickMs, this.promptKind || 'assistant', this.persona));
    for (const [role, content] of history) p += role === 'user' ? f.user(content) : content + f.end + f.sep;
    p += f.user(heard);
    return p;
  }

  // One micro-turn. Returns { said, pListen, ms, tokensCached, rerun, repeat, capped, midSentence, bar, blocked }.
  // listenSample: listen with exactly the model's P(listen) instead of the threshold rule. Adapters trained on real
  // people spread "start talking" over several ticks and are rarely confident enough to clear a threshold.
  // wordSampling: e.g. { temperature: 0.7, top_p: 0.9 }: once it has decided to speak, its words are drawn instead of
  // always the likeliest (greedy brought the same nudge back word for word every time the line went quiet). The
  // speak-or-listen decision and each tick's first word stay greedy. null: greedy throughout (the tests).
  async tick(history, heard, { tickMs, listenThreshold, listenSample = false, repeatGuard = true, backchannelGapTicks = BACKCHANNEL_GAP_TICKS, wordSampling = null, carryOnP = CARRY_ON_P }) {
    const prompt = this.renderPrompt(history, heard, tickMs);
    // Carry-on bar: right after its own words that did not end a sentence, with the line quiet, listening takes more
    // certainty than the threshold. The timing adapters stop mid-sentence into a silence at P(listen) 0.6-0.8 and one
    // quiet tick later are sure (0.98) they have finished; 21 of g26b4's 26 such stops were under 0.9 (2026-10-28).
    const last = history.length ? history[history.length - 1] : null;
    const midSentence = heard === NOVOICE && last && last[0] === 'assistant' && last[1] !== LISTEN && !SENTENCE_END.test(last[1].trim());
    if (midSentence && carryOnP > listenThreshold) listenThreshold = carryOnP;
    const base = { temperature: 0, stop: [FORMATS[this.format || 'qwen3.6'].end], cache_prompt: true };
    const words = wordSampling ? { ...base, ...wordSampling, seed: Math.floor(Math.random() * 2 ** 31) } : base;
    const t0 = Date.now();
    // Decide on the first token: three ticks in four are <listen>, and a listen tick then costs one decoded token
    // instead of four. A speaking tick continues after that first token (prompt + first), never resending the bare
    // prompt, because the model's recurrent layers make rolling the prompt cache back expensive.
    const j = await this.complete({ ...base, prompt, n_predict: 1, n_probs: 8 });
    const first = j.content || '';
    const top = j.completion_probabilities?.[0]?.top_logprobs || [];
    const listenIds = top.filter((p) => p.token.startsWith('<')).map((p) => p.id);
    const pListen = top.filter((p) => p.token.startsWith('<')).reduce((s, p) => s + Math.exp(p.logprob), 0);
    let rerun = false;
    let out;
    const listen = listenSample && listenIds.length ? Math.random() < pListen : null; // null: use the threshold rule
    if (listen === true) out = LISTEN;
    else if (first.startsWith('<')) {
      if (listen === null && (pListen >= listenThreshold || !listenIds.length)) out = LISTEN;
      else {
        // greedy picked "<" but the threshold (or the draw) says speak: ban it and make the model speak
        rerun = true;
        const j2 = await this.complete({ ...words, prompt, n_predict: 64, logit_bias: listenIds.map((i) => [i, false]) });
        out = (j2.content || '').trim();
      }
    } else {
      const j2 = await this.complete({ ...words, prompt: prompt + first, n_predict: 63 });
      out = (first + (j2.content || '')).trim();
    }
    let said = spoken(out);
    // Repeat guard: a reply that is word for word what it said on one of the last two ticks ("I'm here.
    // I'm here. I'm here.") is a decoding loop, not speech; listen instead. Loops showed up in 2-22% of utterances of
    // adapters trained on real conversations, most when a sampled decision forces speech the model did not want.
    let repeat = false, blocked = null;
    if (said !== LISTEN && repeatGuard) {
      const norm = (s) => s.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim();
      // back to back: anything (a "Yeah." loop); within the last 20 ticks (~12 s): any chunk of two or more words. Real
      // people never repeat a 3+ word reply within four replies (0 of 421 on 24 recorded calls); the adapters did in
      // 27-67% of them. A lone "Yeah." after a pause is speech and passes.
      const mine = history.filter(([role]) => role === 'assistant').map(([, c]) => c);
      const n = norm(said);
      const last2 = mine.slice(-2).filter((c) => c !== LISTEN).map(norm);
      const recent = mine.slice(-20).filter((c) => c !== LISTEN).map(norm);
      if (last2.includes(n) || (n.split(' ').length >= 2 && recent.includes(n))) { blocked = said; said = LISTEN; repeat = true; }
      // a nudge (speaking up into silence) that opens the way an earlier nudge of the last ~5 minutes opened: greedy
      // decoding brings back the same line each time the line goes quiet ("So, are you still cleaning or did you stop
      // the bleach?" nine times in one call); stay quiet instead, and it says something else or waits
    }
    // Back-channel cap: a lone "mm-hm" / "oh wow" while the caller is talking, within ~10 s of the previous one,
    // becomes a listen. Training weights swung the rate from 0.2 to 4.5 a minute (a person: 0.8) on a weight change
    // of 1.8 to 2.05; on run 10's recorded-call replays a 10 s gap brings 4.6 a minute down to 1.8.
    let capped = false;
    if (said !== LISTEN && backchannelGapTicks && heard !== NOVOICE && BACKCHANNEL.test(said.trim())) {
      const mine = history.filter(([role]) => role === 'assistant').map(([, c]) => c).slice(-backchannelGapTicks);
      if (mine.some((c) => c !== LISTEN && BACKCHANNEL.test(c.trim()))) { said = LISTEN; capped = true; }
    }
    // midSentence: this tick came straight after its own unfinished words with the line quiet; bar: the listen
    // threshold it used; blocked: the words the repeat guard turned into a listen (the loop logs mid-sentence stops)
    return { said, pListen, ms: Date.now() - t0, tokensCached: j.tokens_cached, rerun, repeat, capped, midSentence: !!midSentence, bar: listenThreshold, blocked };
  }
}

module.exports = { Llama, LISTEN, NOVOICE, BACKCHANNEL, CARRY_ON_P };
