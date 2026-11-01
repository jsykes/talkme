'use strict';
// The GPU voices (Chatterbox Turbo, Orpheus 3B) run as local HTTP servers in their own Python environments
// (tts_bench/<voice>/.venv, tts_bench/serve_*.py); Orpheus's language model runs in a second llama-server. This
// starts them, waits until they answer, and stops them with everything they started.
const { spawnTree, killTree } = require('./proc');
const path = require('path');
const fs = require('fs');
const { ROOT } = require('./config');

const BENCH = path.join(ROOT, 'tts_bench');

class VoiceServer {
  constructor(log) { this.log = log || (() => {}); this.procs = []; }

  async healthy(url) {
    try { const r = await fetch(url + '/health', { signal: AbortSignal.timeout(1500) }); return r.ok ? await r.json() : null; } catch { return null; }
  }

  spawnLogged(cmd, args, opts, tag) {
    const p = spawnTree(cmd, args, opts);
    // Orpheus's llama-server reports every piece it generates (slot picked, timings, released), like the app's own
    // model server (llama.js): those lines are left out of the log; progress bars too
    const ROUTINE = /\bslot +(print_timing|release|get_availabl|launch_slot_)\b|it\/s\]|^\s*\d+%\||^\s*[\d.]+( [IWE])?\s*$/;
    const out = (d) => d.toString().split(/\r?\n/).filter((l) => l.trim() && !ROUTINE.test(l)).forEach((l) => this.log(`${tag}: ${l}`));
    p.stdout.on('data', out);
    p.stderr.on('data', out);
    p.on('exit', (code) => this.log(`${tag}: exited ${code}`));
    this.procs.push(p);
    return p;
  }

  async waitFor(url, what, ms) {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      if (this.procs.some((p) => p.exitCode !== null)) throw new Error(`${what} stopped during startup (see the log)`);
      const h = await this.healthy(url);
      if (h) return h;
      await new Promise((r) => setTimeout(r, 1000));
    }
    throw new Error(`${what} did not start in ${ms / 1000} s`);
  }

  // cfg from voices.js: { kind: 'chatterbox' | 'orpheus', url, port, speaker, llamaPort, llamaModel, llamaExe }
  async start(cfg) {
    const reuse = await this.healthy(cfg.url);
    if (reuse && reuse.voice === cfg.kind) { this.log(`tts: the ${reuse.voice} server is already running; switching its voice`); return reuse; }
    await this.stop();
    if (cfg.kind === 'orpheus') {
      const llamaUrl = `http://127.0.0.1:${cfg.llamaPort}`;
      if (!(await this.healthy(llamaUrl))) {
        this.spawnLogged(cfg.llamaExe, ['-m', cfg.llamaModel, '-ngl', '99', '-c', '8192', '--port', String(cfg.llamaPort), '--host', '127.0.0.1', '-np', String(cfg.parallel || 1)],
          { cwd: path.dirname(cfg.llamaExe), env: process.platform === 'linux' ? { ...process.env, LD_LIBRARY_PATH: [path.dirname(cfg.llamaExe), process.env.LD_LIBRARY_PATH].filter(Boolean).join(':') } : process.env }, 'orpheus-llama');
        await this.waitFor(llamaUrl, 'the Orpheus model server', 120000);
      }
      this.spawnLogged('uv', ['run', '--project', path.join(BENCH, 'orpheus'), 'python', path.join(BENCH, 'serve_orpheus.py'),
        '--port', String(cfg.port), '--llama-url', llamaUrl, '--voice', cfg.speaker], { cwd: path.join(BENCH, 'orpheus') }, 'orpheus');
    } else {
      this.spawnLogged('uv', ['run', '--project', path.join(BENCH, 'chatterbox'), 'python', path.join(BENCH, 'serve_chatterbox.py'),
        '--port', String(cfg.port)], { cwd: path.join(BENCH, 'chatterbox') }, 'chatterbox');
    }
    const h = await this.waitFor(cfg.url, `the ${cfg.kind} voice`, 240000);
    this.log(`tts: ${cfg.kind} voice ready (${h.sample_rate} Hz)`);
    return h;
  }

  async stop() {
    for (const p of this.procs) killTree(p);
    this.procs = [];
  }
}

// Whether a GPU voice can be offered: its environment and weights are in place
function available(kind) {
  const models = path.join(ROOT, 'models', 'tts_bench');
  if (kind === 'chatterbox') return fs.existsSync(path.join(BENCH, 'chatterbox', '.venv')) && fs.existsSync(path.join(models, 'chatterbox-turbo', 't3_turbo_v1.safetensors'));
  return fs.existsSync(path.join(BENCH, 'orpheus', '.venv')) && fs.existsSync(path.join(models, 'orpheus', 'orpheus-3b-0.1-ft-Q4_K_M.gguf'));
}

module.exports = { VoiceServer, available };
