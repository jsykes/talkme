'use strict';
// The echo canceller: WebRTC AEC3 in a small Python helper (aec/aec_server.py, its own uv environment in aec/.venv).
// Every ~100 ms mic frame goes in with what the speaker played over the same stretch (the renderer takes both in one
// audio callback) and comes back with the assistant's voice removed, before the speech recogniser hears it. Without
// it, a call on speakers hears itself: whatever echo gets through is transcribed as the caller talking.
//
// Frames are answered strictly in order (one pipe, one queue). If the helper is missing or stops, frames pass through
// untouched and the call goes on as before.
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { ROOT, venvPython } = require('./config');

const DIR = path.join(ROOT, 'aec');
const PY = venvPython(DIR);

class Aec {
  constructor(log) {
    this.log = log || (() => {});
    this.proc = null;
    this.ready = false;
    this.queue = []; // { n, done } per frame sent, oldest first
    this.buf = Buffer.alloc(0);
  }

  installed() { return fs.existsSync(PY) || fs.existsSync(path.join(DIR, 'pyproject.toml')); }

  async start({ delayMs = 40, noiseSuppression = true, gainControl = true } = {}) {
    await this.stop();
    const args = [path.join(DIR, 'aec_server.py'), '--rate', '16000', '--delay-ms', String(delayMs),
      ...(noiseSuppression ? ['--ns'] : []), ...(gainControl ? ['--agc'] : [])];
    // the environment's own python when it is there (starts at once); uv makes it on first use otherwise
    const [cmd, argv] = fs.existsSync(PY) ? [PY, args] : ['uv', ['run', '--project', DIR, 'python', ...args]];
    const p = spawn(cmd, argv, { cwd: DIR, windowsHide: true });
    this.proc = p;
    p.stdout.on('data', (d) => this.onData(d));
    p.stderr.on('data', (d) => d.toString().split(/\r?\n/).filter(Boolean).forEach((l) => this.log(l)));
    p.on('exit', (code) => {
      if (this.proc !== p) return;
      this.log(`aec: helper exited (${code}); the mic goes to the recogniser as it is`);
      this.proc = null; this.ready = false;
      this.flushQueue();
    });
    await new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error('the echo canceller did not start in 60 s')), 60000);
      const onErr = (d) => { if (/aec: ready/.test(d.toString())) { clearTimeout(t); p.stderr.off('data', onErr); resolve(); } };
      p.stderr.on('data', onErr);
      p.once('exit', (code) => { clearTimeout(t); reject(new Error(`the echo canceller stopped while starting (${code})`)); });
    });
    this.ready = true;
  }

  async stop() {
    const p = this.proc;
    this.proc = null; this.ready = false;
    this.flushQueue();
    if (p) { try { p.stdin.end(); } catch {} setTimeout(() => { try { p.kill(); } catch {} }, 1000); }
  }

  // frames still waiting when the helper goes away: hand them on unprocessed, in order
  flushQueue() { const q = this.queue; this.queue = []; this.buf = Buffer.alloc(0); q.forEach((f) => f.done(f.mic)); }

  setDelay(ms) { if (this.ready) this.write(3, Math.max(0, Math.round(ms))); }
  reset() { if (this.ready) this.write(2, 0); }

  write(kind, n, body) {
    const head = Buffer.alloc(8); head.writeUInt32LE(kind, 0); head.writeUInt32LE(n, 4);
    this.proc.stdin.write(body ? Buffer.concat([head, body]) : head);
  }

  // mic, ref: Float32Array of the same length (ref null: nothing is known about the speaker, pass the mic through).
  // done(clean, level): level is the RMS before gain control, or undefined when the frame passed through
  process(mic, ref, done) {
    if (!this.ready || !ref || ref.length !== mic.length) { done(mic); return; }
    // copy each frame out of the IPC message's buffer, so nothing aliases it once it has been written
    const body = Buffer.concat([Buffer.from(mic.buffer, mic.byteOffset, mic.byteLength), Buffer.from(ref.buffer, ref.byteOffset, ref.byteLength)]);
    this.queue.push({ n: mic.length, mic, done });
    this.write(1, mic.length, body);
  }

  onData(d) {
    this.buf = Buffer.concat([this.buf, d]);
    // each reply: the cleaned samples, then their level before gain control (what the app ducks on)
    while (this.queue.length && this.buf.length >= this.queue[0].n * 4 + 4) {
      const f = this.queue.shift();
      const bytes = this.buf.subarray(0, f.n * 4);
      const level = this.buf.readFloatLE(f.n * 4);
      this.buf = this.buf.subarray(f.n * 4 + 4);
      const out = new Float32Array(f.n);
      Buffer.from(out.buffer).set(bytes); // an aligned copy for the recogniser's worker
      f.done(out, level);
    }
  }
}

module.exports = { Aec };
