'use strict';
// "Save call audio" (Settings, off by default): each call's audio as three 16 kHz WAV files in the app's data folder
// (calls/<time>-mic.wav, -speaker.wav, -clean.wav: what the mic heard, what the speaker played, the mic after the echo
// canceller), so a glitch heard in a call can be found and looked at afterwards. Only the last KEEP calls are kept.
const fs = require('fs');
const path = require('path');

const KEEP = 5;

class WavWriter {
  constructor(file, rate = 16000) {
    this.fd = fs.openSync(file, 'w');
    this.rate = rate;
    this.bytes = 0;
    fs.writeSync(this.fd, this.header(0));
  }
  header(n) {
    const h = Buffer.alloc(44);
    h.write('RIFF', 0); h.writeUInt32LE(36 + n, 4); h.write('WAVE', 8); h.write('fmt ', 12);
    h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22); h.writeUInt32LE(this.rate, 24);
    h.writeUInt32LE(this.rate * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34); h.write('data', 36); h.writeUInt32LE(n, 40);
    return h;
  }
  write(f32) {
    if (!f32 || this.fd === null) return;
    const b = Buffer.alloc(f32.length * 2);
    for (let i = 0; i < f32.length; i++) b.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32767))), i * 2);
    fs.writeSync(this.fd, b);
    this.bytes += b.length;
  }
  close() {
    if (this.fd === null) return;
    fs.writeSync(this.fd, this.header(this.bytes), 0, 44, 0);
    fs.closeSync(this.fd);
    this.fd = null;
  }
}

class CallRecorder {
  constructor(dir, log) { this.dir = dir; this.log = log || (() => {}); this.w = null; }

  start() {
    this.stop();
    fs.mkdirSync(this.dir, { recursive: true });
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    this.w = Object.fromEntries(['mic', 'speaker', 'clean'].map((k) => [k, new WavWriter(path.join(this.dir, `${stamp}-${k}.wav`))]));
    this.t0 = Date.now();
    this.log(`callrec: saving this call's audio to ${path.join(this.dir, stamp)}-*.wav`);
    // older calls go: keep the last KEEP (three files each)
    const calls = [...new Set(fs.readdirSync(this.dir).filter((f) => /-(mic|speaker|clean)\.wav$/.test(f)).map((f) => f.replace(/-(mic|speaker|clean)\.wav$/, '')))].sort();
    for (const c of calls.slice(0, Math.max(0, calls.length - KEEP))) for (const k of ['mic', 'speaker', 'clean']) fs.rmSync(path.join(this.dir, `${c}-${k}.wav`), { force: true });
  }

  // mic, speaker: the raw frame and the speaker reference; clean: the frame the recogniser got
  add({ mic, speaker, clean }) {
    if (!this.w) return;
    this.w.mic.write(mic); this.w.speaker.write(speaker || new Float32Array(mic.length)); this.w.clean.write(clean);
  }

  stop() {
    if (!this.w) return;
    for (const w of Object.values(this.w)) w.close();
    this.log(`callrec: saved ${((Date.now() - this.t0) / 1000).toFixed(0)} s of call audio`);
    this.w = null;
  }
}

module.exports = { CallRecorder };
