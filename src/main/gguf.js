'use strict';
// Reads the key/value metadata at the start of a GGUF file (format: github.com/ggml-org/ggml docs/gguf.md), without the
// tensors. Adapters carry talkme.* keys (tune/adapter_meta.py) saying which base and prompt they were trained on.
const fs = require('fs');

const T = { U8: 0, I8: 1, U16: 2, I16: 3, U32: 4, I32: 5, F32: 6, BOOL: 7, STRING: 8, ARRAY: 9, U64: 10, I64: 11, F64: 12 };
const SIZE = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };

// keys: only these are decoded (others are skipped), e.g. a base model's huge tokenizer arrays; null = all
// stopAfter: stop reading once every wanted key is found (a base model's header runs to megabytes)
function readMeta(file, { keys = null, maxBytes = 64 << 20 } = {}) {
  const fd = fs.openSync(file, 'r');
  let buf = Buffer.alloc(0), base = 0; // buf holds file bytes [base, base + buf.length)
  const need = (pos, n) => {
    if (pos + n <= base + buf.length) return;
    if (pos + n - base > maxBytes) throw new Error('GGUF header too large');
    const more = Buffer.alloc(Math.max(n, 1 << 20));
    const got = fs.readSync(fd, more, 0, more.length, base + buf.length);
    if (!got) throw new Error('GGUF header cut short');
    buf = Buffer.concat([buf, more.subarray(0, got)]);
    need(pos, n);
  };
  let pos = 0;
  const u32 = () => { need(pos, 4); const v = buf.readUInt32LE(pos - base); pos += 4; return v; };
  const u64 = () => { need(pos, 8); const v = Number(buf.readBigUInt64LE(pos - base)); pos += 8; return v; };
  const str = () => { const n = u64(); need(pos, n); const s = buf.toString('utf8', pos - base, pos - base + n); pos += n; return s; };
  const scalar = (t) => {
    need(pos, SIZE[t]); const o = pos - base; pos += SIZE[t];
    switch (t) {
      case T.U8: return buf.readUInt8(o); case T.I8: return buf.readInt8(o);
      case T.U16: return buf.readUInt16LE(o); case T.I16: return buf.readInt16LE(o);
      case T.U32: return buf.readUInt32LE(o); case T.I32: return buf.readInt32LE(o);
      case T.F32: return buf.readFloatLE(o); case T.BOOL: return buf.readUInt8(o) !== 0;
      case T.U64: return Number(buf.readBigUInt64LE(o)); case T.I64: return Number(buf.readBigInt64LE(o));
      case T.F64: return buf.readDoubleLE(o);
      default: throw new Error(`GGUF value type ${t}`);
    }
  };
  const skip = (t) => {
    if (t === T.STRING) { const n = u64(); pos += n; return; }
    if (t === T.ARRAY) {
      const et = u32(), n = u64();
      if (SIZE[et]) pos += n * SIZE[et]; else for (let i = 0; i < n; i++) skip(et);
      return;
    }
    pos += SIZE[t];
  };
  const value = (t) => {
    if (t === T.STRING) return str();
    if (t === T.ARRAY) { const et = u32(), n = u64(), a = []; for (let i = 0; i < n; i++) a.push(value(et)); return a; }
    return scalar(t);
  };
  try {
    need(0, 24);
    if (buf.toString('latin1', 0, 4) !== 'GGUF') throw new Error('not a GGUF file');
    pos = 4; u32(); u64(); // version, tensor count
    const n = u64(), out = {};
    let left = keys ? keys.length : Infinity;
    for (let i = 0; i < n && left > 0; i++) {
      const k = str(), t = u32();
      if (!keys || keys.includes(k)) { out[k] = value(t); left--; } else skip(t);
    }
    return out;
  } finally { fs.closeSync(fd); }
}

module.exports = { readMeta };
