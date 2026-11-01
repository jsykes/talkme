#!/usr/bin/env node
'use strict';
// Downloads what the app needs that git does not carry, into the repo layout, and skips what is already there:
// llama.cpp's server (the CUDA build for this system), the base model v1 runs on, the speech recogniser, punctuation
// and the Kokoro voice. Optional: other bases, the Orpheus voice. Interrupted downloads resume. Model files come from a
// fixed revision and are checked against their sha256 (models/bases.json). A failed step does not stop the others:
// everything that failed is listed at the end and the exit code is 1, and running it again skips what is there.
//   npm run setup                         the minimum for a first call (~19 GB)
//   npm run setup -- --base ministral3    another base model as well (see --list); --base all for every one
//   npm run setup -- --orpheus            the Orpheus voice (GPU, ~2 GB, needs uv)
//   npm run setup -- --train gemma4a4bstd what training on a base needs: its full-precision checkpoint (models/hf),
//                                         llama.cpp's source for the adapter converter, the trainer's Python packages,
//                                         and the voices the live tests use
//   npm run setup -- --check              only says what is there and what each download would be, downloads nothing
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawnSync } = require('child_process');
const { BASES } = require('../src/main/config');

const ROOT = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
const CHECK = args.includes('--check');
const has = (cmd) => (process.platform === 'win32' ? spawnSync(`${cmd} --version`, { stdio: 'ignore', shell: true }) : spawnSync(cmd, ['--version'], { stdio: 'ignore' })).status === 0; // (one string with the shell: Node warns about args with it)

const LLAMA_TAG = 'b11696'; // the build every adapter and test here ran on
const LLAMA_URL = `https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/`;
// per system: the server and the CUDA runtime it needs (only the NVIDIA driver has to be installed)
const LLAMA = {
  'win32-x64': { files: [`llama-${LLAMA_TAG}-bin-win-cuda-12.4-x64.zip`, 'cudart-llama-bin-win-cuda-12.4-x64.zip'], cuda: '12.4', driver: 551 },
  'linux-x64': { files: [`llama-${LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz`, `cudart-llama-${LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz`], cuda: '12.8', driver: 570 },
  'linux-arm64': { files: [`llama-${LLAMA_TAG}-bin-ubuntu-cuda-13.4-arm64.tar.gz`, `cudart-llama-${LLAMA_TAG}-bin-ubuntu-cuda-13.4-arm64.tar.gz`], cuda: '13.4', driver: 580 },
};
const SHERPA = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/';
const SPEECH = [
  { what: 'speech recogniser (Nemotron 0.6B, 560 ms)', dir: 'models/asr', name: 'sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25', url: SHERPA + 'asr-models/' },
  { what: 'punctuation', dir: 'models/punct', name: 'sherpa-onnx-online-punct-en-2024-08-06', url: SHERPA + 'punctuation-models/' },
  { what: 'Kokoro voices (CPU)', dir: 'models/tts', name: 'kokoro-multi-lang-v1_0', url: SHERPA + 'tts-models/' },
];
// the live tests' callers and the voice they check against
const TEST_VOICES = [
  { what: 'test voice: Piper Amy', dir: 'models/tts', name: 'vits-piper-en_US-amy-medium', url: SHERPA + 'tts-models/' },
  { what: 'test voice: Piper Lessac', dir: 'models/tts', name: 'vits-piper-en_US-lessac-medium', url: SHERPA + 'tts-models/' },
];
const HF = (repo, file, rev = 'main') => `https://huggingface.co/${repo}/resolve/${rev}/${file}`;
const ORPHEUS_REV = 'e2b00302c46af8205f521f60016600aa25a068e6', SNAC_REV = 'd73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec';
const ORPHEUS = [
  { url: HF('unsloth/orpheus-3b-0.1-ft-GGUF', 'orpheus-3b-0.1-ft-Q4_K_M.gguf', ORPHEUS_REV), to: 'models/tts_bench/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf', sha256: '4e94a925593c8fc1c20ee50e5b0e4bf915ea106a0bda3b917ff69b02559f7ab9' },
  { url: HF('hubertsiuzdak/snac_24khz', 'config.json', SNAC_REV), to: 'models/tts_bench/snac_24khz/config.json' },
  { url: HF('hubertsiuzdak/snac_24khz', 'pytorch_model.bin', SNAC_REV), to: 'models/tts_bench/snac_24khz/pytorch_model.bin', sha256: '4b8164cc6606bfa627f1a784734c1e539891518f1191ed9194fe1e3b9b4bff40' },
];
// sent only to huggingface.co, for a gated repo
const AUTH = process.env.HF_TOKEN ? { Authorization: `Bearer ${process.env.HF_TOKEN}` } : {};

// files of a training checkpoint the trainer does not read (Mistral's single-file copy of the same weights, other
// formats); each base's checkpoint repo is its trainFrom in models/bases.json, its folder its hf
const TRAIN_SKIP = /^(consolidated.*\.safetensors|original\/.*|.*\.(gguf|pth|bin|pt))$|^\./;
// a base by its name or an alias ("original" is the first Qwen in recipes)
const baseNamed = (k) => (BASES[k] ? k : Object.keys(BASES).find((n) => (BASES[n].aliases || []).includes(k)));

const gb = (n) => (n / 1e9 >= 1 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.round(n / 1e6)} MB`);
const say = (s) => console.log(s);

// what went wrong, for the list at the end
const failed = [];
function fail(what, why) {
  say(`FAIL  ${what}: ${why}`);
  failed.push(`${what}: ${why}`);
}
async function attempt(what, step) {
  try { await step(); } catch (e) { fail(what, e.message); }
}

async function sha256Of(file) {
  const h = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(file, { highWaterMark: 1 << 24 })) h.update(chunk);
  return h.digest('hex');
}

async function size(url) {
  try { const r = await fetch(url, { method: 'HEAD', redirect: 'follow' }); return r.ok ? Number(r.headers.get('content-length')) || 0 : -r.status; } catch { return -1; }
}

// to `file` through `file.part`, resuming a partial one; with `sha256`, a file that does not match never reaches `file`
async function download(url, file, headers = {}, sha256 = null) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const part = file + '.part';
  const have = fs.existsSync(part) ? fs.statSync(part).size : 0;
  const r = await fetch(url, { headers: have ? { ...headers, Range: `bytes=${have}-` } : headers, redirect: 'follow' })
    .catch((e) => { throw new Error(`${url}: ${(e.cause && e.cause.message) || e.message}`); });
  // nothing past the end of `part`: it holds the whole file already, or more than there is
  if (r.status === 416 && have) { fs.rmSync(part); return download(url, file, headers, sha256); }
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const resumed = r.status === 206;
  const total = Number(r.headers.get('content-length') || 0) + (resumed ? have : 0);
  const out = fs.createWriteStream(part, { flags: resumed ? 'a' : 'w' });
  let done = resumed ? have : 0, shown = 0;
  for await (const chunk of r.body) {
    if (!out.write(chunk)) await new Promise((res) => out.once('drain', res));
    done += chunk.length;
    if (process.stdout.isTTY && Date.now() - shown > 500) { shown = Date.now(); process.stdout.write(`\r  ${gb(done)} of ${gb(total)}   `); }
  }
  await new Promise((res, rej) => out.end((e) => (e ? rej(e) : res())));
  if (process.stdout.isTTY) process.stdout.write('\r' + ' '.repeat(40) + '\r');
  if (sha256 && (await sha256Of(part)) !== sha256) {
    fs.rmSync(part);
    throw new Error(`${url}: the file is not the one expected (sha256), so it was not kept`);
  }
  fs.renameSync(part, file);
}

// unpacks into `dest`; an archive holding one folder gives that folder's contents
function unpack(archive, dest) {
  const tmp = dest + '.unpack';
  fs.rmSync(tmp, { recursive: true, force: true });
  fs.mkdirSync(tmp, { recursive: true });
  // Windows' own tar (bsdtar) reads .zip and .tar.bz2; Git Bash's would be found first on the PATH and reads neither
  const tar = process.platform === 'win32' ? path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe') : 'tar';
  const r = spawnSync(tar, ['-xf', archive, '-C', tmp], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error(`could not unpack ${archive}`);
  let from = tmp;
  const top = fs.readdirSync(tmp);
  if (top.length === 1 && fs.statSync(path.join(tmp, top[0])).isDirectory()) from = path.join(tmp, top[0]);
  fs.mkdirSync(dest, { recursive: true });
  for (const f of fs.readdirSync(from)) fs.renameSync(path.join(from, f), path.join(dest, f));
  fs.rmSync(tmp, { recursive: true, force: true });
}

async function fetchTo(url, file, what, sha256 = null) {
  if (fs.existsSync(file)) { say(`ok    ${what}`); return; }
  const n = await size(url);
  if (CHECK) {
    if (n < 0) fail(what, `unreachable (${-n})  ${url}`);
    else say(`need  ${what}: ${n > 0 ? gb(n) : '?'}  ${url}`);
    return;
  }
  say(`get   ${what} (${n > 0 ? gb(n) : '?'})`);
  await download(url, file, url.startsWith('https://huggingface.co/') ? AUTH : {}, sha256);
}

async function fetchUnpacked(url, archiveName, dest, ready, what) {
  if (ready()) { say(`ok    ${what}`); return; }
  const archive = path.join(ROOT, 'vendor', 'downloads', archiveName);
  await fetchTo(url, archive, what + ' (archive)');
  if (CHECK) return;
  // a damaged archive would otherwise be found again by the next run and fail the same way
  try { unpack(archive, dest); } catch (e) { fs.rmSync(archive, { force: true }); throw e; }
  fs.rmSync(archive, { force: true });
  if (!ready()) throw new Error(`${what}: unpacked, but not where expected (${dest})`);
}

function gpu() {
  const r = spawnSync('nvidia-smi', ['--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'], { encoding: 'utf8' });
  if (r.status !== 0) return null;
  const [name, mem, driver] = r.stdout.trim().split('\n')[0].split(',').map((s) => s.trim());
  return { name, memGb: Number(mem) / 1024, driver };
}

async function main() {
  if (args.includes('--list')) {
    for (const [k, b] of Object.entries(BASES)) {
      const train = b.hf && fs.existsSync(path.join(ROOT, b.hf, 'config.json')) ? '  [trainable]' : b.trainFrom ? '  (--train for its checkpoint)' : '';
      say(`${k.padEnd(13)} ${b.label}  (${b.from})${fs.existsSync(path.join(ROOT, b.gguf)) ? '  [downloaded]' : ''}${train}`);
    }
    return;
  }
  const sys = LLAMA[`${process.platform}-${process.arch}`];
  if (!sys) { say(`talkme runs on Windows or Linux with an NVIDIA GPU; ${process.platform}-${process.arch} is not set up here.`); process.exit(1); }

  // what the steps below cannot install
  const g = gpu();
  if (!g) say(`warn  no NVIDIA driver found (nvidia-smi): install one, ${sys.driver} or newer`);
  else {
    say(`ok    ${g.name}, ${g.memGb.toFixed(0)} GB, driver ${g.driver}`);
    if (parseInt(g.driver, 10) < sys.driver) say(`warn  llama.cpp's CUDA ${sys.cuda} build needs driver ${sys.driver} or newer`);
    if (g.memGb < 15) say('warn  under 16 GB of VRAM: raise "experts on the CPU" in Settings to fit the model (slower ticks)');
  }
  for (const [cmd, why] of [['git-lfs', 'the trained adapters are kept in Git LFS'], ['uv', 'the echo canceller, GPU voices and training are Python (uv)'], ['ffmpeg', 'making a voice from a clip (Chatterbox)']]) {
    if (!has(cmd)) say(`warn  ${cmd} not found: ${why}`);
  }

  // adapters: real files, not Git LFS pointers
  const lora = path.join(ROOT, 'models', 'lora');
  const pointers = fs.readdirSync(lora).filter((f) => f.endsWith('.gguf') && fs.statSync(path.join(lora, f)).size < 1024);
  if (!pointers.length) say('ok    adapters (models/lora)');
  else if (CHECK) say(`need  adapters: ${pointers.length} are Git LFS pointers (git lfs pull)`);
  else {
    say('get   adapters (git lfs pull)');
    const r = spawnSync('git', ['lfs', 'pull', '--include', 'models/lora/*'], { cwd: ROOT, stdio: 'inherit' });
    if (r.status !== 0) fail('adapters', 'git lfs pull failed: install Git LFS (git lfs install), then run this again');
  }

  // llama.cpp. Its Linux builds need glibc 2.38 or newer (Ubuntu 24.04, Debian 13, Fedora 39); on an older system
  // the server would not start, so say so instead of downloading it
  const exe = path.join(ROOT, 'vendor', 'llama', process.platform === 'win32' ? 'llama-server.exe' : 'llama-server');
  const [server, cudart] = sys.files;
  const glibc = process.platform === 'linux' && (process.report.getReport().header.glibcVersionRuntime || '');
  const [maj, min] = glibc ? glibc.split('.').map(Number) : [];
  const onPath = process.platform === 'linux' && spawnSync('llama-server', ['--version'], { stdio: 'ignore' }).status === 0;
  if (fs.existsSync(exe) || onPath) say(`ok    llama.cpp server${fs.existsSync(exe) ? '' : ' (on the PATH)'}`);
  else if (glibc && (maj < 2 || (maj === 2 && min < 38))) {
    say(`warn  this system has glibc ${glibc}; llama.cpp's Linux release needs 2.38 or newer (Ubuntu 24.04, Debian 13, Fedora 39 or newer).`);
    say('      Build llama.cpp yourself with CUDA (needs the CUDA toolkit), put llama-server on the PATH, and run this again.');
  } else await attempt(`llama.cpp ${LLAMA_TAG}`, () => fetchUnpacked(LLAMA_URL + server, server, path.dirname(exe), () => fs.existsSync(exe), `llama.cpp ${LLAMA_TAG} (CUDA ${sys.cuda})`));
  const runtime = (f) => (process.platform === 'win32' ? /^cudart64_.*\.dll$/.test(f) : /^libcudart\.so/.test(f));
  if (fs.existsSync(exe)) await attempt(`CUDA ${sys.cuda} runtime`, () => fetchUnpacked(LLAMA_URL + cudart, cudart, path.dirname(exe), () => fs.existsSync(path.dirname(exe)) && fs.readdirSync(path.dirname(exe)).some(runtime), `CUDA ${sys.cuda} runtime for llama.cpp`));
  if (process.platform !== 'win32' && fs.existsSync(exe)) fs.chmodSync(exe, 0o755);

  // base models: v1's unless named
  const named = args.flatMap((a, i) => (a === '--base' ? [args[i + 1]] : []));
  const bases = named.includes('all') ? Object.keys(BASES) : named.length ? named : ['gemma4a4b'];
  for (const k of bases) {
    const b = BASES[k];
    if (!b) { fail(`base ${k}`, 'no base with that name (npm run setup -- --list)'); continue; }
    await attempt(`base model: ${b.label}`, () => fetchTo(HF(b.from, path.basename(b.gguf), b.rev), path.join(ROOT, b.gguf), `base model: ${b.label}`, b.sha256));
  }

  for (const s of [...SPEECH, ...(args.includes('--train') ? TEST_VOICES : [])]) {
    const dest = path.join(ROOT, s.dir, s.name);
    await attempt(s.what, () => fetchUnpacked(`${s.url}${s.name}.tar.bz2`, `${s.name}.tar.bz2`, dest, () => fs.existsSync(dest) && fs.readdirSync(dest).some((f) => f.endsWith('.onnx')), s.what));
  }

  if (args.includes('--orpheus')) {
    for (const o of ORPHEUS) await attempt(`Orpheus: ${path.basename(o.to)}`, () => fetchTo(o.url, path.join(ROOT, o.to), `Orpheus: ${path.basename(o.to)}`, o.sha256));
  }
  // training: each named base's checkpoint, file by file (the repo's list from the Hugging Face API; HF_TOKEN is sent
  // when set, for a gated repo), llama.cpp's source at the server's tag (export_gguf.py uses its converter), and the
  // trainer's packages
  const trains = args.flatMap((a, i) => (a === '--train' ? [args[i + 1]] : []));
  if (trains.length) {
    const trainable = Object.keys(BASES).filter((n) => BASES[n].trainFrom && BASES[n].hf);
    for (const k of trains) await attempt(`training checkpoint: ${k}`, async () => {
      const b = BASES[baseNamed(k)];
      if (!b) throw new Error('no base with that name (npm run setup -- --list)');
      const repo = b.trainFrom, dir = b.hf, rev = b.trainRev || 'main';
      if (!repo || !dir) throw new Error(`it has none to download${b.note ? ` (${b.note})` : ''}; these do: ${trainable.join(', ')}`);
      const r = await fetch(`https://huggingface.co/api/models/${repo}/tree/${rev}?recursive=true`, { headers: AUTH });
      if (!r.ok) throw new Error(`${repo}: HTTP ${r.status} (a gated repo needs HF_TOKEN and its licence accepted on huggingface.co)`);
      const files = (await r.json()).filter((f) => f.type === 'file' && !TRAIN_SKIP.test(f.path));
      const missing = files.filter((f) => !fs.existsSync(path.join(ROOT, dir, f.path)));
      const total = files.reduce((s, f) => s + (f.lfs ? f.lfs.size : f.size), 0);
      if (!missing.length) { say(`ok    training checkpoint: ${repo} (${dir})`); return; }
      const need = missing.reduce((s, f) => s + (f.lfs ? f.lfs.size : f.size), 0);
      if (CHECK) { say(`need  training checkpoint: ${repo}, ${missing.length} of ${files.length} files, ${gb(need)} of ${gb(total)}  -> ${dir}`); return; }
      say(`get   training checkpoint: ${repo} (${gb(need)}) -> ${dir}`);
      for (const f of missing) { say(`      ${f.path}`); await download(HF(repo, f.path, rev), path.join(ROOT, dir, f.path), AUTH, f.lfs ? f.lfs.oid : null); }
    });
    const src = path.join(ROOT, 'vendor', 'llama.cpp');
    if (fs.existsSync(path.join(src, 'convert_lora_to_gguf.py'))) say('ok    llama.cpp source (vendor/llama.cpp)');
    else if (CHECK) say(`need  llama.cpp source ${LLAMA_TAG} (git clone, ~0.1 GB) -> vendor/llama.cpp`);
    else {
      say(`get   llama.cpp source ${LLAMA_TAG} -> vendor/llama.cpp`);
      const g = spawnSync('git', ['clone', '--depth', '1', '--branch', LLAMA_TAG, 'https://github.com/ggml-org/llama.cpp', src], { stdio: 'inherit' });
      if (g.status !== 0) fail('llama.cpp source', 'git clone failed');
    }
    if (!has('uv')) fail('trainer packages', 'uv not found: the trainer needs it (https://docs.astral.sh/uv/)');
    else if (!CHECK) {
      say('get   trainer packages (uv sync --project tune --extra train)');
      if (spawnSync('uv', ['sync', '--project', path.join(ROOT, 'tune'), '--extra', 'train'], { cwd: ROOT, stdio: 'inherit' }).status !== 0) fail('trainer packages', 'uv sync failed');
    }
  }
  // the Python environments, when uv is there: the echo canceller always, the Orpheus voice when asked for
  if (has('uv') && !CHECK) {
    for (const [dir, want] of [['aec', true], ['tts_bench/orpheus', args.includes('--orpheus')]]) {
      if (!want || fs.existsSync(path.join(ROOT, dir, '.venv'))) continue;
      say(`get   Python environment: ${dir}`);
      if (spawnSync('uv', ['sync', '--project', path.join(ROOT, dir)], { cwd: ROOT, stdio: 'inherit' }).status !== 0) fail(`Python environment: ${dir}`, 'uv sync failed');
    }
  }
  if (failed.length) {
    say(`\n${failed.length} failed:`);
    for (const f of failed) say(`  ${f}`);
    say(CHECK ? '(--check: nothing was downloaded)' : 'The rest is in place. Once fixed, run the same command again: it skips what is there.');
    process.exitCode = 1;
  } else say(CHECK ? '\n(--check: nothing was downloaded)' : '\nDone. npm start, then Load models.');
}

if (require.main === module) main().catch((e) => { console.error("setup: " + e.message); process.exit(1); });
module.exports = { unpack, download };
