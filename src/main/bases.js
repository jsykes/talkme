'use strict';
// "Add a base model" in the Tune tab: looks a model up on Hugging Face (the quantized GGUF the app serves and the
// full-precision checkpoint the trainer streams), works out its chat format and which bases it shares weight shapes
// with, adds it to models/bases.json (config.js addBase), and downloads it with scripts/setup.js.
const path = require('path');
const fs = require('fs');
const { ROOT, BASES, addBase, storageLink } = require('./config');
const { spawnTree } = require('./proc');

const API = 'https://huggingface.co/api/models/';
const auth = () => (process.env.HF_TOKEN ? { Authorization: `Bearer ${process.env.HF_TOKEN}` } : {});
// the families the trainer has a profile for (tune/profiles.py model_family) and the app a chat format for (config.js
// FORMATS), by the checkpoint's model_type
const FAMILIES = { gemma4: 'gemma4', mistral3: 'mistral', ministral3: 'mistral', qwen3_5_moe: 'qwen3.6', gpt_oss: 'harmony' };
const repoOk = (r) => /^[\w.-]+\/[\w.-]+$/.test(String(r || '').trim());

async function tree(repo) {
  const r = await fetch(API + repo + '/tree/main', { headers: auth() });
  if (r.status === 401 || r.status === 403) throw new Error(`${repo} is gated: accept its licence on huggingface.co and start the app with HF_TOKEN set`);
  if (!r.ok) throw new Error(`${repo}: not found on Hugging Face (HTTP ${r.status})`);
  return r.json();
}
const sizeOf = (f) => (f.lfs ? f.lfs.size : f.size) || 0;

// the GGUF files of a repo at its top level (not a vision projector or a speculative-decoding draft model), the usual
// Q4_K_M first (MXFP4 for gpt-oss, which ships its experts in it)
async function ggufFiles(repo) {
  if (!repoOk(repo)) throw new Error('a repo is owner/name, as on huggingface.co');
  const files = (await tree(repo.trim())).filter((f) => f.type === 'file' && /\.gguf$/i.test(f.path) && !/mmproj|eagle|draft/i.test(f.path));
  if (!files.length) throw new Error(`${repo} has no .gguf files at its top level`);
  const rank = (p) => (/Q4_K_M|MXFP4/i.test(p) ? 0 : /Q4_K_S|IQ4_XS/i.test(p) ? 1 : /Q5_K_M/i.test(p) ? 2 : 3);
  return files.map((f) => ({ file: f.path, size: sizeOf(f) })).sort((a, b) => rank(a.file) - rank(b.file) || a.size - b.size);
}

// the layer shapes that decide whether adapters fit: the text model's type, layers, width and experts
function shape(cfg) {
  const t = cfg.text_config || cfg;
  return { modelType: cfg.model_type, sig: [t.model_type || cfg.model_type, t.num_hidden_layers, t.hidden_size, t.num_experts || t.num_local_experts || 0].join('/') };
}
function localShape(b) {
  try { return shape(JSON.parse(fs.readFileSync(path.join(ROOT, b.hf, 'config.json'), 'utf8'))).sig; } catch { return null; }
}

// What a new base would be, without changing anything: { row, family, sameAs, sizes }. ggufRepo/ggufFile: the
// quantized model; trainRepo: the full-precision checkpoint.
async function inspect({ name, label, ggufRepo, ggufFile, trainRepo }) {
  name = String(name || '').trim();
  if (!/^[a-z0-9]+$/.test(name)) throw new Error('the name is lower-case letters and digits (it ends adapter file names), e.g. gemma4e4b');
  if (BASES[name]) throw new Error(`there is already a base called ${name}`);
  if (!repoOk(ggufRepo) || !repoOk(trainRepo)) throw new Error('both repos are owner/name, as on huggingface.co');
  const ggufs = await ggufFiles(ggufRepo);
  const g = ggufs.find((f) => f.file === ggufFile) || ggufs[0];
  const files = (await tree(trainRepo.trim())).filter((f) => f.type === 'file');
  if (!files.some((f) => f.path === 'config.json')) throw new Error(`${trainRepo} has no config.json: it is not a Transformers checkpoint`);
  const weights = files.filter((f) => /\.safetensors$/.test(f.path) && !/^consolidated/.test(f.path)).reduce((s, f) => s + sizeOf(f), 0);
  const r = await fetch(`https://huggingface.co/${trainRepo.trim()}/resolve/main/config.json`, { headers: auth() });
  if (!r.ok) throw new Error(`${trainRepo}: could not read config.json (HTTP ${r.status})`);
  const cfg = await r.json();
  const { modelType, sig } = shape(cfg);
  const family = FAMILIES[modelType];
  if (!family) {
    throw new Error(`${trainRepo} is a ${modelType} model, a family the trainer does not know yet. It knows ${Object.keys(FAMILIES).join(', ')}. ` +
      'A new family needs code: its layer layout in tune/profiles.py and its chat format in src/main/config.js FORMATS (README, Training your own adapter).');
  }
  // the trainer reads full-precision weights: OpenAI's gpt-oss stores the experts in MXFP4, Mistral's plain repos are FP8
  if (modelType === 'gpt_oss' && /mxfp4/i.test(JSON.stringify(cfg.quantization_config || ''))) {
    throw new Error(`${trainRepo} stores the experts in MXFP4; the trainer needs a BF16 copy, e.g. unsloth/${trainRepo.trim().split('/')[1]}-BF16`);
  }
  if (FAMILIES[modelType] === 'mistral' && /fp8/i.test(JSON.stringify(cfg.quantization_config || ''))) {
    throw new Error(`${trainRepo} is stored in FP8; the trainer needs the BF16 release (Mistral publishes it as <name>-BF16)`);
  }
  // a base already here with the same shapes: same arch (they share adapters) and the same call settings
  const twinKey = Object.keys(BASES).find((k) => BASES[k].hf && localShape(BASES[k]) === sig);
  const twin = twinKey && BASES[twinKey];
  const folder = trainRepo.trim().split('/')[1];
  // the same GGUF as a base already listed: the same file, not a second copy
  const shared = Object.values(BASES).find((b) => b.from === ggufRepo.trim() && path.basename(b.gguf) === g.file);
  const row = {
    label: String(label || '').trim() || folder,
    gguf: shared ? shared.gguf : `models/base/${name}/${g.file}`, from: ggufRepo.trim(),
    hf: `models/hf/${folder}`, trainFrom: trainRepo.trim(),
    format: family, arch: twin ? twin.arch : `${modelType}-${sig.split('/').slice(1).join('x')}`,
    ...(twin ? Object.fromEntries(['ctx', 'historyTicks', 'expertsOnGpu', 'gpuLayersWithVoice'].filter((k) => twin[k] != null).map((k) => [k, twin[k]])) : { ctx: 16384, historyTicks: 400 }),
    note: twin ? `added in the Tune tab; the same shapes as ${twinKey}, so it shares its adapters and call settings`
      : 'added in the Tune tab; call settings are defaults (ctx 16384, 400 ticks kept), not measured on this model',
  };
  const have = { gguf: fs.existsSync(path.join(ROOT, row.gguf)), checkpoint: fs.existsSync(path.join(ROOT, row.hf, 'config.json')) };
  return { name, row, family, sameAs: twinKey || null, ggufs, sizes: { gguf: g.size, checkpoint: weights }, have };
}

async function add(spec) {
  const { name, row } = await inspect(spec);
  addBase(name, row);
  return { name, row };
}

// downloads a base (its GGUF and its training checkpoint, plus what training needs) with scripts/setup.js, streaming
// its output to `emit`; resumes where a stopped download was. storage: the folder for large files (Settings), where a
// checkpoint, and a GGUF in a folder of its own, go when they are not here yet
let proc = null;
function download(name, emit, storage) {
  if (proc) throw new Error('a download is already running');
  const b = BASES[name];
  if (!b) throw new Error(`no base called ${name}`);
  const dirs = [b.trainFrom && b.hf, path.dirname(b.gguf) !== 'models/base' && path.dirname(b.gguf)].filter(Boolean);
  for (const dir of dirs) {
    const there = storageLink(dir, storage);
    if (there) emit('log', `${dir} goes to ${there}`);
  }
  const argv = [path.join(ROOT, 'scripts', 'setup.js'), '--base', name, ...(b.trainFrom ? ['--train', name] : [])];
  emit('log', `$ node ${argv.map((a) => path.relative(ROOT, a) || a).join(' ')}`);
  proc = spawnTree(process.execPath, argv, { cwd: ROOT, env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' } });
  proc.stdout.on('data', (d) => emit('log', d.toString().trimEnd()));
  proc.stderr.on('data', (d) => emit('log', d.toString().trimEnd()));
  proc.on('exit', (code) => { proc = null; emit('done', { stage: 'download', recipe: name, code }); });
  return { message: `downloading ${name}` };
}

module.exports = { ggufFiles, inspect, add, download, FAMILIES };
