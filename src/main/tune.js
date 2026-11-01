'use strict';
// The Tune tab's back end: recipes in tune/recipes/*.json, run stage by stage through tune/pipeline.py (the same
// script the command line uses), with the output streamed to the renderer and the results read back from tune/out.
const { spawnTree, killTree, killPortOwner } = require('./proc');
const path = require('path');
const fs = require('fs');
const { ROOT, BASES, storageLink } = require('./config');

const TUNE = path.join(ROOT, 'tune');
const OUT = path.join(TUNE, 'out');
const RECIPES = path.join(TUNE, 'recipes');
const STAGES = ['dialogues', 'data', 'train', 'replays', 'live', 'all'];

class Tune {
  // beforeRun: called before a stage starts (the pipeline runs its own llama-server, so the app's must be stopped);
  // storage: the folder for large files (Settings), where a run's training cache goes
  constructor(log, emit, { beforeRun, storage = () => '' } = {}) {
    this.log = log; this.emit = emit; this.beforeRun = beforeRun; this.storage = storage; this.proc = null; this.stage = null; this.recipe = null;
  }

  status() { return { running: !!this.proc, stage: this.stage, recipe: this.recipe }; }

  recipes() {
    fs.mkdirSync(RECIPES, { recursive: true });
    // runs that have produced something come first, the most recent at the top; untouched templates after
    const list = fs.readdirSync(RECIPES).filter((f) => f.endsWith('.json')).map((f) => {
      let r = {};
      try { r = JSON.parse(fs.readFileSync(path.join(RECIPES, f), 'utf8')); } catch {}
      const name = r.name || f.replace(/\.json$/, '');
      let ran = 0;
      try { ran = fs.statSync(path.join(OUT, `${name}.pipeline.log`)).mtimeMs; } catch {}
      return { file: f, name, about: r._about || '', ran };
    });
    return list.sort((a, b) => (b.ran - a.ran) || a.file.localeCompare(b.file));
  }

  // the bases a run can train on: those in models/bases.json with a full-precision checkpoint folder ("hf"), which the
  // trainer streams; one whose checkpoint is not on disk yet is listed but cannot train. aliases: other names recipes
  // use for it ("original" for the first Qwen)
  bases() {
    return Object.entries(BASES).filter(([, b]) => b.hf).map(([name, b]) => ({
      name, label: b.label, aliases: b.aliases || [], trainable: fs.existsSync(path.join(ROOT, b.hf, 'config.json')),
      download: !!b.trainFrom, gguf: fs.existsSync(path.join(ROOT, b.gguf)), format: b.format,
    }));
  }

  // the loss weight of every tick tag a recipe leaves out (common.py WEIGHTS)
  weights() {
    const src = fs.readFileSync(path.join(TUNE, 'common.py'), 'utf8');
    const block = (src.match(/^WEIGHTS = \{([\s\S]*?)^\}/m) || [])[1] || '';
    return Object.fromEntries([...block.matchAll(/"(\w+)":\s*([\d.]+)/g)].map((m) => [m[1], Number(m[2])]));
  }

  // one line on where a running stage is, from the newest of its logs: caching (Phase A), training steps with the time
  // left, or the replays' call count
  progress(file) {
    let name;
    try { name = JSON.parse(this.readRecipe(file)).name; } catch { return ''; }
    const logs = [['train', `${name}.log`], ['recorded calls', `eval_yap_${name}.log`], ['held-out set', `eval_ours_${name}.log`]]
      .map(([what, f]) => { try { return { what, text: fs.readFileSync(path.join(OUT, f), 'utf8'), t: fs.statSync(path.join(OUT, f)).mtimeMs }; } catch { return null; } })
      .filter(Boolean).sort((a, b) => b.t - a.t);
    if (!logs.length || Date.now() - logs[0].t > 10 * 60e3) return '';
    const { what, text } = logs[0];
    const lines = text.split(/\r?\n/);
    const last = (re) => { for (let i = lines.length - 1; i >= 0; i--) { const m = lines[i].match(re); if (m) return m; } return null; };
    if (what === 'train') {
      const step = last(/^step (\d+)\/(\d+) .* ([\d.]+) s\/step/);
      if (step) return `training: step ${step[1]} of ${step[2]}, about ${Math.ceil(((step[2] - step[1]) * step[3]) / 60)} min left`;
      const chunk = last(/^ chunk (\d+)\/(\d+)/);
      return chunk ? `caching the calls through the base model: chunk ${chunk[1]} of ${chunk[2]}` : 'training: loading the base model';
    }
    const call = last(/^call (\d+)\/(\d+)/);
    return call ? `${what}: call ${call[1]} of ${call[2]}` : what;
  }

  readRecipe(file) { return fs.readFileSync(path.join(RECIPES, path.basename(file)), 'utf8'); }

  // saves the text (must be valid JSON); saveAs: a new file name, and the recipe's name follows it
  saveRecipe(file, text, saveAs) {
    let r;
    try { r = JSON.parse(text); } catch (e) { throw new Error('not valid JSON: ' + e.message); }
    if (!r.name) throw new Error('the recipe needs a "name" (its outputs are named after it)');
    let target = path.basename(file);
    let note = '';
    if (saveAs) {
      target = saveAs.replace(/[^\w.-]+/g, '-').replace(/\.json$/, '') + '.json';
      if (fs.existsSync(path.join(RECIPES, target))) throw new Error(`there is already a run called ${target}`);
      const src = JSON.parse(this.readRecipe(file));
      if (r.name === src.name) r.name = target.replace(/\.json$/, '');
      // the adapter's file name carries the run, its prompt and its base (the file itself carries them too, for the app:
      // tune/adapter_meta.py at export); the source's name in the app is not the new run's
      const base = r.base || 'original';
      if (r.label && r.label === src.label) delete r.label;
      r.export_as = `talkme-${r.name}-${r.prompt || 'assistant'}${base === 'original' ? '' : '-' + base}.gguf`;
      // only the weights, the steps or the tests changed (not the layers it learns in): the new run borrows the source's
      // training cache (the calls run through the base model, Phase A), which saves the first hour of training
      const DATA = ['base', 'writer', 'prompt', 'dialogues', 'dialogue_files', 'build', 'mix', 'seed'];
      const tr = r.train || {}; const st = src.train || {};
      const same = DATA.every((k) => JSON.stringify(r[k]) === JSON.stringify(src[k])) && tr.max_tokens === st.max_tokens && (tr.top ?? 4) === (st.top ?? 4);
      const cache = src.cache || src.name;
      if (same && fs.existsSync(path.join(OUT, `cache_${cache}`))) { r.cache = cache; note = `reuses ${cache}'s training cache (skips the first hour)`; }
      else { delete r.cache; note = 'builds its own training cache (about an hour on the 4090)'; }
      text = JSON.stringify(r, null, 2) + '\n';
    }
    fs.writeFileSync(path.join(RECIPES, target), text);
    return { file: target, name: r.name, note };
  }

  // what the run has produced so far: the pipeline's own log lines, the quiz, the recorded-conversation table and
  // the live report, plus the exported adapter's path in models/lora when it exists
  results(file) {
    let r;
    try { r = JSON.parse(this.readRecipe(file)); } catch { return { text: '', adapter: null }; }
    const name = r.name;
    const read = (p) => { try { return fs.readFileSync(p, 'utf8'); } catch { return ''; } };
    const parts = [];
    // the results first (quiz, held-out set, recorded calls, live tests), then what the pipeline is doing
    const quiz = read(path.join(OUT, `probe_${name}.log`)).trim().split('\n').pop();
    if (quiz) parts.push(`Quiz: ${quiz}`);
    const log = read(path.join(OUT, `${name}.pipeline.log`)).split(/\r?\n/).filter((l) => l.startsWith(`[${name}]`) || l.startsWith('[bench]'));
    const ours = log.filter((l) => / ours: /.test(l)).pop();
    if (ours) parts.push('Held-out set: ' + ours.replace(/^.* ours: /, '').replace(/\s{2,}/g, '  '));
    const yap = read(path.join(OUT, `yap_${name}.md`));
    if (yap) parts.push('\nRecorded conversations (a person\'s number in brackets):\n' + yap.trim());
    const live = read(path.join(ROOT, 'tests', 'out', 'live_bench.md'));
    if (live && live.includes(name)) parts.push('\nLive tests (tests/out/live_bench.md):\n' + live.split(/\r?\n\r?\n/)[0]);
    if (log.length) parts.push('\nPipeline log:\n' + log.slice(-25).join('\n'));
    const exported = path.join(ROOT, 'models', 'lora', r.export_as || `talkme-${name}.gguf`);
    // trained already: the train stage skips it, so changing what it learns needs a new run
    const trained = fs.existsSync(path.join(OUT, `${name}.gguf`));
    return { text: parts.join('\n'), adapter: fs.existsSync(exported) ? exported : null, name, trained };
  }

  async run(stage, file) {
    if (this.proc) throw new Error(`tune: ${this.stage} of ${this.recipe} is still running`);
    if (!STAGES.includes(stage)) throw new Error('tune: unknown stage ' + stage);
    const recipe = path.join(RECIPES, path.basename(file));
    if (!fs.existsSync(recipe)) throw new Error('tune: no such recipe ' + file);
    fs.mkdirSync(OUT, { recursive: true });
    // a run that will train makes its cache (tens of GB) in the folder for large files, when one is set
    const r = JSON.parse(fs.readFileSync(recipe, 'utf8'));
    if (['train', 'all'].includes(stage) && !fs.existsSync(path.join(OUT, `${r.name}.gguf`))) {
      const there = storageLink(path.relative(ROOT, path.join(OUT, `cache_${r.cache || r.name}`)), this.storage());
      if (there) this.emit('log', `the training cache goes to ${there}`);
    }
    if (this.beforeRun) await this.beforeRun();
    const argv = ['run', '--project', TUNE, 'python', path.join(TUNE, 'pipeline.py'), '--recipe', recipe, '--stage', stage];
    this.stage = stage; this.recipe = path.basename(file);
    this.emit('log', `$ uv ${argv.join(' ')}`);
    this.proc = spawnTree('uv', argv, { cwd: TUNE, env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' } });
    this.proc.stdout.on('data', (d) => this.emit('log', d.toString().trimEnd()));
    this.proc.stderr.on('data', (d) => this.emit('log', d.toString().trimEnd()));
    this.proc.on('exit', (code) => { this.proc = null; this.emit('done', { stage, recipe: this.recipe, code }); this.stage = null; });
    return { message: `${stage} started for ${this.recipe}` };
  }

  // stops the stage and everything it started (the model server, the trainer); a stopped stage resumes next time
  stop() {
    if (!this.proc) return;
    killTree(this.proc);
    // on Windows a model server the pipeline started can outlive the tree kill (on Linux it is in the stopped group):
    // the llama-server on the pipeline's port, and no other (the Orpheus voice's, another program's: this used to stop
    // every llama-server there was)
    if (process.platform === 'win32') {
      const pid = killPortOwner(8080, 'llama-server');
      if (pid) this.log(`tune: stopped the pipeline's model server (pid ${pid})`);
    }
  }
}

module.exports = { Tune, OUT, STAGES };
