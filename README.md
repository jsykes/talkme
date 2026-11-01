# talkme

A voice partner you talk to like a friend on the phone, running entirely on your own computer. It listens while it
talks: it says "mm-hm" while you tell a story, stops when you cut in, picks up again when you go quiet, and speaks up
if your plan sounds dangerous. There is no voice activity detector deciding when it may speak. Every 600 ms a local
text LLM reads the words the microphone just heard and writes either a few words to say next or `<listen>`. A small
LoRA adapter teaches it to do that.

https://github.com/user-attachments/assets/bd064b85-6166-4895-9d2d-93c0ee5281f1

It comes as an Electron app (the call, the voices, the settings) and a Python pipeline for training your own adapters
from the app's **Tune** tab.

The current version (v1) is the `g26b4` adapter on Gemma 4 26B-A4B. Five more adapters ship for comparison, on Gemma,
Qwen3.6, Ministral and gpt-oss. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how it fits together,
[docs/RESULTS.md](docs/RESULTS.md) has every training run, and [docs/ROADMAP.md](docs/ROADMAP.md) says what's next.

## What you need

| | |
|---|---|
| **System** | Windows 10 or 11, or Linux (x64) as new as Ubuntu 24.04, Debian 13 or Fedora 39 (llama.cpp's Linux build needs glibc 2.38; on an older system build llama.cpp yourself and put `llama-server` on the PATH). You need a desktop session and a microphone. Headphones are easiest; speakers work too (see [Speakers](#speakers)). Linux support is new and not yet tested on a Linux machine; please report what breaks. |
| **GPU** | An NVIDIA card. It was built on an RTX 4090 (24 GB), which runs v1 with room for the Orpheus voice beside it. On a smaller card, raise **n-cpu-moe** in Settings so more of the model sits in system RAM; each tick then takes longer. Nothing below 24 GB has been measured yet. |
| **NVIDIA driver** | 551 or newer on Windows, 570 or newer on Linux (`nvidia-smi` shows the version). You do **not** need to install CUDA: the llama.cpp build ships its CUDA runtime, and the Python parts install PyTorch builds that carry their own. |
| **RAM and disk** | 32 GB of RAM or more. About 19 GB of disk for the minimum setup; each extra base model is 8 to 21 GB more. |
| **Tools** | [Node.js](https://nodejs.org) 20 or newer, Git with [Git LFS](https://git-lfs.com) (the adapters are stored in it), and [uv](https://docs.astral.sh/uv/) for the Python parts (the echo canceller, the GPU voices, training). [ffmpeg](https://ffmpeg.org) only if you want to make a voice from your own clip. |

## Setup

```bash
git lfs install
git clone https://github.com/jsykes/talkme.git
cd talkme
npm install
npm run setup
```

`npm run setup` ([scripts/setup.js](scripts/setup.js)) downloads what git doesn't carry. It puts each file where the
app looks for it, skips anything already there, and resumes a download that was cut off:

| What | From | Size | Goes to |
|---|---|---|---|
| llama.cpp b11696, CUDA build for your system, plus its CUDA runtime | [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases/tag/b11696) | 0.7 GB | `vendor/llama/` |
| Gemma 4 26B-A4B heretic, Q4_K_M (v1's base model) | [mradermacher/gemma-4-26B-A4B-it-heretic-GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-heretic-GGUF) | 16.8 GB | `models/base/gemma4/` |
| Speech recogniser (Nemotron 0.6B streaming, 560 ms) | [sherpa-onnx asr-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | 0.5 GB | `models/asr/` |
| Punctuation | [sherpa-onnx punctuation-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models) | 31 MB | `models/punct/` |
| Kokoro voices (on the CPU) | [sherpa-onnx tts-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models) | 0.35 GB | `models/tts/` |

The script also:
- checks your driver, VRAM and tools;
- fetches the adapters with `git lfs pull` if the clone only holds LFS pointers;
- sets up the echo canceller's Python environment when uv is installed.

Model files come from a fixed revision of each repo and are checked against their sha256, both listed in
`models/bases.json`. If a download fails, setup carries on with the rest, lists what failed at the end and exits with
an error. Run the same command again once it is fixed: it skips what is there.

Options:

```bash
npm run setup -- --check
```

Shows what is there and what each download would be, without downloading anything.

```bash
npm run setup -- --list
```

Lists the base models and which are downloaded.

```bash
npm run setup -- --base ministral3
```

Adds another base model. `--base all` gets every one.

```bash
npm run setup -- --orpheus
```

Adds the Orpheus voice (GPU, about 2 GB, needs uv).

To train your own adapters, `npm run setup -- --train <base>` gets what training needs (see
[Training your own adapter](#training-your-own-adapter)).

If you'd rather download the files yourself, the table above and `models/bases.json` (each base model's `from`, and
`trainFrom` for its training checkpoint) give the sources. Keep the file names.

**Linux notes:**
- Electron needs the usual desktop libraries; a desktop install of Ubuntu or Fedora has them.
- The microphone goes through PulseAudio or PipeWire.
- On Ubuntu 24.04, `npm start` may stop with "The SUID sandbox helper binary was found, but is not configured
  correctly". Either start it with `npm start -- --no-sandbox`, or give the sandbox helper its rights once:
  `sudo chown root node_modules/electron/dist/chrome-sandbox && sudo chmod 4755 node_modules/electron/dist/chrome-sandbox`.
- You can also use a `llama-server` you built yourself: put it on your PATH, or in `vendor/llama/`.

## Your first call

```bash
npm start
```

1. Click **Load models**. The first load reads the 16 GB model from disk and takes a minute or so; later loads are
   faster.
2. The adapter list shows **Partner v1** for the model that is loaded. Leave it.
3. Click **Start call** and talk.

The **tick tape** on the right shows every 600 ms decision: what it heard, what it said, how sure it was that it should
stay quiet (P(listen)), and how long the model took. Type into the box under the transcript for a call without the
microphone. [tests/manual_script.md](tests/manual_script.md) is a 10-minute script that goes through every behaviour.

If an adapter's base model isn't downloaded, **Load models** says which one it needs and the setup command that gets it.

## What the default will and won't refuse

Partner v1 runs on a decensored model: Gemma 4 26B-A4B with its refusals removed by Heretic. It was also trained with no
examples of declining, so that it talks like a friend and not like an assistant. It was trained to speak up when a plan
sounds dangerous, but it is not a safe-for-everyone product: it will go along with many requests a standard model
turns down. In the live tests it declined 6 of 12 requests it should have refused. There is no content filter, and it
has not been tested for use by children or by anyone at risk. Partner v0 on Gemma and on Qwen (HauhauCS) and the Gemma
4 31B base are decensored too.

For a model that declines, use **Partner v1 (standard Gemma)**: the same recipe on Gemma 4 26B-A4B as Google released
it, 12 of 12 declines in the same tests, 85% of all live checks against v1's 86%, and a little slower to take its turn.

```bash
npm run setup -- --base gemma4a4bstd
```

## Adapters

Choose one in the Call tab. Each adapter file records which base model it was trained on, the system prompt it
expects, its listen threshold and its name in the list. The app restarts the model server on the right base when you
switch (about a minute). Thresholds you change with the sliders are remembered per adapter.

| In the list | File (`models/lora/`) | Base model |
|---|---|---|
| **Partner v1** (default) | `talkme-partner-v1-gemma4a4b.gguf` | Gemma 4 26B-A4B, decensored (heretic) |
| Partner v1 (standard Gemma) | `talkme-partner-v1-gemma4a4bstd.gguf` | Gemma 4 26B-A4B as Google released it |
| Partner v1 (Ministral) | `talkme-partner-v1-ministral3.gguf` | Ministral 3 14B |
| Partner v1 (gpt-oss) | `talkme-partner-v1-gptoss.gguf` | gpt-oss 20B |
| First Gemma partner (g26) | `talkme-partner-v0-gemma4a4b.gguf` | Gemma 4 26B-A4B, decensored |
| Best Qwen partner (run 12) | `talkme-partner-v0-hauhau.gguf` | Qwen3.6 35B-A3B, abliterated (HauhauCS) |
| Published micro-turn adapter | `microturn-duplex-lora.gguf` (not in git: [download it from codacus](https://huggingface.co/codacus/Qwen3.6-35B-A3B-MicroTurn-Duplex-LoRA/resolve/main/microturn-duplex-lora.gguf) into `models/lora/`) | Qwen3.6 35B-A3B |

To try an adapter on another base, get that base with `npm run setup -- --base <name>`. An adapter from somewhere else
needs its base written in: use `tune/adapter_meta.py`, or end its file name with the base's name, like
`-gemma4a4b.gguf`. Otherwise the app lists it greyed out instead of guessing.

The **Adapters** tab lists every adapter with its base, prompt, recipe, size, training date, thresholds and test
results, and shows which test run is closest to how the file runs. There you can write notes, change its name in the
list, use it, rename its file, or move it to the Recycle Bin. Notes and that name go in a small JSON file beside the
adapter (`talkme-partner-v1-gemma4a4b.json`), because changing the GGUF would rewrite the whole file. The pipeline adds
the run, its results and a suggested name there after each run. A rename also updates saved settings and the
`export_as` of the recipes that export the file.

## Voices

**Orpheus Tara** (GPU) is the default when it is installed (`npm run setup -- --orpheus`); otherwise the app uses
Kokoro, which runs on the CPU and needs no setup. The **voice** list in the Call tab has every voice, and switching is
instant.

**Chatterbox** (GPU) makes a voice from a 10-second clip or recording with **+ voice**. It needs ffmpeg, `uv sync --project
tts_bench/chatterbox`, and its weights in `models/tts_bench/chatterbox-turbo` (`tts_bench/download.sh` shows how they
were fetched).

## Speakers

On speakers the mic hears the assistant's own voice, and the recogniser would pass it on as if you had said it. Two
things stop that. Both are on by default (`aec` and `ownWordsFilter` in the settings file).

- **The echo canceller** (`aec/`, WebRTC AEC3): every 100 ms of mic audio is cleaned against what the speaker played
  over the same stretch, before the recogniser hears it. It needs uv; setup installs it, and the **aec** pill shows
  whether it is running. Offline, with a simulated room, it takes the echo down 24-30 dB and keeps your voice when you
  cut in.
- **The own-words filter** (`src/main/loop.js`): while it talks and for 1.5 s after, heard words that repeat its own
  last words in order are dropped as leftover echo.

The **echo test** button in the Call tab (models loaded, no call running) plays a sentence with the mic open and reports
how many of its words the recogniser heard. The goal is none.

## Sharing the GPU

**n-cpu-moe** in Settings is how many layers' experts sit in system RAM instead of VRAM. Raise it to make room for a
game or another model; ticks get slower. To stop a training run that is using the GPU:

```bash
bash tune/gpu_pause.sh
```

Running the same pipeline command again resumes it.

## Training your own adapter

A training run is a recipe, `tune/recipes/<name>.json`. It sets:
- the base model and system prompt;
- what dialogues the base model writes;
- how they are timed into calls and mixed with other data (real recorded conversations, calls with warnings in them);
- the loss weight per kind of tick, the LoRA size, and which tests to run.

`tune/recipes/g26b4.json` is the recipe v1 was made with. Training it from a fresh clone gets close to v1, not v1 itself:
[The training data](#the-training-data) says why, and what to add for the full recipe.

### What training needs

Every base model comes in two forms. The **GGUF** is a compressed copy that llama.cpp runs: the app uses it for calls,
and the pipeline for its tests. The **full-precision checkpoint** (BF16 safetensors) is what the trainer learns
on, since working out how to adjust the model needs the exact weights. Training ends in a small adapter file (a few MB)
that runs on top of the GGUF, so once you are done training on a base, its checkpoint can be deleted; calls never use
it.

Training needs more than calling does. One command gets it for a base model:

```bash
npm run setup -- --train gemma4a4bstd
```

It downloads four things, skipping what is already there:
- the base's full-precision checkpoint into `models/hf/` (49 GB for Gemma 4 26B-A4B, 28 GB for Ministral 3 14B, 67 GB
  for Qwen3.6 35B-A3B);
- llama.cpp's source at the server's tag into `vendor/llama.cpp/`, for the converter that exports the adapter;
- the trainer's Python packages (`uv sync --project tune --extra train`);
- the two Piper voices the live tests use.

Add `--check` to see what it would download and how big it is. The bases you can train on are `gemma4a4bstd` (Gemma 4
26B-A4B), `gemma4a4b` (its heretic version, v1's base), `gemma4` (Gemma 4 31B heretic), `ministral3`, `gptoss`
(gpt-oss 20B) and `qwen` (Qwen3.6). The base model a run is tested on, its GGUF, comes from `npm run setup -- --base <name>` as for calls.

Plan for the disk and the GPU too:
- **Disk:** a run on a new base first builds a training cache in `tune/out/`, about 54 GB for Gemma 4 26B-A4B. Runs
  that only change the loss weights, the training steps or the tests share their source run's cache.
- **Another drive:** checkpoints and caches go inside the project folder. If its drive is short of space, set
  **Settings, Storage, folder for large files** (e.g. `E:\talkme`): from then on Download in Add a base model puts a
  checkpoint there (and a new base's GGUF, when it has a folder of its own), and a run that trains puts its cache
  there. Each goes at the same path as in the project (`E:\talkme\models\hf\...`, `E:\talkme\tune\out\cache_<run>`)
  and is linked into the project, so everything finds it; what is already on disk stays where it is. It is a link
  per folder: on Windows a directory junction, on Linux a symlink. Without the app (the pipeline on the command
  line), make the link yourself before downloading or training. For example, for the checkpoint of `gptoss` and the
  cache of a run named `gptoss1`:

  ```bat
  mklink /J models\hf\gpt-oss-20b-BF16 E:\talkme\hf\gpt-oss-20b-BF16
  ```

  ```bat
  mklink /J tune\out\cache_gptoss1 E:\talkme\cache\cache_gptoss1
  ```

  Run these in Command Prompt from the project folder, after making the folders on the other drive. A run's cache is
  `tune/out/cache_<run name>`, and **Check** in Add a base model shows where a checkpoint goes.
- **GPU:** about 14 GB while training. The stages start their own model server, so the app's is stopped while a
  stage runs.

### The training data

Everything a run needs to start is in the repository, in `tune/data/train/`, except the real conversations:

| File | What it is | In the repository |
|---|---|---|
| `dial_run10`, `dial_partner`, `dial_partner2` | Friend dialogues written by the base models | Yes |
| `calls_built_r5_nodecline`, `calls_train_masked_nodecline` | Warnings and cut-ins, from the codacus dataset | Yes |
| `calls_special_r8_nodecline` | Resumed answers and other special calls | Yes |
| `calls_oto_all` | Real conversations from otoSpeech, 250 of about 1,800 calls in v1's mix | No: you build it |

Without `calls_oto_all`, a run says so and goes on. It is saved as `<name>-no-oto.gguf` with "(no otoSpeech)" in its
label, so it never replaces an adapter that came with the repository. Its test results do replace the recipe's in
`tune/out`; `git checkout` brings those back.

The adapter works, but it learns from written calls only and it is pushier. `g26b4` trained this way from a fresh clone
took 72 minutes on an RTX 4090. One run of each:

| Test | Partner v1 | Without `calls_oto_all` |
|---|---|---|
| Recorded calls: takes the turn within about 2.4 s | 58% | 61% |
| Recorded calls: starts talking over the caller, per minute | 1.0 | 2.1 |
| Recorded calls: quiet when the person gave way to a cut-in | 89% | 74% |
| Held-out calls: answers when it should | 83% | 80% |
| Held-out calls: stops when cut in on | 89% | 94% |
| Held-out calls: back-channels when the person did | 26% | 17% |

To add the real conversations, ask for access to
[otoSpeech](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-processed-141h) (a short form, CC BY 4.0), set
`HF_TOKEN`, and download it (19 GB):

```bash
uvx --from huggingface_hub hf download otoearth/otoSpeech-full-duplex-processed-141h --repo-type dataset --local-dir data/otospeech
```

Then transcribe it. This runs on the CPU and takes about 12 hours with three workers. Run this once for each worker,
with `--worker 0`, `1` and `2`, each in its own terminal; a stopped worker picks up where it was:

```bash
node scripts/oto_transcribe.js --worker 0 --workers 3
```

Then build the calls. Runs pick them up from `tune/out/` from then on:

```bash
node scripts/oto_to_calls.js --out tune/out/calls_oto_all.jsonl --dev-out tune/out/calls_oto_all_dev.jsonl
```

A few recipes need more than this. Before it starts anything, a run lists any file it does not have and where that file
comes from:
- Some `g26m` recipes, the memory experiments, read dialogues left by earlier runs that are not in the repository.
- A recipe with `init_from` carries on from another run's adapter, so that run comes first. `gptoss15`, for example,
  needs `gptoss2`, then `gptoss12`.
- Side tools that compare adapters, such as `tune/compare.py`, default to the codacus dataset in `models/lora/dataset/`,
  which setup does not fetch.

### Training in the Tune tab

1. Open the **Tune** tab and pick a run to start from in the **run** list. `g26b4` is v1; `g26b4std` is the same
   recipe on the standard Gemma 4 26B-A4B.
2. Change what you want in the form: the model, the loss weights under "What it learns" (each says what it does and
   which test shows it), the training steps, how many layers it learns in, or the tests' listen threshold. The whole
   recipe is under "Everything in the recipe (JSON)".
3. Type a name in "new run's name" and click **New run from this**. A run that is trained already can't be trained
   again with changes; the banner at the top says so. The note beside the button says whether the new run reuses the
   source's training cache, which skips about an hour.
4. Click **Run everything**. It runs the run picked in the list as saved: with a new run's name typed but not made,
   or changes not saved, it says so and waits. The line beside the buttons shows the stage and where it is
   ("training: step 120 of 600, about 35 min left"), and **Progress** shows the pipeline's output. You can also run
   the stages one at a time: write dialogues, build data, train and export, quiz and replays, live tests.
5. **Pause** stops the stage. Running it again resumes: each stage skips work that is already done, and training goes
   on from its last saved step.
6. When it is done, **Results** shows the quiz, the held-out set, the recorded calls and the live tests, next to the
   runs named in "compare with". **Use this run's adapter** loads it in the Call tab, and **Load models** brings the
   app's model server back.

On a 4090, a run of Gemma 4 26B-A4B that reuses a cache takes about 45 minutes to train (600 steps at about 4.3 s)
and about two hours of tests. A run on a new base adds about an hour for the cache.

The command line does the same thing:

```bash
cd tune && uv run python pipeline.py --recipe recipes/g26b4.json
```

Add `--stage train` to run one stage, or `--dry-run` to see the commands. The pipeline starts its own model server, so
close the app's first. If it needs something you don't have, it stops before doing any work and says what is missing.

The stages, if you want to run them yourself:

1. **Dialogues** (`gen_dialogues.py`): the base model writes conversations in the friend's style, with hazards,
   interrupted stories and "something the friend can't know" moments spliced in.
2. **Calls** (`build_calls.py`, `mix_calls.py`): the dialogues are cut into 0.6 s ticks with simulated recogniser
   noise, fillers, back-channels, cut-ins and silences, and mixed with the files in
   [The training data](#the-training-data).
3. **Train** (`train_lora.py`): a rank-32 LoRA on the model's last four layers (`train.top` in the recipe, "layers it
   learns in" in the form). The lower layers run once and are cached, then only the top ones train. More layers can
   learn more, but take more GPU memory, and another number needs its own cache: the cache is the input of the first
   trained layer (its `layers.json` says which), so the trainer refuses one made for other layers. On most bases the
   LoRA is on attention and the feed-forward layers. gpt-oss keeps its experts in fused tensors PEFT cannot adapt, so
   it trains attention only, unless `train.experts` is on ("learns in the experts too"). Then each trained layer also
   gets one adapter shared by all 32 of its experts (`profiles.ExpertLoRA`). The export writes it once for every expert,
   as llama.cpp expects, so the file is about 600 MB. `check_expert_lora.py` compares such an adapter with llama.cpp.
4. **Export** (`export_gguf.py`) to a llama.cpp adapter, stamped with its base, prompt and threshold
   (`adapter_meta.py`).
5. **Tests**:
   - `probe_answers.py`: a 12-question quiz;
   - `evaluate.py` on 24 recorded real calls (`data/heldout_yap.jsonl`, scored against what the person did) and on
     scripted calls (`data/heldout_ours.jsonl`: warnings caught, stopping when cut in on);
   - `scripts/live_bench.sh`: 28 live scenarios × 3 synthetic callers through the real audio path.

### Adding a base model

The base models are listed in one file, `models/bases.json`, which the app, the pipeline and setup all read. To add
one, open **Add a base model** at the bottom of the Tune tab:

1. Give it a short name (lower-case letters and digits; adapters trained on it end with it) and a name for the app.
2. Paste the Hugging Face repo of its GGUF and click **Look up**; the usual Q4_K_M file is picked first.
3. Paste the repo of its full-precision checkpoint, the same model unquantized. It has to be BF16: **Check** refuses
   OpenAI's own gpt-oss (its experts are stored in MXFP4) and Mistral's plain repos (FP8), and names the BF16 copy.
4. Click **Check**. It reads the checkpoint's config and says its chat format, whether it has the same layer shapes as
   a base you have (then it shares that base's adapters and call settings), and what is already on disk. Nothing
   changes yet.
5. Click **Add**, then **Download**. The download runs `npm run setup -- --base <name> --train <name>`, with its
   output in Progress, and resumes if stopped. The new base is then in the run's model list and in the Call tab.

For example, gpt-oss-20b: GGUF repo `ggml-org/gpt-oss-20b-GGUF` (`gpt-oss-20b-MXFP4.gguf`, 12.1 GB), checkpoint repo
`unsloth/gpt-oss-20b-BF16` (41.8 GB).

This works for any model of a family the trainer knows: Qwen3.6, Gemma 4, Ministral 3 and gpt-oss, each proven with a partner adapter above. **Check** says so when a
model is of another family. A new family needs code: its layer layout in `tune/profiles.py` and its chat format in
`src/main/config.js` (`FORMATS`). A gated repo needs `HF_TOKEN` set when you start the app, and its licence
accepted on huggingface.co.

## What is in the box

| Path | What |
|---|---|
| `src/main/` | Electron main process: `llama.js` (runs llama-server and one micro-turn per tick), `loop.js` (the 600 ms scheduler), `stt.js` (sherpa-onnx streaming recogniser), `tts.js` / `voices.js` / `voice-server.js` (the voices), `aec.js`, `config.js` (settings; the base models are in `models/bases.json`), `bases.js` (Add a base model), `gguf.js` (reads adapter metadata), `tune.js` (the Tune tab), `adapters.js` (the Adapters tab) |
| `src/renderer/` | The UI: Call tab (transcript, tick tape, typed input), Tune tab, Adapters tab, Settings |
| `tune/` | The training pipeline (`pipeline.py`, `recipes/`, and the stages above), held-out sets in `data/` |
| `scripts/` | `setup.js`, the live audio tests (`audio_test.js`, `live_bench.sh`, `live_bench_report.js`), conversation importers, benchmarks |
| `tests/` | Live scenarios and `manual_script.md`; results go to `tests/out/` |
| `aec/` | The echo canceller helper (Python, uv) |
| `tts_bench/` | The voice bake-off, and the Orpheus and Chatterbox servers the app uses |
| `models/` | `bases.json` lists the base models. `lora/` holds the adapters (Git LFS). Base GGUFs, speech models and training checkpoints are downloaded (git-ignored). |
| `vendor/` | `llama/`: the llama.cpp server build. `llama.cpp/`: its source, for training only. Both git-ignored. |

## Licence

Apache 2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE). Base models and speech models are downloaded from their
publishers and keep their own licences.

## Credits

- Micro-turn idea: DuplexCascade (Yang, Fujita and Sudo, 2026, [arXiv:2603.09180](https://arxiv.org/abs/2603.09180))
- Adapter format, first dataset and the listen-probability rule: codacus/Qwen3.6-35B-A3B-MicroTurn-Duplex-LoRA (Apache 2.0)
- Base models:
  - Gemma 4 (Google), decensored with Heretic by coder3101, GGUFs by mradermacher;
  - Qwen3.6-35B-A3B (Qwen), GGUF by unsloth; abliterated by HauhauCS;
  - Ministral 3 (Mistral AI), GGUF by bartowski;
  - gpt-oss-20b (OpenAI), GGUF by ggml-org, BF16 checkpoint by unsloth.
- Inference: llama.cpp (ggml-org)
- Speech: sherpa-onnx (k2-fsa), Nemotron streaming ASR (NVIDIA), Kokoro (hexgrad), Orpheus (Canopy Labs) with SNAC, Chatterbox (Resemble AI)
- Echo cancelling: WebRTC AEC3, through pywebrtc-audio
- Live test voices: Piper
- Data: otoSpeech (oto.earth, CC BY 4.0), Open Yap (CC BY 4.0); see [tune/data/ATTRIBUTION.md](tune/data/ATTRIBUTION.md)
