# Data attribution

- `heldout_yap.jsonl` is derived from the **Open Yap 1K sample** by The Agentic Data Company
  (https://huggingface.co/datasets/TheAgenticDataCompany/open-yap-1k), licensed CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). Changes: 3-minute windows of 16 conversations; one speaker's track
  transcribed by the app's recogniser (Nemotron streaming 0.6B) into 600 ms fragments; the other speaker's words and
  timings taken from the dataset's machine transcripts, so both speakers' words are in the file. `tune/out/eval_*`
  hold only the models' replies to these calls. Built by `scripts/yap_to_calls.js`.
  The audio itself is not in this repository (`data/openyap/`, ignored).
  The dataset's licence file carries this request from The Agentic Data Company, on the speakers' behalf, and it applies
  to anything built on this file: do not attempt to identify any speaker, and do not create voice clones, replicas or
  generative reproductions identifiable as a speaker.
- `heldout_ours.jsonl` and the training calls are built from the dialogues in
  codacus/Qwen3.6-35B-A3B-MicroTurn-Duplex-LoRA (Apache 2.0).
- Calls built by `scripts/oto_to_calls.js` (kept in `tune/out/`, not committed) are derived from
  **otoSpeech-full-duplex-processed-141h** by otoearth
  (https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-processed-141h), licensed CC BY 4.0. Changes: both
  channels transcribed by the app's recogniser (`scripts/oto_transcribe.js`), cut into 460-tick windows, turn-taking
  tags from word timings; the audio is not in this repository (`data/otospeech/`, ignored).
- `train/` holds the training inputs that are free to share. `calls_built_r5_nodecline.jsonl` and
  `calls_train_masked_nodecline.jsonl` are built from the codacus dialogues and calls (Apache 2.0).
  `calls_special_r8_nodecline.jsonl` and the `dial_*.jsonl` files were written by the base models for this project.
  None of them holds a real conversation.
- `heldout_memory*.jsonl` and `quiz_wide.json` were written for this project, the memory calls by the base models.
  None of them holds a real conversation.
- The adapters in `models/lora/` were trained on those calls and dialogues, so these credits apply to them too.
