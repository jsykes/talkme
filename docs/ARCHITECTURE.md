# talkme: full-duplex voice chat from a plain text LLM

talkme is an Electron app that turns an ordinary local text LLM (by default Gemma 4 26B-A4B; `models/bases.json` lists the others) into a
"speaks while it listens" voice partner, and ships the tooling to tune its behaviour yourself.

The idea is the micro-turn from the DuplexCascade paper (arXiv 2603.09180): instead of waiting for a voice activity
detector to say "the user is done", the LLM gets a turn every 600 ms. Each turn it sees the transcript fragment the
microphone produced in that window (or `<no voice>`) and answers with up to three words to say next, or `<listen>`.
There is no VAD anywhere in the loop. The model decides, every tick, whether to talk.

## Components

```
 mic ──► renderer AudioWorklet (16 kHz f32) ──IPC──► main/stt.js  (sherpa-onnx Nemotron streaming, CPU, worker)
                                                          │ drain(): new words since last tick
                                                          ▼
                                          main/loop.js  600 ms scheduler, never overlapping ticks
                                                          │ heard  ──► main/llama.js ──► llama-server /completion
                                                          │                 ▲ listen rule, repeat guard, logit_bias ban
                                                          │ said   ◄────────┘
                                                          ▼
                                          main/tts.js  (Orpheus, Chatterbox, Kokoro or Piper)
                                                          │ f32 PCM chunks
 speakers ◄── renderer Web Audio queue ◄──IPC─────────────┘
```

| Piece | Where | Notes |
|---|---|---|
| LLM | `vendor/llama/llama-server` (b11696; CUDA 12.4 on Windows, 12.8 on Linux, or one on the PATH; `scripts/setup.js` fetches it) | Base GGUF + LoRA adapter, spawned and supervised by `src/main/llama.js`. The bases are one file, `models/bases.json`, read by `config.js` (`BASES`), `tune/pipeline.py` and `scripts/setup.js`, and added to from the Tune tab (`src/main/bases.js`): each entry has its GGUF, chat format (`FORMATS`), weight shape (which adapters load beside it) and what the server and loop need (context, history kept, GPU layers with a voice), plus where its GGUF and training checkpoint come from. Today: the original Qwen3.6-35B-A3B, HauhauCS's abliterated build, Gemma 4 31B and 26B-A4B heretic, the standard Gemma 4 26B-A4B and Ministral 3 14B. An adapter runs on the base it records in its own file (`talkme.base_model`, the base GGUF's own name, then `talkme.base`), else the one its file name ends with (`talkme-partner-v1-gemma4a4b.gguf`: `gemma4a4b`), and the app restarts the server when that changes. One that says neither is listed greyed out under the models of its architecture and refused, rather than guessed (`config.js adapterInfo`). Adding a base: a row there and in `pipeline.py`, and for a new family a format in both and a training profile in `tune/profiles.py`. |
| Adapters | `models/lora/*.gguf`, `tune/out/talkme-*.gguf` | Every adapter found is loaded at once with scale 0 (`--lora-init-without-apply`); each request names the active one, so the dropdown swaps adapters instantly, even mid-call. Default: v1, `talkme-partner-v1-gemma4a4b` (Gemma 4 26B-A4B heretic, layers 27-30). Kept beside it (2026-10-28): the author's `microturn-duplex-lora`, `talkme-partner-v0-gemma4a4b` (the first Gemma partner) and `talkme-partner-v0-hauhau` (the best Qwen partner, layers 36-39); the rest are in git history. Each adapter file carries its base, system prompt, listen threshold, carry-on bar, name in the list and run (`talkme.*` keys in its GGUF header, written at export by `tune/adapter_meta.py`, read by `src/main/gguf.js`; llama-server ignores them), so it works under any file name; the author's adapter, which has none, is in `config.js KNOWN_ADAPTERS`, and older files fall back to their names (`promptFor`). What you set with the sliders is remembered per adapter and comes first. All are rank 32 on the last 4 layers. Blends (`blend:a=0.5,b=0.5`, `BLENDS`) mix adapters per request. Beside each file, a small JSON card (`talkme-x.json`) holds what is not worth rewriting the GGUF for: notes and a name from the Adapters tab, and the run, results and suggested name from `tune/pipeline.py write_card`; the app prefers the card's name. `src/main/adapters.js` builds the tab's entries, matching test results by what each test ran on (`args.adapter`, threshold, bar, the file's former names), not by recipe name, because the same weights are often tested at several settings. |
| STT | sherpa-onnx Nemotron speech streaming 0.6B, 560 ms chunks (int8) | CPU, 4 threads, in a worker thread (`stt-worker.js`) so decoding never stalls the loop. Chosen by `scripts/stt_bench.js`: 0% word errors on the test voice, last word ~0.2 s after speech ends, real-time factor ~0.3. It cases and punctuates natively, so fragments look like the adapter's training data; words still growing are handed over as pieces and late punctuation goes with the next piece, both as in training. The worker is restarted if it dies (at most three times a minute) and reset at every call start, so nothing half-heard carries over. |
| TTS | Orpheus (the default once installed) or Chatterbox on the GPU, Kokoro 82M v1.0 (speaker 7) or Piper on the CPU, chosen in the voice dropdown (`src/main/voices.js`) | Kokoro and Piper run in a worker thread. Piper synthesises a phrase in ~50 ms; Kokoro sounds far more natural and takes ~0.3 s per phrase on 8 threads. Both spell "Mm-hm" letter by letter, so `tts.js` rewrites it (`SAY_AS`). `tts_bench/` holds the bake-off behind these choices. |
| UI | `src/renderer/` | Vanilla HTML/JS. Call tab: transcript, tick tape (heard / said / P(listen) / latency), adapter, voice, threshold, carry-on bar (both remembered per adapter) and "human timing" controls, typed input. Tune tab: train an adapter from the app: a form over a recipe (base model, prompt, the loss weights explained with the tests that show them, training size, tests), "New run from this" (reuses the source's training cache when only weights or tests change), the pipeline stages with a progress line, and Pause / resume (below). Settings. |

## The protocol (must match the adapter byte for byte)

System prompt, interpolating only the tick length (`src/main/config.js systemPrompt`, `tune/common.py SYSTEMS`):

```
assistant (every adapter up to run 8):
You are a voice assistant in a live phone call; both sides can talk at the same time. Every 0.6 seconds you get what the microphone heard in that moment, as a streaming transcript fragment, or <no voice>. You reply with up to three new words to say next, or <listen> to add nothing.

partner (run 9 on):
You are a friend on a live phone call; both of you can talk at the same time. Every 0.6 seconds you get what the microphone heard in that moment, as a streaming transcript fragment, or <no voice>. You reply with up to three new words to say next, or <listen> to add nothing.

persona (g26m on, when Settings > Persona is filled in; config.js personaPrompt, common.py persona_system):
<the partner prompt>

You are Maya, 29, a night-shift nurse who loves horror movies; dry and teasing.
What you remember about the caller from earlier calls:
- Their name is Alex.
- Alex's dog Biscuit hurt his paw last week.
Only bring up a shared past that is in this list.
(with no memories, the last three lines are: You know the caller as a friend, but nothing about their life from earlier calls.)
```

The app sends each adapter the prompt it was trained with (`index.js promptFor`: partner adapters from run 9 on and
every Gemma adapter get the partner prompt; those in `PERSONA_ADAPTERS` get the persona one when a persona is set, and
the plain partner prompt otherwise, which they were also trained on). History is rendered in the base's own format (`models/bases.json`) and sent to `/completion` (not `/chat/completions`). Each tick
appends one `user` message (`heard`) and one `assistant` message (`said`). Nothing already sent is ever edited, so
llama.cpp's prompt cache resumes every request and a tick costs ~130-250 ms instead of a full re-read.

**Speak-or-listen rule.** `<listen>` is a single token, while "keep talking" is spread across many candidate words, so
greedy decoding stops mid-sentence. The client reads the probability mass of tokens starting with `<` at the first
position. If it is ≥ the listen threshold the model listens; otherwise those token ids are banned with `logit_bias`
and the request is re-run so the model must pick a word. The threshold is per adapter: 0.3 for the assistant line
(run 2; run 4 0.45), 0.6 for everything trained on real conversation timing (run 6 on, all partner adapters), which
spreads "start talking" over several ticks and never reaches 70% on one. "Human timing" samples the decision with the
model's own probability instead; it over-talks on recorded calls, so it is off by default.

**Carry-on bar** (`llama.js tick`, config `carryOnP` 0.9, a Call-tab slider remembered per adapter; `evaluate.py --carry-on-p`): right after its own words
that did not end a sentence, with the line quiet, listening needs P(listen) ≥ 0.9 instead of the threshold. The timing
adapters stop mid-sentence into a silence at 0.6–0.8 and, one quiet tick later, are sure (0.98) they have finished;
21 of g26b4's 26 such stops were under 0.9 (2026-10-28). A threshold that depends on the state, not a timer; the
model's own carry-on is the trained fix (recipe g26b5, `carry_on_first`).
Each such stop is logged (`loop.js noteStop`): its P(listen) against the bar, or the words the repeat guard
blocked, then how many quiet ticks followed (with their P(listen)) and whether it went on or the caller spoke.

**Repeat guard** (`llama.js tick`, `evaluate.py --repeat-guard`): a reply word for word what was said on one of the
last two ticks, or a 2+ word chunk repeated within 20 ticks, becomes `<listen>`. A lone back-channel after a pause
passes. Loops fell from 21% of utterances to under 1%.

**Back-channel cap** (same places): a lone back-channel ("mm-hm", "oh wow", "really?") while the caller is talking,
within ~10 s (17 ticks) of the previous one, becomes `<listen>`. Training weights could not hold the rate (0.2, 4.5
and 1.4 a minute across runs 9, 10 and 10-r64; a person 0.8); on run 10's recorded-call replays the cap brings 4.6 a
minute to 1.8. A short "Yeah." after the caller stops is a reply and is never capped.

## Final punctuation

Nemotron punctuates a word only once the next word gives it context, so the last word of an utterance never gets its
"?" or "." on its own. `stt.js` closes a quiet utterance after 450 ms of mic silence with no new words: "?" if its
sentence opens like a question, "." otherwise, nothing after a word that leaves the thought hanging ("but", "and",
"um"). The model still decides whether to speak. The transcript pane joins that late mark onto the user's last line.
`scripts/check_final_punct.js` checks the rule; the `finalPunct` STT setting turns it off.

## Timing budget (measured, RTX 4090, 8 MoE layers' experts on CPU)

| Step | Median |
|---|---|
| llama-server tick (cached prompt, 4-6 new tokens) | ~130 ms (8 MoE layers on CPU) / ~300 ms (24 on CPU, 11 GB VRAM) |
| first tick of a call (prompt not cached) | ~2 s |
| Kokoro synthesis of a phrase | ~300 ms (Piper ~60 ms) |
| STT decode per 560 ms chunk (Nemotron, 4 threads) | ~150 ms, in a worker |
| reply after the caller stops (live, partner adapters) | 0.3-1.1 s |
| stop after a cut-in starts (live) | 1.3-2.0 s |

A 600 ms tick leaves headroom. The scheduler is `next = max(now, next + tick)`: if a tick runs long, the next one
starts immediately but never piles up. The cut-in stop has a floor of about 1.3 s in this design: the recogniser's
560 ms chunk, up to one tick, and the TTS chunk already queued. Faster needs the 160 ms recogniser, a shorter tick,
or stopping playback on the recogniser's partial before the model confirms.

## Audio handling decisions

- **Electron and native buffers.** sherpa-onnx returns audio in externally allocated buffers, which Electron refuses to pass between threads. The TTS worker generates with `enableExternalBuffer: false` and posts a plain copy.
- **Echo.** The model is never told what audio is playing, so on speakers it must not hear its own voice at all.
  Chromium's echo canceller does not see audio played through Web Audio (only `<audio>` elements, unless Chromium's
  "Chrome-wide echo cancellation" feature is on in the build), so talkme runs its own: the mic worklet takes the mic
  and the voice going to the speaker (after ducking) in the same render quantum, both go to the main process every
  100 ms, and `aec.js` passes them to `aec/aec_server.py` (WebRTC AEC3, then noise suppression and gain control, in
  that order), whose output is what the recogniser hears. The renderer then opens the mic without Chromium's
  processing. AEC3 estimates the speaker-to-mic delay itself; the renderer's output latency is a hint. Frames are
  answered in order; without the helper they pass through untouched. Behind it, the loop's own-words filter drops
  a stretch of heard words that repeats the assistant's last words in order and has 7+ letters, while its voice plays
  and 1.5 s after; short common words always pass. The echo test (Call tab) measures
  what still gets through.
- **TTS chunking.** Three words per tick synthesised one by one sounds choppy. `tts.js` buffers `said` text and
  flushes when a chunk ends in punctuation, when `ttsFlushWords` (default 6) words are buffered, or when the model
  goes to `<listen>`; the opening words of an utterance are spoken at once. The voice still cannot see where a
  sentence is heading, which is part of why it sounds flat; voices that take streaming text in (Kyutai, CosyVoice)
  were tried in `tts_bench/`.
- **The voice anchor.** The model writes a tick's words at its training pace (about 2.9 words a 0.6 s tick); Orpheus
  says about 1.6. Without an anchor a long reply ran up to 15 s ahead of what the caller had heard: the model took its
  own still-playing speech for silence and spoke up over it ("Anyway,"), and the caller's questions landed in its
  history far behind where the voice was. So while the voice has more than 1.2 s of speech queued (`tts.js`
  `backlogMs`: queued audio plus words not yet synthesised at the voice's measured pace), the next tick waits; it runs
  at once when the recogniser has words from the caller. Each tick keeps the words the model wrote, as in training; a
  tick just lasts as long as the voice takes. The voice stays at most ~2.4 s behind. Thinning the words out instead
  (each tick's history holding only what the voice said in it) was tried first and failed in a real call: one word a
  tick, never seen in training, made it hesitate and restart. `anchorToVoice: false` turns it off;
  `scripts/check_anchor.js` checks it with a stand-in model and voice.
- **Barge-in.** If the model answers `<listen>` on a tick where the user was heard and audio is still playing, that
  is the trained `barge_stop` behaviour: the playback queue is cleared and the current buffer is stopped. If the
  user was silent (`<no voice>`) and the model says `<listen>`, it just finished its answer and playback drains.
- **Context.** ~17 tokens per tick means ~1,700 tokens per minute. With `-c 32768` a call runs about 18 minutes
  before the loop drops the oldest 40% of ticks (one cache miss, ~2 s) and continues; Gemma 26B is capped at 400
  ticks (its tick time grows with the call). The cut lands where the caller starts talking into a silence
  (`loop.js` `trimPoint`), so the history the model sees still opens like a call: a cut at an arbitrary tick left
  it starting mid-reply, and after two such cuts in a 7-minute call it spoke in fragments and stalled mid-sentence
  (2026-10-27). The replays trim the same way. Nothing is kept between calls.

## Testing

Four views of every run, from most to least like real use:

1. **Your own ears.** `tests/manual_script.md`: 14 lines to say, one per behaviour, with what good and bad look like.
2. **Live audio** (`scripts/audio_test.js`, `tests/scenarios.json`, `scripts/live_bench.sh`): a full round trip in
   real time with no microphone or speakers. A synthetic caller (Piper Amy, or Kokoro voices George and Emma at other
   speeds) speaks the user's lines; 100 ms frames go into the real recogniser; the real loop runs against
   llama-server; the assistant's audio plays on a simulated speaker queue, so cut-ins really cut it; every assistant
   chunk is transcribed back and checked. 28 scenarios: reply latency, silence through mid-thought pauses, cut-in stop
   time, keeping going through "mm-hm", keywords, and checks a keyword list scores too strictly, which the base model
   judges as a yes/no question with every adapter off (`judge`): warnings, warmth, not inventing memories or
   addresses, answering then stopping (`maxWords`). The `friend-call` scenario is the manual script as one call.
   `live_bench_report.js` puts runs side by side (`tests/out/live_bench.md`).
3. **Recorded real conversations** (`tune/data/heldout_yap.jsonl`, 24 calls / 72 minutes from Open Yap; `evaluate.py`
   replays the other person's words, `yap_metrics.py` scores where the adapter spoke against where the real person
   did): turns taken where the person took them, back-channels a minute (a person: 0.8), talking over the caller,
   quiet when the person gave way to a cut-in.
4. **Scripted held-out calls** (`tune/data/heldout_ours.jsonl`): warnings caught, cut-in stops, answer starts, fillers,
   and the 12-question answer quiz (`probe_answers.py`): is the knowledge still there.

## Tuning pipeline (`tune/`)

A run is a recipe, `tune/recipes/<name>.json`, run by `tune/pipeline.py` from the command line or the app's Tune
tab (`src/main/tune.js` spawns it and streams the output). Every stage skips what is already done, so the same
command resumes a stopped run; a stopped training continues from its last saved step. The recipes so far were made
with the hand-written `pipeline_r*.sh` scripts (now in `tune/archive/`); `recipes/g26b4.json` is v1's recipe.

1. **Dialogues** (`gen_dialogues.py`): the base model, served without any adapter, writes them. `short` / `long` /
   `special` are the assistant kinds (questions, statements, stories told in parts, a dangerous plan with
   `cut_after` and `interjection`, an answer the user cuts into and the assistant resumes, a request declined out
   loud). `partner` and `partner2` are the friend kinds: nudges after a silence, long told stories with the user's
   sounds over them, casual replies that ask back, and no claims about the user's life or plans together. `partner2`
   writes one special turn per dialogue separately and splices it in (`--focus hazard|resume|unknown`), because the
   writer left it out when asked for it inside the whole dialogue; `--goodbye` ends a share of them with one. Every
   output is checked (fields present, lengths, no assistant-speak, no invented shared life or made-up addresses).
2. **Calls** (`build_calls.py`): the timing script. Pace 2.2-3.2 words/s, recogniser delay, late punctuation, split
   words, the caller's own fillers and restarts (a **speaker profile** per call), back-channels every few seconds of
   user speech, warnings over the user, cut-ins and stops, nudges after 2.6-5.5 s of silence (real people: median
   3.1 s), three words per assistant tick, and the per-tick `tag` and loss weight `w`. `--timing nemotron|mixed`
   delivers words in the recogniser's bursts. `mix_calls.py` combines call files (`file:N` takes N) and splits long
   calls at quiet ticks so every tick fits the trainer's 8,000-token window.
   **Real conversations**: `scripts/oto_transcribe.js` transcribes otoSpeech's 538 two-channel calls (141 h);
   `oto_to_calls.js` turns them into `timing_only` calls, trained on the first token of each reply only (when to
   talk, not what a stranger said next). The recipe's `weights` set the loss weight per tag; the values double as tag
   ids for `train_lora.py --weight-map`.
3. **Train** (`train_lora.py`): LoRA on the top layers only. The lower 36 layers are frozen, so their output for every
   call is computed once and cached to disk by streaming the bf16 checkpoint through the GPU one layer at a time
   (~2 h for 1,900 calls); training then touches the last layers (four by default), the final norm and `lm_head` (~9 s a step of 4 calls on Qwen,
   600 steps). Loss is weighted cross-entropy on assistant tokens. `--system partner|assistant` is the prompt every
   call is trained with; `--resume` continues; `--rank` / `--alpha` set the size. The HauhauCS full-precision
   checkpoint was rebuilt from its Q8 GGUF (`hauhau_rebuild.py`, verified tensor by tensor).
4. **Export** (`export_gguf.py`): the PEFT adapter to a llama.cpp LoRA GGUF, copied into `models/lora` under the name
   the app reads (`adapter_meta.py` stores the base, prompt and listen threshold in the file).
5. **Tests**: the quiz, the recorded conversations and the held-out set (`evaluate.py --repeat-guard --system ...` at
   the recipe's listen threshold), then the live benchmark; the reports compare against the runs the recipe names.

## Why not a purpose-built duplex model

Moshi and similar open full-duplex speech models are 7B and score ~30 on VoiceBench; a plain STT → 7B text LLM
pipeline scores ~70. The micro-turn approach keeps the text model's intelligence and teaches it only the timing.
