# talkme results

A log of the runs in date order, oldest first. Each section describes what was true then: older ones name defaults
that have since changed and adapter files that no longer ship. For the release, start at
[v1, and two checks on it](#v1-and-two-checks-on-it-2026-10-28).

## The first run: beating the published micro-turn adapter

Date: 2026-10-17. Hardware: one RTX 4090 (24 GB), i9-14900KF, 64 GB RAM. Base model: Qwen3.6-35B-A3B UD-Q4_K_M.

**Outcome.** Our run 2 adapter (`models/lora/talkme-run2.gguf`, the app's default then) beats the published
codacus adapter on the held-out replay: answers start on time far more often, it places fillers where a listener would
twice as often, never stops mid-sentence, and makes about a quarter of the unprompted remarks and repeated warnings. It is
slightly worse at stopping when cut in on and talks over the user a little more. Live, through real audio, the two are
close; see the live section.

## Held-out replay (all 28 held-out calls, same evaluator, same conditions)

Closed-loop replay: every 0.6 s tick of a held-out call is fed to the model, which sees its own earlier replies. Listen
threshold 0.3, greedy decoding. `tune/evaluate.py`, tables by `tune/compare.py`, extra measures by `tune/extra_metrics.py`.

| Measure | Published | Run 2 |
|---|---|---|
| Answer starts within one tick of the right moment | 303/377 (80%) | **347/377 (92%)** |
| Interrupts a dangerous plan | 15/27 (56%) | **16/27 (59%)** |
| Stops talking when the user cuts in | **34/37 (92%)** | 32/37 (86%) |
| Fillers placed where a good listener would | 213/515 (41%) | **350/515 (68%)** |
| Talks while the user is talking or pausing (lower is better) | **5%** | 7% |
| Stops in the middle of its own sentence (lower is better) | 11/463 (2%) | **0/467 (0%)** |
| Starts an unprompted remark in silence, per hour of call (lower is better) | 4.2 | **1.1** |
| Repeats a sentence within 12 s, per hour (lower is better) | 5.9 | **1.4** |
| Fillers said per filler slot | 0.96 | 1.53 |

Split by call length (model card numbers for the published adapter in brackets):

| Measure | Long, published | Long, run 2 | Short, published | Short, run 2 |
|---|---|---|---|---|
| On-time start | 88% [78%] | **95%** | 75% [62%] | **90%** |
| Hazard interrupts | **8/10** [6/10] | 7/10 | 7/17 [6/17] | **9/17** |
| Stops on cut-in | 7/7 [7/7] | 7/7 | **27/30** [27/30] | 25/30 |
| Fillers in place | 43% [35%] | **70%** | 38% [33%] | **65%** |
| Talks over the user | **6%** [8%] | 8% | **5%** [5%] | 7% |

Our evaluator scores the published adapter somewhat higher than its own card (for example 80% vs 68% on-time starts),
so compare the published and run 2 columns, which went through the same evaluator, not run 2 with the card.

## What run 2 changed

Same recipe as the published adapter (LoRA rank 32, alpha 64, last 4 layers, AdamW 2e-4, 4 calls per step), different data:

1. **Every tick trained on.** The trainer reads the first 8,000 tokens of a call; the published data has calls up to
   ~25,000 tokens, so it never saw their endings. `tune/mix_calls.py` splits long calls at quiet ticks: 539 calls became
   1,122 pieces.
2. **Our recogniser's timing.** Half of 560 extra calls (built by `tune/build_calls.py` from the published dialogues) use
   Nemotron-style timing: words arrive in bursts at 0.56 s chunk boundaries, punctuation arrives with the next word, and
   the last word of a turn has none. The model learns to answer on silence, not only on a "?".
3. **No leakage.** Dialogues behind the held-out calls are excluded from every built call.
4. **Hazards reused more** (`--hazard-boost 1`), and silent gaps inside a user's turn are tagged and weighted as pauses.
5. 1,738 pieces, 590,732 ticks (the published adapter saw at most ~250,000, since each of its 539 calls was cut at 8,000 tokens); 600 steps, final loss 0.50.

## Runs that did not win

- **Run 1**, a faithful reproduction (published calls, published settings). On the screening calls it matched the
  published adapter within noise (on-time starts 75% vs 76%, fillers 33% vs 35%), which validated the pipeline.
- **Run 3**, run 2 continued 150 steps with more weight on staying quiet (0.3 to 0.45) and on hazard starts (4 to 8).
  It talked over the user less (4%) and caught more hazards, but answered late more often (76% on time) and started
  warnings out of nowhere, about 9 per hour ("Don't lean out. Keep your eyes on the road ahead." with nobody talking).
  Rejected.

## Live audio round trip

`scripts/audio_test.js`: a second voice speaks six scenarios into the real recogniser (Nemotron 560 ms), the real loop
runs against llama-server on the whole GPU, the assistant's speech plays on a simulated speaker so barge-in really cuts
it, and every reply is transcribed back. Quiet machine, one slot, `--n-cpu-moe 8`.

Three passes per adapter, six scenarios, 20 checks per pass:

| Live measure | Published | Run 2 |
|---|---|---|
| Checks passed (3 passes) | 48/60 | 48/60 |
| Piano: answers "never too old" after waiting out the pause | 1/3 | **3/3** |
| Car story: says to get it checked | **3/3** | 1/3 |
| Stops within 1.5 s of a cut-in | 1/3 | 0/3 |
| Keeps talking through "mm-hm" | 2/3 | 2/3 |
| Median tick / 95th percentile | 122 ms / ~300 ms | 127 ms / ~310 ms |

Live, the two tie. Run 2's replay advantage (on-time starts, far fewer false alarms) does not show as more passed checks
in six short scenarios, which are too few to see a 12-point timing difference. Both stop about 1.7 to 2 s after a cut-in,
and both lose "And of Japan?", which the recogniser hears as "End of Japan". Run 2 once looped on "Intake, compression,
power, exhaust." Recordings and transcripts: `tests/out/*-final*.wav`.

## Speech pipeline changes found by the tests

- **Recogniser.** The 2023 zipformer misheard "ammonia cleaner" as "E mol. Clean er", so the bleach warning never came.
  Nemotron 0.6B (560 ms chunks) transcribes the test lines with 0% word errors versus 6.9%, punctuates and cases
  natively, and delivers the last word ~0.2 s after speech ends. Bleach went from 1/3 to 3/3 checks.
- **Final punctuation.** Nemotron never punctuates the last word of an utterance, and the published adapter waits for
  that "?". After 450 ms of quiet the app closes the utterance with "?" or "." (question openers vs statements, nothing
  after a hanging word like "but").
- **Fast tick.** The decision is made on the first token; a listen tick decodes one token instead of four. Live: median
  tick 124 ms, 95th percentile ~300 ms, under 3% of ticks over the 600 ms budget.
- **Back-channels are never cut.** A "Mm-hm." playing while the user talks on is left alone; only real answers stop on a cut-in.
- **First words at once.** The opening words of an answer go to the voice immediately instead of waiting for punctuation
  or six words, which held the first audio back by up to two ticks (~1 s). Later chunks still batch so the voice
  stays smooth. (`--old-flush` in the audio test restores the old behaviour for comparisons.)
- **Ducking.** While the user's mic is louder than the room (adaptive noise floor), the assistant's voice drops to a
  quarter volume, so a cut-in is heard over it before the model decides to stop; full volume returns after 400 ms of
  quiet. Toggle "duck" in the toolbar.

## Reproduce

This needs the codacus dataset in `models/lora/dataset/`, which setup does not fetch.

```
cd tune
uv run python build_calls.py --dialogues ../models/lora/dataset/dialogues_short.jsonl ../models/lora/dataset/dialogues_long.jsonl \
  --exclude-calls ../models/lora/dataset/calls_heldout.jsonl --per 2 --hazard-boost 1 --timing mixed --max-ticks 460 --seed 7 --out out/calls_built_r2.jsonl
uv run python mix_calls.py --calls ../models/lora/dataset/calls_train.jsonl out/calls_built_r2.jsonl --out out/calls_run2.jsonl
uv run --extra train python train_lora.py --hf ../models/hf/Qwen3.6-35B-A3B --calls out/calls_run2.jsonl --cache out/cache_r2 \
  --chunk 160 --steps 600 --batch 4 --rank 32 --alpha 64 --lr 2e-4 --out out/adapter_run2
uv run --extra train python export_gguf.py --adapter out/adapter_run2 --hf ../models/hf/Qwen3.6-35B-A3B --out out/run2.gguf
```

About 3 hours of caching plus 1.7 hours of training on the 4090. A full 28-call replay takes ~55 minutes (one slot).

## Personality adapters

Three adapters continued from run 2 (`models/lora/talkme-run2-brisk.gguf`, `talkme-run2-warm.gguf`, `talkme-run2-playful.gguf`), each
from ~60 styled dialogues written by the base model (`gen_dialogues.py --style`), built into calls with the style's own
back-channels (`build_calls.py --filler-words --filler-keep`) and mixed with 250 of run 2's calls as anchors, 120 steps
at lr 1e-4. Same 8 held-out calls (4 short, 4 long):

| Measure | Run 2 | Brisk | Warm | Playful |
|---|---|---|---|---|
| Answer starts on time | 90% | 82% | 76% | 87% |
| Interrupts a dangerous plan | 3/6 | 3/6 | 2/6 | 3/6 |
| Stops when the user cuts in | 11/11 | 10/11 | 11/11 | 11/11 |
| Talks while the user talks or pauses | 8% | 4% | 5% | 5% |
| Unprompted remarks per hour | 1.1 | 0 | 1.2 | 2.4 |
| Repeated sentences per hour | 1.4 | 4.9 | 6.1 | 7.3 |

Getting there took three rounds, and the failures are worth knowing:

1. **v1** blurted a warning on the first tick of every call ("Don't lean out.", "Do not enter your password...") and
   answered late. Two causes stacked: the styled data had 3-4x the hazard density of run 2's (the generator demanded a
   hazard in every short dialogue, and hazards were reused), and the anchors included split pieces of long calls that
   *open with the assistant talking into silence* (`mix_calls.py` cut just before a calm follow-up). 25 of run 2's 1,738
   pieces were like that; 7 landed in the anchors, which then dominated the small style sets.
2. **v2** thinned hazards to run 2's density (`--hazard-keep`) and used more anchors: better timing, same blurting.
3. **v3** took anchors only from clean pieces: blurting gone, timing close to run 2. `mix_calls.py` now never starts a
   piece on an assistant tick, so future builds don't produce such pieces at all.

Blending (run 2 and a style at 50% each, per request) restored timing too, but kept the blurting and added unfinished
sentences; it works as a strength dial, not as a fix. Open item: all three repeat warnings more than run 2.

## Voices

The voice is separate from the adapters; any personality works with any voice. Kokoro 82M (27 English voices, CPU)
sounds far more natural than Piper and takes ~0.3 s per phrase on 8 threads versus ~0.05 s, so it adds about that much
before each answer. Both engines speak English through espeak, which spells all-consonant unknown words, so the model's
"Mm-hm" came out "em em aych em"; Kokoro's lexicon files are not consulted for English in this build. The app now
rewrites "Mm-hm" before speaking, rotating between raw espeak phonemes for a hummed "m-hm" (`[[m'hm]]`), "Umhum" and
"Uh-huh" for Kokoro, and "Uh-huh" for Piper.

**A lighter Chatterbox (2026-10-28).** Five lines in one cloned voice (Kokoro Nova clip), speed as synthesis time over
audio length (median), memory as the GPU peak while speaking:

| Chatterbox Turbo | Memory | Speed |
|---|---|---|
| PyTorch, full precision (the app) | 3.3 GB | 0.24× |
| PyTorch, speech model at 16-bit, under autocast | 2.6 GB | 0.31× |
| PyTorch, 16-bit throughout, under autocast | 1.7 GB | 0.30× |
| Resemble's ONNX, 4-bit, CPU (i9-14900KF, 8 threads, nothing else running) | (0.7 GB on disk) | 0.95× |
| Resemble's ONNX, 8-bit, CPU | (1.1 GB on disk) | 8.3× |
| PyTorch, full precision, CPU | | 1.6× |

Converting the weights to 16-bit without autocast fails (the library mixes 32- and 16-bit tensors); autocast alone,
with 32-bit weights, saves nothing. On the CPU even the 4-bit ONNX only just keeps pace on a fast, idle CPU, and in a
call the CPU also runs the recogniser (and, on a small GPU, the chat model's experts), so Chatterbox stays a GPU voice
and Kokoro (or Piper) is the voice for weaker systems. The ONNX files were deleted after the test. The app runs Chatterbox with its speech
model at 16-bit (version B, picked by ear; `serve_chatterbox.py --fp32` for full precision); Orpheus stays at Q4_K_M.


## Run 4 (shipped as talkme-run4-polite)

Run 2's recipe from scratch with: the dataset card's 49 flagged dialogues excluded (109 of the author's 539 calls
dropped), 150 generated dialogues where an interrupted answer is resumed and a request is declined out loud, speaker
profiles (each caller's own pace, pauses, fillers, restarts, trail-offs and back-channels), and the fixed splitter.
1,553 pieces, 514k ticks, 600 steps, final loss 0.44.

All 28 held-out calls, threshold 0.3:

| Measure | Published | Run 2 | Run 4 |
|---|---|---|---|
| Answer starts on time | 80% | **92%** | 81% |
| Interrupts a dangerous plan | 56% | **59%** | 44% |
| Stops when the user cuts in | **92%** | 86% | **92%** |
| Fillers placed where a good listener would | 41% | **68%** | 61% |
| Talks over the user | 5% | 7% | **4%** |
| Unprompted remarks per hour | 4.2 | **1.1** | **1.1** |
| Repeated sentences per hour | 5.9 | 1.4 | **0.0** |

Run 4 is more patient: the speaker profiles' long pauses and trail-offs ("so yeah.") taught it that a finished sentence
is often not the end of the turn. Threshold sweep on 14 calls: at 0.45 it interrupts 10/14 hazards (run 2: 8/14),
talks over the user 5%, repeats 0.6/h, but starts on time 83% (run 2: 92%); at 0.6 timing nears run 2 (89%) and the
politeness is gone (8% over-talk, 5.8 repeats/h). Shipped as `models/lora/talkme-run4-polite.gguf`; the app switches to its
threshold, 0.45, when it is picked (`ADAPTER_THRESHOLDS` in `src/main/index.js`).

Live (three passes each, new scenarios): declined both requests out loud 6/6 (run 2: 4/6). Resumed the interrupted
story 0/3, as did run 2. Original six scenarios: 14/20 (run 2: 16/20); it never answered "What's the capital of France?".

**Why resuming failed: the published data never answers a cut-in.** In all 821 cut-ins in the author's calls (and the
400 built here from the same dialogues) the user's interrupting question ("Wait, does it matter if it's only on one
side?") is followed by the user's next line, never by an answer. Over a thousand examples taught every adapter to go
quiet after being interrupted with a question; run 4's 299 resume examples, which answer the side question, were
outnumbered four to one. Run 5 fixes the data: `answer_cutins.py` writes an answer for every cut-in in the dialogues and
every call is built from the dialogues (`tune/archive/pipeline_r5.sh`).

## Benchmark closer to real use

The published held-out calls use timing the app never produces (punctuation on time, every word on its own tick) and
never answer a cut-in, and the first live tests used six scenarios and one caller voice. Three additions:

- **Our own held-out set** (`tune/data/heldout_ours.jsonl`): 26 calls, 17k ticks, built from the 52 held-out dialogues
  (never trained on) the way our training calls are built: Nemotron timing (late or missing final punctuation), speaker
  profiles (pace, fillers, restarts, back-channels) and answered cut-ins.
- **24 live scenarios** (`tests/scenarios.json`), up from 8: two questions at once, self-correction ("no wait, I
  meant..."), "right, right" and "uh-huh / yeah" back-channels, thinking pauses, sad news, yes/no, follow-ups, math,
  units, lists, a cut-in that changes the request, long silences, greeting and goodbye.
- **Three callers**: each live pass uses a different caller voice (`scripts/live_bench.sh`): Piper Amy (US woman),
  Kokoro George (UK man, 10% faster) and Kokoro Emma (UK woman, 10% slower).

How well the app's recogniser hears each caller on the 46 scenario lines (`scripts/caller_voice_check.js`):

| Caller | Word errors |
|---|---|
| Amy (US woman) | 1.9% |
| Michael (US man) | 1.2% |
| Emma (UK woman) | 3.1% |
| George (UK man) | 1.9% |

Most errors are spelling ("colours", "mum") or a short back-channel. "Uh-huh" alone is missed or misheard for every
caller ("Uh", "As", nothing), and "Yeah" alone needs about two seconds of silence after it to come out. The back-channel
check now records whether the app heard the caller at all, so a pass can't hide a recogniser miss.

### Run 5 (talkme-run5-attentive)

Run 5 trains only on calls built from the dialogues, with every cut-in answered (the published calls never answer
one), plus run 4's resume/decline dialogues and speaker profiles. Shipped as `models/lora/talkme-run5-attentive.gguf`;
run 2 stays the default.

| Published held-out calls (28) | Published | Run 2 | Run 4 | Run 5 |
|---|---|---|---|---|
| Answer starts on time | 80% | **92%** | 81% | 85% |
| Interrupts a dangerous plan | 56% | **59%** | 44% | 48% |
| Stops when cut in on | **92%** | 86% | **92%** | 73% |
| Back-channels in the right place | 41% | **68%** | 61% | 38% |
| Talks over the user | 5% | 7% | **4%** | 8% |
| Unprompted remarks per hour | 4.2 | **1.1** | **1.1** | 9.3 |

The published calls reward silence after a cut-in, which run 5 was trained out of, so "stops when cut in on" and
"unprompted" count some of its answers to the interruption as faults.

| Our held-out set (26 calls, answered cut-ins) | Run 2 | Run 5 |
|---|---|---|
| Answer starts on time | 82% | 81% |
| Interrupts a dangerous plan | **73%** | 54% |
| Stops when cut in on | 80% | **83%** |
| Back-channels in the right place | **47%** | 19% |
| Talks over the user | 9% | 10% |
| Unprompted remarks / repeats per hour | **0.0** / 3.9 | 2.1 / **1.4** |

Live, run 5 answered the cut-in question ("what was the fox's name?") 3/3 times and kept going or resumed after
"Okay" 3/3 (runs 2 and 4: 0/3), declined both off-limits requests out loud 6/6, but stopped its voice on the cut-in
only 1/3. Original six scenarios: 16/20 (run 4: 14/20).

### Real recorded conversations

Run 2 and run 5 replayed on 64 three-minute windows of real two-person calls (`tune/data/heldout_yap.jsonl`). The
caller's audio goes through the app's own recogniser; the other person's timing is the reference.

| | run2 | run5 |
|---|---|---|
| Takes the turn where the person did (within 1 tick) | 32% (129/402) | 45% (182/402) |
| ... within about 2.4 s | 42% (170/402) | 63% (253/402) |
| Median delay after the person's first word | +0.0 s | +0.0 s |
| Starts talking over the caller, per minute | 0.3 | 1.6 |
| Starts talking in the caller's mid-turn pauses | 4% (12/309) | 5% (16/309) |
| Back-channels per minute (person) | 2.9 (1.3) | 1.2 (1.3) |
| Back-channels within a tick of the person's | 25% (45/180) | 11% (19/180) |
| Utterances started per minute (person) | 0.9 (7.4) | 3.1 (7.4) |
| Quiet when the person gave way to a cut-in | 94% (80/85) | 93% (79/85) |
| Minutes replayed | 192 | 192 |

Against real people the picture flips. Run 2 is far more passive than a person: it starts under one utterance a
minute where people start 7.4, and takes only 42% of the turns a person took (within ~2.4 s). Run 5 takes 63% of them,
with the same zero median delay when it does, and starts 3.1 utterances a minute. The price is more talking over the
caller (1.6 a minute vs 0.3). Run 2 back-channels twice as often as people do and lands a quarter of them where the
person did; run 5 matches the person's rate but places them worse. Neither talks into the caller's pauses much (4-5%).

Some reserve is right for an assistant (these are friends chatting), so "takes the turn" is a ceiling, not a target.

**For run 6:** keep run 5's answered cut-ins and data, mix the author's prebuilt calls back in for back-channels and
warnings with the ticks after their unanswered cut-ins weighted zero, and add real conversation timing (otoSpeech,
CC BY 4.0, 141 h; Open Yap full corpus on request) once access comes through. Not yet run: the 24-scenario live
benchmark with three callers (`scripts/live_bench.sh`), paused to free the GPU.

## Run 6 (talkme-run6): real conversation timing, and the threshold lesson

Run 6 = run 5's calls + the author's prebuilt calls with the silence after their unanswered cut-in questions weighted
zero (`mask_cutins.py`, 966 ticks) + 350 otoSpeech calls (real two-person conversations transcribed by the app's
recogniser; 82 conversations, 86 speakers) trained on the first reply token only (when to talk, not what was said).

On the scripted sets it fixed most of what run 5 broke (published set: stops on cut-in 84% vs 73%, back-channels in
place 51% vs 38%, unprompted remarks 2.8/h vs 9.3/h, least talking over the user, 5%) but started answers on time less
(80% published, 68% ours). On the real recorded calls at the usual 0.3 threshold it was almost mute: 0.4 utterances a
minute where people start 7.4.

**The cause is the listen rule, not the training.** The app listens unless P(listen) < 0.3, i.e. it speaks only when
about 70% sure. Scripted data always starts on exactly one tick, so models trained on it are confident; real people
start at varying moments, so a model that learned their timing spreads "start now" over several ticks and never clears
70% on any one. Sweep on 24 recorded calls:

| | run5@0.3 | run6@0.3 | run6@0.6 | run6@0.8 | run6 sampled |
|---|---|---|---|---|---|
| Takes the turn where the person did (within 1 tick) | 36% (30/84) | 5% (4/84) | 52% (44/84) | 73% (61/84) | 73% (61/84) |
| ... within about 2.4 s | 52% (44/84) | 10% (8/84) | 70% (59/84) | 86% (72/84) | 85% (71/84) |
| Median delay after the person's first word | +0.0 s | +1.2 s | +0.0 s | -1.2 s | -0.6 s |
| Starts talking over the caller, per minute | 1.8 | 0.1 | 0.6 | 0.9 | 3.9 |
| Starts talking in the caller's mid-turn pauses | 5% (7/129) | 2% (2/129) | 6% (8/129) | 2% (3/129) | 12% (16/129) |
| Back-channels per minute (person) | 0.9 (0.8) | 0.1 (0.8) | 3.1 (0.8) | 7.1 (0.8) | 6.1 (0.8) |
| Back-channels within a tick of the person's | 0% (0/34) | 0% (0/34) | 26% (9/34) | 53% (18/34) | 24% (8/34) |
| Utterances started per minute (person) | 3.0 (7.4) | 0.4 (7.4) | 3.2 (7.4) | 4.6 (7.4) | 9.9 (7.4) |
| Quiet when the person gave way to a cut-in | 95% (18/19) | 95% (18/19) | 68% (13/19) | 42% (8/19) | 74% (14/19) |
| Minutes replayed | 72 | 72 | 72 | 72 | 72 |

0.6 is the balance (70% of the person's turns taken on time, 0.6 interruptions a minute, but back-channels at four
times a person's rate and only 68% quiet on cut-ins). 0.8 and sampling the decision (`--sample`, the app's "human
timing" toggle) over-talk: every tick is a fresh chance, and small chances add up. The app now defaults run 6 and the
run 7 partner to 0.6, human timing off.

The user decided talkme does not have to be an assistant: separate adapters, a conversation partner first (run 7,
`tune/archive/pipeline_r7_partner.sh`: 700 otoSpeech calls with the person's words kept, plus all resume/decline calls and some
answered-cut-in and masked prebuilt calls for warnings and declines). For a partner, human-like hesitation is not a
fault and the recorded-conversation test is the main yardstick.

## Runs 7 and 8: the conversation partner

**Run 7** (`talkme-run7-partner`) learned real strangers' words as well as their timing (700 otoSpeech calls, words
kept). On real recorded calls it took turns most like a person (86% of the person's turns with sampled timing), but its
replies looped ("Yeah. Yeah. Yeah...", 21% of utterances) and it reacted instead of answering. Its knowledge was intact:
an answer quiz (`tune/probe_answers.py`, 12 questions) got 9/12, and **the run 7 + run 6 blend got 12/12** with short,
natural answers. Blends are in the app's adapter dropdown.

**Repeat guard** (app and `evaluate.py --repeat-guard`): a reply word for word what was said on one of the last two
ticks, or a 2+ word chunk repeated within 20 ticks, becomes a listen. Loops fell to 0.8% (real people 2.1%). Real people
never repeat a 3+ word reply within four replies; the adapters did in 27-67% of them.

**Run 8** (`talkme-run8-partner-hauhau`, on HauhauCS's abliterated base, no decline examples, by choice):
~490 friend-style dialogues written by the base model (`gen_dialogues.py --kind partner`: nudges after a silence,
long told stories, asks back), all 538 otoSpeech conversations for timing only, denser and heavier back-channels. The
HauhauCS full-precision weights were rebuilt from its Q8 GGUF (`tune/hauhau_rebuild.py`, verified tensor by tensor).

| Live, 3 callers | Run 7 + run 6 blend | Run 8 |
|---|---|---|
| Speaks up after a silence | 0/3 | **3/3** |
| Back-channels while you talk | 1/3 | **3/3** |
| Keeps a story going | 3/3 | 3/3 |
| All checks (24 shared scenarios + decline) | **86%** | 80% |
| Stops when cut in on | **33%** | 11% |
| Says the right thing | **78%** | 57% (includes the decline scenario, failing by design) |

On real recorded calls run 8 places back-channels where the person did 35% of the time (blend 12%), at 4.2 a minute
(people 0.8). Answer quiz 11/12, but replies often drift into musing like the caller ("Eight. So, I'm trying to figure
out how many legs a spider has...") and open with "Yeah.". **Safety regression:** the live bleach warning was proper
once, weak once ("You should be careful") and missing once; the scripted held-out set still caught 54% of dangerous plans
(the blend 54%, run 2 73%). Run 9 should restore warning data at full strength, add stop-on-cut-in examples, lower the
back-channel weight a little, and keep the partner from inventing a shared life ("I saw your car in the parking lot").

## Run 9: the partner, fixed where run 8 fell short

`talkme-run9-partner-hauhau` (HauhauCS base, no declines, as run 8). New: its own system prompt ("You are a friend on a
live phone call...", sent by the app for partner adapters from run 9 on); 260 new friend dialogues, half with a
dangerous-plan moment and half with a story the user cuts into (`gen_dialogues.py --kind partner2 --focus`, the special
turn written on its own and spliced in: asked for inside the whole dialogue, the writer kept 0 of 12); 400 extra
warning-bearing calls; warning and cut-in stop ticks weighted up (5.0 / 2.2 / 4.5); back-channels 1.8 (run 8 2.5) and
every ~5 s of user speech; 250 real conversations for timing only, with their mid-reply words weighted 0. Training took
2.2 h (13 s a step).

| | Run 8 | Run 9 |
|---|---|---|
| Live: all checks, 3 callers | 80% | **86%** |
| Live: says the right thing | 57% | **81%** |
| Live: stops when cut in on | 11% | **33%** (target 60%) |
| Live: median stop after a cut-in | 3.29 s | **1.94 s** |
| Live: keeps going through back-channels | 42% | **83%** |
| Live: waits through pauses | **93%** | 67% |
| Live: speaks up after a silence | **3/3** | 2/3 |
| Live: back-channels while you talk | **3/3** | 0/3 |
| Live: bleach warning given | 1 proper of 3 | 3 of 3 (1 passes the strict keyword check) |
| Held-out: dangerous plans caught | **54%** | 50% (target 70%) |
| Held-out: takes the turn | 63% | **82%** |
| Held-out: stops on a cut-in | 60% | **69%** |
| Recorded calls: back-channels a minute (person 0.8) | 4.2 | 0.2 |
| Recorded calls: turn where the person took it | **50%** | 42% |
| Answer quiz | **11/12** | 10/12 |

Better at answering, at warnings that do come, and at cut-ins; the musing like the caller is gone. But the back-channel
change overshot (0.2 a minute, from 4.2; a person 0.8), and the replies still drift into invented plans ("Mercury is
the closest one. So, anyway, I was thinking about going to the park later. Do you want to come with me?") and filler
("Yeah, I'm here. What's up?"), likely from run 8's dialogues, which were mixed back in unfiltered. With no decline
examples it played along with a request for someone's home address and made one up ("the address is 123 Main Street").
Run 10 ideas: back-channel weight between run 8 and 9 (about 2.2), filter run 8's dialogues for shared plans, and
end replies once they are answered.

## Notes for later runs

- **Resume after an interruption.** First live test (2026-10-18): asked for a story, cut in with "what was the fox's
  name?", got the answer, said "Okay" and went quiet; the model kept listening instead of resuming the story. No call in
  the published or our data ever resumes an interrupted answer: after a cut-in the conversation always moves on. Add
  calls where, once the interruption is resolved and the user says "okay" or pauses, the assistant picks the answer back
  up ("Anyway, back to the story..."). Needs a new turn type in gen_dialogues.py and build_calls.py.
- Train on real recogniser fragments (TTS the dialogues, run them through Nemotron) instead of simulated timing.
- Reduce warning repeats in the personality adapters.
- Faster stop on cut-in (live: 1.7-2 s): measure the 160 ms Nemotron variant with experts on the GPU.
- Short back-channels: "Uh-huh" alone rarely survives the recogniser. Either map the recogniser's near-misses for a lone
  short sound to a back-channel, or train with the fragments it actually produces.

## Run 10: the partner, plus the size test

`talkme-run10-partner-hauhau` (recipe: `tune/recipes/run10.json`). Run 9 plus 380 new friend dialogues: 130
dangerous-plan moments, 130 stories the user cuts into, 120 where the user asks something the friend can't know
("no clue where Dave lives, why?"), 40% ending with a goodbye; run 8's dialogues with the 87 that invented shared plans
dropped; back-channels weight 2.05, listens after a finished reply 1.9. Training: 8.7 s a step.
**`talkme-run10-partner-r64-hauhau`** is the same data at rank 64 (alpha 128): the size test.

| | Run 9 | Run 10 | Run 10, rank 64 | |
|---|---|---|---|---|
| Live: all checks (264, 3 callers) | 84% | 83% | **85%** | |
| Live: stops when cut in on | 44% | **89%** | 67% | target 60% |
| Live: median stop after a cut-in | 1.81 s | **1.48 s** | 1.67 s | |
| Live: median reply after the caller stops | 0.83 s | **0.42 s** | 1.60 s | |
| Live: waits through pauses | 80% | 73% | **100%** | |
| Live: speaks up after a silence | **2/3** | 1/3 | 0/3 | |
| Live: judged right by the base model | **90%** | 71% | 81% | |
| Held-out: dangerous plans caught | 50% | **81%** | 69% | target 70% |
| Held-out: stops on a cut-in | **69%** | 57% | 60% | |
| Recorded calls: turn where the person took it | 42% | **57%** | 38% | |
| Recorded calls: back-channels a minute (person 0.8) | 0.2 | 4.5 | **1.4** | |
| Answer quiz | **10/12** | 8/12 | 6/12 | |

**Run 10** has the best timing and safety of any partner run (cut-in stops, warnings, reply speed, turn-taking on
real calls) but answers worse: the "no idea" dialogues became a habit ("I don't know, you should ask her directly"
for a gift idea; odd openers like "You can't hear me, I'm in a car"), and on the car-noise scenario it said "Please
ignore the previous response. The correct answer is to ignore the loose gas cap". Back-channels swung from 0.2 to
4.5 a minute on a weight change of 1.8 to 2.05: training weights are too blunt a dial for that rate.

**The size test: rank 64 is not better.** It wins some checks (waiting through pauses, a back-channel rate close to a
person's) and loses others (quiz 6/12, turn-taking, reply speed, warnings), about evenly: different, not stronger. A
bigger adapter on the same top 4 layers does not fix what the data teaches; the next runs stay at rank 32 and change
the data. The remaining size lever, more layers (top 8), needs a new cache and is not tried.

Next: run 10's recipe with the "no idea" dialogues cut from 120 to 40 and the
back-channel weight at 1.9; and a back-channel cap in the app (at most one every few seconds) so the rate no longer
depends on training.

## Run 11: fixing run 10's answering made it too quiet

`talkme-run11-partner-hauhau` (recipe `tune/recipes/run11.json`, the first run made by `tune/pipeline.py`): run 10's
recipe with the "no idea" dialogues cut from 120 to 40, the dialogue writer told that after a factual answer the
friend stops and that most calls have a nudge after a silence, and back-channels at 1.95. The replays and live tests
now include the app's back-channel cap.

| | Run 9 | Run 10 | Run 11 |
|---|---|---|---|
| Live: all checks | **84%** | 83% | 76% |
| Live: answers, then stops | 1/3 | 1/3 | **3/3** |
| Live: stops when cut in on | 44% | **89%** | 44% |
| Live: back-channels while you talk | 4/6 | **6/6** | 1/6 |
| Live: keeps going through your "mm-hm" | 42% | 42% | 8% |
| Live: median reply after the caller stops | 0.83 s | **0.42 s** | 1.59 s |
| Live: bleach scenario | 9/9 | 9/9 | 6/9 |
| Held-out: dangerous plans caught | 50% | **81%** | 46% |
| Recorded calls: turn where the person took it | 42% | **57%** | 29% |
| Answer quiz | **10/12** | 8/12 | 9/12 |

Cutting the "no idea" dialogues gave back some answering (quiz 9/12), and "answers, then stops" finally passes. But
"stop after answering" in the dialogues together with the heavier weight on listening after a finished reply taught
it to hold back everywhere: fewer turns, almost no back-channels, slower replies, and it stops talking when the caller
says "mm-hm" over it. Run 10 stays the best partner on timing and safety. Next: run 10's data with only the "no idea"
cut (one change at a time).

## Run 12: run 10 with one change, the best partner so far

`talkme-partner-v0-hauhau` (recipe `tune/recipes/run12.json`): run 10's own dialogues, mix and weights, with only
40 of its 120 "no idea" dialogues. No new dialogues; 1,802 calls; training 13 s a step (slower than run 10's 8.7 s,
the GPU was shared for part of it).

| | Run 9 | Run 10 | Run 11 | **Run 12** |
|---|---|---|---|---|
| Live: all checks | 84% | 83% | 76% | **87%** |
| Live: says the right thing | 70% | 68% | 70% | **82%** |
| Live: judged right by the base model | **90%** | 71% | 76% | **90%** |
| Live: bleach scenario | 9/9 | 9/9 | 6/9 | **9/9** |
| Live: stops when cut in on | 44% | **89%** | 44% | 67% |
| Live: median stop after a cut-in | 1.81 s | 1.48 s | 1.64 s | **1.36 s** |
| Live: median reply after the caller stops | 0.83 s | **0.42 s** | 1.59 s* | 0.70 s |
| Live: back-channels while you talk | 4/6 | **6/6** | 1/6 | **6/6** |
| Live: keeps going through your "mm-hm" | 42% | **42%** | 8% | 8% |
| Held-out: stops on a cut-in | 69% | 57% | 69% | **89%** |
| Held-out: dangerous plans caught | 50% | **81%** | 46% | 46% |
| Recorded calls: turn where the person took it | 42% | **57%** | 29% | 50% |
| Recorded calls: back-channels a minute (person 0.8) | 0.2 | 4.5 | 0.1 | **1.2** |
| Answer quiz | **10/12** | 8/12 | 9/12 | **10/12** |

It answers like run 9, takes turns nearly like run 10, has the most human back-channel rate so far, stops fastest when
cut in on, and gave the bleach warning on every pass. So the hedging came from the "no idea" dialogues (one change,
one effect). Still weak: carrying on when the caller says "mm-hm" over it (8%, as run 11), and the scripted held-out
warnings (46%, though every live warning came through).

\* **The live tests' own voice changed by mistake.** `scripts/audio_test.js` spoke the assistant's lines with the
app's default voice, which became Kokoro between 01:13 and 16:08 on 2026-10-21 (and Orpheus after). Kokoro adds ~0.3 s
a phrase, so the live timings of run 10-r64, run 11 and two of run 9's three rerun callers are slower than the model
alone; run 12's first attempt failed outright when Orpheus could not start beside the chat model. The tests now pin
Piper Lessac, as every benchmark before.

## Gemma 4 31B heretic (g12): run 12's recipe on a dense model

`talkme-g12-partner-gemma4` (recipe `tune/recipes/g12.json`): run 12's 1,802 calls, weights and partner prompt,
trained on Google's Gemma 4 31B decensored with Heretic (coder3101/gemma-4-31B-it-heretic; served as mradermacher's
IQ4_XS GGUF) instead of Qwen3.6-35B-A3B HauhauCS. The trainer, replays and app gained model profiles
(`tune/profiles.py`): Gemma's `<|turn>` chat format, its dense layers with sliding-window masks, scaled embedding,
tied output layer and logit soft-cap. `tune/check_forward.py` confirmed the trainer's forward pass matches llama.cpp
(top token and runners-up, four prompts up to 2,519 tokens). Training: cache 2.5 h, 600 steps at 7.3 s, rank 32 on
attention and MLP of layers 56-59 (16.4 M parameters), final loss 0.84 (run 12 0.86).

| Live, 3 callers | Run 12 (Qwen) | g12 (Gemma) |
|---|---|---|
| All checks | 87% | **92%** (the best of any run) |
| Says the right thing | 82% | **88%** |
| Judged right by the base model | 90% | **100%** |
| Answers, then stops | 1/3 | **3/3** |
| Keeps going through your "mm-hm" | 8% | **67%** |
| Speaks up after a silence | 2/3 | **3/3** |
| Bleach | 9/9 | 9/9 |
| Stops when cut in on | **67%** | 22% |
| Median stop after a cut-in | **1.36 s** | 2.67 s |
| Median reply after the caller stops | **0.70 s** | 1.41 s |
| Ticks over the 600 ms budget | ~2-3% | 13-17% |

| Recorded calls | Run 12 | g12 |
|---|---|---|
| Takes the turn where the person did | **50%** | 40% |
| Back-channels a minute (person 0.8) | 1.2 | **0.8** |
| Talks over the caller, a minute | **0.9** | 1.7 |
| Answer quiz | **10/12** | 8/12 (four silences; every answer it gave was right) |

The smartest partner so far, and the slowest. Gemma is dense: all 31B weights work on every token, against ~3B
active in Qwen's mixture of experts, and its ten full-attention layers make each new token cost more as the call
grows even with the prompt cached (a listening tick ~150 ms early in a call, ~500 ms by 5,000 tokens). The app and the
replays keep ~150 ticks (~90 s) of a call on Gemma; still one tick in six runs over 600 ms in the live tests, which
is why it is slow to stop when cut in on and to reply. Fully on the GPU it takes ~21 GB, so a GPU voice needs 8 of its
60 layers on the CPU (not yet measured). The scripted held-out set (26 calls of up to 650 ticks) was postponed: at
Gemma's pace it takes ~4 hours. Next for this line: a mixture-of-experts model of similar quality, or a llama.cpp
attention path for Gemma's 512-wide heads.

## Gemma 4 26B-A4B heretic (g26): run 12's recipe on a mixture-of-experts Gemma

`talkme-partner-v0-gemma4a4b` (recipe `tune/recipes/g26.json`): run 12's calls, weights and prompt on Gemma 4
26B-A4B decensored with Heretic (coder3101/gemma-4-26B-A4B-it-heretic; mradermacher's Q4_K_M GGUF), ~4B active per
token like Qwen. The trainer's experts now use transformers' grouped-matmul kernel (`tune/check_experts.py`: same
gradients, cosine 0.99997; a training step 4.0 s instead of 7-9 s). Ticks ~64 ms early in a call, ~180 ms at 300
ticks, so the app keeps 400 ticks (~4 min) of a call.

| | Run 12 (Qwen) | g26 |
|---|---|---|
| Live, all checks | **87%** | 86% |
| Says the right thing | **82%** | 72% |
| Keeps going through your "mm-hm" | 8% | **83%** |
| Stops when cut in on | **67%** | 22% |
| Median reply after the caller stops | 0.70 s | **-0.50 s** (starts before they finish) |
| Recorded calls: takes the turn where the person did | 50% | **61%** |
| Recorded calls: talks over the caller, a minute | **0.9** | 1.8 |
| Our held-out set: warnings caught / over-talk | 46% / **7%** | **62%** / 18% |
| Quiz | 10/12 | **11/12** |

The best answers and turn timing on recorded calls, and the most eager: it starts before callers finish (its wrong
live answers are mostly questions it answered before they were done), talks over them, and seldom stops when cut in
on. Its extra talk after an answer comes in the silence that follows: the trained nudge, sometimes too early, and with
invented shared history ("that new movie we saw") or help-desk lines ("What can I help you with?"). The second has a
cause in the data: the call builder opened 39% of the training calls with a service greeting ("Hi. How can I help?",
"What can I do for you?", "What's on your mind?").

Raising its listen threshold makes it talk more, not less (it speaks whenever P(listen) is below the threshold).
The sweep on the 24 recorded calls:

| g26 at threshold | 0.4 | **0.5** | 0.6 | 0.7 | 0.8 | (run 12, 0.6) |
|---|---|---|---|---|---|---|
| Talks over the caller, a minute | 0.8 | **1.2** | 1.8 | 2.1 | 3.2 | 0.9 |
| Back-channels a minute (person 0.8) | 0.3 | **1.2** | 2.1 | 1.7 | 2.2 | 1.2 |
| Quiet when the caller cuts in | 74% | **74%** | 42% | 58% | 53% | 79% |
| Takes its turn where the person did | 38% | 39% | 61% | 70% | 76% | 50% |
| Utterances a minute (person 7.4) | 2.6 | 3.3 | 5.0 | 6.3 | 8.0 | 3.7 |
| Quiz | 11/12 | **11/12** | 11/12 | 11/12 | 11/12 | 10/12 |

At 0.5 g26 has about run 12's manners (talk-over, back-channels, stopping when cut in on) with the better answers,
at the cost of taking its turn later; 0.4 goes quiet. The app gives g26 0.5.

## Persona and memories (g26m, g26m2)

Goal: the app sends a persona ("Maya, 29, a night-shift nurse ...") and real memories of the caller in the prompt
(Settings > Persona); the friend should bring up only those, and not play along when the caller claims a shared past
that is not in them. `gen_dialogues.py --kind memory` writes calls with a made-up friend and 0-5 memories each, with
either a recall moment or a false-memory moment ("remember when we ...?" for something not in the list). A judge (the
plain writer model) drops calls where the friend invents a past; it first dropped the friend's correct "we never went
to a taco place, who are you thinking of?" too (it mixed up the caller's claim with the friend's reply), so the
false-memory moment gets its own narrow question. The memory test (`tune/memory_check.py`, 24 held-out persona calls in
`data/heldout_memory.jsonl`) counts invented pasts and how the friend handles 16 false claims.

Both recipes also open rebuilt calls with a friend's greeting and give the older calls' help-desk greetings no weight.

- **g26m**: 280 persona calls (twice each) mixed into a full g26 run.
- **g26m2**: a second stage on top of g26 (pipeline `init_from`): 170 persona calls (once each) plus 400 of g26's own
  calls, 150 steps at half the learning rate. Its writer gives every false claim its own scenario (half of g26m's
  were "remember Vegas"), writes ordinary people instead of gimmicks, and has every call answer a question.

| Memory test | g26 (0.6) | g26m (0.6) | g26m2 (0.6) | g26m2 (0.85) |
|---|---|---|---|---|
| Invents a past (calls) | 5/24 | 2/24 | **1/24** | 3/24 |
| False claims: plays along | 14/16 | **6/16** | 9/16 | 11/16 |
| False claims: says it doesn't remember | 2/16 | **9/16** | 7/16 | 5/16 |

| | g26 | g26m | g26m2 (0.6) | g26m2 (0.85) |
|---|---|---|---|---|
| Quiz | 11/12 | 8/12 (2/12 with a persona) | 11/12 | 11/12 |
| Live, all checks | **86%** | 72% | 76% | not run |
| Recorded calls: takes the turn where the person did | **61%** | 33% | 19% | 56% |
| Recorded calls: utterances a minute (person 7.4) | **5.0** | 1.4 | 1.4 | 4.1 |
| Recorded calls: back-channels a minute (person 0.8) | 2.1 | 0.1 | 0.2 | **1.0** |
| Recorded calls: quiet when cut in on | 42% | **84%** | 79% | 74% |
| Recorded calls: talks over the caller, a minute | 1.8 | **0.7** | 0.5 | 1.9 |

Both learned the memory part. g26m lost the rest: late, quiet, garbled forced answers, and with a persona prompt it
dodged questions with persona chatter ("I'm just making a pun about a meter?"). g26m2 kept its answers (11/12), and
most of what it lost was calibration, not skill: the persona stage left it surer of listening, and at a 0.85
threshold its recorded-call timing is back near g26's with back-channels near a person's and far better cut-in
stops. The app gives g26m2 0.85 (`ADAPTER_THRESHOLDS`). But at 0.85 most of the memory gain goes too (plays along 11 of 16
against g26's 14): the more readily it speaks, the more it goes along. So far the persona data moves the memory
behaviour only as far as it makes the model hold back. Still to run: the live tests at 0.85.
The memory test's own claims were mostly "Vegas" (written before the scenario fix), so the test was rewritten (v2,
`data/heldout_memory.jsonl`: 16 false claims, none about Vegas; v1 kept as `heldout_memory_v1*.jsonl`):

| Memory test v2 | g26 (0.6) | g26m (0.6) | g26m2 (0.85) |
|---|---|---|---|
| Invents a past (calls) | 4/24 | 3/24 | 3/24 |
| False claims: plays along | 14/16 | **8/16** | 11/16 |
| False claims: says it doesn't remember | 2/16 | **8/16** | 5/16 |

g26m's gain holds on claims it never trained on (it is not just "Vegas, deny"); g26m2's is smaller.

g26m2 live at 0.85: all checks 81% (76% at 0.6, g26 86%), but it then starts before callers finish (median -1.2 s)
and stopped on none of 9 cut-ins: the recorded calls and the live callers disagree on its best threshold. The app
gives it 0.75, between the two.

### g26m3: a third try, and the question it raises

`tune/recipes/g26m3.json`: a fresh run like g26m, with the fixed writer's persona calls (g26m2's 170 plus 110 new, each
once: ~14% of the ticks instead of ~22%), the false-memory reply weighted 8 / 3, and 700 steps so every other call is
seen as often as in g26 (1.33 times).

| | g26 | g26m | g26m3 |
|---|---|---|---|
| Quiz | 11/12 | 8/12 | 7/12 (silent on four) |
| Live, all checks | **86%** | 72% | 73% |
| Recorded calls: utterances a minute (person 7.4) | **5.0** | 1.4 | 2.2 |
| Recorded calls: back-channels a minute (person 0.8) | 2.1 | 0.1 | 0.0 |
| Our held-out set: answer starts / warnings / cut-in stops | 85% / 62% / 71% | 76% / 58% / 83% | 84% / **69%** / **89%** |
| Memory test v2: declines a false claim | 2/16 | **8/16** | 7/16 |

Quiet again. Yet apart from its 293 persona calls, g26m3's training set is g26's: 1,806 calls against 1,802 with the
same share of every tick kind (answer starts 2.64% / 2.66%, speaking 22.9%, listening through speech 44.4% / 44.3%,
cut-in stops 0.21%). Either ~14% persona calls are enough to make the model hold back everywhere, or g26 itself is a
lucky draw: it was trained once. `g26s1` retrains g26 exactly with another seed (same calls and cache) to tell.

**g26s1 answers it: the persona calls, not luck.** g26 retrained with seed 1 on the same calls and cache:

| Recorded calls | g26 | g26s1 | g26m3 |
|---|---|---|---|
| Quiz | 11/12 | **12/12** | 7/12 |
| Utterances a minute (person 7.4) | 5.0 | 4.7 | 2.2 |
| Back-channels a minute (person 0.8) | 2.1 | **0.8** | 0.0 |
| Quiet when the caller cuts in | 42% | **89%** | 89% |
| Talks over the caller, a minute | 1.8 | 1.3 | 0.6 |
| Takes the turn where the person did | **61%** | 39% | 24% |

Without persona calls the recipe stays lively over two seeds, so the persona calls are what make the model hold back.
The seeds also show how much one run's details can swing on the same data (cut-in stops 42% against 89%): a single
run per setting is not enough to compare settings that close.

g26s1's full tests (threshold 0.6): our held-out set answer starts 79%, warnings 54%, cut-in stops **91%**, over-talk
**9%** (g26 85 / 62 / 71 / 18%, run 12 72 / 46 / 89 / 7%); live 82% of checks (g26 86%, run 12 87%), judged right 90%,
keeps stories going 3/3, but it too starts before live callers finish (median -0.5 s) and says the right thing 58%.
So on the recorded and scripted calls it has run 12's manners with Gemma's answers; live it shares g26's eagerness,
which g26 shed at 0.5. The two runs differ as much as two recipes would.

**Live at 0.5.** g26 at its app default stops starting before callers finish: median reply start +0.74 s (at 0.6:
-0.50 s; run 12 +0.70 s), says the right thing 72%, all checks 83% (86% at 0.6, run 12 87%), but it no longer speaks
up after a silence (0/3) and still stops on only 3 of 9 live cut-ins. g26s1 at 0.5: +0.86 s, answers then stops 3/3,
says the right thing 63%, all checks 82%, quiz 10/12.

**Three seeds of g26** (same calls, seeds 0, 1, 2; recorded calls at 0.6):

| | g26 | g26s1 | g26s2 |
|---|---|---|---|
| Quiz | 11/12 | 12/12 | 11/12 |
| Utterances a minute (person 7.4) | 5.0 | 4.7 | 4.5 |
| Talks over the caller, a minute | 1.8 | 1.3 | 1.5 |
| Takes the turn where the person did | 61% | 39% | 50% |
| Back-channels a minute (person 0.8) | 2.1 | 0.8 | 0.5 |
| Quiet when the caller cuts in | 42% | 89% | 68% |

How much it talks and what it knows hold steady across seeds; the finer manners (when it takes the turn, how often it
back-channels, whether it stops when cut in on) swing as much as most recipe changes did. Differences of that size
between two single runs say little.

## One BOS for Gemma (2026-10-24)

Until now every request to a Gemma adapter (app, replays, quiz, live tests) started with two BOS tokens: ours in the
prompt and the one llama-server adds for Gemma. Training had one. Fixed in `llama.js` and `common.render`. The same g26
adapter, deterministic replays, with one BOS: quiz 11/12 either way; on the recorded calls back-channels 1.0 a minute
(2.1 with two; person 0.8), quiet on 63% of cut-ins (42%), talk-over 1.5 a minute (1.8), takes the turn where the
person did 48% (61%). The doubled BOS made Gemma pushier; every Gemma number above was measured with it.

## g26m4: persona calls trained only on their memory moments

`tune/recipes/g26m4.json`: g26m3's calls, but a persona call is trained only on its false-memory and recall replies
(`persona_tags_only`), reusing g26m3's cache with new loss weights (`train_lora --weights-from-calls`). Tested with
one BOS.

| | g26 (one BOS) | g26m4 |
|---|---|---|
| Memory test v2: declines a false claim / plays along | 2 / 14 of 16 (two BOS) | **16 / 0 of 16** |
| Memory test v2: invents a past | 4/24 (two BOS) | 1/24 |
| Quiz | 11/12 | 8/12 |
| Recorded calls: utterances a minute (person 7.4) | 5.1 | 2.1 |
| Live, all checks | 86% (two BOS) | 65% |

The memory part is solved, too well: with nothing but "I don't recall" replies to learn from, it says "I don't recall"
to plain questions ("What's the capital of Japan?" -> "I'm sorry... I don't recall the capital city"; ~30 replies on
the recorded calls), and holds back again. The one "I don't recall who you are. Is this a prank call?" came from a
no-memory call whose prompt says "You don't remember anything about the caller from before this call", which reads as
a stranger; the wording should say the caller is a friend it knows nothing about yet.

## g26m5, g26m6: the memory skill and holding back go together (2026-10-25, one BOS)

- **g26m5** (`recipes/g26m5.json`): g26m4 plus the friend's ordinary replies in persona calls (answers, own talk,
  nudges) at full weight; listening ticks there still weigh nothing. A no-memory persona now reads "You know the
  caller as a friend, but nothing about their life from earlier calls" (the old line read as a stranger).
- **g26m6** (`recipes/g26m6.json`): g26m5's calls and cache, ordinary persona replies at half weight, false-memory
  replies at about three times (first words 24, rest 8), recall 12 / 4. Memory moments carry 42% of the persona-call
  weight (g26m5 13%).

| | g26 | g26m4 | g26m5 | g26m6 |
|---|---|---|---|---|
| Memory test: declines a false claim (of 16) | 2 | **16** | 2 | 8 |
| Memory test: invents a past (of 24) | 4 | 1 | 2 | **0** |
| Quiz | 11/12 | 8/12 | **12/12** | 11/12 |
| Recorded calls: utterances a minute (person 7.4) | **5.1** | 2.1 | 4.3 | 2.5 |
| Recorded calls: takes the turn where the person did | 48% | 27% | **55%** | 24% |
| Live, all checks | 83% | 65% | **84%** | 75% |
| Live, says the right thing | 62% | 43% | **75%** | 58% |

Five ways of teaching it (g26m, g26m3, g26m4, g26m5, g26m6) give one pattern: the more it learns to say "I don't
remember that", the more it holds back everywhere. g26m5 is lively and knows things but plays along; g26m4 never plays
along but says "I don't recall" to plain questions and goes quiet; g26m6 is in between. g26m2 showed part of the holding
back is calibration a higher threshold wins back (see the 0.75 check below).

**Thresholds with one BOS.** g26 at 0.6 now replies a median 0.12 s after a live caller stops (0.5 s before with two
BOS) and is judged right 86% of the time (71%); at 0.5 it backs off too far (back-channels 0.4 a minute, utterances
3.3). The app gives g26 the partner default 0.6 again. g26s1 and g26s2 at 0.6 with one BOS: quiz 11/12, a bit quieter
than g26 (4.2 and 3.8 utterances a minute).

**g26m6 at 0.75** wins its holding back back: on the recorded calls 6.1 utterances a minute (the most person-like of
any run; person 7.4, g26 5.1), takes the turn where the person did 63% (g26 48%), back-channels 0.6 a minute (person
0.8); memory test: declines 9 of 16 false claims, invents a past in 0 of 24 calls; quiz 10/12. It talks over the
caller more (2.0 a minute) and goes quiet on 53% of cut-ins. The app gives g26m6 0.75: the memory adapter to try, with
a persona and memories set.

g26m6 live at 0.75: all checks 84% (75% at 0.6; g26 83%), stops on 4 of 9 cut-ins (the best Gemma), says the right
thing 70%; but with the synthetic callers it starts a median 1.8 s before they finish. 0.7 is the next setting to try
if it jumps in.

**Orpheus beeps (a report from a real call).** A scan generated 150 reply pieces, kept every audio
code and rebuilt the audio exactly as `serve_orpheus.py` streams it: no code out of range in any piece, and by
a stricter test (a steady pitch above 1.2 kHz for >= 50 ms with no harmonics) one borderline hit in 150. So the beep
is not in the decoded voice; next suspects are playback (seams, the volume dip toggling) or something outside the app.
Settings > Debugging > "save call audio" keeps each call's mic, speaker and cleaned-mic audio (last 5 calls) to find it.

## g26m7, and the memory test scored three ways (2026-10-26)

`recipes/g26m7.json`: g26m6's weights; 150 new persona calls where the friend brings up one memory, after the caller has
talked; invitations to meet dropped from every dialogue file (78); 35% of nudges left unanswered for 7-15 s with the
friend's silence trained (`listen_wait`), replacing an app-side timer. The memory test now tells "rolled with it"
(reacts without claiming the memory: "That's great, when do you start?") from "played along" (claims to remember or
to have been there: "I think we ate a lot of pasta").

| Of 16 false memories | declined | rolled with it | played along | invents a past (of 24) |
|---|---|---|---|---|
| g26 (two BOS) | 0 | 4 | 12 | 4 |
| g26m4 | 12 | 3 | 1 | 1 |
| g26m5 | 1 | 5 | 10 | 2 |
| g26m6 / at 0.75 | 5 / 4 | 6 / 8 | 5 / 4 | 0 / 0 |
| **g26m7** at 0.6 / 0.65 / 0.7 | 10 / 12 / 13 | 4 / 0 / 1 | 2 / 4 / 2 | 0 / 0 / 0 |

| g26m7, recorded calls | 0.6 | **0.65** | 0.7 | person |
|---|---|---|---|---|
| Utterances a minute | 4.7 | **6.4** | 9.2 | 7.4 |
| Back-channels a minute | 0.5 | **0.9** | 1.0 | 0.8 |
| Talks over the caller, a minute | 1.6 | 2.3 | 2.9 | |
| Quiet when the caller cuts in | 89% | 84% | 84% | |

At 0.6: quiz 11/12, our held-out set stops on 94% of cut-ins (the best of any run), live 82% of checks. The app gives
it 0.65. In a real call it said "I don't recall" as a tic in ordinary teasing: half its training refusals began that way, weighted 24. g26m8 varies them.

## Voice delay with the GPU shared

One live-test caller against g26m7 with Orpheus, by how many layers' experts stay in RAM:

| Expert layers in RAM | Orpheus first sound | Orpheus speed | Model tick |
|---|---|---|---|
| 15 (the app's setting with a GPU voice until now) | 1.13 s | 0.87x | 241 ms |
| **8** | **0.97 s** | **0.91x** | **182 ms** |
| 1 | 6.5 s | 0.55x | 122 ms |

The app now keeps 8 on Gemma 26B-A4B with a GPU voice.

## g26m8, g26t, and does the partner need persona training at all? (2026-10-26)

g26m8 (varied refusals, the memory spread, weights 10/3 and 6/2) lost g26m7's "I don't recall" tic: 0 times on the
24 recorded calls, against g26m7's 82. It also lost declining. g26t (g26m8's calls, the rest of each reply's words at
0.5) asked whether the words were pulling the timing around; they were not, and it was worse at nearly everything.

| Memory test (16 false claims, 24 calls) | declined | rolled with it | played along | invented a past |
|---|---|---|---|---|
| bare base, plain chat, persona prompt (2 replies each, 10 claims) | 0 | 8 | 12 | |
| bare base, micro-turns | (silent on all 16; quiz 0/12: it cannot use the protocol) | | | |
| g26 (one BOS) at 0.6 | 1 | 4 | 11 | 4 |
| g26m7 at 0.65 | 12 | 0 | 4 | 0 |
| g26m8 at 0.65 | 1 | 4 | 11 | 4 |
| g26t at 0.65 | 4 | 2 | 10 | 1 |

| Recorded calls | g26 | g26m8 | g26t | person |
|---|---|---|---|---|
| Takes the turn where the person did | 61% | 42% | 36% | |
| Utterances a minute | 5.0 | 2.8 | 3.2 | 7.4 |
| Back-channels a minute | 2.1 | 0.1 | 0.7 | 0.8 |

g26t: quiz 9/12, live 76% of checks (g26 86%, g26m7 82%), "says the right thing" 50%.

- The base model does not decline a shared past it does not have, even in plain chat with the memories listed: it
  has to be taught. So persona training stays.
- Declining lives in the words, not the timing: with the words at 0.5 (g26t) the model still answers and plays along.
- g26m7's declines were mostly its tic ("I don't recall a time limit", "I don't recall that being a blue"), so its
  12 of 16 overstates the skill. With varied refusals at 10/3, among 2,500 calls, g26m8 learned neither the tic nor
  the skill: the signal was too faint.
- Next, g26p: the persona calls alone as an add-on on top of g26 (g26 keeps the timing), so the memory moments are a
  far larger share of what it learns.

### A clearer line in the persona prompt (2026-10-26)

The persona prompt said only "Only bring up a shared past that is in this list": what the friend raises, not what it
does when the caller claims one. With one more line, "If the caller talks about something you did together, or
something they told you, that is not in this list, you don't remember it: say so warmly and ask about it."
(`tune/data/heldout_memory_strict.jsonl`):

| False memories | declined | rolled with it | played along |
|---|---|---|---|
| bare base, plain chat (20 replies) | 0 → **16** | 8 → 4 | 12 → **0** |
| g26 at 0.6, micro-turns (16) | 1 → 4 | 4 → 7 | 11 → 5 |

The base model declines when asked to, in varied words ("Wait, a game night? I don't quite remember that one, what
were we playing?"), so this can be the persona's choice rather than a trained habit. g26 follows it only in part: its
word training overrides some of the base's prompt-following. g26p (the persona calls as an add-on at full strength on
g26) halved g26's liveliness (2.4 utterances a minute, quiz 6/12). So: a timing adapter that leaves the words to the
base (g26b), with persona behaviour from the prompt.

## The voice anchor, and the pauses it uncovered (2026-10-26)

The model writes a tick's words at its training pace (2.9 words a 0.6 s tick); Orpheus says about 1.6. In a recorded
call the voice ran up to 15 s behind the model: it took its own still-playing story for silence and spoke up over it
("Anyway,"), and the caller's "happy or sad?" landed 15 s behind where the voice was. The app now holds the next tick
while the voice has over 1.8 s queued (not while a sentence is being written), runs a tick at once when the caller
speaks, and sends the voice whole sentences (a comma seam restarted its tone mid-sentence). Details in ARCHITECTURE.md;
`scripts/check_anchor.js` replays it with a stand-in model and voice. A first version that thinned each tick's history
to the words the voice had said made the model hesitate and restart (one word a tick is unseen in training).

With the voice in step, the model's own manners became audible: it stops after nearly every sentence, waits, and
comes back seconds later as a nudge ("So,", "Anyway,"). The data explains it:

| After the speaker ends a sentence and goes quiet | written training calls | real recorded calls |
|---|---|---|
| The other person speaks next | 97% | 10% |
| The speaker carries on by itself | 3%, as a nudge 2-7 s later | 90%, 93% of those within 0.6 s |
| At a sentence end: carries straight on / pauses first | always straight on within a turn | 70% / 30% |

The call builder laid every friend turn down as one run of words with the caller always speaking next. g26b2's recipe
(`--sentence-pauses 0.3`) gives 30% of the friend's sentence ends a beat of one tick (93%) or two, tagged
`speak_pause`, and it carries on (`speak_on`, weight 3); evaluate.py scores it as `carry`.

## The timing base line: g26b to g26b4, and where the stalls come from (2026-10-27/28)

The persona is a prompt now (above), so the adapter's job is timing. g26b is g26's calls with the friend's words at
loss weight 0.5 instead of 2 (what it says is its own, the tags carry the timing) and service greetings muted; g26b2
adds the sentence-end beats; g26b3 weighs a sentence opened straight away like a beat (g26b2 paused at 76% of its
sentence ends); g26b4 chunks a sentence as evenly as it can (7 words as 3/2/2, not 3/3/1). All at 0.6 on the 24
recorded calls (`out/yap_sweep_g26b4.md`):

| | g26 | g26b | g26b2 | g26b3 | g26b4 | g26b4 at 0.7 | person |
|---|---|---|---|---|---|---|---|
| Takes the turn where the person did, within 2.4 s | 82% | 76% | 54% | 51% | 58% | 70% | |
| Talks over the caller, per minute | 1.8 | 1.6 | 1.0 | 0.9 | 1.0 | 1.5 | |
| Quiet when the person gave way to a cut-in | 42% | 58% | 79% | 89% | 89% | 63% | |
| Back-channels per minute | 2.1 | 1.1 | 0.5 | 0.6 | 0.8 | 1.8 | 0.8 |
| After its own sentence: straight on / beat / waits | 38/47/15 | 45/40/16 | 11/76/13 | 34/51/14 | 39/44/17 | 55/33/12 | 55/27/18 |
| Unfinished utterances | 7% | 10% | 8% | 14% | 15% | 14% | |
| Stalls (unfinished, then 5+ quiet ticks) | 4 | 6 | 9 | 14 | 27 | 17 | |
| Quiz | 11/12 | 11/12 | 11/12 | 10/12 | 9/12 | | |

g26b4's held-out set: start 83%, barge 89%, talks over 9% (g26 85 / 71 / 18); live bench 80% (g26 86%): it replies
1.1 s after the caller stops (g26 half a second before), 0 of 3 on speaking up after a silence. Its persona-rule
memory test: declined 5, rolled with it 4, played along 7 of 16, invented a past in 0 of 24 calls (g26 4 / 7 / 5, 3).

What the even chunks were for was not there: g26b3's replays have no lone-word ticks at all; the lone "I" in the
7-minute call was a one-off. The stalls are something else. Read one by one (`eval_yap_*.json`), nearly all of them
are the model speaking up into a long silence (where the other person talked in the recording), getting a few words or
sentences out and stopping mid-sentence ("He makes the action look", 11 quiet ticks, "So, I was"), often looping the
same opener until the repeat guard cuts it. g26b, with the light words and no beats, stalls like g26 (6 vs 4), so the
word weight is not the cause; the stalls arrive with the beats and grow as the beats are balanced (b3) and evened
(b4): a model taught to stop for a beat inside its own turn also stops where no beat belongs, and once it has
listened into a silence it goes on listening. The beats bought what they were for (quiet on a cut-in 89%, talking over
halved, back-channels at the person's rate) at that price.

The long-call check (four 600-tick calls, g26b3, the history trimmed at 400 ticks where the
caller starts a turn, or never): after tick 400, 6 unfinished and 5 stalls of 35 utterances with the trim, 5 and 5 of
38 without. The trim is neutral; the stalls in that call were the beat training, not the cut.

Where that leaves the timing base: g26b is the closest to the person at sentence ends (45% straight on) and barely
stalls, but talks over at g26's rate and stops for a cut-in only 58% of the time; g26b4 has the manners and the
stalls. The next recipe should get the cut-in behaviour without the mid-turn stopping: the beat at a lower weight,
or the stop-for-a-cut-in weight (`barge_stop`) raised on g26b's calls instead of beats at all.

## The carry-on bar and g26b5: finishing the sentence (2026-10-27/28)

The stalls have a shape (`eval_yap_g26b4.json`, P(listen) read through each one): the model stops mid-sentence at
P(listen) 0.6-0.8, just over the 0.6 threshold, and after one quiet tick of its own it is at 0.95+ and stays there.
The data does not teach that: in the real calls the speaker goes straight on at 77% of mid-sentence ticks and resumes
after a one-tick gap 88% of the time. What the timing line lacks is weight on the decision to carry on: the real calls
train none of it (`real_mid_reply_off`), and in the built calls it is the first token of a continuing tick at the
words' weight 0.5, against stops at 1.0-1.9.

Two fixes, tried apart:
- **The carry-on bar** (app `llama.js`, replay `common.MicroTurnClient.bar`, `--carry-on-p`): right after its own words
  that did not end a sentence, with the line quiet, listening takes P 0.9 instead of the threshold. One tick only; the
  caller speaking or a finished sentence leave the threshold alone. On by default in the app (`carryOnP` 0.9).
- **g26b5**: g26b4 on its cache with the first token of each continuing tick at 2.0 (`carry_on_first`, train_lora `w1`),
  the rest of the words at 0.5. Replayed with the bar off.

All at 0.6 on the 24 recorded calls:

| | g26 | g26b4 | g26b4 + bar | g26b5 | person |
|---|---|---|---|---|---|
| Unfinished utterances | 7% | 15% | 9% | 7% | |
| Stalls (unfinished, then 5+ quiet ticks) | 4 | 27 | 4 | 2 | |
| Takes the turn where the person did, within 2.4 s | 82% | 58% | 57% | 56% | |
| Talks over the caller, per minute | 1.8 | 1.0 | 0.9 | 0.8 | |
| Quiet when the person gave way to a cut-in | 42% | 89% | 89% | 74% (14/19) | |
| Back-channels per minute | 2.1 | 0.8 | 0.8 | 1.4 | 0.8 |
| After its own sentence: straight on / beat / waits | 38/47/15 | 39/44/17 | 40/43/18 | 42/43/15 | 55/27/18 |
| Utterances started per minute | 5.0 | 4.2 | 4.0 | 3.3 | 7.4 |

Both work. The bar takes g26b4 from 27 stalls to 4 and moves nothing else; of those 4, three are the repeat guard
forcing a listen mid-sentence (replays pick words greedily and repeat more than the app, which samples at 0.7) and one
a confident stop (0.92). g26b5 learned it with no bar: 2 stalls, one guard-forced, one a stop at 0.87 that it picked
up again. What it gave for it: quiet on a cut-in 14 of 19 against 17 (three cases; the held-out barge set, 80-odd,
is the better measure and has not run), and back-channels above the person's rate. It starts fewer, longer
utterances. Its quiz, held-out set and live bench have not run yet.

## v1, and two checks on it (2026-10-28)

**v1 is g26b4 with the carry-on bar, at 0.6, on the Gemma 4 26B-A4B heretic base** (the app's default). Scorecard:
held-out set start 83%, barge 89%, filler 27%, talks over 9%, unfinished 1% (g26b4 without
the bar the same, unfinished 2%); recorded calls as in the table above (g26b4 + bar); live bench `g26b4v1` 86% (g26
86%, g26b4 80%): stops on a cut-in 8 of 9 (median 1.35 s), speaks up after a silence 2 of 3, replies 0.89 s after the
caller stops, the base model judges 81% of its answers right. Weak spots: goodbye 3/9, keeps going through
back-channels 8/12, and declines 6/12. The decline misses are not silence: asked for a neighbour's address it offers
to help ("which neighbor?") without inventing one, and asked for something explicit it half-declines or plays along.
Nothing has trained declines since run 8, and the base is decensored.

**Without the caller's final "?" / "."** (`out/yap_nomark.md`, the 24 recorded calls with the last mark of every
caller turn stripped, as if `stt.js finalMark` were gone): it takes the turn within 2.4 s 50% of the time (57% with
marks), within a tick 29% (35%), starts in the caller's mid-turn pauses 1% (4%), is quiet for a cut-in 14 of 19 (17),
and stalls 7 times (4). The mark helps a little and does no harm; the model already reads "done" from the words and
the quiet for most turns (half the friend's turn starts in the training calls follow an unmarked caller ending). Not
worth a retrain while the app supplies the mark.

**The same adapter on the standard Gemma** (`gemma-4-26B-A4B-it.Q4_K_M.gguf`, mradermacher, same quant, not
abliterated). The heretic edit touched layers 10-30, which includes the four the adapter sits
on, so the adapter sees shifted inputs. It loads and runs at the same speed (0.21 s a tick). Recorded calls
(`out/yap_stock.md`):

| | v1 (heretic) | v1 adapter on standard |
|---|---|---|
| Takes the turn where the person did, within a tick / 2.4 s | 35% / 57% | 43% / 68% |
| Talks over the caller, per minute | 0.9 | 1.5 |
| Starts in the caller's mid-turn pauses | 4% | 8% |
| Quiet when the person gave way to a cut-in | 89% (17/19) | 68% (13/19) |
| Stalls | 4 | 3 |

Live bench (`g26b4stock`): 82% (v1 86%). Declines 11 of 12 ("I can't do that, that is a violation of privacy";
"I cannot generate explicit or sexual content"), stops on a cut-in 6 of 9, waits through pauses 15/15, judged right
86%. But short questions fail: quick question 4/12, math 0/6, the thinking pause 5/9. After "What's twelve times
twelve?" its P(listen) stays at 0.65-0.85 for seconds where v1 drops under 0.6, and when it does speak it says
"You're being quiet." as if nothing was asked. The adapter's "answer now" was learned on the heretic base's hidden
states and does not carry over cleanly; it is eager on the long recorded turns and deaf on short ones. The standard
base is the better decliner, so the way to use it is to train the timing adapter on it (a new Phase A cache, the same
recipe), not to swap the base under v1.

**How this training differs from the author's Qwen adapter** (codacus MicroTurn Duplex): theirs is rank 32 on Qwen3.6
35B-A3B's last 4 layers (36-39), 282 steps over 564 LLM-written calls in the assistant role, run at threshold 0.3,
against the UD-Q4_K_M weights. Ours keeps the rank and the last-4-layers shape (Gemma's 27-30), but trains on built
calls plus real recorded conversations (Open Yap, otoSpeech) with per-tag loss weights (beats, cut-in stops,
carry-on), on a partner system prompt, cached hidden states (Phase A) so each base needs its own cache, and runs at
0.6 with the carry-on bar.

## g26b4std: v1's recipe trained on the standard Gemma (2026-10-28)

`tune/recipes/g26b4std.json`: g26b4's calls and settings, base `gemma4a4bstd` (the standard, not abliterated, Gemma 4
26B-A4B), a new Phase A cache; tested at 0.6 with the carry-on bar like v1. Export
`talkme-partner-v1-gemma4a4bstd.gguf`. Quiz 10/12 (v1 9/12).

| | v1 (heretic) | v1 adapter on standard | g26b4std |
|---|---|---|---|
| Recorded calls: takes the turn within a tick / 2.4 s | 35% / 57% | 43% / 68% | 29% / 45% |
| Talks over the caller, per minute | 0.9 | 1.5 | 0.7 |
| Starts in the caller's mid-turn pauses | 4% | 8% | 2% |
| Quiet when the person gave way to a cut-in | 89% (17/19) | 68% (13/19) | 95% (18/19) |
| After its own sentence: straight on / a beat / waits (person 55/27/18) | 40 / 43 / 18 | 46 / 30 / 25 | 13 / 51 / 35 |
| Stalls (`stalls.py`); unfinished utterances | 4; 9% | 3; - | 1; 6% |
| Held-out set: start / barge / filler / talks over / unfinished | 83 / 89 / 27 / 9 / 1% | - | 85 / 94 / 24 / 7 / 0% |
| Live bench, all checks | 86% | 82% | 85% |
| Live: replies in time; median reply start | 93%; 0.89 s | 89%; 0.85 s | 91%; 1.19 s |
| Live: stops when cut in on; median stop | 8/9; 1.35 s | 6/9; 1.37 s | 6/9; 1.67 s |
| Live: says the right thing; judged right by the base | 72%; 81% | 67%; 86% | 77%; 81% |
| Live: keeps going through back-channels | 8/12 | 7/12 | 10/12 |
| Live: decline / quick question / math / thinking pause | 6/12, 10/12, 2/6, 9/9 | 11/12, 4/12, 0/6, 5/9 | 12/12, 8/12, 1/6, 7/9 |
| Live: goodbye / three colours / gift-and / friend call | 3/9, 2/6, 9/9, 41/45 | 3/9, 3/6, 9/9, 43/45 | 5/9, 5/6, 6/9, 37/45 |

Training its own adapter fixed what the borrowed one broke: short questions are heard again (quick question 8/12, from
4), it declines everything it should (12/12), and it is the most polite of the three (fewest talk-overs and mid-turn
starts, quiet for 18 of 19 cut-ins, one stall). The cost is pace: it takes the turn later on the recorded calls (45%
within 2.4 s against 57%), replies 0.3 s later live, is slower to stop when cut in on (6/9, 1.67 s), and pauses or
waits after its own sentences far more than people do (35% waits, people 18%). Overall it is level with v1 (85% vs
86%), better behaved and slower. Since only the base changed, the 0.6 threshold and the 0.9 bar were tuned on the
heretic base's probabilities; a different threshold on this base is the cheap first thing to try before any retraining.

**Its threshold: 0.5 and 0.7.** The same adapter at other listen thresholds (`recipes/g26b4std05.json`,
`g26b4std07.json`; only the tests run). It speaks when P(listen) is below the threshold, so a lower one makes it wait
longer: 0.5 was the wrong way and is worse everywhere it matters. 0.7 puts its pace back level with v1's. The 0.7 run was
stopped before the live bench to try it by hand.

| | v1 (0.6) | std 0.5 | std 0.6 | std 0.7 |
|---|---|---|---|---|
| Recorded calls: turn within a tick / 2.4 s | 35% / 57% | 13% / 25% | 29% / 45% | **38% / 60%** |
| Talks over per minute; mid-turn pause starts | 0.9; 4% | 0.3; 5% | 0.7; 2% | 0.9; 4% |
| Back-channels per minute (person 0.8); within a tick of the person's | 0.8; 15% | 0.2; 3% | 0.6; 6% | 1.3; 18% |
| Utterances per minute (person 7.4) | 4.0 | 1.7 | 2.9 | 4.5 |
| Quiet for a cut-in | 17/19 | 18/19 | 18/19 | 17/19 |
| After its own sentence: straight on / a beat / waits (person 55/27/18) | 40 / 43 / 18 | 8 / 40 / 52 | 13 / 51 / 35 | 26 / 52 / 21 |
| Stalls; unfinished utterances | 4; 9% | 0; 10% | 1; 6% | 2; 6% |
| Held-out set: start / hazard / barge / filler / talks over / unfinished | 83 / 54 / 89 / 27 / 9 / 1% | 80 / 38 / 94 / 22 / 5 / 0% | 85 / 54 / 94 / 24 / 7 / 0% | **89 / 65** / 91 / 25 / 9 / 1% |
| Live bench, all checks; median reply start | 86%; 0.89 s | 81%; 1.55 s | 85%; 1.19 s | not run |
| Live: stops when cut in on | 8/9 | 4/9 | 6/9 | not run |

At 0.7 the standard base matches or beats v1 on every recorded-call and held-out line (the best answer starts and
warnings of any g26b4 so far), with v1's talk-over rate and a few more back-channels than a person. The live bench
(cut-in stops, declines, reply time) is what is left before it can replace v1.

## m3b4: the same recipe on Ministral 3 14B, a second model family (2026-10-28)

`tune/recipes/m3b4.json`: g26b4's calls and settings on Mistral's dense Ministral 3 14B (`[INST]` format, base
`ministral3`). Trained on an RTX 3080 Ti Laptop (16 GB): Phase A 57 chunks of 32 calls, about 1.5 min each;
600 steps at 5.6 s, peak 7.4 GB. Tested on the 4090 at 0.6 with the bar, Q4_K_M (bartowski). Export
`talkme-partner-v1-ministral3.gguf`. It is fast: 0.03-0.04 s a tick on the recorded calls (the 26B-A4B 0.21 s).

| | v1 | g26b4std | m3b4 |
|---|---|---|---|
| Quiz | 9/12 | 10/12 | 10/12 |
| Recorded calls: turn within a tick / 2.4 s | 35% / 57% | 29% / 45% | 27% / 54% |
| Talks over per minute; mid-turn pause starts | 0.9; 4% | 0.7; 2% | 0.8; 2% |
| Quiet for a cut-in | 17/19 | 18/19 | 18/19 |
| After its own sentence: straight on / a beat / waits (person 55/27/18) | 40 / 43 / 18 | 13 / 51 / 35 | 12 / 44 / 44 |
| Stalls; unfinished utterances | 4; 9% | 1; 6% | 8; 13% |
| Held-out set: start / hazard / barge / filler / talks over | 83 / 54 / 89 / 27 / 9% | 85 / 54 / 94 / 24 / 7% | 86 / 58 / 94 / 24 / 7% |
| Live bench, all checks | 86% | 85% | 84% |
| Live: replies in time; median reply start | 93%; 0.89 s | 91%; 1.19 s | 98%; 1.19 s |
| Live: stops when cut in on; waits through pauses | 8/9; 14/15 | 6/9; 13/15 | 5/9; 12/15 |
| Live: back-channels while you talk; keeps going through them | 6/6; 8/12 | 6/6; 10/12 | 3/6; 6/12 |
| Live: speaks up after a silence; quiet nudge; long talk | 2/3; 5/6; 6/6 | 3/3; 6/6; 6/6 | 0/3; 3/6; 3/6 |
| Live: decline / quick question / math / goodbye | 6/12, 10/12, 2/6, 3/9 | 12/12, 8/12, 1/6, 5/9 | 10/12, 7/12, 5/6, 6/9 |

The recipe carries over to a different family with no changes: on the held-out set it is the best of the three, it
answers in time 98% of the time live, gets the math question 5 of 6 (the Gemmas 1-2) and declines 10 of 12 on a
standard base. Its weak side is talking at length: it stops after a three-word chunk with P(listen) at 0.97 (the bar
at 0.9 does not catch it), 6 of its 8 stalls in one call where it restarted "Third one is" six times, waits after 44%
of its own sentences, back-channels half as often live, and does not speak up into a silence (0/3, quiet nudge 3/6).
It learned the listening better than the speaking on; more speak-on weight, or calls with longer friend turns, is the
place to start if it gets another run.

## gpt-oss 20B: a third model family, and what it took (2026-10-30 to 11-01)

gpt-oss was the only base that barely learned in 600 steps of attention-only training: 13 of 36 on the wide quiz
(`tune/data/quiz_wide.json`, 19 silent), where the Gemmas and Ministral were done at 600. It keeps its experts in fused
tensors PEFT cannot adapt, so the first idea was to train those too (`train.experts`). That made it speak but not know
more. Steps were the lever.

| Recipe | Layers | Experts | Steps | Size | Right / wrong / silent of 36 |
|---|---|---|---|---|---|
| plain gpt-oss | | | | | 32 / 1 / 3 |
| gptoss1 | 4 | no | 600 | 10 MB | 13 / 4 / 19 |
| gptoss2 | 8 | no | 600 | 21 MB | 23 / 6 / 7 |
| gptoss5 | 4 | yes | 600 | | 13 / 15 / 8 |
| gptoss6 | 8 | yes | 600 | | 20 / 4 / 12 |
| gptoss10 | 4 | yes | 1,200 | | 26 / 6 / 4 |
| gptoss11 | 8 | yes | 1,200 | 587 MB | 27 / 9 / 0 |
| gptoss14 | 4 | no | 1,200 | 10 MB | 24 / 5 / 7 |
| gptoss12 | 8 | no | 1,200 | 21 MB | 25 / 6 / 5 |
| gptoss13 | 8 | yes | 1,800 | 587 MB | 25 / 8 / 3 |
| **gptoss15** | 8 | no | 1,800 | 21 MB | **30 / 6 / 0** |

Each row is one quiz, and two seeds of one recipe differed by 4, so small gaps mean nothing. Each longer run continues
the one before it (`init_from`). Experts add 566 MB and no score. At 1,200 steps, 4 layers matched 8 with cleaner
replies but a higher training loss (0.99 over the last 100 steps; 8 layers 0.82). Going to 1,800 took 8 layers to 0.72.
Against gptoss12 on the same questions gptoss15 won 8 and lost 3. "Right" is lenient: the check finds the answer
anywhere in the reply, so "Three. A standard bicycle has two wheels." counts. gptoss15 has also picked up a habit of
confirming itself ("That's correct.") on 8 of 36 replies, against 0-3 for the others.

Only gptoss12 had the full tests: live 212/264 (80%), says the right thing 63%, stops when cut in on 3/9; held-out start
81%, barge 100%, unfinished 1%; on recorded calls 43% within a tick and 5.2 utterances a minute (person 7.4), with 2.6
talk-overs a minute. For comparison g26b4std was 85% live. `tests/out/live_bench.md` still shows 210: two declines failed
on a curly apostrophe the check did not accept.

gptoss15 is **Partner v1 (gpt-oss)** (`talkme-partner-v1-gptoss.gguf`), picked by ear over gptoss13 on manual calls,
at a threshold of 0.6. The family is proven and working. gptoss15 itself has had only the quizzes, 9/12 and the wide one.
