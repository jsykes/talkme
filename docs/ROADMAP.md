# talkme roadmap

Where the project is headed. Results of what has been tried are in [RESULTS.md](RESULTS.md); how it works is in
[ARCHITECTURE.md](ARCHITECTURE.md). Principle for behaviour: teach it through training data so the model decides
(when to be quiet, how many memories to bring up), not through rules in the app; keep app-side code simple.

## Version 1: a partner you would call

Done: the full-duplex app (600 ms micro-turns, cut-ins, back-channels), Orpheus and Chatterbox voices, the echo
canceller for speakers, the training pipeline and its tests, Gemma 4 26B-A4B partners (g26 the general one), persona
and memories in the prompt (g26m6/g26m7).

Checklist:
- [x] Pick the default adapter from real calls. **v1 is g26b4 (2026-10-28, chosen after a second 12-minute
      live call; the app default now).** Memory adapters (g26m7) stay for the persona line. The first call as a candidate: g26b4 (`talkme-partner-v1-gemma4a4b`) at 0.6 with the carry-on
      bar. A 13-minute live call: stopped and answered nearly every cut-in, one greeting, 4 mid-sentence stalls of
      6-10 s (fewer than without the bar); twice it said it would "be right back", once then silent for 2 minutes.
      Stops are now logged with their P(listen) and guard state to read the remaining stalls.
- [ ] Memory mentions as a spread, not a rule (agreed 2026-10-25, g26m8): of the persona calls, ~30% bring up no
      memory unprompted, ~50% one, ~20% two (spaced, each only after the caller has talked); it always answers when
      the caller brings one up. A tendency it learns, tuned from real calls (g26m7's writer said "only one").
- [x] A timing base that stops for a cut-in without stopping mid-turn (2026-10-28, RESULTS.md): the sentence-end
      beats (g26b2-b4) bought quiet on cut-ins (89%) and half the talking over, but taught stopping mid-sentence into
      silences (g26b4 27 stalls on the recorded calls, g26b 6). Even chunks (g26b4) fixed nothing: there were no
      lone-word ticks. The cause was the carry-on decision trained at the words' weight 0.5 (2026-10-28): the app's
      carry-on bar (P 0.9 to stop mid-sentence into a silence) takes g26b4 to 4 stalls, and g26b5 (that decision at 2.0)
      to 2 with no bar, at some cost to quiet on a cut-in (74% of 19). Live (2026-10-28): g26b5 talked over a question
      mid-story and said "Hello." 2-3 times; g26b4 with the bar handled cut-ins well (v1, above). Second call, with the
      stop log: 3 mid-sentence stops in 12 minutes, none longer than 6 s (one P 0.98 after it spoke its own listen tag,
      one the repeat guard on "talk about", one P 0.97 at the end of a question).
- [x] Long calls (2026-10-28): the trim at a caller's turn start is neutral on four 600-tick replays;
      the 7-minute call's fragments were the beat training, not the cut.
- [x] Beats at sentence ends (g26b2, g26b3 ran 2026-10-26: the beat is learned; g26b3 balances it, 34% straight on): with the voice anchor the model's own
      turn-taking became audible, a hand-over after nearly every sentence; the written calls taught it (the caller
      speaks next after 97% of the friend's sentence ends; real speakers carry on 90% of the time). Needs a new
      cache (~53 GB) and ~5 h of GPU.
- [x] The beep heard in calls (fixed 2026-10-28): one render quantum (128 samples at 48 kHz, 375 Hz) repeating for
      ~0.25 s after a voice part ends. Replaying saved voice parts (`TALKME_DUMP_TTS`) reproduced it at the same spots
      with no model and no voice server, idle or under GPU load, and only for buffers at another sample rate than the
      context's (24 kHz in 48 kHz) that end part-way through a quantum: Chromium's resampler in Electron 44 keeps
      repeating the last quantum instead of going silent. Streamed Orpheus parts end loud mid-word, so it was heard;
      a piece's last part ends on a fade, so Kokoro's repeats were silent. The app now brings every part to the
      context's rate itself (`app.js atContextRate`), so Chromium never resamples; zero repeats in replays after.
- [x] Varied ways to say "I don't remember that": g26m8 says "don't recall" 0 times on recorded calls (g26m7 82),
      but it also stopped declining false memories (1 of 16).
- [ ] Teach declining a shared past it does not have without a tic: the base model never does it on its own, even in
      plain chat. g26p (persona add-on on top of g26) first; preference training if weights alone keep failing.
- [ ] Varied nudges after a silence (about 1 in 9 training nudges open "So, are you...").
- [ ] Voice delay: Orpheus runs below real time while Gemma shares the GPU (first sound ~1-1.2 s). Measure expert
      offload settings (8 / 15 / 0) during a call; compare a smaller Orpheus quant and Kokoro by ear.
- [ ] Echo canceller in real calls: the echo test hears 0 words; check longer calls on speakers.
- [x] An adapter browser (2026-10-31): the Adapters tab shows each adapter's metadata, results and notes in one place; notes
      and names live in a JSON card beside the file, and rename and remove keep saved settings and recipes right.
- [ ] Confirm the chosen adapters over 2-3 training seeds (one run's manners swing as much as recipe changes).
- [x] One-command setup and a README for someone new (2026-10-28): `npm run setup` (`scripts/setup.js`) fetches llama.cpp,
  v1's base, the speech models and the aec/ environment; `--orpheus` and `--base` for more.
- [ ] Linux with NVIDIA (2026-10-28): tested in Docker on the 4090. Fresh clone, `npm install` and `npm run setup`
  worked (Debian 12); Kokoro and Nemotron run; on Ubuntu 24.04 the app loaded v1 on the GPU (66 s) and answered a
  typed call aloud. llama.cpp's Linux build needs glibc 2.38 (Ubuntu 24.04, Debian 13, Fedora 39), which setup checks.
  Still untested: a real Linux desktop with a microphone and speakers, and Electron's sandbox on Ubuntu 24.04.

## Next training ideas

- Preference training for memory (show "that wasn't me" against playing along) if data and weights alone keep
  trading memory skill against liveliness.
- Clean the older partner data with the model judge (invented shared past it still has; the phrase filter misses most).
- A rolling summary of the call instead of dropping its oldest ticks when the history fills up.
- Several personas switched by prompt with one adapter (persona variety in training already points this way).

## After v1

In order: training on the standard Gemma, the caller's end of turn without the final mark (low), then tone and tags;
remembering a long call when long calls start feeling forgetful; several people on the call is a maybe, further down
the road.

### Knowing the caller is done without the "?" or "." (2026-10-28, low priority)

The recogniser (Nemotron) never punctuates the last word, so `stt.js finalMark` adds "?" / "." from word lists
(question openers, words that leave a thought hanging, a wh-question ending on a preposition). A wrong mark is harmless
(both say "done"); a missing one can leave the model waiting before an "Anyway," (live, 2026-10-28).
Measured (docs/RESULTS.md, "v1, and two checks on it"): with every caller's final mark stripped, v1 still takes the
turn within 2.4 s 50% of the time against 57%, and is quiet for a cut-in 14 of 19 against 17. The model mostly reads
"done" from the words and the quiet already; the mark adds a little. Keep `finalMark`; retrain only if its word
lists start missing often in live calls (then: drop the mark on a share of caller turns in the training calls and
check the replays as above).

### Training on the standard Gemma (2026-10-28)

v1's adapter on the standard (not decensored) Gemma 4 26B-A4B declines properly (11/12 live, v1 6/12) but goes deaf
on short questions (quick question 4/12, math 0/6): the adapter was learned on the heretic base's hidden states.
- Train the timing recipe (g26b4 + the bar) on the standard base: a new Phase A cache, the same calls.
- Compare against v1 on the recorded calls, held-out set and live bench; if it matches, it becomes the base (and the
  "decensored base" line in Sharing goes away). The app and pipeline have the base as
  `gemma4a4bstd` (config.js BASES); its adapters end in `-gemma4a4bstd.gguf`.
- Done (g26b4std, docs/RESULTS.md): level with v1 live (85% vs 86%), declines 12/12, hears short questions again,
  but slower: takes the turn later, replies 0.3 s later, stops on a cut-in 6/9. Threshold 0.5 was worse (it waits
  longer); 0.7 is level with or better than v1 on the recorded calls and held-out set. Next: the live bench (or real
  calls) at 0.7 before deciding it is the release base.
- Second family (m3b4, Ministral 3 14B, docs/RESULTS.md): the same recipe, 84% live, 6x faster ticks; listens well,
  stops short when talking at length (8 stalls, speaks up after a silence 0/3). Showcase-ready after one more run on its
  speaking-on.
- Third family, proven and working (gpt-oss 20B, 2026-11-01, docs/RESULTS.md): Partner v1 (gpt-oss), gptoss15, attention
  only on the last 8 layers, 1,800 steps. Picked by ear over the expert version. It needed three times the steps
  of the other families. Quiz 30 of 36 with no silences (plain gpt-oss 32); it has not had the full tests.

### Remembering a long call (2026-10-28, when long calls feel forgetful)

Every 0.6 s tick adds a heard/said pair to the history (~17 tokens), mostly `<no voice>` / `<listen>`. Those quiet
pairs are the model's clock (how long a pause has run) and cost little per tick with the prompt cache, but they fill the
window: the 26B-A4B and Ministral keep 400 ticks (~4 min; config.js BASES `historyTicks`, then the oldest 40% go), Qwen
~1,600. On a long call v1 forgets anything said more than a few minutes back. Two ways to stretch it, both trained so
the model sees the same thing in training as in the app:
- [ ] **A running summary of the call** in the system prompt: what the trimmed history held, as "earlier in this call:
      ...", beside the persona and memories (same machinery). First choice; train with summaries in the calls' prompts
      and check the model brings up earlier parts of the call without making them up.
- [ ] **Collapsed silences:** one line for a run of quiet ticks ("<no voice> x20") instead of 20 pairs, so the same
      window holds far more talk. The training calls need the same shorthand so it still keeps time; check pauses,
      nudges and tick time stay as they are.

### Hearing how things are said (emotion and intent)

Today the model only gets words. People also answer the tone: a sigh, a laugh, a flat "fine", excitement.
- Run a speech-emotion / paralinguistic model on the mic audio next to the recogniser (candidates to evaluate:
  emotion-recognition models that work on short streaming windows, plus laughter / sigh detection), and pass what it
  hears into the heard text as light tags, e.g. `(laughing)`, `(sounds tired)`.
- Training data needs the same tags: label the real recorded calls (otoSpeech, Open Yap) with the same model, and
  write dialogues where the tone changes the right reply ("I'm fine." said flatly).
- Questions: latency (it must not slow the tick), accuracy on one speaker's voice, and whether tags help or distract.
- The other direction: Orpheus performs emotion tags in its text (`<laugh>`, `<chuckle>`, `<sigh>`, `<gasp>`,
  `<groan>`, `<yawn>`). Train the partner to write them as `(laughs)` / `(sighs)` (anything starting with `<` reads as
  `<listen>` in the micro-turn protocol) and have `tts.js` map them to Orpheus's tags just before speaking.

### Several people on the call (who said what): maybe, lower priority

- Speaker diarization on the mic stream: speaker embeddings per stretch of speech, clustered into voices; optional
  enrollment ("this is Sam") to put names on them.
- The heard text carries the speaker ("Alex: wait, really?"); the system prompt says who is on the call.
- Training data with three-way calls: the writer makes two callers, the call builder interleaves and overlaps them.
- Questions: overlapping speech, a voice the diarizer has not heard before, echo from the speaker counted as a person.

## Sharing it

- Open-source release with a technical report (arXiv) first; a workshop paper needs a blind listener study (10-20
  people) and repeated runs. Credit DuplexCascade (Yang, Fujita and Sudo, 2026), codacus's adapter and dataset
  (Apache 2.0), otoSpeech and Open Yap (CC BY 4.0).
- A note on the codacus Hugging Face repo with the findings.

## Parked

- Gemma 4 31B with Orpheus (52 GPU layers): untested; the 31B is out of focus.
- The Gemma "listen" leak in the forced-speak path.
- A gaming mode (smaller GPU footprint), a faster cut-in stop (stop
  playback on the recogniser's partial before the model confirms).
