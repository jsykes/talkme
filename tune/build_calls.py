"""Turn written dialogues into simulated 0.6 s micro-turn calls, in the schema of the published dataset.

This re-implements the timing rules described in the dataset card (models/lora/dataset/README.md):
  - a call strings 2-4 short dialogues or 1-2 long ones together; ~6 in 10 open with a fixed greeting exchange
  - the user's pace is drawn once per call (2.2-3.2 five-letter words per second) with pauses after punctuation,
    occasional mid-sentence hesitations (0.6-1.8 s) and occasional "um" / "you know"
  - words reach the model ~0.5 s after they are spoken; long words are sometimes split across two ticks, punctuation
    sometimes arrives a tick late, letters are occasionally dropped
  - the assistant says three words per tick and starts on the tick after the user's last word; after a warning the
    calmer follow-up starts two ticks after both have finished
  - hazard warnings start on the tick carrying the dangerous phrase, over the user
  - when the user cuts in the assistant goes to <listen> on the tick where the cut-in is expected and listens two more
  - fillers ("Mm-hm.") at clause breaks once the user has talked >= 3.6 s in the turn, >= 4.8 s since the last one, and
    >= 4 words still to come
  - every tick gets a tag and the loss weight for that tag

    uv run python build_calls.py --dialogues a.jsonl b.jsonl --per 2 --out out/calls_train.jsonl
"""
import argparse
import math
import random
import re

from common import LISTEN, NOVOICE, WEIGHTS, persona_system, read_ids, read_jsonl, write_jsonl

TICK = 0.6
STT_DELAY = 0.5
# Streaming recogniser timing models:
#   kyutai   - each word arrives ~0.5 s after it is spoken (what the published adapter was trained on)
#   nemotron - words arrive in bursts at 0.56 s chunk boundaries, ~0.15 s after the chunk that completes them
#              (what the app's Nemotron 560 ms recogniser does); long words split less often
NEMO_CHUNK, NEMO_LAG = 0.56, 0.15
FILLERS = ["Mm-hm.", "Right.", "Uh-huh.", "Yeah.", "Okay.", "I see.", "Got it.", "Sure.", "Oh, okay.", "Gotcha."]
GREETINGS = [
    ("Hey, how are you?", "I'm good, thanks. What's on your mind?"),
    ("Hi there.", "Hi. How can I help?"),
    ("Hello?", "Hello. I'm here. Go ahead."),
    ("Hey, you there?", "Yes, I'm here. What's up?"),
    ("Good morning.", "Good morning. What can I do for you?"),
    ("Hi, got a second?", "Of course. What do you need?"),
]
# --greetings partner: a friend picking up. The service greetings above ("How can I help?", "What can I do for you?")
# opened ~6 in 10 partner calls up to g26, which then said "What can I help you with?" to break a silence.
PARTNER_GREETINGS = [
    ("Hey!", "Hey you! What's up?"),
    ("Hi, it's me.", "Oh hey! How's it going?"),
    ("Hey, you got a minute?", "Yeah, totally. What's going on?"),
    ("Hello?", "Hey! Good timing, I was just making tea."),
    ("Hey, how are you?", "Pretty good! You?"),
    ("Hi there.", "Hey hey. How are you doing?"),
    ("Yo.", "Yo! What's new?"),
    ("Hey, sorry, is this a bad time?", "Nah, it's fine. What's up?"),
]
REACTIONS = ["yeah", "mm-hm", "right", "okay", "uh-huh"]

# Speaker profiles: every call gets a caller with their own habits, so the model hears many kinds of talkers instead of
# one average one. Back-channels are what the caller says over the assistant's answers; fillers and restarts are inside
# their own sentences; a trail-off ends a statement ("so yeah."); pause_scale stretches their pauses.
BACKCHANNELS = ["yeah", "mm-hm", "right", "okay", "uh-huh", "sure", "got it", "mhm", "right, right", "yep", "oh okay",
                "ah", "uh huh", "totally", "for sure", "I see"]
USER_FILLERS = ["um", "uh", "um,", "uh,", "like,", "you know,", "I mean,", "so,", "kind of", "sort of", "basically,",
                "honestly,", "well,"]
TRAIL_OFFS = ["so yeah.", "and yeah.", "you know?", "or something.", "I guess.", "anyway.", "so..."]


def sample_profile(rng):
    talkative = rng.random()
    return {
        "pace": rng.uniform(2.0, 3.4) * 6.0,                          # characters (incl. space) per second
        "filler_rate": rng.choice([0.0, 0.01, 0.03, 0.05, 0.08]),     # per word
        "fillers": rng.sample(USER_FILLERS, rng.randint(2, 4)),
        "restart_rate": rng.choice([0.0, 0.0, 0.01, 0.025, 0.04]),    # "I I think", "the the"
        "trail_rate": rng.choice([0.0, 0.0, 0.05, 0.1]),              # per statement-ending turn
        "pause_scale": rng.uniform(0.75, 1.25),                       # run 4 used 0.6-1.6 and learned to wait too long
        "bc_rate": 0.0 if talkative < 0.2 else rng.choice([0.15, 0.3, 0.5, 0.8]),  # some callers never back-channel
        "bc_words": rng.sample(BACKCHANNELS, rng.randint(1, 3)),      # most people have one or two favourites
        "mirror": rng.random() < 0.4,                                # the assistant echoes the caller's back-channels
    }


def even_chunks(words, size=3):
    """The words in chunks of at most `size`, as even as they can be: 7 words as 3/2/2, not 3/3/1. Chunking by sentence
    (--sentence-pauses) left 10% of the friend's ticks one word long (4% before), and the model took a lone word for a
    sentence end and stopped after it (g26b3: a lone "I", then a long pause)."""
    n = -(-len(words) // size) or 1  # chunks
    base, extra = divmod(len(words), n)
    out, i = [], 0
    for k in range(n):
        m = base + (1 if k < extra else 0)
        out.append(words[i:i + m])
        i += m
    return out


class Call:
    """Accumulates heard fragments and assistant words per tick index."""
    greetings = GREETINGS  # set from --greetings
    fillers = FILLERS      # set from --filler-words
    filler_keep = 1.0      # set from --filler-keep: share of filler opportunities actually used
    hazard_keep = 1.0      # set from --hazard-keep: share of hazard turns built as hazards (the rest as statements)
    filler_min_talk = 3.6  # --filler-min-talk: seconds the user must have talked in this turn before a back-channel
    filler_gap = 4.8       # --filler-gap: seconds between two back-channels
    nudge_quiet = (4.0, 9.0)  # --nudge-quiet: seconds of silence before the partner picks the call back up
    nudge_unanswered = 0.0  # --nudge-unanswered: share of nudges the caller leaves unanswered for 7-15 s
    sentence_pauses = 0.0  # --sentence-pauses: share of the friend's sentence ends where it stops for a beat and carries on
    even_chunks = False  # --even-chunks: a sentence's chunks as even as they can be (7 words: 3/2/2, not 3/3/1)

    def __init__(self, rng, timing="kyutai"):
        self.rng = rng
        self.timing = timing
        self.heard = {}      # tick -> [fragments]
        self.said = {}       # tick -> (text, tag)
        self.listen_tag = {}  # tick -> tag for silent assistant ticks that are not the default
        self.user_spans = []  # (first tick, last tick) of every user turn: silent ticks inside are pauses, not idle
        self.profile = sample_profile(rng)
        self.pace = self.profile["pace"]
        self.t = 0.0          # current spoken time
        self.last_filler_t = -99.0
        self.turn_start_t = 0.0
        self.max_tick = 0

    def tick_of(self, t):
        if self.timing == "nemotron":
            arrival = math.ceil(t / NEMO_CHUNK) * NEMO_CHUNK + NEMO_LAG
            return int(arrival // TICK)
        return int((t + STT_DELAY) // TICK)

    def hear(self, tick, frag):
        self.heard.setdefault(tick, []).append(frag)
        self.max_tick = max(self.max_tick, tick)

    def say(self, tick, text, tag):
        self.said[tick] = (text, tag)
        self.max_tick = max(self.max_tick, tick)

    # ---------- user speech ----------
    def speak_user(self, text, allow_fillers=True, is_hazard_cut=None):
        """Speaks `text` from self.t on. Returns (first_tick, last_tick, cut_tick) where cut_tick is the tick carrying the
        end of the hazard phrase (if given), and leaves self.t at the end of the last word."""
        rng = self.rng
        words = text.split()
        if not words:
            return None, None, None
        # the caller's own habits: fillers, restarts ("I I think"), and trailing off at the end of a statement
        prof = self.profile
        out = []
        for w in words:
            if allow_fillers and len(out) > 1 and rng.random() < prof["filler_rate"]:
                out.append(rng.choice(prof["fillers"]))
            if len(out) > 0 and len(w) <= 5 and rng.random() < prof["restart_rate"]:
                out.append(re.sub(r"[.,!?;:]+$", "", w))
            out.append(w)
        if allow_fillers and out and out[-1].endswith(".") and rng.random() < prof["trail_rate"]:
            out.append(rng.choice(TRAIL_OFFS))
        words = out
        cut_idx = None
        if is_hazard_cut:
            cut_words = is_hazard_cut.split()
            plain = [re.sub(r"[^\w']", "", w).lower() for w in words]
            cplain = [re.sub(r"[^\w']", "", w).lower() for w in cut_words]
            for i in range(len(plain) - len(cplain), -1, -1):
                if plain[i:i + len(cplain)] == cplain:
                    cut_idx = i + len(cplain) - 1
                    break
            if cut_idx is None:
                cut_idx = min(len(words) - 1, max(0, int(len(words) * 0.6)))
        first_tick, last_tick, cut_tick = None, None, None
        self.turn_start_t = self.t
        late_punct = ""
        n_sent = 0
        for i, w in enumerate(words):
            dur = (len(w) + 1) / self.pace
            start, end = self.t, self.t + dur
            self.t = end
            emit = w
            # recogniser noise
            if len(w) >= 4 and rng.random() < 0.02:
                k = rng.randrange(1, len(w) - 1)
                emit = w[:k] + w[k + 1:]
            punct = ""
            m = re.search(r"[.,!?;:]+$", emit)
            # kyutai: punctuation now and then a tick late. nemotron: punctuation needs the next word as context, so it
            # usually arrives with the next word (". So"), and the last word of a turn usually never gets it at all.
            if m and rng.random() < (0.7 if self.timing == "nemotron" else 0.15):
                punct = m.group(0)
                emit = emit[: -len(punct)]
            tk_end = self.tick_of(end)
            if late_punct:
                emit = late_punct + " " + emit
                late_punct = ""
            # long words sometimes arrive split across two ticks
            tk_mid = self.tick_of(start + dur * 0.45)
            if len(w) >= 6 and tk_mid < tk_end and rng.random() < (0.5 if self.timing == "kyutai" else 0.25) and not punct:
                k = max(2, len(emit) // 2)
                self.hear(tk_mid, emit[:k])
                self.hear(tk_end, emit[k:])
            else:
                self.hear(tk_end, emit)
            if punct:
                late_punct = punct
            first_tick = first_tick if first_tick is not None else self.tick_of(start + dur * 0.45) if (len(w) >= 6 and tk_mid < tk_end) else tk_end
            if first_tick is None:
                first_tick = tk_end
            last_tick = tk_end
            if cut_idx is not None and i == cut_idx:
                cut_tick = tk_end
            # fillers at clause breaks
            remaining = len(words) - i - 1
            if allow_fillers and re.search(r"[,.;:]$", w) and (end - self.turn_start_t) >= self.filler_min_talk and (end - self.last_filler_t) >= self.filler_gap and remaining >= 4:
                if tk_end not in self.said and rng.random() < self.filler_keep:
                    filler = rng.choice(self.fillers)
                    if self.profile["mirror"] and self.fillers is FILLERS and rng.random() < 0.5:
                        bc = rng.choice(self.profile["bc_words"])  # echo the caller's own "right" / "yeah"
                        filler = bc[0].upper() + bc[1:].rstrip(".") + "."
                    self.say(tk_end, filler, "continuer")
                    self.last_filler_t = end
            # pauses
            if re.search(r"[.!?]$", w):
                self.t += rng.uniform(0.4, 0.8) * prof["pause_scale"]
                n_sent += 1
            elif re.search(r"[,;:]$", w):
                self.t += rng.uniform(0.15, 0.35) * prof["pause_scale"]
            if remaining >= 3 and rng.random() < 0.03:
                # hesitation in the middle of a thought: the tick with these words is listen_last, the gap listen_pause
                gap = rng.uniform(0.6, 1.8)
                self.listen_tag[tk_end] = "listen_last"
                for tk in range(tk_end + 1, self.tick_of(self.t + gap) + 1):
                    self.listen_tag.setdefault(tk, "listen_pause")
                self.t += gap
        if late_punct and (self.timing != "nemotron" or rng.random() < 0.1):
            self.hear(last_tick + 1, late_punct)
            last_tick += 1
        elif self.timing == "nemotron" and not late_punct and rng.random() < 0.9:
            # the last word's punctuation arrived with it; strip it, Nemotron would not have had it yet
            for tk in range(last_tick, -1, -1):
                frags = self.heard.get(tk)
                if frags:
                    frags[-1] = re.sub(r"[.,!?;:]+$", "", frags[-1]) or frags[-1]
                    break
        if first_tick is not None:
            self.user_spans.append((first_tick, last_tick))
        return first_tick, last_tick, cut_tick

    # ---------- assistant speech ----------
    def speak_assistant(self, text, start_tick, first_tag="start", rest_tag="speak", reactions=None, stop_after_words=None):
        """Three words per tick from start_tick. Returns the last tick spoken. `reactions` is a list of
        (after_words, text) the user says over the answer (keep_talking ticks). `stop_after_words` cuts the answer
        for a barge-in and returns (last_tick, True)."""
        words = text.split()
        if stop_after_words is not None:
            words = words[:stop_after_words]
        if Call.sentence_pauses:
            # chunks follow the sentences, so a beat can fall at a sentence end. On the recorded calls a speaker who ends
            # a sentence in silence runs straight on 70% of the time and stops for a beat 30%, one tick (93%) or two,
            # then carries on. The written calls had every friend turn as one unbroken run and the caller always
            # speaking next, so the model learned "sentence end = hand over" and only came back as a nudge seconds later.
            sentences, cur = [], []
            for w in words:
                cur.append(w)
                if re.search(r"[.!?]['\")]*$", w):
                    sentences.append(cur)
                    cur = []
            if cur:
                sentences.append(cur)
            chunks = []  # (text, beat before it, opens a sentence)
            for si, sent in enumerate(sentences):
                beat = 0 if si == 0 or self.rng.random() >= Call.sentence_pauses else (1 if self.rng.random() < 0.93 else 2)
                parts = even_chunks(sent) if Call.even_chunks else [sent[i:i + 3] for i in range(0, len(sent), 3)]
                chunks.extend((" ".join(p), beat if k == 0 else 0, si > 0 and k == 0) for k, p in enumerate(parts))
        else:
            chunks = [(" ".join(words[i:i + 3]), 0, False) for i in range(0, len(words), 3)]
        tk = start_tick
        said_words = 0
        reactions = sorted(reactions or [], key=lambda r: r[0])
        for ci, (ch, beat, opens) in enumerate(chunks):
            for _ in range(beat):  # its own beat at a sentence end: it says nothing and the caller stays quiet
                while tk in self.said:
                    tk += 1
                self.listen_tag[tk] = "speak_pause"
                tk += 1
            while tk in self.said:
                tk += 1
            # at a sentence end the two choices, a beat (speak_pause) or straight on (speak_next), weigh the same, so the
            # data's share of each sets the model's preference (g26b2 had the words at 0.5 against the beat at 1.0 and
            # paused at 76% of sentence ends; people 27%)
            tag = first_tag if ci == 0 else ("speak_on" if beat else "speak_next" if opens else rest_tag) if rest_tag == "speak" else rest_tag
            # user reaction over the answer
            while reactions and reactions[0][0] <= said_words:
                _, rtext = reactions.pop(0)
                self.hear(tk, rtext)
                tag = "keep_talking" if ci > 0 else tag
            self.say(tk, ch, tag)
            said_words += len(ch.split())
            tk += 1
        last = tk - 1
        # the assistant's spoken time: it starts at start_tick and speaks 3 words per tick
        self.t = max(self.t, (last + 1) * TICK - STT_DELAY)
        return last

    def finish(self):
        ticks = []
        last_said_tick = -1
        for tk in range(0, self.max_tick + 1):
            heard = " ".join(self.heard.get(tk, [])).strip() or NOVOICE
            if tk in self.said:
                text, tag = self.said[tk]
                if tag not in ("continuer",):
                    last_said_tick = tk
            else:
                text = LISTEN
                if tk in self.listen_tag:
                    tag = self.listen_tag[tk]
                elif heard == NOVOICE and any(a < tk < b for a, b in self.user_spans):
                    tag = "listen_pause"  # a gap between sentences of one turn: the user is not done
                elif heard != NOVOICE:
                    tag = self.listen_tag.get(tk, "listen_speech")
                elif 0 <= tk - last_said_tick <= 4 and last_said_tick >= 0 and tk != last_said_tick:
                    tag = "listen_done"
                else:
                    tag = "listen_idle"
            ticks.append({"heard": heard, "said": text, "w": WEIGHTS[tag], "tag": tag})
        return ticks


def build_call(dialogues, rng, with_greeting=True, timing="kyutai"):
    c = Call(rng, timing)
    c.t = rng.uniform(0.3, 1.5)
    if with_greeting and rng.random() < 0.6:
        u, a = rng.choice(Call.greetings)
        _, last, _ = c.speak_user(u, allow_fillers=False)
        end = c.speak_assistant(a, last + 1)
        c.t = (end + 1 + rng.randint(3, 7)) * TICK - STT_DELAY
    for d in dialogues:
        turns = d["turns"]
        i = 0
        while i < len(turns):
            t = turns[i]
            kind = t.get("kind", "question")
            user = (t.get("user") or "").strip()
            assistant = (t.get("assistant") or "").strip()
            if kind == "nudge" and assistant:
                # the line goes quiet (after a trailing "yeah..." or nothing), then the partner picks it back up
                if user:
                    _, last, _ = c.speak_user(user, allow_fillers=False)
                    base = last + 1
                else:
                    base = c.tick_of(c.t)
                quiet = round(rng.uniform(*Call.nudge_quiet) / TICK)
                end = c.speak_assistant(assistant, base + quiet, first_tag="nudge")
                if Call.nudge_unanswered and rng.random() < Call.nudge_unanswered:
                    # the caller does not answer for a while (busy, thinking, gone to get something): the friend has
                    # said its piece and waits, every tick of that silence a trained listen (g26m6 talked on alone)
                    wait = rng.randint(12, 25)
                    for tk in range(end + 1, end + 1 + wait):
                        c.listen_tag[tk] = "listen_wait"
                    end += wait
                c.t = (end + 1 + rng.randint(3, 7)) * TICK - STT_DELAY
                i += 1
                continue
            if not user:
                i += 1
                continue
            if kind == "false_memory" and assistant:
                # the caller talks as if the friend remembers something it does not; the friend says so, not plays along
                _, last, _ = c.speak_user(user)
                end = c.speak_assistant(assistant, last + 1, first_tag="false_memory_start", rest_tag="false_memory")
                c.t = (end + 1 + rng.randint(3, 7)) * TICK - STT_DELAY
                i += 1
                continue
            if kind == "hazard" and rng.random() >= Call.hazard_keep:
                kind = "statement"  # thin out hazards: the user's plan stays, the assistant answers it after they finish
            if kind == "hazard" and t.get("cut_after") and t.get("interjection"):
                first, last, cut = c.speak_user(user, is_hazard_cut=t["cut_after"])
                warn_end = c.speak_assistant(t["interjection"], cut, first_tag="hazard_start", rest_tag="hazard")
                if assistant:
                    start = max(last, warn_end) + 3
                    end = c.speak_assistant(assistant, start)
                else:
                    end = max(last, warn_end)
                c.t = (end + 1 + rng.randint(3, 7)) * TICK - STT_DELAY
            elif kind == "barge_in" and t.get("barge"):
                first, last, _ = c.speak_user(user)
                end = c.speak_assistant(assistant or "Well, yes, as long as", last + 1)
                # the user cuts in right after the answer's last chunk; the stop lands where the cut-in is expected
                stop_tick = end + 1
                c.say(stop_tick, LISTEN, "barge_stop")
                c.listen_tag[stop_tick + 1] = "barge_listen"
                c.listen_tag[stop_tick + 2] = "barge_listen"
                c.t = (stop_tick + 1) * TICK - STT_DELAY - rng.uniform(0.0, 0.5)
                _, last, _ = c.speak_user(t["barge"], allow_fillers=False)
                if t.get("barge_answer"):
                    # answer the cut-in right away, as a person would (the published data never did, and taught silence)
                    end = c.speak_assistant(t["barge_answer"], last + 1)
                    c.t = (end + 1 + rng.randint(3, 6)) * TICK - STT_DELAY
                else:
                    c.t = (last + 1 + rng.randint(2, 5)) * TICK - STT_DELAY
            elif kind == "resume" and t.get("barge") and t.get("side_answer") and t.get("resume"):
                # a long answer, cut into with a side question, answered, then picked back up once the user hands back
                first, last, _ = c.speak_user(user)
                n_words = len(assistant.split())
                cut = int(t.get("cut_at_words") or n_words // 3) + rng.randint(-2, 2)
                # the side question must be about something already said: if it names a word of the answer that comes
                # after the cut ("Wait, what wall?"), move the cut to just after that word
                plain = [re.sub(r"[^\w']", "", w).lower() for w in assistant.split()]
                stop = {"what", "wait", "does", "that", "this", "with", "about", "which", "where", "when", "there", "they",
                        "have", "mean", "sorry", "hold", "actually", "your", "from", "just", "like", "really"}
                keys = [k for k in (re.sub(r"[^\w']", "", w).lower() for w in t["barge"].split()) if len(k) >= 4 and k not in stop]
                hits = [plain.index(k) for k in keys if k in plain]
                if hits and min(hits) + 1 > cut:
                    cut = min(hits) + 2
                cut = max(4, min(cut, n_words - 6))
                end = c.speak_assistant(assistant, last + 1, stop_after_words=cut)
                stop_tick = end + 1
                c.say(stop_tick, LISTEN, "barge_stop")
                c.listen_tag[stop_tick + 1] = "barge_listen"
                c.listen_tag[stop_tick + 2] = "barge_listen"
                c.t = (stop_tick + 1) * TICK - STT_DELAY - rng.uniform(0.0, 0.5)
                _, last, _ = c.speak_user(t["barge"], allow_fillers=False)
                end = c.speak_assistant(t["side_answer"], last + 1)
                c.t = (end + 1 + rng.randint(1, 3)) * TICK - STT_DELAY
                if rng.random() < 0.65 and t.get("ack"):
                    _, last, _ = c.speak_user(t["ack"], allow_fillers=False)  # "Okay." / "Got it, go on."
                    resume_at = last + 1 + rng.randint(0, 2)
                else:
                    resume_at = end + 1 + rng.randint(3, 5)  # the user just goes quiet: pick it up after a beat
                end = c.speak_assistant(t["resume"], resume_at)
                c.t = (end + 1 + rng.randint(3, 7)) * TICK - STT_DELAY
            elif kind == "story" and not assistant:
                first, last, _ = c.speak_user(user)
                # the user is not done: a pause, then the next part
                c.listen_tag[last] = "listen_last"
                gap_ticks = rng.randint(1, 3)
                for tk in range(last + 1, last + 1 + gap_ticks):
                    c.listen_tag[tk] = "listen_pause"
                c.t = (last + 1 + gap_ticks) * TICK - STT_DELAY
            else:
                # a reaction "after 0 words" would land on the answer's first tick, before there is anything to react to
                reactions = [(r["after_words"], r["text"]) for r in t.get("user_reactions", []) if isinstance(r, dict) and isinstance(r.get("after_words"), int) and r["after_words"] >= 3 and r.get("text")]
                n = len(assistant.split())
                if not reactions and n >= 12 and rng.random() < c.profile["bc_rate"]:
                    # the caller's own back-channels over the answer: one, or two on a long answer from a frequent one
                    k = 2 if n >= 30 and c.profile["bc_rate"] >= 0.5 and rng.random() < 0.5 else 1
                    spots = sorted(rng.sample(range(6, max(8, n - 3)), k))
                    reactions = [(at, rng.choice(c.profile["bc_words"])) for at in spots]
                first, last, _ = c.speak_user(user)
                if not assistant:
                    c.t = (last + 1 + rng.randint(2, 5)) * TICK - STT_DELAY
                    i += 1
                    continue
                interrupt = t.get("interrupt") if isinstance(t.get("interrupt"), dict) else None
                if interrupt and isinstance(interrupt.get("after_words"), int) and interrupt.get("text"):
                    end = c.speak_assistant(assistant, last + 1, reactions=reactions, stop_after_words=interrupt["after_words"])
                    stop_tick = end + 1
                    c.say(stop_tick, LISTEN, "barge_stop")
                    c.listen_tag[stop_tick + 1] = "barge_listen"
                    c.listen_tag[stop_tick + 2] = "barge_listen"
                    c.t = (stop_tick + 1) * TICK - STT_DELAY - rng.uniform(0.0, 0.5)
                    _, last, _ = c.speak_user(interrupt["text"], allow_fillers=False)
                    if interrupt.get("answer"):
                        end = c.speak_assistant(interrupt["answer"], last + 1)
                        c.t = (end + 1 + rng.randint(3, 6)) * TICK - STT_DELAY
                    else:
                        c.t = (last + 1 + rng.randint(2, 5)) * TICK - STT_DELAY
                else:
                    # a reply from the friend's memories gets tags of its own (weighted like any answer; a recipe can
                    # train persona calls on just these and the false-memory replies). Same timing, same random draws.
                    tags = {"first_tag": "recall_start", "rest_tag": "recall"} if kind == "recall" else {}
                    end = c.speak_assistant(assistant, last + 1, reactions=reactions, **tags)
                    c.t = (end + 1 + rng.randint(3, 7)) * TICK - STT_DELAY
            i += 1
    return c.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", nargs="+", required=True)
    ap.add_argument("--per", type=int, default=2, help="how many calls each dialogue appears in")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ticks", type=int, default=520, help="keep calls under ~8k tokens")
    ap.add_argument("--heldout", type=float, default=0.0, help="fraction of dialogue ids set aside as eval calls")
    ap.add_argument("--timing", choices=["kyutai", "nemotron", "mixed"], default="kyutai",
                    help="recogniser timing to simulate; mixed draws one per call")
    ap.add_argument("--exclude-calls", nargs="*", default=[],
                    help="calls files (e.g. the published held-out calls) whose dialogue ids must not be used")
    ap.add_argument("--exclude-ids", default=None, help="file of dialogue ids never to use (e.g. data/exclude_dialogues.txt)")
    ap.add_argument("--only-from-calls", nargs="*", default=[], help="use only the dialogues these calls use (to build a held-out set)")
    ap.add_argument("--hazard-boost", type=int, default=0, help="extra calls for every dialogue with a hazard turn")
    ap.add_argument("--filler-words", default=None, help="the assistant's back-channels, separated by |")
    ap.add_argument("--filler-keep", type=float, default=1.0, help="share of back-channel opportunities used (personality)")
    ap.add_argument("--filler-min-talk", type=float, default=3.6, help="seconds the user talks in a turn before a back-channel")
    ap.add_argument("--filler-gap", type=float, default=4.8, help="seconds between back-channels")
    ap.add_argument("--nudge-quiet", default="4,9", help="seconds of silence (min,max) before a nudge turn")
    ap.add_argument("--hazard-keep", type=float, default=1.0, help="share of hazard turns kept as hazards (keeps warning density realistic)")
    ap.add_argument("--greetings", choices=["service", "partner"], default="service", help="how the assistant picks up (partner: a friend)")
    ap.add_argument("--nudge-unanswered", type=float, default=0.0, help="share of nudges the caller leaves unanswered for 7-15 s (the friend waits)")
    ap.add_argument("--sentence-pauses", type=float, default=0.0, help="share of the friend's sentence ends where it stops for a beat (1-2 ticks) and carries on (real calls: 0.3)")
    ap.add_argument("--even-chunks", action="store_true", help="a sentence's chunks as even as they can be (7 words: 3/2/2), with --sentence-pauses")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if a.filler_words:
        Call.fillers = [w.strip() for w in a.filler_words.split("|") if w.strip()]
    Call.filler_keep = a.filler_keep
    Call.filler_min_talk, Call.filler_gap = a.filler_min_talk, a.filler_gap
    Call.nudge_quiet = tuple(float(x) for x in a.nudge_quiet.split(","))
    Call.hazard_keep = a.hazard_keep
    Call.greetings = PARTNER_GREETINGS if a.greetings == "partner" else GREETINGS
    Call.nudge_unanswered = a.nudge_unanswered
    Call.sentence_pauses = a.sentence_pauses
    Call.even_chunks = a.even_chunks
    dialogues = []
    for f in a.dialogues:
        dialogues += read_jsonl(f)
    excluded = {i for f in a.exclude_calls for c in read_jsonl(f) for i in c.get("dialogues", [])} | read_ids(a.exclude_ids)
    if excluded:
        n0 = len(dialogues)
        dialogues = [d for d in dialogues if d["id"] not in excluded]
        print(f"excluded {n0 - len(dialogues)} dialogues (held-out calls {a.exclude_calls}, id list {a.exclude_ids})")
    if a.only_from_calls:
        keep = {i for f in a.only_from_calls for c in read_jsonl(f) for i in c.get("dialogues", [])}
        dialogues = [d for d in dialogues if d["id"] in keep]
        print(f"kept only the {len(dialogues)} dialogues used by {a.only_from_calls}")
    timing = (lambda: rng.choice(["kyutai", "nemotron"])) if a.timing == "mixed" else (lambda: a.timing)
    # a dialogue with its own system prompt (a persona and its memories, gen_dialogues --kind memory) is a call of its
    # own, trained with that prompt: the friend in it is nobody else's
    own = [d for d in dialogues if d.get("system")]
    dialogues = [d for d in dialogues if not d.get("system")]
    own_calls = []
    for d in [d for d in own for _ in range(a.per)]:
        tm = timing()
        own_calls.append({"split": "train", "dialogues": [d["id"]], "timing": tm, "ticks": build_call([d], rng, timing=tm),
                          # the prompt as common.persona_system writes it now (its wording changed at g26m5), not as stored
                          "system": persona_system(d["persona"], d.get("memories", [])) if d.get("persona") else d["system"],
                          "memories": d.get("memories", []), "focus": d.get("focus"),
                          "claim": next((t["user"] for t in d["turns"] if t.get("kind") == "false_memory"), "")})
    has_hazard = lambda d: any(t.get("kind") == "hazard" and t.get("cut_after") and t.get("interjection") for t in d["turns"])
    short = [d for d in dialogues if not any(t.get("kind") in ("explain", "chat") for t in d["turns"])]
    long_ = [d for d in dialogues if d not in short]
    print(f"{len(short)} short and {len(long_)} long dialogues")
    rng.shuffle(short)
    rng.shuffle(long_)
    n_hs, n_hl = int(len(short) * a.heldout), int(len(long_) * a.heldout)
    splits = {"eval": (short[:n_hs], long_[:n_hl]), "train": (short[n_hs:], long_[n_hl:])}
    calls = own_calls
    for split, (S, L) in splits.items():
        pool_s = [d for d in S for _ in range(a.per + (a.hazard_boost if has_hazard(d) else 0))]
        pool_l = [d for d in L for _ in range(a.per + (a.hazard_boost if has_hazard(d) else 0))]
        rng.shuffle(pool_s)
        rng.shuffle(pool_l)
        while pool_s:
            k = min(len(pool_s), rng.randint(2, 4))
            group, pool_s = pool_s[:k], pool_s[k:]
            tm = timing()
            for _ in range(5):
                ticks = build_call(group, rng, timing=tm)
                if len(ticks) <= a.max_ticks or len(group) == 1:
                    break
                group = group[:-1]
            calls.append({"split": split, "dialogues": [d["id"] for d in group], "timing": tm, "ticks": ticks})
        while pool_l:
            k = min(len(pool_l), rng.randint(1, 2))
            group, pool_l = pool_l[:k], pool_l[k:]
            tm = timing()
            ticks = build_call(group, rng, timing=tm)
            if len(ticks) > a.max_ticks and len(group) > 1:
                ticks = build_call(group[:1], rng, timing=tm)
                group = group[:1]
            calls.append({"split": split, "dialogues": [d["id"] for d in group], "timing": tm, "ticks": ticks})
    rng.shuffle(calls)
    write_jsonl(a.out, calls)
    tags = {}
    n = 0
    for c in calls:
        for t in c["ticks"]:
            tags[t["tag"]] = tags.get(t["tag"], 0) + 1
            n += 1
    print(f"wrote {len(calls)} calls, {n} ticks to {a.out}")
    for k, v in sorted(tags.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14} {v:7d}  {100 * v / n:5.1f}%")


if __name__ == "__main__":
    main()
