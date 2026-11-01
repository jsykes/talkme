"""Shared protocol constants and the micro-turn client used by every tune step."""
import json
import math
import os
import random
import re
import requests

TICK = 0.6
LISTEN, NOVOICE = "<listen>", "<no voice>"
# The adapter was trained on this exact text. Keep it byte for byte.
SYSTEM = ("You are a voice assistant in a live phone call; both sides can talk at the same time. Every "
          f"{TICK:g} seconds you get what the microphone heard in that moment, as a streaming transcript fragment, or "
          f"{NOVOICE}. You reply with up to three new words to say next, or {LISTEN} to add nothing.")
# The partner adapters (run 9 on) are trained with this one instead; the app sends whichever an adapter was trained with.
PARTNER_SYSTEM = ("You are a friend on a live phone call; both of you can talk at the same time. Every "
                  f"{TICK:g} seconds you get what the microphone heard in that moment, as a streaming transcript fragment, "
                  f"or {NOVOICE}. You reply with up to three new words to say next, or {LISTEN} to add nothing.")
SYSTEMS = {"assistant": SYSTEM, "partner": PARTNER_SYSTEM}


def persona_system(persona, memories):
    """The partner prompt with who the friend is and what it remembers about the caller from earlier calls (g26m on).
    The app builds the same text (src/main/config.js personaPrompt); keep the two byte for byte."""
    s = f"{PARTNER_SYSTEM}\n\nYou are {persona.strip()}"
    if memories:
        return (s + "\nWhat you remember about the caller from earlier calls:\n" + "\n".join(f"- {m.strip()}" for m in memories)
                + "\nOnly bring up a shared past that is in this list.")
    # g26m5 on: "You don't remember anything about the caller from before this call" read as a stranger ("I don't recall
    # who you are. Is this a prank call?"); the friend knows the caller, just nothing about their life yet
    return s + "\nYou know the caller as a friend, but nothing about their life from earlier calls."

# loss weights per tick tag, from the dataset card
# the app's back-channel cap (src/main/llama.js BACKCHANNEL): at most one lone back-channel over the caller per ~10 s
BACKCHANNEL = re.compile(r"^(mm-?hm|mhm|uh-?huh|yeah|yep|right|okay|ok|i see|got it|sure|huh|totally|really\??|oh( (yeah\??|wow|nice|no|really\??))?|ha,? yeah|wow)[.!?,]*$", re.I)
SENTENCE_END = re.compile(r"[.!?]['\")]*$")  # a tick's words that end a sentence (the carry-on bar applies after the others)


def spoken(out):
    """A tick's words as spoken: cut at the model's own tag if it wrote one mid-reply ("How would you define '<listen>",
    2026-10-28: the voice read it out), with an opening quote or bracket left hanging dropped; nothing left is a listen.
    Mirrors llama.js spoken()."""
    out = re.sub(r"\s+[‘“\"'(]+$", "", out.split("<", 1)[0].rstrip())
    return out.strip() or LISTEN
BACKCHANNEL_GAP_TICKS = 17

WEIGHTS = {
    "listen_speech": 0.3, "speak": 2.0, "listen_idle": 0.2, "listen_done": 1.5, "listen_pause": 2.0,
    "listen_last": 0.5, "continuer": 1.5, "start": 6.0, "keep_talking": 2.0, "barge_listen": 1.0,
    "hazard": 1.5, "barge_stop": 3.0, "hazard_start": 4.0,
    "nudge": 3.5,  # the partner speaks first after a silence
    # the caller talks as if the friend remembers something that is not in its memories; the friend does not play along
    "false_memory_start": 6.0, "false_memory": 2.2,
    # the caller brings up something the friend remembers, and it answers from the memory (weighted like an answer)
    "recall_start": 6.0, "recall": 2.0,
    # the caller leaves a nudge unanswered: the friend keeps quiet through the silence instead of talking on alone
    "listen_wait": 1.0,
    # the friend's own beat at a sentence end (it says nothing for a tick or two, the caller quiet), carrying on after it,
    # and opening the next sentence straight away instead (weighed like the beat: the data's shares set the preference)
    "speak_pause": 1.0, "speak_on": 3.0, "speak_next": 1.0,
}


def render(history, heard=None, system=None, fmt="qwen3.6"):
    """The prompt for /completion in the base model's chat format (profiles.FORMATS: ChatML for Qwen, <|turn> for
    Gemma 4). history: list of (role, content). system: the system prompt (default SYSTEM)."""
    from profiles import FORMATS
    return FORMATS[fmt].prompt(system or SYSTEM, history, heard, bos=False)  # llama-server adds the BOS itself


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def read_ids(path):
    """Dialogue ids from a file like data/exclude_dialogues.txt (whitespace separated, # comments)."""
    if not path:
        return set()
    return {int(x) for line in open(path, encoding="utf-8") if not line.lstrip().startswith("#") for x in line.split()}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class MicroTurnClient:
    @property
    def end(self):
        """the marker that closes a reply in this client's chat format"""
        from profiles import FORMATS
        return FORMATS[self.format].end

    """One micro-turn per tick() against llama-server, with the P(listen) threshold rule."""

    def __init__(self, url, listen_p=0.3, lora_scale=None, timeout=120, adapter=None, fast=False, sample=False, seed=0):
        self.url = url.rstrip("/")
        self.listen_p = listen_p
        # sample=True: listen with exactly the model's P(listen) instead of the threshold rule. A model trained on real
        # people spreads "start talking" over several ticks, so it is rarely confident enough to clear a threshold.
        self.sample = sample
        self.rng = random.Random(seed)
        self.repeat_guard = False
        # carry_on_p: the P(listen) needed to stop right after its own words that did not end a sentence, the line
        # quiet (the app's carry-on bar, llama.js tick); 0 = the threshold throughout
        self.carry_on_p = 0.0
        self.system = SYSTEM  # PARTNER_SYSTEM for adapters trained with it
        self.format = "qwen3.6"  # the base model's chat format (profiles.FORMATS); set .format to switch
        # speak_opts: decoding options for the words of a speaking tick (e.g. temperature, repeat_penalty,
        # dry_multiplier). The listen/speak decision itself stays greedy on the first token; when set, a speaking tick
        # regenerates its words from the prompt with "<" banned and these options (one extra request).
        self.speak_opts = None
        self.fast = fast
        self.timeout = timeout
        self.s = requests.Session()
        # Per-request adapter scales. None = whatever the server applies by default. adapter=<path> picks one of the
        # loaded adapters (the rest get 0), adapter="none" = base model, lora_scale sets every loaded adapter's scale.
        self.lora = None
        if adapter is not None or lora_scale is not None:
            loaded = self.s.get(self.url + "/lora-adapters", timeout=10).json()
            # the server reports paths as given on its command line, relative to its own folder: compare file names
            name = lambda p: os.path.normcase(os.path.basename(p))
            if adapter is not None and ":" in adapter.replace(":\\", "").replace(":/", ""):
                # a blend: "run2.gguf:0.5,warm.gguf:0.5" (scales per adapter file name, others 0)
                want = {}
                for part in adapter.split(","):
                    f, sc = part.rsplit(":", 1)
                    want[name(f)] = float(sc)
                missing = [f for f in want if not any(name(a["path"]) == f for a in loaded)]
                if missing:
                    raise SystemExit(f"blend adapters not loaded: {missing}; loaded: {[a['path'] for a in loaded]}")
                self.lora = [{"id": a["id"], "scale": want.get(name(a["path"]), 0.0)} for a in loaded]
            elif adapter is not None and adapter != "none":
                hits = [a for a in loaded if name(a["path"]) == name(adapter)]
                if len(hits) != 1:
                    raise SystemExit(f"adapter {adapter} is not loaded (or ambiguous); loaded: {[a['path'] for a in loaded]}")
                self.lora = [{"id": a["id"], "scale": 1.0 if a is hits[0] else 0.0} for a in loaded]
            else:
                s = 0.0 if adapter == "none" else lora_scale
                self.lora = [{"id": a["id"], "scale": s} for a in loaded]

    def guard(self, history, said, heard=None):
        """Repeat guard and back-channel cap (as in the app, src/main/llama.js): a reply that is word for word what it said
        on one of the last two ticks is a decoding loop, and a lone back-channel over the caller within ~10 s of the
        previous one is one too many; listen instead."""
        if not self.repeat_guard or said == LISTEN:
            return said
        if heard is not None and heard != NOVOICE and BACKCHANNEL.match(said.strip()):
            mine = [c for r, c in history if r == "assistant"][-BACKCHANNEL_GAP_TICKS:]
            if any(c != LISTEN and BACKCHANNEL.match(c.strip()) for c in mine):
                return LISTEN
        norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", s.lower())).strip()
        # back to back: anything (a "Yeah." loop); within the last 20 ticks (~12 s): any chunk of two or more words.
        # Real people never repeat a 3+ word reply within four replies (0 of 421 on 24 recorded calls); the adapters did
        # in 27-67% of them. A lone "Yeah." after a pause is speech and passes.
        mine = [norm(c) for r, c in history if r == "assistant"]
        n = norm(said)
        last2 = [c for c in mine[-2:] if c != norm(LISTEN)]
        recent = [c for c in mine[-20:] if c != norm(LISTEN)]
        return LISTEN if n in last2 or (len(n.split()) >= 2 and n in recent) else said

    def bar(self, history, heard):
        """This tick's listen threshold: carry_on_p right after its own unfinished sentence into a silence."""
        if self.carry_on_p > self.listen_p and heard == NOVOICE and history and history[-1][0] == "assistant" \
                and history[-1][1] != LISTEN and not SENTENCE_END.search(history[-1][1].strip()):
            return self.carry_on_p
        return self.listen_p

    def complete(self, body):
        if self.lora is not None:
            body = {**body, "lora": self.lora}
        # a busy machine can stall a connection now and then; retry before giving up on a long replay
        for attempt in range(4):
            try:
                r = self.s.post(self.url + "/completion", json=body, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 3:
                    raise
                self.s = requests.Session()

    def tick(self, history, heard):
        if self.fast:
            return self.tick_fast(history, heard)
        body = {"prompt": render(history, heard, self.system, self.format), "n_predict": 64, "temperature": 0, "stop": [self.end],
                "cache_prompt": True, "n_probs": 8}
        j = self.complete(body)
        out = j["content"].strip()
        top = j["completion_probabilities"][0]["top_logprobs"]
        listen_ids = [p["id"] for p in top if p["token"].startswith("<")]
        p_listen = sum(math.exp(p["logprob"]) for p in top if p["token"].startswith("<"))
        rerun = False
        if out.startswith("<") and p_listen < self.bar(history, heard) and listen_ids:
            rerun = True
            body.update({"logit_bias": [[i, False] for i in listen_ids], "n_probs": 0})
            out = self.complete(body)["content"].strip()
        return spoken(out), p_listen, rerun

    def tick_fast(self, history, heard):
        """Same decisions and text as tick(), with less decoding. Three ticks in four are <listen>, and the decision
        is made on the first token alone, so ask for one token first. Only a speaking tick asks for the rest, and it
        continues after that first token rather than resending the bare prompt: the model's recurrent layers make
        rolling the prompt cache back even one token expensive."""
        prompt = render(history, heard, self.system, self.format)
        base = {"temperature": 0, "stop": [self.end], "cache_prompt": True}
        j = self.complete({**base, "prompt": prompt, "n_predict": 1, "n_probs": 8})
        first = j["content"]
        top = j["completion_probabilities"][0]["top_logprobs"]
        listen_ids = [p["id"] for p in top if p["token"].startswith("<")]
        p_listen = sum(math.exp(p["logprob"]) for p in top if p["token"].startswith("<"))
        def speak():
            ban = {"logit_bias": [[i, False] for i in listen_ids]} if listen_ids else {}
            o = self.complete({**base, **self.speak_opts, "prompt": prompt, "n_predict": 64, **ban})["content"].strip()
            return spoken(o), p_listen, True
        if self.sample and listen_ids:
            if self.rng.random() < p_listen:
                return LISTEN, p_listen, False
            if self.speak_opts:
                return speak()
            if first.startswith("<"):  # the draw says speak although "<" is the top token: ban it
                out = self.complete({**base, "prompt": prompt, "n_predict": 64, "logit_bias": [[i, False] for i in listen_ids]})["content"].strip()
                return spoken(out), p_listen, True
        elif first.startswith("<"):
            if p_listen >= self.bar(history, heard) or not listen_ids:
                return LISTEN, p_listen, False
            if self.speak_opts:
                return speak()
            # below the threshold: ban "<" and make the model speak (this one does roll back a token)
            out = self.complete({**base, "prompt": prompt, "n_predict": 64, "logit_bias": [[i, False] for i in listen_ids]})["content"]
            out = out.strip()
            return spoken(out), p_listen, True
        if self.speak_opts:
            return speak()
        rest = self.complete({**base, "prompt": prompt + first, "n_predict": 63})["content"]
        out = (first + rest).strip()
        return spoken(out), p_listen, False

    def chat(self, messages, temperature=0.8, max_tokens=2048, json_mode=False):
        """Ordinary chat completion (used to write dialogues with the base model)."""
        body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.lora is not None:
            body["lora"] = self.lora
        r = self.s.post(self.url + "/v1/chat/completions", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
