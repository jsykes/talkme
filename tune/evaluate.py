"""Closed-loop replay of held-out calls against a running llama-server.

For every tick the model sees the call's `heard` text and its OWN earlier replies (not the reference ones), exactly as
in a live call. Its replies are then scored against the reference tags with the same measures as the model card:

  start      answer begins within one tick of the reference `start` tick
  hazard     a warning begins within two ticks of the reference `hazard_start` tick
  barge      the model goes to <listen> within one tick of the reference `barge_stop` tick
  filler     a short filler is said within one tick of a reference `continuer` tick (and how many fillers overall)
  over-talk  share of listen_speech / listen_pause ticks on which the model spoke
  unfinished model utterances that end without terminal punctuation and without a barge-in
  carry      it carries on within a tick of a reference `speak_on` tick (its own beat at a sentence end, calls built with --sentence-pauses)

    uv run python evaluate.py --url http://127.0.0.1:8080 --calls ../models/lora/dataset/calls_heldout.jsonl --n 4
"""
import argparse
import json
import re
import time

from common import LISTEN, NOVOICE, SYSTEMS, MicroTurnClient, persona_system, read_jsonl

FILLER_RE = re.compile(r"^(mm-?hm|uh-?huh|right|yeah|okay|ok|I see|got it|sure|oh)[.!,]?$", re.I)


def near(tags_or_flags, i, lo, hi):
    return any(0 <= j < len(tags_or_flags) and tags_or_flags[j] for j in range(i + lo, i + hi + 1))


def replay(client, call, log=None, history_ticks=0):
    history, said_list, p_list = [], [], []
    t_sum = 0.0
    for k, tick in enumerate(call["ticks"]):
        t0 = time.time()
        said, p, _ = client.tick(history, tick["heard"])
        said = client.guard(history, said, tick["heard"])
        t_sum += time.time() - t0
        history += [("user", tick["heard"]), ("assistant", said)]
        if history_ticks and len(history) // 2 > history_ticks:  # as the app's loop (loop.js trimPoint): drop the oldest
            want = (len(history) * 2 // 10) * 2  # 40 %, then on to where the caller starts talking into a silence
            cut = next((i for i in range(want, min(len(history), want + 120), 2)
                        if history[i][1] != NOVOICE and history[i - 2][1] == NOVOICE and history[i - 1][1] == LISTEN), want)
            del history[:cut]
        said_list.append(said)
        p_list.append(p)
        if log and (said != LISTEN or tick["said"] != LISTEN):
            log(f"{k:5d} {tick['tag']:<14} heard={tick['heard']!r:<24} ref={tick['said']!r:<24} model={said!r:<24} P={p:.2f}")
    return said_list, p_list, t_sum / max(1, len(call["ticks"]))


def score(call, said):
    ticks = call["ticks"]
    tags = [t["tag"] for t in ticks]
    spoke = [s != LISTEN for s in said]
    fill = [bool(FILLER_RE.match(s)) for s in said]
    m = {"start": [0, 0], "hazard": [0, 0], "barge": [0, 0], "filler": [0, 0], "over": [0, 0], "unfinished": [0, 0], "carry": [0, 0]}
    for i, tag in enumerate(tags):
        if tag in ("start", "false_memory_start", "recall_start"):
            m["start"][1] += 1
            m["start"][0] += near(spoke, i, -1, 1)
        elif tag == "speak_on":  # carries on after its own beat at a sentence end (within a tick of the reference)
            m["carry"][1] += 1
            m["carry"][0] += near(spoke, i, -1, 1)
        elif tag == "hazard_start":
            m["hazard"][1] += 1
            m["hazard"][0] += near(spoke, i, 0, 2)
        elif tag == "barge_stop":
            m["barge"][1] += 1
            m["barge"][0] += (not spoke[i]) or (i + 1 < len(spoke) and not spoke[i + 1])
        elif tag == "continuer":
            m["filler"][1] += 1
            m["filler"][0] += near(fill, i, -1, 1)
        if tag in ("listen_speech", "listen_pause"):
            m["over"][1] += 1
            m["over"][0] += spoke[i] and not fill[i]
    # utterances: runs of speaking ticks; unfinished if the run ends without .!? and the user was not talking
    i = 0
    while i < len(said):
        if spoke[i] and not fill[i]:
            j = i
            while j + 1 < len(said) and spoke[j + 1]:
                j += 1
            m["unfinished"][1] += 1
            user_cut_in = any(ticks[k]["heard"] != NOVOICE for k in range(max(i, j - 1), min(len(ticks), j + 2)))
            if not re.search(r"[.!?]['\"]?$", said[j]) and not user_cut_in:
                m["unfinished"][0] += 1
            i = j + 1
        else:
            i += 1
    m["fillers_said"] = sum(fill)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--calls", required=True)
    ap.add_argument("--n", type=int, default=4, help="how many calls to replay (0 = all)")
    ap.add_argument("--listen-p", type=float, default=0.3)
    ap.add_argument("--lora-scale", type=float, default=None, help="0 = evaluate the base model without the adapter")
    ap.add_argument("--adapter", default=None, help="path of a loaded adapter to evaluate (the others get scale 0); 'none' = base model")
    ap.add_argument("--max-ticks", type=int, default=0, help="cut each call after this many ticks (0 = whole call)")
    ap.add_argument("--out", default="out/eval.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--parallel", type=int, default=1, help="replay this many calls at once (start the server with -np >= this)")
    ap.add_argument("--stratified", type=int, default=0, help="replay only the first K short and the first K long calls")
    ap.add_argument("--plain", action="store_true", help="generate every reply in full (default: decide on the first token)")
    ap.add_argument("--sample", action="store_true", help="listen with the model's own P(listen) instead of the --listen-p threshold")
    ap.add_argument("--speak-opts", default="", help='decoding options for spoken words, JSON, e.g. {"temperature":0.7,"repeat_penalty":1.3,"dry_multiplier":0.8}')
    ap.add_argument("--carry-on-p", type=float, default=0.0, help="P(listen) needed to stop right after its own unfinished sentence into a silence (the app's carry-on bar, 0.9); 0 = the threshold")
    ap.add_argument("--history-ticks", type=int, default=0, help="keep at most this many ticks of the call, as the app does for slower models (Gemma 4: 150)")
    ap.add_argument("--format", default="qwen3.6", help="the base model's chat format (profiles.FORMATS): qwen3.6 or gemma4")
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="assistant", help="system prompt the adapter was trained with")
    ap.add_argument("--persona", default=None, help="a JSON file {persona, memories}: test with that persona prompt (common.persona_system), as the app sends persona adapters")
    ap.add_argument("--repeat-guard", action="store_true", help="turn a reply that repeats one of the last two spoken replies into <listen> (the app does)")
    a = ap.parse_args()

    calls = read_jsonl(a.calls)
    indices = list(range(len(calls)))
    if a.stratified:
        long_ = [i for i, c in enumerate(calls) if any(d >= 10000 for d in c.get("dialogues", []))]
        short = [i for i in indices if i not in long_]
        indices = sorted(short[: a.stratified] + long_[: a.stratified])
    elif a.n:
        indices = indices[: a.n]
    calls = [calls[i] for i in indices]
    MicroTurnClient(a.url, listen_p=a.listen_p, lora_scale=a.lora_scale, adapter=a.adapter, fast=not a.plain)  # fail early on a missing adapter
    totals = {}
    per_call = []
    replies = []

    def run_one(ci, call):
        ticks = call["ticks"][: a.max_ticks] if a.max_ticks else call["ticks"]
        call = {**call, "ticks": ticks}
        # one client per call; the server keeps a prompt cache per slot, so each call stays on its own slot
        cl = MicroTurnClient(a.url, listen_p=a.listen_p, lora_scale=a.lora_scale, adapter=a.adapter, fast=not a.plain, sample=a.sample, seed=ci)
        cl.repeat_guard = a.repeat_guard
        cl.system = call.get("system") or SYSTEMS[a.system]  # a persona call carries its own prompt
        if a.persona and not call.get("system"):
            p = json.load(open(a.persona, encoding="utf-8"))
            cl.system = persona_system(p["persona"], p.get("memories", []))
        cl.format = a.format
        cl.speak_opts = json.loads(a.speak_opts) if a.speak_opts else None
        cl.carry_on_p = a.carry_on_p
        said, p, avg = replay(cl, call, log=(print if a.verbose else None), history_ticks=a.history_ticks)
        m = score(call, said)
        m["avg_tick_s"] = round(avg, 3)
        return ci, call, m, said, p

    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = [None] * len(calls)
    with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as ex:
        for fut in as_completed([ex.submit(run_one, ci, c) for ci, c in enumerate(calls)]):
            ci, call, m, said, p = fut.result()
            done[ci] = m
            replies.append({"call": ci, "said": said, "p_listen": [round(x, 3) for x in p]})
            print(f"call {ci + 1}/{len(calls)}: {len(call['ticks'])} ticks, dialogues {call.get('dialogues')}", flush=True)
            print("   " + "  ".join(f"{k}={v[0]}/{v[1]}" if isinstance(v, list) else f"{k}={v}" for k, v in m.items()), flush=True)
    for m in done:
        per_call.append(m)
        for k, v in m.items():
            if isinstance(v, list):
                t = totals.setdefault(k, [0, 0])
                t[0] += v[0]
                t[1] += v[1]
            else:
                totals[k] = totals.get(k, 0) + v
    print("\n== totals ==")
    for k, v in totals.items():
        if isinstance(v, list):
            pct = f" ({100 * v[0] / v[1]:.0f}%)" if v[1] else ""
            print(f"{k:<12} {v[0]}/{v[1]}{pct}")
        else:
            print(f"{k:<12} {v}")
    with open(a.out, "w", encoding="utf-8") as f:
        # every reply and its P(listen) are kept for later analysis
        json.dump({"totals": totals, "calls": per_call, "indices": indices, "args": vars(a), "replies": sorted(replies, key=lambda r: r["call"])}, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
