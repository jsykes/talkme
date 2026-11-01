"""Two failure modes the model-card measures miss, computed from the replies evaluate.py saves.

  unprompted  the model starts an utterance (not a filler, not a follow-up within 8 ticks of its own last one) in
              silence, when the reference has nobody talking and no
              answer due (listen_idle, or listen_done more than two ticks after the reference answer ended): false
              alarms and random remarks. Reported per hour of call.
  repeats     the model says nearly the same sentence again within 20 ticks (warning loops). Reported per hour.

    uv run python extra_metrics.py out/eval_published.json out/eval_run2.json out/eval14_run3_t0.45.json
"""
import json
import os
import re
import sys

from common import LISTEN, read_jsonl

FILLER = re.compile(r"^(mm-?hm|uh-?huh|right|yeah|okay|ok|i see|got it|sure|oh)[.!,]?$", re.I)


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def utterances(said):
    out, i = [], 0
    while i < len(said):
        if said[i] == LISTEN:
            i += 1
            continue
        j = i
        while j < len(said) and said[j] != LISTEN:
            j += 1
        out.append((i, j, " ".join(said[i:j])))
        i = j
    return out


def main():
    cache = {}
    print(f"{'adapter':<22}{'hours':>7}{'unprompted/h':>14}{'repeats/h':>11}   examples")
    for f in sys.argv[1:]:
        ev = json.load(open(f))
        # the calls file the evaluation replayed (older evals all used the published held-out calls)
        cf = ev.get("args", {}).get("calls") or "../models/lora/dataset/calls_heldout.jsonl"
        calls = cache.setdefault(cf, read_jsonl(cf))
        idx = ev.get("indices", list(range(len(ev["calls"]))))
        unprompted = repeats = 0
        ticks_total = 0
        ex = []
        for r in ev["replies"]:
            call = calls[idx[r["call"]]]
            tags = [t["tag"] for t in call["ticks"]]
            said = r["said"]
            ticks_total += len(said)
            # when does the reference last speak before each tick
            last_ref_speak = -999
            ref_last = []
            for k, t in enumerate(call["ticks"]):
                if t["said"] != LISTEN:
                    last_ref_speak = k
                ref_last.append(last_ref_speak)
            utts = utterances(said)
            prev_end = -999
            for a, b, text in utts:
                if FILLER.match(text.strip()):
                    continue
                # a follow-up shortly after the model's own utterance (the calmer advice after a warning) is not unprompted
                follow_up = a - prev_end <= 8
                prev_end = b
                if follow_up:
                    continue
                tag = tags[a]
                if tag == "listen_idle" or (tag == "listen_done" and a - ref_last[a] > 2):
                    if not any(call["ticks"][k]["heard"] != "<no voice>" for k in range(max(0, a - 3), a + 1)):
                        unprompted += 1
                        if len(ex) < 2:
                            ex.append(text[:50])
            sents = [(a, s.strip()) for a, b, text in utts for s in re.split(r"(?<=[.!?])\s+", text) if len(words(s)) >= 3]
            for n, (a, s) in enumerate(sents):
                w = set(words(s))
                for a2, s2 in sents[:n]:
                    if a - a2 <= 20 and a2 != a:
                        w2 = set(words(s2))
                        if len(w & w2) / max(1, len(w | w2)) >= 0.6:
                            repeats += 1
                            break
        hours = ticks_total * 0.6 / 3600
        name = os.path.basename(f).replace("eval_", "").replace("eval14_", "").replace(".json", "")
        print(f"{name:<22}{hours:>7.2f}{unprompted / hours:>14.1f}{repeats / hours:>11.1f}   {ex}")


if __name__ == "__main__":
    main()
