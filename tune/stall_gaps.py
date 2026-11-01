"""What the training calls say about a speaker stopping mid-sentence into a silence. For each file: of the friend's
ticks that end without .!? and are followed by quiet listen ticks (caller silent), the gap length and what follows
(the friend resumes / the caller speaks / the call ends); the tags on those quiet ticks; and how long the friend's
runs of talking into silence get (ticks, beats included).

This is where "real speakers resume 88% of the time after a one-tick gap" came from (docs/RESULTS.md, 2026-10-28): what
a timing adapter should learn about carrying on, to compare with what it does on the replays (stalls.py).

  uv run python stall_gaps.py out/calls_oto_all.jsonl out/calls_built_r5_nodecline.jsonl
"""
import json
import re
import sys
from collections import Counter

END = re.compile(r"[.!?]['\")]*$")
for f in sys.argv[1:]:
    gaps = Counter()          # (gap bucket, what follows)
    tags = Counter()
    runs = Counter()
    unfinished_then_speak = 0  # unfinished tick, friend speaks straight on next tick
    unfinished = 0
    calls = 0
    sample = []
    for line in open(f, encoding="utf-8"):
        c = json.loads(line)
        calls += 1
        t = c["ticks"]
        n = len(t)
        # runs of friend talk into silence: consecutive ticks with caller silent, from a speaking tick to the last
        # speaking tick before 3+ quiet ticks or the caller
        i = 0
        while i < n:
            if t[i]["said"] != "<listen>" and t[i]["heard"] == "<no voice>":
                j, last = i, i
                while j + 1 < n and t[j + 1]["heard"] == "<no voice>" and (t[j + 1]["said"] != "<listen>" or j + 1 - last <= 2):
                    j += 1
                    if t[j]["said"] != "<listen>":
                        last = j
                runs[min(last - i + 1, 30)] += 1
                i = last + 1
            else:
                i += 1
        for i, tk in enumerate(t):
            s = tk["said"]
            if s == "<listen>" or END.search(s):
                continue
            unfinished += 1
            if i + 1 < n and t[i + 1]["said"] != "<listen>":
                unfinished_then_speak += 1
                continue
            k = i + 1
            while k < n and t[k]["said"] == "<listen>" and t[k]["heard"] == "<no voice>":
                tags[t[k].get("tag", "?")] += 1
                k += 1
            gap = k - i - 1
            if gap == 0:
                what = "caller speaks at once"
            elif k >= n:
                what = "call ends"
            elif t[k]["said"] != "<listen>":
                what = "friend resumes"
            else:
                what = "caller speaks"
            b = "0" if gap == 0 else "1" if gap == 1 else "2" if gap == 2 else "3-4" if gap <= 4 else "5+"
            gaps[(b, what)] += 1
            if what == "friend resumes" and gap >= 3 and len(sample) < 4:
                sample.append((s, gap, t[k]["said"]))
    print(f"\n{f}: {calls} calls; friend ticks ending mid-sentence {unfinished}, of which straight on next tick {unfinished_then_speak} ({100*unfinished_then_speak/max(1,unfinished):.0f}%)")
    print("  the rest, by quiet gap then what follows:")
    for key in sorted(gaps, key=lambda k: (["0", "1", "2", "3-4", "5+"].index(k[0]), k[1])):
        print(f"    gap {key[0]:>3} ticks, {key[1]:22s} {gaps[key]}")
    print("  tags on those quiet ticks:", dict(tags.most_common(6)))
    tot = sum(runs.values())
    print("  runs of talking into silence (ticks):", " ".join(f"{k}:{v}" for k, v in sorted(runs.items())), f"; 8+ ticks: {sum(v for k, v in runs.items() if k >= 8)} of {tot} ({100*sum(v for k, v in runs.items() if k >= 8)/max(1,tot):.0f}%)")
    for s in sample:
        print("  e.g.", s)
