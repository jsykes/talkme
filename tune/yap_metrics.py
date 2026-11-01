"""Turn-taking scores on real recorded conversations (data/heldout_yap.jsonl, from scripts/yap_to_calls.js).

The reference is a real person's timing, not a scripted assistant, so the questions are about WHEN:
  takes the turn      the model starts talking within 1 tick (and within ~2.4 s) of where the person took the turn
  onset delay         median time from the person's first word to the model's, when both took that turn
  talks over caller   model starts a real utterance (not a back-channel) while the caller is mid-speech, per minute
  waits through pause share of the caller's mid-turn pauses where the model started a real utterance
  back-channels       model back-channels per minute next to the person's, and how many land within a tick of theirs
  barge               the model is quiet within a tick where the person gave way to the caller cutting in

    uv run python yap_metrics.py data/heldout_yap.jsonl out/eval_yap_run2.json out/eval_yap_run5.json [--md out/yap.md]
"""
import argparse
import json
import re
import statistics

from common import LISTEN, NOVOICE, read_jsonl
from evaluate import FILLER_RE

TICK = 0.6


def metrics(calls, ev, first=0):
    replies = {r["call"]: r["said"] for r in ev["replies"]}
    m = dict(start=[0, 0], start_wide=[0, 0], over=0, pause=[0, 0], bc_model=0, bc_ref=0, bc_hit=0, barge=[0, 0],
             onsets_model=0, onsets_ref=0, bc_ref_all=0, minutes=0.0, delays=[], ends=[0, 0, 0], ends_ref=[0, 0, 0])
    for ci, idx in enumerate(ev["indices"]):
        if first and idx >= first:
            continue
        call, said = calls[idx], replies.get(ci)
        if said is None:
            continue
        tags = [t["tag"] for t in call["ticks"]][: len(said)]
        spoke = [s != LISTEN for s in said]
        fill = [bool(FILLER_RE.match(s)) for s in said]
        real = [s and not f for s, f in zip(spoke, fill)]
        onset = [real[i] and (i == 0 or not spoke[i - 1]) for i in range(len(said))]
        m["minutes"] += len(said) * TICK / 60
        # the person, counted the same way as the model
        ref = [t["said"] for t in call["ticks"]][: len(said)]
        r_spoke = [x != LISTEN for x in ref]
        r_fill = [bool(FILLER_RE.match(x)) for x in ref]
        m["onsets_ref"] += sum(r_spoke[i] and not r_fill[i] and (i == 0 or not r_spoke[i - 1]) for i in range(len(ref)))
        m["bc_ref_all"] += sum(r_fill[i] and (i == 0 or not r_spoke[i - 1]) for i in range(len(ref)))
        m["onsets_model"] += sum(onset)
        m["bc_model"] += sum(f and (i == 0 or not spoke[i - 1]) for i, f in enumerate(fill))
        for i, tag in enumerate(tags):
            if tag == "start":
                m["start"][1] += 1
                m["start_wide"][1] += 1
                m["start"][0] += any(spoke[j] for j in range(max(0, i - 1), min(len(said), i + 2)))
                hit = next((j for j in range(max(0, i - 2), min(len(said), i + 5)) if real[j] or fill[j]), None)
                if hit is not None and hit <= i + 3:
                    m["start_wide"][0] += 1
                if hit is not None:
                    m["delays"].append((hit - i) * TICK)
            elif tag == "continuer":
                m["bc_ref"] += 1
                m["bc_hit"] += any(fill[j] for j in range(max(0, i - 1), min(len(said), i + 2)))
            elif tag == "barge_stop":
                m["barge"][1] += 1
                m["barge"][0] += (not spoke[i]) or (i + 1 < len(said) and not spoke[i + 1])
            if tag == "listen_speech" and onset[i]:
                m["over"] += 1
            if tag == "listen_pause":
                m["pause"][1] += 1
                m["pause"][0] += onset[i]
        heard = [t["heard"] for t in call["ticks"]][: len(said)]
        sentence_ends(said, heard, m["ends"])
        sentence_ends(ref, heard, m["ends_ref"])
    return m


def sentence_ends(said, heard, out):
    """What a speaker does after ending a sentence while the other is quiet: out[0] runs straight on, out[1] pauses a
    beat or more and carries on, out[2] waits until the other speaks. People run straight on at 70% of sentence ends and
    pause first at 27% (nearly all one tick); the written training calls had the caller always speak next."""
    i = 0
    while i < len(said) - 1:
        if said[i] != LISTEN and re.search(r"[.!?]['\")]*$", said[i]) and heard[i] == NOVOICE:
            if said[i + 1] != LISTEN:
                out[0] += 1
                i += 1
                continue
            j = i + 1
            while j < len(said) and said[j] == LISTEN and heard[j] == NOVOICE:
                j += 1
            if j < len(said):
                out[2 if said[j] == LISTEN else 1] += 1
            i = j
        else:
            i += 1


def ends(e):
    n = max(1, sum(e))
    return " / ".join(f"{100 * x / n:.0f}%" for x in e)


def row(m):
    pct = lambda p: f"{100 * p[0] / p[1]:.0f}% ({p[0]}/{p[1]})" if p[1] else "-"
    mins = max(m["minutes"], 1e-9)
    return {
        "Takes the turn where the person did (within 1 tick)": pct(m["start"]),
        "... within about 2.4 s": pct(m["start_wide"]),
        "Median delay after the person's first word": f"{statistics.median(m['delays']):+.1f} s" if m["delays"] else "-",
        "Starts talking over the caller, per minute": f"{m['over'] / mins:.1f}",
        "Starts talking in the caller's mid-turn pauses": pct(m["pause"]),
        "Back-channels per minute (person)": f"{m['bc_model'] / mins:.1f} ({m['bc_ref_all'] / mins:.1f})",
        "Back-channels within a tick of the person's": pct([m["bc_hit"], m["bc_ref"]]),
        "Utterances started per minute (person)": f"{m['onsets_model'] / mins:.1f} ({m['onsets_ref'] / mins:.1f})",
        "Quiet when the person gave way to a cut-in": pct(m["barge"]),
        "After its own sentence: straight on / a beat then on / waits (person)": ends(m["ends"]) + f" ({ends(m['ends_ref'])})",
        "Minutes replayed": f"{m['minutes']:.0f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calls")
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--md", default=None)
    ap.add_argument("--first", type=int, default=0, help="score only the first N calls (to compare with a partial replay)")
    ap.add_argument("--names", default="", help="column names, comma separated (default: adapter file names)")
    a = ap.parse_args()
    calls = read_jsonl(a.calls)
    cols = {}
    for f in a.evals:
        ev = json.load(open(f, encoding="utf-8"))
        name = (ev["args"].get("adapter") or "base").replace("\\", "/").split("/")[-1].replace(".gguf", "")
        cols[name if name not in cols else f] = row(metrics(calls, ev, a.first))
    if a.names:
        cols = dict(zip(a.names.split(","), cols.values()))
    names = list(cols)
    lines = [f"| | {' | '.join(names)} |", "|---|" + "---|" * len(names)]
    for k in next(iter(cols.values())):
        lines.append(f"| {k} | {' | '.join(cols[n][k] for n in names)} |")
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(out + "\n")


if __name__ == "__main__":
    main()
