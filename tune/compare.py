"""Side-by-side table of held-out replay results, in the layout of the published model card.

    uv run python compare.py out/eval_published.json out/eval_run1.json out/eval_run2.json [--md out/compare.md]

Each eval JSON comes from evaluate.py on the same held-out calls file (the published calls_heldout.jsonl). Calls whose
dialogue ids are 10000 or more are the long calls, the rest are short, as in the dataset card.
"""
import argparse
import json
import os

from common import read_jsonl

# The model card's own numbers for the published adapter (closed-loop replay, threshold 0.3).
CARD = {
    "long": {"start": (121, 156), "hazard": (6, 10), "barge": (7, 7), "filler": (123, 351), "over_pct": 8, "unfinished_pct": 1},
    "short": {"start": (137, 221), "hazard": (6, 17), "barge": (27, 30), "filler": (54, 164), "over_pct": 5, "unfinished_pct": 0},
}
ROWS = [
    ("start", "Starts its answer within one tick of the right moment", True),
    ("hazard", "Interrupts a dangerous plan", True),
    ("barge", "Stops talking when the user cuts in", True),
    ("filler", "Fillers placed where a good listener would", True),
    ("over", "Talks while the user is talking or pausing (lower is better)", False),
    ("unfinished", "Stops in the middle of its own sentence (lower is better)", False),
]


def split_totals(ev, calls):
    out = {"long": {}, "short": {}, "all": {}}
    for m, c in zip(ev["calls"], calls):
        kind = "long" if any(d >= 10000 for d in c.get("dialogues", [])) else "short"
        for k, v in m.items():
            if isinstance(v, list):
                for g in (kind, "all"):
                    t = out[g].setdefault(k, [0, 0])
                    t[0] += v[0]
                    t[1] += v[1]
    return out


def fmt(v):
    a, b = v
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--calls", default="../models/lora/dataset/calls_heldout.jsonl")
    ap.add_argument("--md", default=None, help="also write the table as markdown")
    a = ap.parse_args()
    calls = read_jsonl(a.calls)
    names = [os.path.splitext(os.path.basename(e))[0].replace("eval_", "") for e in a.evals]
    evs = [json.load(open(e)) for e in a.evals]
    idx = [tuple(ev.get("indices", range(len(ev["calls"])))) for ev in evs]
    if len(set(idx)) > 1:
        print("warning: the evals replayed different calls; compare like with like")
    res = [split_totals(ev, [calls[i] for i in ev.get("indices", range(len(ev["calls"])))]) for ev in evs]
    calls = [calls[i] for i in idx[0]]
    lines = []
    for group in ("long", "short", "all"):
        n = sum(1 for c in calls if (any(d >= 10000 for d in c.get("dialogues", [])) == (group == "long")) or group == "all")
        head = ["metric"] + (["model card"] if group != "all" else []) + names
        lines.append(f"\n### {group} calls ({n})\n")
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        for key, label, higher in ROWS:
            row = [label]
            if group != "all":
                c = CARD[group]
                row.append(fmt(c[key]) if key in c else f"{c[key + '_pct']}%")
            vals = [r[group].get(key, [0, 0]) for r in res]
            pcts = [v[0] / v[1] if v[1] else None for v in vals]
            best = (max if higher else min)([p for p in pcts if p is not None], default=None)
            for v, p in zip(vals, pcts):
                s = fmt(v)
                row.append(f"**{s}**" if p is not None and best is not None and abs(p - best) < 1e-9 and len(res) > 1 else s)
            lines.append("| " + " | ".join(row) + " |")
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print("\nwrote", a.md)


if __name__ == "__main__":
    main()
