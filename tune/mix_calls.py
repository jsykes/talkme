"""Combine calls files into one training set, splitting calls that are too long to be trained on whole.

The trainer reads only the first --max-tokens of a call (8,000, about 470 ticks), so the published adapter never saw
the ends of its longer calls. Here every call longer than --max-ticks is cut into pieces at quiet ticks (listen_idle,
nobody talking and nothing pending), so each piece starts cleanly and every tick is trained on. A piece drops the
history before it, which is also what the app does when a very long call outgrows the context.

    uv run python mix_calls.py --calls ../models/lora/dataset/calls_train.jsonl out/calls_built.jsonl --out out/calls_run2.jsonl
"""
import argparse
import random

from common import read_ids, read_jsonl, write_jsonl


def split_call(call, max_ticks, min_ticks=120):
    ticks = call["ticks"]
    if len(ticks) <= max_ticks:
        return [call]
    pieces, start = [], 0
    while len(ticks) - start > max_ticks:
        # the last quiet tick inside the window, preferring the end of a run of quiet ticks
        cut = None
        for i in range(start + max_ticks - 1, start + min_ticks, -1):
            # the next piece must not open with the assistant talking into silence: a model trained on that learns
            # to say something on the first tick of a call (it blurted a split-off "Don't force it, ..." on calls)
            opens_with_assistant = i + 1 < len(ticks) and ticks[i + 1]["said"] != "<listen>"
            if ticks[i]["tag"] == "listen_idle" and not opens_with_assistant and (i + 1 >= len(ticks) or ticks[i + 1]["tag"] != "listen_idle"):
                cut = i + 1
                break
        if cut is None:
            cut = start + max_ticks
        pieces.append({**call, "ticks": ticks[start:cut], "piece": len(pieces)})
        start = cut
    if len(ticks) - start >= 20:
        pieces.append({**call, "ticks": ticks[start:], "piece": len(pieces)})
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ticks", type=int, default=460, help="~17 tokens per tick: 460 ticks fit in 8,000 tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-ids", default=None, help="drop every call that contains one of these dialogue ids")
    a = ap.parse_args()
    out = []
    bad = read_ids(a.exclude_ids)
    for spec in a.calls:
        # "file.jsonl:N" keeps a random N of that file's pieces (after splitting), e.g. to cap a large real-data set
        f, take = (spec.rsplit(":", 1) if spec.rsplit(":", 1)[-1].isdigit() else (spec, None))
        calls = read_jsonl(f)
        if bad:
            n0 = len(calls)
            calls = [c for c in calls if not set(c.get("dialogues", [])) & bad]
            print(f"{f}: dropped {n0 - len(calls)} calls containing excluded dialogues")
        pieces = [p for c in calls for p in split_call(c, a.max_ticks)]
        print(f"{f}: {len(calls)} calls, {sum(len(c['ticks']) for c in calls)} ticks -> {len(pieces)} pieces")
        if take is not None and int(take) < len(pieces):
            pieces = random.Random(a.seed).sample(pieces, int(take))
            print(f"  kept a random {len(pieces)} of them ({sum(len(p['ticks']) for p in pieces)} ticks)")
        for p in pieces:
            p["source"] = f
        out += pieces
    random.Random(a.seed).shuffle(out)
    write_jsonl(a.out, out)
    tags = {}
    for c in out:
        for t in c["ticks"]:
            tags[t["tag"]] = tags.get(t["tag"], 0) + 1
    n = sum(tags.values())
    print(f"wrote {len(out)} calls, {n} ticks to {a.out}")
    for k, v in sorted(tags.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14} {v:7d}  {100 * v / n:5.1f}%")


if __name__ == "__main__":
    main()
