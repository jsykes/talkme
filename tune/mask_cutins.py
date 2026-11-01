"""Zero the loss on the silence that follows an unanswered cut-in in the author's prebuilt calls.

In those calls the user's interrupting question ("Wait, does it matter if it's only on one side?") is never answered:
the user's next line follows and the assistant replies to that instead. The quiet ticks between the end of the
interrupting question and the user carrying on are exactly where an adapter learns to stay silent after being asked
something mid-answer, which run 5's answered cut-ins are meant to undo. Everything else in those calls (stopping on the
cut-in, listening while the user talks, back-channels, warnings) keeps its weight.

    uv run python mask_cutins.py --calls ../models/lora/dataset/calls_train.jsonl --out out/calls_train_masked.jsonl
"""
import argparse

from common import LISTEN, NOVOICE, read_jsonl, write_jsonl


def mask(call, max_quiet=15):
    t = call["ticks"]
    masked = 0
    for k, tick in enumerate(t):
        if tick["tag"] != "barge_stop":
            continue
        # the interrupting utterance: from the cut-in until the caller first goes quiet
        j = k + 1
        while j < len(t) and t[j]["heard"] != NOVOICE:
            j += 1
        # only a question asks for an answer; a statement or a trailing-off pause is a pause to wait through
        if "?" not in " ".join(x["heard"] for x in t[k:j]):
            continue
        # the quiet ticks after it, until the caller carries on or the assistant speaks
        m = j
        while m < len(t) and m - j < max_quiet and t[m]["heard"] == NOVOICE and t[m]["said"] == LISTEN:
            t[m] = {**t[m], "w": 0.0, "masked": "unanswered_cutin"}
            masked += 1
            m += 1
    return masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    calls = read_jsonl(a.calls)
    total = sum(mask(c) for c in calls)
    cutins = sum(t["tag"] == "barge_stop" for c in calls for t in c["ticks"])
    write_jsonl(a.out, calls)
    print(f"{len(calls)} calls, {cutins} cut-ins: {total} quiet ticks after them set to weight 0 -> {a.out}")


if __name__ == "__main__":
    main()
