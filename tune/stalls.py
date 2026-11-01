"""Mid-sentence stalls on the recorded-call replays (out/eval_yap_<name>.json, evaluate.py on data/heldout_yap.jsonl).

A stall is an utterance whose last tick does not end a sentence, followed by 5+ quiet ticks (the model listening while
the caller is silent). This is how the timing line g26b-g26b5 and the carry-on bar were scored (docs/RESULTS.md,
2026-10-28); compare a new timing adapter against g26b4_bar9 (v1).

  uv run python stalls.py g26 g26b4 g26b4_bar9 g26b5     one line per replay
  uv run python stalls.py --show g26b4 [--limit 12]       each stall: what came before, the utterance with P(listen)
                                                          per tick, the quiet ticks after it and what followed

Summary columns: unfinished = utterances ending mid-sentence; stalls, split into guard-forced (P(listen) at the stop
under the tick's bar: the repeat guard turned words into a listen) and chosen (with the median P(listen) at the stop and
two ticks later); repeats = the stalled utterance opens like an earlier one in the call (a loop); into silence = nothing
heard in the 4 ticks before it; after a beat = it started within 3 quiet ticks of its own finished sentence.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

TUNE = Path(__file__).resolve().parent
LISTEN, NOVOICE = "<listen>", "<no voice>"
END = re.compile(r"[.!?]['\")]*$")  # common.SENTENCE_END
QUIET_TICKS = 5


def utterances(said):
    """(first, last) tick of each run of speaking ticks."""
    i = 0
    while i < len(said):
        if said[i] == LISTEN:
            i += 1
            continue
        j = i
        while j + 1 < len(said) and said[j + 1] != LISTEN:
            j += 1
        yield i, j
        i = j + 1


def scan(name):
    """The replay's utterances, unfinished count and its stalls (one dict each)."""
    ev = json.loads((TUNE / "out" / f"eval_yap_{name}.json").read_text(encoding="utf-8"))
    args = ev.get("args", {})
    calls = [json.loads(l) for l in (TUNE / args.get("calls", "data/heldout_yap.jsonl")).open(encoding="utf-8")]
    # the stop tick comes right after its own unfinished words with the line quiet, so the carry-on bar applied there
    bar = max(args.get("listen_p", 0.6), args.get("carry_on_p") or 0.0)
    n_utt = n_unf = 0
    stalls = []
    for r in ev["replies"]:
        t = calls[ev["indices"][r["call"]]]["ticks"]
        said, p = r["said"], r.get("p_listen") or [None] * len(r["said"])
        seen = []
        for i, j in utterances(said):
            n_utt += 1
            opening = " ".join(said[i:j + 1]).split()[:5]
            if not END.search(said[j]):
                n_unf += 1
                k = j + 1
                while k < len(said) and said[k] == LISTEN and t[k]["heard"] == NOVOICE:
                    k += 1
                if k - j - 1 >= QUIET_TICKS:
                    b = i - 1
                    while b >= 0 and said[b] == LISTEN:
                        b -= 1
                    stalls.append({
                        "call": r["call"], "i": i, "j": j, "k": k, "said": said, "p": p, "t": t,
                        "forced": p[j + 1] is not None and p[j + 1] < bar,
                        "repeat": opening in seen,
                        "silence": all(t[x]["heard"] == NOVOICE for x in range(max(0, i - 4), i)),
                        "after_beat": b >= 0 and i - b - 1 <= 3 and i - b - 1 > 0 and END.search(said[b]) is not None,
                    })
            seen.append(opening)
    return n_utt, n_unf, stalls, bar


def summary(name):
    n_utt, n_unf, stalls, bar = scan(name)
    chosen = [s for s in stalls if not s["forced"]]
    med = lambda v: f"{statistics.median(v):.2f}" if v else "-"
    p_at = [s["p"][s["j"] + 1] or 0 for s in chosen]
    p_two = [s["p"][s["j"] + 3] or 0 for s in chosen if s["j"] + 3 < len(s["p"])]
    print(f"{name:12s} utterances {n_utt:4d}  unfinished {n_unf:3d} ({100 * n_unf / max(1, n_utt):.0f}%)  "
          f"stalls {len(stalls):3d}: guard-forced {len(stalls) - len(chosen)}, chosen {len(chosen)} "
          f"(P at the stop {med(p_at)}, two ticks later {med(p_two)}; bar {bar:.2f})  "
          f"repeats {sum(s['repeat'] for s in stalls)}  into silence {sum(s['silence'] for s in stalls)}  "
          f"after a beat {sum(s['after_beat'] for s in stalls)}")


def show(name, limit):
    _, _, stalls, bar = scan(name)
    fmt = lambda x: "-" if x is None else f"{x:.2f}"
    for s in stalls[:limit]:
        said, p, t, i, j, k = s["said"], s["p"], s["t"], s["i"], s["j"], s["k"]
        heard = [t[x]["heard"] for x in range(max(0, i - 4), i) if t[x]["heard"] != NOVOICE]
        why = "guard-forced" if s["forced"] else "chosen"
        print(f"call {s['call']} tick {i} ({why}): before {' | '.join(said[max(0, i - 4):i])!r}  heard {heard}")
        print("   said:  " + "  ".join(f"[{w[:24]}] {fmt(p[x])}" for x, w in zip(range(i, j + 1), said[i:j + 1])))
        print(f"   quiet {k - j - 1} ticks: " + " ".join(fmt(p[x]) for x in range(j + 1, min(k, j + 13)))
              + (" ..." if k > j + 13 else "") + f"; then {next((w for w in said[k:k + 6] if w != LISTEN), '-')!r}")
    print(f"{name}: {len(stalls)} stalls (bar {bar:.2f}), {min(limit, len(stalls))} shown")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="+", help="replay names: out/eval_yap_<name>.json")
    ap.add_argument("--show", action="store_true", help="list each stall instead of one summary line")
    ap.add_argument("--limit", type=int, default=12)
    a = ap.parse_args()
    for nm in a.names:
        if not (TUNE / "out" / f"eval_yap_{nm}.json").exists():
            print(f"{nm:12s} (no out/eval_yap_{nm}.json)")
            continue
        show(nm, a.limit) if a.show else summary(nm)
