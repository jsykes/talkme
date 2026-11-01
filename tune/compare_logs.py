"""Compare evaluate.py logs (even of interrupted runs) on the calls every log finished.
    uv run python compare_logs.py out/eval14_published.log out/eval14_run1.log
"""
import os
import re
import sys

FIELDS = ["start", "hazard", "barge", "filler", "over", "unfinished"]


def parse(path):
    calls, cur = {}, None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"call \d+/\d+: (\d+) ticks, dialogues (\[.*\])", line)
        if m:
            cur = m.group(2)
            continue
        if cur and line.startswith("   start="):
            calls[cur] = {k: (int(a), int(b)) for k, a, b in re.findall(r"(\w+)=(\d+)/(\d+)", line)}
            cur = None
    return calls


def main():
    logs = sys.argv[1:]
    parsed = [parse(p) for p in logs]
    common = set.intersection(*(set(p) for p in parsed))
    long_ = {k for k in common if any(int(d) >= 10000 for d in re.findall(r"\d+", k))}
    print(f"{len(common)} calls finished in every log ({len(long_)} long, {len(common) - len(long_)} short)\n")
    names = [os.path.basename(p).replace("eval14_", "").replace("eval_", "").replace(".log", "") for p in logs]
    print("metric".ljust(12) + "".join(n.rjust(20) for n in names))
    for f in FIELDS:
        row = f.ljust(12)
        for p in parsed:
            a = sum(p[k][f][0] for k in common if f in p[k])
            b = sum(p[k][f][1] for k in common if f in p[k])
            row += (f"{a}/{b} ({100 * a / b:.0f}%)" if b else "-").rjust(20)
        print(row)


if __name__ == "__main__":
    main()
