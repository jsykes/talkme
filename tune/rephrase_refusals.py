"""Rewrites the friend's replies to false memories in persona dialogues so no one phrase dominates. Half of g26m7's
training refusals began "I don't recall", weighted 24 times, and it then said "I don't recall you ever..." to anything.
Each reply is rewritten by the plain writer model in the friend's own voice, nudged by two examples drawn from
gen_dialogues.REFUSALS, keeping what it means and any question it asks; a rewrite that still opens with "I don't
recall" / "I don't remember", or runs long, keeps the original. Writes <file stem>_varied.jsonl next to each file.
    uv run python rephrase_refusals.py --url http://127.0.0.1:8080 out/dial_g26m2.jsonl out/dial_g26m3.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path

from common import MicroTurnClient
from gen_dialogues import REFUSALS

PROMPT = """A friend on the phone, who is {persona}, was told by the other person: "{claim}"
The friend does not remember that (it is not something it knows about them), and replied: "{reply}"

Rewrite the friend's reply so it means the same thing in different words, the way this friend talks, for instance
{hints}, but in its own words. One or two short sentences; keep any question it asks. Do not start with "I don't
recall" or "I don't remember". Return ONLY the new reply."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    client = MicroTurnClient(a.url, lora_scale=0.0, timeout=600)
    for f in a.files:
        out, n, changed = [], 0, 0
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            for t in d["turns"]:
                if t.get("kind") != "false_memory" or not t.get("assistant", "").strip():
                    continue
                n += 1
                hints = " or ".join(f'"{x}"' for x in rng.sample(REFUSALS, 2))
                try:
                    new = client.chat([{"role": "user", "content": PROMPT.format(persona=d.get("persona", "a friend"), claim=t["user"].strip(),
                                                                                 reply=t["assistant"].strip(), hints=hints)}],
                                      temperature=0.9, max_tokens=200).strip().strip('"').strip()
                except Exception as e:
                    print(f"  kept (error {e})", flush=True)
                    continue
                if new and not re.match(r"i (don.?t|do not) (recall|remember)", new, re.I) and 2 <= len(new.split()) <= 40:
                    t["assistant"] = new
                    changed += 1
            out.append(d)
        dst = Path(f).with_name(Path(f).stem + "_varied.jsonl")
        dst.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in out), encoding="utf-8")
        print(f"{f}: rewrote {changed} of {n} refusals -> {dst.name}", flush=True)


if __name__ == "__main__":
    main()
