"""Adds an answer to every cut-in in the dialogues, so built calls teach "answer the interruption".

In the published dialogues a cut-in ("Wait, does it need to be unplugged first?") is never answered: the call just
carries on with the user's next line. Over a thousand such examples taught every adapter to go quiet after being
interrupted with a question, which also blocked resuming an interrupted answer. This writes copies of the dialogues
with `barge_answer` on short barge-in turns and `answer` on long-dialogue interrupts, written by the base model.

    uv run python answer_cutins.py --url http://127.0.0.1:8080 \
        --dialogues ../models/lora/dataset/dialogues_short.jsonl ../models/lora/dataset/dialogues_long.jsonl --outdir out
"""
import argparse
import json
import os
import re

from common import MicroTurnClient, read_ids, read_jsonl, write_jsonl

PROMPT = """This is a phone call between a USER and a voice ASSISTANT about: {topic}.
The user said: "{user}"
The assistant began to answer: "{partial}..."
and the user cut in: "{cutin}"
Write what the assistant says next, in reply to the cut-in: one or two short spoken sentences, at most 25 words. Answer
a question directly; if the cut-in is a correction or a comment, acknowledge it briefly. Plain, friendly, correct and
cautious: never approve something dangerous. Reply with the spoken words only, no quotes."""


def clean(text):
    text = text.strip().strip('"').strip()
    text = re.sub(r"^(assistant|ASSISTANT)\s*:\s*", "", text)
    return " ".join(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--dialogues", nargs="+", required=True)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--exclude-ids", default=None, help="skip these dialogue ids (they are not built anyway)")
    a = ap.parse_args()
    client = MicroTurnClient(a.url, lora_scale=0.0, timeout=300)
    skip = read_ids(a.exclude_ids)
    for path in a.dialogues:
        out_path = os.path.join(a.outdir, os.path.basename(path).replace(".jsonl", "_answered.jsonl"))
        done = {d["id"]: d for d in read_jsonl(out_path)} if os.path.exists(out_path) else {}
        dialogues = read_jsonl(path)
        n_new = 0
        for d in dialogues:
            if d["id"] in done or d["id"] in skip:
                continue
            for t in d["turns"]:
                cutin, partial = None, None
                if t.get("kind") == "barge_in" and t.get("barge"):
                    cutin, partial = t["barge"], t.get("assistant") or ""
                elif isinstance(t.get("interrupt"), dict) and t["interrupt"].get("text"):
                    n = t["interrupt"].get("after_words") or 20
                    cutin, partial = t["interrupt"]["text"], " ".join((t.get("assistant") or "").split()[:n])
                if not cutin:
                    continue
                try:
                    ans = clean(client.chat([{"role": "user", "content": PROMPT.format(
                        topic=d.get("topic", ""), user=t.get("user", ""), partial=partial, cutin=cutin)}],
                        temperature=0.6, max_tokens=120))
                except Exception as e:
                    print(f"  dialogue {d['id']}: {e}", flush=True)
                    continue
                if not ans or len(ans.split()) > 40:
                    continue
                if t.get("kind") == "barge_in":
                    t["barge_answer"] = ans
                else:
                    t["interrupt"]["answer"] = ans
                n_new += 1
                print(f"{d['id']}: {cutin!r} -> {ans!r}", flush=True)
            done[d["id"]] = d
            write_jsonl(out_path, [done[k] for k in sorted(done)])  # checkpoint after every dialogue
        print(f"{path}: {n_new} cut-ins answered -> {out_path}")


if __name__ == "__main__":
    main()
