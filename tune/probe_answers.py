"""Does an adapter still answer? A text-only quiz through the micro-turn protocol: each question is fed as the
recogniser would deliver it (a few words per tick, then quiet ticks), the reply is collected tick by tick, and it is
checked for the right answer. Timing tests can't see this: a partner that says "Oh, okay." on time scores well there.

    uv run python probe_answers.py --adapter out/run7.gguf --listen-p 0.6 --repeat-guard
    uv run python probe_answers.py --adapter "run7.gguf:0.5,run5.gguf:0.5" --listen-p 0.6     (a blend)
"""
import argparse
import json
import re
import sys

from common import LISTEN, NOVOICE, SYSTEMS, MicroTurnClient, persona_system

QUESTIONS = [
    ("What's the capital of Japan?", r"tokyo"),
    ("What's the tallest mountain in the world?", r"everest"),
    ("How many legs does a spider have?", r"eight|\b8\b"),
    ("What's twelve times twelve?", r"144|hundred (and )?forty[- ]?four|one forty[- ]?four"),
    ("What day comes after Tuesday?", r"wednesday"),
    ("Name the three primary colors.", r"(?=.*red)(?=.*blue)(?=.*yellow)"),
    ("How many centimeters are in a meter?", r"100|hundred"),
    ("What's the boiling point of water in Celsius?", r"100|hundred"),
    ("Who wrote Romeo and Juliet?", r"shakespeare"),
    ("What planet is closest to the sun?", r"mercury"),
    ("My mom loves gardening. What's a good birthday gift for her?", r"seed|plant|glove|tool|pot|flower|garden|herb|kneel|bulb|pruner|shear"),
    ("Is it safe to mix bleach and ammonia?", r"\bno\b|not safe|never|don't|do not|toxic|dangerous"),
]


def ask(cl, question, quiet=14):
    words = question.split()
    heard = [" ".join(words[i:i + 3]) for i in range(0, len(words), 3)] + [NOVOICE] * quiet
    history, said = [], []
    for h in heard:
        s, _, _ = cl.tick(history, h)
        s = cl.guard(history, s)
        history += [("user", h), ("assistant", s)]
        if s != LISTEN:
            said.append(s)
    return " ".join(said)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--listen-p", type=float, default=0.3)
    ap.add_argument("--repeat-guard", action="store_true")
    ap.add_argument("--speak-opts", default="")
    ap.add_argument("--format", default="qwen3.6", help="the base model's chat format (profiles.FORMATS): qwen3.6 or gemma4")
    ap.add_argument("--history-ticks", type=int, default=0, help="accepted for the pipeline; quiz calls are short")
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="assistant")
    ap.add_argument("--persona", default=None, help="a JSON file {persona, memories}: test with that persona prompt (common.persona_system), as the app sends persona adapters")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--questions", default=None, help="a JSON file of [question, answer pattern] pairs to ask instead (data/quiz_wide.json: 36 more, to tell close adapters apart)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # a reply can hold what the console code page cannot (CO₂)
    cl = MicroTurnClient(a.url, listen_p=a.listen_p, adapter=a.adapter, fast=True)
    cl.repeat_guard = a.repeat_guard
    cl.system = SYSTEMS[a.system]
    if a.persona:
        p = json.load(open(a.persona, encoding="utf-8"))
        cl.system = persona_system(p["persona"], p.get("memories", []))
    cl.format = a.format
    cl.speak_opts = json.loads(a.speak_opts) if a.speak_opts else None
    questions = json.load(open(a.questions, encoding="utf-8")) if a.questions else QUESTIONS
    rows, ok = [], 0
    for q, pat in questions:
        reply = ask(cl, q)
        hit = bool(re.search(pat, reply, re.I))
        ok += hit
        rows.append({"q": q, "reply": reply, "ok": hit})
        print(f"{'OK ' if hit else '-- '} {q}\n     {reply[:160] or '(silence)'}")
    name = a.name or a.adapter
    silent = sum(not r["reply"] for r in rows)
    print(f"{name}: {ok}/{len(questions)} answered, {silent} silent")
    if a.out:
        json.dump({"name": name, "score": ok, "of": len(questions), "silent": silent, "rows": rows}, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
