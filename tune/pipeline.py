"""One training run from a recipe (tune/recipes/<name>.json), the way the hand-written pipeline_r*.sh scripts (now in archive/) did
it, so the app's Tune tab and the command line run the same thing:

  uv run python pipeline.py --recipe recipes/run10.json [--stage all|dialogues|data|train|replays|live] [--dry-run]

Stages, each skipped when its output already exists (so the same command resumes a stopped run):
  dialogues  the base model writes new dialogues (gen_dialogues.py), several writers at once; with tests.memory, also the
             held-out persona calls of the memory test once (data/heldout_memory.jsonl), shared by every run after
  data       dialogues -> timed calls (build_calls.py), mixed with other call files (mix_calls.py), tick weights
  train      train_lora.py (--resume) on the recipe's base checkpoint, export_gguf.py, copy into models/lora
  replays    the answer quiz, the recorded conversations and our held-out set, at the recipe's listen threshold; with
             tests.memory, the memory test (memory_check.py)
  live       scripts/live_bench.sh (three synthetic callers) and the comparison report
The log is out/<name>.pipeline.log. Everything runs llama-server itself; stop the app's model server first. After a run
the adapter's card (models/lora/<its name>.json, write_card) has the recipe, the results and a suggested label.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TUNE = Path(__file__).resolve().parent
ROOT = TUNE.parent
OUT = TUNE / "out"
# The inputs that are free to share, as they were when the adapters were made: dialogues the base models wrote, and calls
# built from the codacus dataset (data/ATTRIBUTION.md). The real conversations stay with otoSpeech, which asks people to
# sign up: anyone who wants them builds them (README, The training data), and a run goes on without them.
TRAIN_DATA = TUNE / "data" / "train"
REAL_CALLS = "calls_oto_all.jsonl"
NO_REAL_CALLS = ["no out/calls_oto_all.jsonl: training without the real conversations (otoSpeech)",
                 "  the adapter gets close to the recipe's, not the same, so it is saved under a name of its own",
                 "  to add them, see The training data in the README"]
LORA = ROOT / "models" / "lora"
WIN = sys.platform == "win32"
# llama.cpp's server: the release unpacked into vendor/llama (scripts/setup.js), else on Linux one on the PATH
LLAMA = ROOT / "vendor" / "llama" / ("llama-server.exe" if WIN else "llama-server")
if not WIN and not LLAMA.exists() and shutil.which("llama-server"):
    LLAMA = Path(shutil.which("llama-server"))
def load_bases():
    """The GGUF the app serves and the full-precision checkpoint the trainer streams, per base: models/bases.json, the
    file the app and setup read too (its _about says what each field means). A base is also listed under its aliases
    (the first Qwen is "original" in recipes, "qwen" in the app); CANONICAL maps every name to the app's."""
    rows = {k: v for k, v in json.loads((ROOT / "models" / "bases.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}
    bases, canonical = {}, {}
    for name, b in rows.items():
        row = {"gguf": b["gguf"], "hf": b.get("hf"), "format": b.get("format", "qwen3.6"), "label": b.get("label", name)}
        if b.get("ctx"):
            row["ctx"] = b["ctx"]
        if b.get("historyTicks"):
            row["history_ticks"] = b["historyTicks"]
        for n in [name, *b.get("aliases", [])]:
            bases[n], canonical[n] = row, name
    return bases, canonical


BASES, CANONICAL = load_bases()
SHARED_LIFE = (r"\b(we talked about|are we still on|we're still on|our (trip|plan|plans|weekend)|we could (go|do|meet)|"
               r"let's (go|meet|do it)|i'll (come|meet you|pick you up|be there|bring)|see you (there|then|tomorrow|on)|"
               r"i saw (you|your)|when we (went|were|met)|remember when we|last time we|i was (there|at your))\b")
# a call that opens like a help desk ("Hi. How can I help?"): the older assistant-era calls the partner recipes mix in
SERVICE_GREETING = r"(how can i help|what can i do for you|what do you need|what's on your mind|go ahead)[.?!]*$"
# the friend inviting the caller to meet up (gen_dialogues.MEET_UP): drop_meetups drops such dialogues from every file
MEETUP = r"\b(let's (meet|go|get|grab|hang|do something)|we (could|should) (go|grab|get|meet|hang)|do you want to (grab|get|meet|hang|go out|come)|wanna (grab|get|meet|hang|go out|come)|want me to (come|bring|swing|stop|help)|i can (come over|swing by|stop by|bring|help you (move|carry|paint))|i'll (come|bring|swing|stop by|pick you)|(come|swing|stop) (over|by)|hang out|meet up|i'm (totally )?free (on )?(saturday|sunday|tonight|this weekend|tomorrow)|catch up properly|grab a drink this weekend)\b"
DRY = False
LOG = None
NAME = "run"


def say(msg):
    line = f"[{NAME}] {datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    if LOG and not DRY:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run(cmd, cwd=TUNE, log=None, env=None, check=True):
    """Runs a command, streaming its output here (and into `log` when given). Returns the exit code."""
    shown = " ".join(str(c) for c in cmd)
    if DRY:
        print(f"  $ {shown}" + (f"  > {log}" if log else ""))
        return 0
    out = open(log, "a", encoding="utf-8") if log else None
    cmd = [program(str(cmd[0])), *cmd[1:]]
    p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **(env or {})})
    for line in p.stdout:
        if out:
            out.write(line)
            out.flush()
        else:
            print("  " + line.rstrip(), flush=True)
    p.wait()
    if out:
        out.close()
    if check and p.returncode != 0:
        say(f"failed ({p.returncode}): {shown}")
        sys.exit(1)
    return p.returncode


def program(name):
    """The program to start, found on PATH. On Windows "bash" is Git for Windows' bash: the one in System32 (and
    WindowsApps) is WSL's Linux bash, whose Linux node cannot load the app's Windows modules, and the PATH the app has
    may not include Git's own bash at all (the live tests failed that way when started from the Tune tab)."""
    if os.name == "nt" and name == "bash":
        git = shutil.which("git")
        tops = [*Path(git).resolve().parents][:3] if git else []
        for bash in [t / "bin" / "bash.exe" for t in tops] + [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"]:
            if bash.is_file():
                return str(bash)
        say("the live tests need Git for Windows' bash (git-scm.com); WSL's bash runs Linux programs")
        sys.exit(1)
    return shutil.which(name) or name


PORT = 8080  # the pipeline's model server
SERVER = None  # the one it started


def port_owner(port):
    """The pid and program name of what listens on 127.0.0.1:`port`, or (None, None)."""
    try:
        if WIN:
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
            pid = next((int(l.split()[-1]) for l in out.splitlines()
                        if "LISTENING" in l and re.search(rf"(127\.0\.0\.1|0\.0\.0\.0):{port}\s", l)), None)
            if not pid:
                return None, None
            row = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
            return pid, row.split('","')[0].strip('"').lower() or None
        out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True).stdout
        m = re.search(r'\(\("([^"]+)",pid=(\d+)', out)
        return (int(m.group(2)), m.group(1)) if m else (None, None)
    except (OSError, StopIteration):
        return None, None


def kill_server():
    """Stops the pipeline's model server: the one it started, and a llama-server still holding its port (left by a
    stopped run, or the app's). Other llama-servers on the machine (the Orpheus voice's, another program's) are left
    alone: this used to stop every llama-server there was."""
    global SERVER
    if DRY:
        return
    if SERVER and SERVER.poll() is None:
        SERVER.kill()
        SERVER.wait(timeout=30)
    SERVER = None
    for _ in range(30):
        pid, name = port_owner(PORT)
        if not pid:
            return
        if "llama-server" not in (name or ""):
            sys.exit(f"port {PORT} is taken by {name} (pid {pid}), not a llama-server: stop it and run this again")
        if WIN:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            subprocess.run(["kill", str(pid)], capture_output=True)
        time.sleep(1)


def server(gguf, *args, ctx=32768):
    """Starts llama-server on `gguf` (a path from the repo root) with the usual options, and waits until it answers."""
    kill_server()
    cmd = [str(LLAMA), "-m", str(ROOT / gguf), *args, "-ngl", "99", "--n-cpu-moe", "8", "-c", str(ctx), "-ub", "512",
           "--port", str(PORT), "--host", "127.0.0.1", "--no-reasoning-preserve"]
    if DRY:
        print("  $ " + " ".join(str(c) for c in cmd))
        return
    env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [str(LLAMA.parent), os.environ.get("LD_LIBRARY_PATH")]))} if not WIN else None
    global SERVER
    SERVER = subprocess.Popen([str(c) for c in cmd], cwd=str(LLAMA.parent), env=env, stdout=open(ROOT / "vendor" / "llama-server.log", "w"), stderr=subprocess.STDOUT)
    import urllib.request
    for _ in range(200):
        try:
            if b'"status":"ok"' in urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2).read():
                return
        except Exception:
            pass
        time.sleep(3)
    say("the model server did not start (vendor/llama-server.log)")
    sys.exit(1)


def exists(p):
    return not DRY and Path(p).is_file() and Path(p).stat().st_size > 0


def tested(p):
    """A test result that is about this run's adapter: the repo ships results under the recipes' names, and those say
    nothing about one trained since."""
    gguf = OUT / f"{NAME}.gguf"
    return exists(p) and not (gguf.is_file() and Path(p).stat().st_mtime < gguf.stat().st_mtime)


def uv(*args, train=False):
    return ["uv", "run", "--project", str(TUNE), *(["--extra", "train"] if train else []), "python", *args]


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def data_file(rel):
    """A file a recipe reads: tune/<rel> when this machine has it, else the copy shipped in data/train/."""
    here, shipped = TUNE / rel, TRAIN_DATA / Path(rel).name
    return here if here.exists() or not shipped.exists() else shipped


def data_inputs(r):
    return [f if isinstance(f, str) else f["file"] for f in r.get("dialogue_files", []) + r.get("mix", [])]


def check_inputs(r, stage):
    """Before any work: names what this run needs and does not have, and how to get it. Without the real conversations
    (they stay with otoSpeech, see REAL_CALLS) a run still trains, under a name of its own: it is not its recipe's adapter
    any more, and must not replace one that was made with them."""
    absent = [f for f in data_inputs(r) if not data_file(f).exists()]
    real = [f for f in absent if Path(f).name == REAL_CALLS]
    if real:
        for line in NO_REAL_CALLS:
            say(line)
        if not r.get("no_oto"):
            r["export_as"], r["label"] = app_name(r).replace(".gguf", "-no-oto.gguf"), (r.get("label") or suggested_label(r)) + " (no otoSpeech)"
            r["no_oto"] = True
    missing = [f for f in absent if f not in real]
    if stage in ("all", "data") and missing and not (OUT / f"calls_{NAME}.jsonl").exists():
        say("this run needs files that are not here, so nothing was started:")
        for f in missing:
            made = Path(f).stem.removeprefix("dial_")
            say(f"  {f}: " + (f"written by recipe {made}: run it first, --stage dialogues" if (TUNE / "recipes" / f"{made}.json").exists() else "left by an earlier run and not shipped; see tune/archive for how it was made, or take it out of the recipe"))
        sys.exit(1)
    gguf = OUT / f"{NAME}.gguf"
    if stage in ("all", "train") and not gguf.exists():
        for key in ("init_from", "base_adapter"):
            if r.get(key) and not (OUT / f"adapter_{r[key]}").exists():
                say(f"this run continues run {r[key]}, which has not been trained here (no out/adapter_{r[key]}): run recipes/{r[key]}.json first")
                sys.exit(1)


def write_rows(path, items):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")


# ---- stages ----
MEMORY_DIAL = TUNE / "data" / "heldout_memory_dial.jsonl"
MEMORY_CALLS = TUNE / "data" / "heldout_memory.jsonl"
# the memory test's persona calls: written once, by seeds and ids no training run uses. v2 (2026-10-23) is written
# with each false claim's own scenario; v1 (data/heldout_memory_v1*.jsonl, seeds 7001-7003) had 9 of 16 about Vegas
MEMORY_TEST = [{"kind": "memory", "focus": "recall", "count": 8, "seed": 7101, "start_id": 393000},
               {"kind": "memory", "focus": "false_memory", "count": 10, "seed": 7102, "start_id": 394000},
               {"kind": "memory", "focus": "blank", "count": 6, "seed": 7103, "start_id": 395000}]


def stage_dialogues(r):
    out = OUT / f"dial_{NAME}.jsonl"
    jobs = []  # (spec, part file, the file its dialogues go to)
    if r.get("dialogues") and exists(out):
        say(f"dialogues: {out.name} exists, skipping")
    elif r.get("dialogues"):
        jobs += [(s, OUT / f"dial_{NAME}_{i}.part.jsonl", out) for i, s in enumerate(r["dialogues"])]
    if r.get("tests", {}).get("memory") and not exists(MEMORY_CALLS):
        jobs += [(s, OUT / f"dial_memtest_{i}.part.jsonl", MEMORY_DIAL) for i, s in enumerate(MEMORY_TEST)]
    # rephrase_refusals: persona dialogue files whose replies to false memories get rewritten in varied words first
    # (rephrase_refusals.py; each writes <stem>_varied.jsonl, which the recipe's dialogue_files then use)
    rephrase = [TUNE / f for f in r.get("rephrase_refusals", []) if not exists(TUNE / f.replace(".jsonl", "_varied.jsonl"))]
    if not jobs and not rephrase:
        return
    if not jobs:
        server(BASES[r.get("writer", "original")]["gguf"], "-np", "1")
    if jobs:
        say(f"writing {sum(s['count'] for s, _, _ in jobs)} dialogues with the base model ({len(jobs)} writers)")
        server(BASES[r.get("writer", "original")]["gguf"], "-np", str(min(4, len(jobs))))
    procs = []
    for i, (s, part, _) in enumerate(jobs):
        log = OUT / (part.name.replace(".part.jsonl", ".log").replace("dial_", "gen_"))
        cmd = uv("gen_dialogues.py", "--url", "http://127.0.0.1:8080", "--kind", s["kind"], "--count", str(s["count"]),
                 "--seed", str(s.get("seed", 1000 + i)), "--start-id", str(s.get("start_id", 200000 + i * 1000)),
                 "--temperature", str(s.get("temperature", 0.9)), "--out", str(part))
        if s["kind"] == "memory":  # what the judge drops, to read a sample of
            cmd += ["--rejects", str(part.with_name(part.name.replace(".part.jsonl", ".rejects.jsonl")))]
        if s.get("focus"):
            cmd += ["--focus", s["focus"]]
        if s.get("goodbye"):
            cmd += ["--goodbye", str(s["goodbye"])]
        if s.get("style"):
            cmd += ["--style", s["style"]]
        if DRY:
            print("  $ " + " ".join(cmd) + f"  > {log}")
            continue
        part.unlink(missing_ok=True)
        procs.append(subprocess.Popen(cmd, cwd=str(TUNE), stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    if rephrase:
        say(f"rewriting the refusals of {len(rephrase)} persona dialogue files in varied words")
        run(uv("rephrase_refusals.py", "--url", "http://127.0.0.1:8080", *[str(f) for f in rephrase]), log=OUT / f"rephrase_{NAME}.log")
    if DRY:
        return
    kill_server()
    for dst in dict.fromkeys(d for _, _, d in jobs):
        items = [l for _, part, d in jobs if d == dst and part.exists() for l in open(part, encoding="utf-8") if l.strip()]
        dst.write_text("".join(items), encoding="utf-8")
        say(f"{len(items)} new dialogues in {dst.name}")
    if MEMORY_DIAL in (d for _, _, d in jobs):  # the memory test's calls: one per persona, a friend's greeting
        run(uv("build_calls.py", "--dialogues", str(MEMORY_DIAL), "--per", "1", "--seed", "7000", "--timing", "mixed",
               "--greetings", "partner", "--nudge-quiet", "2.6,5.5", "--out", str(MEMORY_CALLS)), log=OUT / f"build_{NAME}.log")
        say(f"memory test calls: {MEMORY_CALLS.relative_to(TUNE)}")


def filtered_dialogues(src, how, limit=None, seed=0):
    """A dialogue file with a filter applied (and at most `limit` of what passes, picked at random), written next to
    the original."""
    dst = OUT / (Path(src).stem + f"_{how}" + (f"_{limit}" if limit else "") + ".jsonl")
    if how == "no_shared_life":
        keep = [d for d in rows(src) if not re.search(SHARED_LIFE, " ".join(t.get("user", "") + " " + t.get("assistant", "") for t in d["turns"]), re.I)]
    elif how == "no_meetups":  # the friend is only on the phone: no "want to grab a drink?" or "I can come over"
        friend = lambda d: " ".join(" ".join(str(t.get(k, "")) for k in ("assistant", "interjection", "side_answer", "resume")) for t in d["turns"])
        keep = [d for d in rows(src) if not re.search(MEETUP, friend(d), re.I)]
    elif how == "no_unknown":  # run 10's "no idea" dialogues, which taught it to hedge (run 11 writes 40 new ones)
        keep = [d for d in rows(src) if not any(t.get("kind") == "unknown" for t in d["turns"])]
    elif how == "only_unknown":  # just the "no idea" dialogues (run 12 keeps 40 of run 10's 120)
        keep = [d for d in rows(src) if any(t.get("kind") == "unknown" for t in d["turns"])]
    else:
        raise SystemExit(f"unknown dialogue filter {how}")
    if limit and len(keep) > limit:
        import random
        keep = random.Random(seed).sample(keep, limit)
    write_rows(dst, keep)
    say(f"{Path(src).name}: kept {len(keep)} of {len(rows(src))} ({how}{f', at most {limit}' if limit else ''})")
    return dst


def filtered_calls(src, how):
    dst = OUT / (Path(src).stem + f"_{how}.jsonl")
    if how == "hazard_only":
        keep = [c for c in rows(src) if any(t["tag"] == "hazard_start" for t in c["ticks"])]
    else:
        raise SystemExit(f"unknown calls filter {how}")
    write_rows(dst, keep)
    say(f"{Path(src).name}: kept {len(keep)} of {len(rows(src))} ({how})")
    return dst


def stage_data(r):
    out = OUT / f"calls_{NAME}.jsonl"
    if exists(out):
        say(f"data: {out.name} exists, skipping")
        return
    b = r.get("build", {})
    common = ["--timing", b.get("timing", "mixed"), "--max-ticks", str(b.get("max_ticks", 460)), "--exclude-ids", "data/exclude_dialogues.txt",
              "--filler-min-talk", str(b.get("filler_min_talk", 3.0)), "--filler-gap", str(b.get("filler_gap", 5.0)), "--nudge-quiet", b.get("nudge_quiet", "2.6,5.5"), "--greetings", b.get("greetings", "service"),
              "--nudge-unanswered", str(b.get("nudge_unanswered", 0.0)), "--sentence-pauses", str(b.get("sentence_pauses", 0.0)),
              *(["--even-chunks"] if b.get("even_chunks") else [])]
    if b.get("fillers"):
        common += ["--filler-words", b["fillers"]]
    sources = []
    files = list(r.get("dialogue_files", []))
    if r.get("dialogues"):
        files.insert(0, {"file": f"out/dial_{NAME}.jsonl", "per": b.get("per", 2)})
    log = OUT / f"build_{NAME}.log"
    for i, f in enumerate(files):
        f = {"file": f} if isinstance(f, str) else f
        src = data_file(f["file"])
        if f.get("filter") and not DRY:
            src = filtered_dialogues(src, f["filter"], f.get("limit"), r.get("seed", 99))
        if r.get("drop_meetups") and not DRY:
            src = filtered_dialogues(src, "no_meetups")
        calls = OUT / f"calls_{NAME}_d{i}.jsonl"
        run(uv("build_calls.py", "--dialogues", str(src), "--per", str(f.get("per", b.get("per", 2))), "--seed", str(100 + i), *common, "--out", str(calls)), log=log)
        sources.append(str(calls))
    for m in r.get("mix", []):
        m = {"file": m} if isinstance(m, str) else m
        src = data_file(m["file"])
        if not src.exists() and Path(m["file"]).name == REAL_CALLS:
            continue
        if m.get("filter") and not DRY:
            src = filtered_calls(src, m["filter"])
        sources.append(f"{src}:{m['take']}" if m.get("take") else str(src))
    if not DRY:
        say(" ".join(l.strip() for l in open(log, encoding="utf-8") if l.startswith("wrote")))
    run(uv("mix_calls.py", "--calls", *sources, "--exclude-ids", "data/exclude_dialogues.txt", "--seed", str(r.get("seed", 99)), "--out", str(out)), log=OUT / f"mix_{NAME}.log")
    if DRY:
        return
    w = r.get("weights", {})
    n = {}
    calls = rows(out)
    for c in calls:
        if r.get("mute_service_greetings"):  # the words of its first reply are not trained on (the timing still is)
            first = []
            for t in c["ticks"]:
                if t["said"] != "<listen>":
                    first.append(t)
                elif first:
                    break
            if first and re.search(SERVICE_GREETING, " ".join(t["said"] for t in first), re.I):
                for t in first:
                    t["w"], t["tag"] = 0.0, "muted"
                n["service greeting 0"] = n.get("service greeting 0", 0) + 1
        if r.get("only_persona_calls") and not c.get("system"):
            # an add-on learns only from persona calls; the others stay in the file at weight 0 (the trainer skips them)
            # so its calls match the run whose cache it borrows
            for t in c["ticks"]:
                t["w"] = 0.0
            n["other calls 0"] = n.get("other calls 0", 0) + 1
            continue
        if r.get("persona_tags_only") and c.get("system"):
            # a persona call teaches only its memory moments: its other ticks (g26m, g26m3: the timing of these calls
            # made the whole model hold back) are not trained on
            keep = set(r["persona_tags_only"])
            # persona_ordinary_scale: the kept ticks that are not memory moments (its ordinary answers) at this share of
            # their weight (g26m5: at full weight they drowned the memory moments out)
            scale = float(r.get("persona_ordinary_scale", 1.0))
            # persona_other_scale: the ticks not kept at this share of their weight instead of none (an add-on trained
            # only on speaking would learn to speak and nothing pushing back: keep a little of when not to)
            other = float(r.get("persona_other_scale", 0.0))
            for t in c["ticks"]:
                if t["tag"] not in keep:
                    t["w"] = float(w.get(t["tag"], t["w"])) * other
                    continue
                if t["tag"] in w:
                    t["w"] = float(w[t["tag"]])
                if not t["tag"].startswith(("false_memory", "recall")):
                    t["w"] *= scale
            n["persona ticks 0"] = n.get("persona ticks 0", 0) + sum(t["tag"] not in keep for t in c["ticks"])
            continue
        real = bool(c.get("timing_only"))
        for t in c["ticks"]:
            tag = t["tag"]
            if real and r.get("real_mid_reply_off", True) and tag in ("speak", "self_resume", "overlap_start"):
                t["w"] = 0.0
                n["real mid-reply words 0"] = n.get("real mid-reply words 0", 0) + 1  # when to talk, not what a stranger said next
            elif tag in w and not (real and tag == "listen_done"):
                t["w"] = float(w[tag])
                n[f"{tag} {w[tag]}"] = n.get(f"{tag} {w[tag]}", 0) + 1
            # carry_on_first: the first token of a continuing tick is the decision to carry on rather than listen;
            # weighed like a decision while the rest of the words stay light (the real calls train no carry-on at all)
            if not real and tag == "speak" and r.get("carry_on_first"):
                t["w1"] = float(r["carry_on_first"])
                n[f"carry-on first token {r['carry_on_first']}"] = n.get(f"carry-on first token {r['carry_on_first']}", 0) + 1
    write_rows(out, calls)
    say(f"{len(calls)} calls, {sum(len(c['ticks']) for c in calls)} ticks in {out.name}; reweighted {n}")


def stage_train(r):
    gguf = OUT / f"{NAME}.gguf"
    if exists(gguf):
        say(f"train: {gguf.name} exists, skipping")
        return
    t = r.get("train", {})
    if not BASES[r.get("base", "original")]["hf"]:
        sys.exit(f"base {r.get('base', 'original')} has no training checkpoint in models/bases.json (hf): it can be called, not trained")
    hf = ROOT / BASES[r.get("base", "original")]["hf"]
    if not (hf / "config.json").exists() and not DRY:
        sys.exit(f"no training checkpoint in {hf.relative_to(ROOT)}: npm run setup -- --train {CANONICAL[r.get('base', 'original')]}")
    cache = OUT / f"cache_{r.get('cache', NAME)}"
    if cache.is_dir() and not DRY:  # a cache file cut off by a stop is removed, the rest is reused
        bad = []
        for f in sorted(cache.glob("call*.pt*"), key=lambda f: f.stat().st_mtime)[-40:]:
            if f.suffix != ".pt" or f.stat().st_size < 1000:
                bad.append(f.name)
                f.unlink()
        say(f"cache check: removed {bad or 'nothing'}")
    say(f"training {NAME} (rank {t.get('rank', 32)}, {t.get('steps', 600)} steps, base {r.get('base', 'original')}, prompt {r.get('prompt', 'assistant')}; resumes from saved steps)")
    code = run(uv("train_lora.py", "--hf", str(hf), "--calls", str(OUT / f"calls_{NAME}.jsonl"), "--cache", str(cache), "--max-tokens", str(t.get("max_tokens", 8000)),
                  "--chunk", str(t.get("chunk", 160)), "--steps", str(t.get("steps", 600)), "--batch", str(t.get("batch", 4)), "--rank", str(t.get("rank", 32)),
                  "--alpha", str(t.get("alpha", 2 * t.get("rank", 32))), "--lr", str(t.get("lr", 2e-4)), "--system", r.get("prompt", "assistant"),
                  "--top", str(t.get("top", 4)),  # how many of the last layers learn; another number needs another cache
                  *(["--experts"] if t.get("experts") else []),  # gpt-oss: its experts learn too (profiles.ExpertLoRA)
                  "--seed", str(t.get("seed", 0)),  # the adapter's starting values and the order calls are drawn in
                  "--lr-decay", t.get("lr_decay", "none"), "--val-calls", str(t.get("val_calls", 0)),
                  "--out", str(OUT / f"adapter_{NAME}"), "--resume",
                  # init_from: continue from another run's adapter (a second stage on top of it) instead of from zero
                  *(["--init-adapter", str(OUT / f"adapter_{r['init_from']}")] if r.get("init_from") else []),
                  # base_adapter: an add-on trained on top of another run's adapter, frozen (played together in the app)
                  *(["--base-adapter", str(OUT / f"adapter_{r['base_adapter']}")] if r.get("base_adapter") else []),
                  # another run's cache, this recipe's loss weights (the calls must be the same apart from weights)
                  *(["--weights-from-calls"] if r.get("cache") and r.get("cache") != NAME else []), train=True),
               log=OUT / f"{NAME}.log", check=False)
    if code != 0:
        say(f"training failed (out/{NAME}.log); the same command resumes it")
        sys.exit(1)
    run(uv("export_gguf.py", "--adapter", str(OUT / f"adapter_{NAME}"), "--hf", str(hf), "--out", str(gguf), train=True), log=OUT / f"{NAME}.export.log")
    # what the app needs to run it, inside the file (adapter_meta.py), so it works under any name; the app's name for
    # the base (the original Qwen is "qwen" there)
    base, tests = r.get("base", "original"), r.get("tests", {})
    run(uv("adapter_meta.py", str(gguf), "--base", CANONICAL[base], "--base-gguf", str(ROOT / BASES[base]["gguf"]),
           "--prompt", r.get("prompt", "assistant"), "--listen", str(tests.get("listen_p", 0.6)),
           *(["--carry-on", str(tests["carry_on_p"])] if tests.get("carry_on_p") else []),
           "--label", r.get("label") or suggested_label(r), "--recipe", NAME), log=OUT / f"{NAME}.export.log")
    if not DRY:
        shutil.copy(gguf, LORA / app_name(r))
    write_card(r, exported=True)
    say(f"exported out/{gguf.name}; in the app as {app_name(r)} (base {r.get('base', 'original')}, prompt {r.get('prompt', 'assistant')})")


def stage_replays(r):
    t = r.get("tests", {})
    p = str(t.get("listen_p", 0.6))
    gguf = OUT / f"{NAME}.gguf"
    want = [f for f, on in (("probe", t.get("quiz", True)), ("eval_yap", t.get("yap", 24)), ("eval_ours", t.get("ours", True)),
                            ("mem", t.get("memory", False))) if on and not tested(OUT / f"{f}_{NAME}.json")]
    if not want:
        say("replays: done already, skipping")
        return
    base = BASES[r.get("base", "original")]
    # an add-on is tested together with its base adapter, both at scale 1 (MicroTurnClient's "a.gguf:1,b.gguf:1")
    loras = ([str(OUT / f"{r['base_adapter']}.gguf")] if r.get("base_adapter") else []) + [str(gguf)]
    server(base["gguf"], "-np", "1", *[x for f in loras for x in ("--lora", f)], "--lora-init-without-apply", ctx=base.get("ctx", 32768))
    adapter = ",".join(f"{f}:1" for f in loras) if len(loras) > 1 else str(gguf)
    sysargs = ["--listen-p", p, "--repeat-guard", "--system", r.get("prompt", "assistant"), "--format", base.get("format", "qwen3.6")]
    if base.get("history_ticks"):  # the app keeps only this much of a call on this model (config.js BASES)
        sysargs += ["--history-ticks", str(base["history_ticks"])]
    # the app's carry-on bar (tests.carry_on_p, v1: 0.9) for the replays; the live bench has it from the app's own code
    bar = ["--carry-on-p", str(t["carry_on_p"])] if t.get("carry_on_p") else []
    if "probe" in want:
        run(uv("probe_answers.py", "--adapter", adapter, *sysargs, "--name", NAME, "--out", str(OUT / f"probe_{NAME}.json")), log=OUT / f"probe_{NAME}.log")
        if not DRY:
            say("quiz: " + open(OUT / f"probe_{NAME}.log", encoding="utf-8").read().strip().splitlines()[-1])
    if "eval_yap" in want:
        say(f"recorded conversations ({t.get('yap', 24)} calls)")
        run(uv("evaluate.py", "--url", "http://127.0.0.1:8080", "--calls", "data/heldout_yap.jsonl", "--n", str(t.get("yap", 24)), "--parallel", "1",
               "--adapter", adapter, *sysargs, *bar, "--out", str(OUT / f"eval_yap_{NAME}.json")), log=OUT / f"eval_yap_{NAME}.log")
        names = [c for c in t.get("compare", []) if exists(OUT / f"eval_yap_{c}.json")] + [NAME]
        run(uv("yap_metrics.py", "data/heldout_yap.jsonl", *[f"out/eval_yap_{c}.json" for c in names], "--first", str(t.get("yap", 24)), "--names", ",".join(names),
               "--md", f"out/yap_{NAME}.md"), check=False)
        say(f"recorded conversations: out/yap_{NAME}.md; {recall_tic(OUT / f'eval_yap_{NAME}.json')}")
    if "eval_ours" in want:
        say("our held-out set")
        run(uv("evaluate.py", "--url", "http://127.0.0.1:8080", "--calls", "data/heldout_ours.jsonl", "--n", "0", "--parallel", "1",
               "--adapter", adapter, *sysargs, *bar, "--out", str(OUT / f"eval_ours_{NAME}.json")), log=OUT / f"eval_ours_{NAME}.log")
        if not DRY:
            lines = open(OUT / f"eval_ours_{NAME}.log", encoding="utf-8").read().splitlines()
            i = next((k for k, l in enumerate(lines) if "== totals" in l), None)
            say("ours: " + " ".join(l.strip() for l in lines[i + 1:i + 8]) if i is not None else "ours: see the log")
    if "mem" in want:
        say("memory test (persona calls)")
        run(uv("evaluate.py", "--url", "http://127.0.0.1:8080", "--calls", str(MEMORY_CALLS), "--n", "0", "--parallel", "1",
               "--adapter", adapter, *sysargs, "--out", str(OUT / f"eval_mem_{NAME}.json")), log=OUT / f"eval_mem_{NAME}.log")
        run(uv("memory_check.py", "--calls", str(MEMORY_CALLS), "--eval", str(OUT / f"eval_mem_{NAME}.json"), "--out", str(OUT / f"mem_{NAME}.json")),
            log=OUT / f"mem_{NAME}.log")
        if not DRY:
            say(open(OUT / f"mem_{NAME}.log", encoding="utf-8").read().strip().splitlines()[-1])
    kill_server()


def recall_tic(eval_json):
    """How often it says "don't recall / don't remember" on the recorded calls, which have no persona (g26m7: 81, a tic
    from its memory training; g26 0)."""
    if not exists(eval_json):
        return "no replays"
    said = " ".join(" ".join(r["said"]) for r in json.loads(Path(eval_json).read_text(encoding="utf-8"))["replies"])
    return f"'don't recall/remember' said {len(re.findall(r'(don.?t|do not) (recall|remember)', said, re.I))} times"


def stage_live(r):
    t = r.get("tests", {})
    if not t.get("live", True):
        return
    kill_server()
    say("live tests (three callers)")
    env = {"MODEL": BASES[r.get("base", "original")]["gguf"], "PROMPT": r.get("prompt", "assistant")}
    adapter = f"tune/out/{NAME}.gguf"
    if r.get("base_adapter"):  # audio_test plays a blend of files from models/lora: the base's and this run's exports
        base_export = json.loads((TUNE / "recipes" / f"{r['base_adapter']}.json").read_text(encoding="utf-8"))["export_as"]
        adapter = f"blend:{base_export}=1,{r.get('export_as') or f'talkme-{NAME}.gguf'}=1"
    run(["bash", "scripts/live_bench.sh", NAME, adapter, str(t.get("listen_p", 0.6))], cwd=ROOT, env=env, log=LOG, check=False)
    names = [c for c in t.get("compare", []) if (ROOT / "tests" / "out" / f"report-bench-{c}-amy.json").exists() or DRY] + [NAME]
    run(["node", "scripts/live_bench_report.js", *names], cwd=ROOT, log=OUT / f"live_{NAME}.log", check=False)
    kill_server()
    say("done; GPU free (live report: tests/out/live_bench.md)")


def app_name(r):
    """The adapter's file name in models/lora."""
    return r.get("export_as") or f"talkme-{NAME}.gguf"


def suggested_label(r):
    """Its name in the app when the recipe gives none: what it is, not just the run's name."""
    return f"{r.get('prompt', 'assistant').capitalize()} {NAME} ({BASES[r.get('base', 'original')]['label']})"


def eval_result(path):
    j = json.loads(path.read_text(encoding="utf-8"))
    a = j.get("args", {})
    return {"run": NAME, "adapter": Path(str(a.get("adapter", ""))).name, "listen": a.get("listen_p"), "carryOn": a.get("carry_on_p"),
            "calls": len(j.get("calls", [])), "totals": j["totals"], "date": int(path.stat().st_mtime * 1000)}


def yap_table(path):
    """This run's column of the table yap_metrics.py wrote."""
    head, out = None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not line.startswith("|") or not cells or set(cells[0]) == {"-"}:
            continue
        if head is None:
            head = cells
        elif NAME in head:
            out[cells[0]] = cells[head.index(NAME)]
    return out


def run_results(r):
    """What this run's tests found, as the app's Adapters tab reads it (src/main/adapters.js), so the results stay with
    the adapter when tune/out and tests/out are not there."""
    out = {}
    for kind, file in (("quiz", OUT / f"probe_{NAME}.json"), ("wide", OUT / f"wide_{NAME}.json")):
        if tested(file):
            j = json.loads(file.read_text(encoding="utf-8"))
            out[kind] = {"run": NAME, "score": j["score"], "of": j["of"], "silent": j.get("silent"), "date": int(file.stat().st_mtime * 1000)}
            if kind == "quiz":
                out[kind]["listen"] = r.get("tests", {}).get("listen_p", 0.6)
    if tested(OUT / f"eval_ours_{NAME}.json"):
        out["ours"] = eval_result(OUT / f"eval_ours_{NAME}.json")
    if tested(OUT / f"eval_yap_{NAME}.json"):
        out["yap"] = eval_result(OUT / f"eval_yap_{NAME}.json")
        if tested(OUT / f"yap_{NAME}.md"):
            out["yap"]["table"] = yap_table(OUT / f"yap_{NAME}.md")
    tests = ROOT / "tests" / "out"
    live = {"run": NAME, "passed": 0, "total": 0, "callers": {}, "date": 0}
    reply, stop = [], []
    for file in filter(tested, tests.glob(f"report-bench-{NAME}-*.json")):
        j = json.loads(file.read_text(encoding="utf-8"))
        caller = file.stem[len(f"report-bench-{NAME}-"):]
        passed, total = sum(x["passed"] for x in j["results"]), sum(x["total"] for x in j["results"])
        live["adapter"], live["listen"] = Path(str(j.get("adapter", ""))).name, j.get("threshold")
        live["passed"] += passed
        live["total"] += total
        live["callers"][caller] = [passed, total]
        live["date"] = max(live["date"], int(file.stat().st_mtime * 1000))
        for c in (c for x in j["results"] for c in x["checks"]):  # as scripts/live_bench_report.js does
            m = re.search(r"first audio (-?[\d.]+) s", c["detail"]) if c["check"].startswith("reply within") else None
            n = re.search(r"cut ([\d.]+) s", c["detail"]) if c["check"].startswith("stops talking") else None
            if m:
                reply.append(float(m.group(1)))
            if n:
                stop.append(float(n.group(1)))
    if live["callers"]:
        if reply:
            live["medianReplyS"] = sorted(reply)[len(reply) // 2]
        if stop:
            live["medianStopS"] = sorted(stop)[len(stop) // 2]
        out["live"] = live
    return out


def write_card(r, exported=False):
    """The adapter's card, a small JSON file beside it in models/lora: which run made it, what that run's tests found and
    a suggested label, so a new adapter does not arrive as a bare run name. The app keeps notes and a label there too
    (src/main/config.js readCard): those stay, except that a fresh export with the recipe's own label goes by that.
    Nothing here may stop a run that has done its work."""
    gguf = LORA / app_name(r)
    if DRY or not gguf.exists():
        return
    path = gguf.with_suffix(".json")
    try:
        card = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        card.update(recipe=NAME, suggestedLabel=r.get("label") or suggested_label(r), results={**({} if exported else card.get("results", {})), **run_results(r)})
        if exported:
            card["trained"] = datetime.now().isoformat(timespec="seconds")
            if r.get("label"):
                card.pop("label", None)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        say(f"card: {path.name}")
    except (OSError, ValueError, KeyError) as e:
        say(f"could not write {path.name}: {e}")


def main():
    global DRY, LOG, NAME
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--stage", default="all", choices=["all", "dialogues", "data", "train", "replays", "live"])
    ap.add_argument("--dry-run", action="store_true", help="print the commands each stage would run")
    a = ap.parse_args()
    r = json.loads(Path(a.recipe).read_text(encoding="utf-8"))
    NAME = r["name"]
    DRY = a.dry_run
    LOG = OUT / f"{NAME}.pipeline.log"
    OUT.mkdir(exist_ok=True)
    check_inputs(r, a.stage)
    stages = {"dialogues": stage_dialogues, "data": stage_data, "train": stage_train, "replays": stage_replays, "live": stage_live}
    try:
        for name, fn in stages.items():
            if a.stage in ("all", name):
                if DRY:
                    print(f"--- {name}")
                fn(r)
    finally:  # what a stage finished before another failed still counts
        write_card(r)


if __name__ == "__main__":
    main()
