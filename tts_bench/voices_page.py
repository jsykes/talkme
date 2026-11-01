"""One listening page for the voice samples (sample_voices.py): every voice of both engines on the same four lines,
audio embedded as MP3.   uv run --project ../tune python voices_page.py"""
import base64
import json
import shutil
import subprocess
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out" / "voices"
KOKORO = {"3": "Heart (US, female)", "2": "Bella (US, female)", "7": "Nova (US, female; your Kokoro voice now)", "11": "Adam (US, male)",
          "16": "Michael (US, male)", "21": "Emma (UK, female)", "26": "George (UK, male)"}
ORPHEUS = {"tara": "female", "leah": "female", "jess": "female", "mia": "female", "zoe": "female", "leo": "male", "dan": "male", "zac": "male"}
ffmpeg = shutil.which("ffmpeg")


def src(wav):
    mp3 = subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(wav), "-ac", "1", "-b:a", "64k", "-f", "mp3", "-"], capture_output=True).stdout
    return "data:audio/mpeg;base64," + base64.b64encode(mp3).decode()


def label(engine, voice):
    if engine == "orpheus":
        return f"Orpheus {voice.capitalize()}", ORPHEUS.get(voice, ""), f"orpheus:{voice}"
    fam = "Chatterbox" if engine == "chatterbox" else "Original"
    if voice == "built-in":
        return f"{fam} built-in", "its own voice", f"{engine}:built-in"
    sid = voice.split("-")[1]
    return f"{fam} as {KOKORO.get(sid, sid).split(' (')[0]}", "cloned from Kokoro " + KOKORO.get(sid, sid), f"{engine}:{voice}"


sections = []
lines = None
for engine in ("orpheus", "chatterbox", "chatterbox-original"):
    f = OUT / f"{engine}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    lines = d["lines"]
    rows = []
    for v in d["voices"]:
        name, kind, vid = label(engine, v["voice"])
        cells = "".join(f'<td><audio controls preload="none" src="{src(OUT / v["dir"] / f"{i}.wav")}"></audio></td>' for i in range(len(lines)))
        rows.append(f'<tr><th><b>{escape(name)}</b><span>{escape(kind)}</span><code>{escape(vid)}</code></th>{cells}</tr>')
    head = "".join(f"<th>{escape(l)}</th>" for l in lines)
    title = {"orpheus": "Orpheus 3B", "chatterbox": "Chatterbox Turbo (350M)", "chatterbox-original": "Chatterbox original (500M)"}[engine]
    rt = d.get("realtime_x") or {"orpheus": 2.08}.get(engine)
    title += f" · {rt:.1f}x real time" + (" (too slow for live calls as it is)" if rt and rt < 1.1 else "") if rt else ""
    sections.append(f'<section><h2>{escape(title)}</h2><div class="scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>')

# Orpheus at three quantizations (orpheus_quants.py): the same lines, Tara and Leo, with speed
qf = OUT / "orpheus-quants.json"
if qf.exists():
    q = json.loads(qf.read_text(encoding="utf-8"))
    head = "".join(f"<th>{escape(l)}</th>" for l in q["lines"])
    rows = []
    for r in q["results"]:
        for sp in q["speakers"]:
            d = OUT / f"orpheus-{r['quant'].lower()}-{sp}"
            cells = "".join(f'<td><audio controls preload="none" src="{src(d / f"{i}.wav")}"></audio></td>' for i in range(len(q["lines"])))
            rows.append(f'<tr><th><b>{escape(r["quant"])} {sp.capitalize()}</b><span>{r["size_gb"]} GB · {r["tokens_per_s"]} tokens/s · {r["realtime_x"]}x real time</span></th>{cells}</tr>')
    sections.insert(0, f'<section><h2>Orpheus quantizations: Q8 (now) vs Q6 vs Q4</h2><div class="scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>')

html = f"""<title>Voice Picker</title>
<style>
:root {{ --bg: #f6f5f2; --card: #fff; --fg: #1d1c1a; --muted: #6b6760; --line: #e3e0d9; --accent: #3b6e8f; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #151617; --card: #1e2022; --fg: #ecebe8; --muted: #a09c95; --line: #2e3134; --accent: #8db8d4; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #151617; --card: #1e2022; --fg: #ecebe8; --muted: #a09c95; --line: #2e3134; --accent: #8db8d4; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.45 system-ui, sans-serif; margin: 0; padding-inline: 16px; padding-block: 24px 48px; }}
main {{ max-width: 1300px; margin: 0 auto; display: grid; gap: 28px; }}
h1 {{ margin: 0; font-size: 1.6rem; }} h2 {{ margin: 0 0 8px; font-size: 1.15rem; color: var(--accent); }}
p {{ margin: 4px 0 0; color: var(--muted); }}
.scroll {{ overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 8px; }}
table {{ border-collapse: collapse; min-width: 900px; width: 100%; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
thead th {{ font-weight: 500; font-size: .82rem; color: var(--muted); }}
tbody th {{ width: 200px; }} tbody th span {{ display: block; font-weight: 400; font-size: .8rem; color: var(--muted); }}
tbody th code {{ display: block; font-size: .72rem; color: var(--muted); margin-top: 2px; }}
audio {{ width: 190px; height: 32px; }}
</style>
<main>
<header><h1>Voice picker</h1><p>The same four lines from each voice, whole lines on a free GPU. The code under each name is how the app would call it.</p></header>
{"".join(sections)}
</main>"""
(OUT / "voices.html").write_text(html, encoding="utf-8")
print(f"wrote {OUT / 'voices.html'} ({len(html) / 1e6:.1f} MB)")
