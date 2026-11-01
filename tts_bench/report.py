"""Bake-off summary: out/report.md (numbers) and out/listen.html (every line from every voice, side by side, audio
embedded as MP3 so the page is one file).   uv run --project ../tune python report.py"""
import base64
import json
import shutil
import statistics
import subprocess
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
LINES = json.loads((ROOT / "lines.json").read_text(encoding="utf-8"))
ORDER = ["kokoro", "kyutai-tts", "cosyvoice2", "chatterbox-turbo", "orpheus-3b"]
LABEL = {"kokoro": "Kokoro (today)", "kyutai-tts": "Kyutai TTS", "cosyvoice2": "CosyVoice 2", "chatterbox-turbo": "Chatterbox Turbo", "orpheus-3b": "Orpheus 3B"}
res = {json.loads(f.read_text(encoding="utf-8"))["model"]: json.loads(f.read_text(encoding="utf-8")) for f in OUT.glob("*.json")}
models = [m for m in ORDER if m in res]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def stat(m, mode, key):
    return med([r.get(key) for r in res[m]["rows"] if r["mode"] == mode])


f2 = lambda x, u="": "-" if x is None else f"{x:.2f}{u}"
L = ["# Voice bake-off", "", "Ten lines a friend might say (lines.json), each spoken whole and in the app's ~3-word pieces. "
     "Times on the RTX 4090 with nothing else running (Kokoro on the CPU, as in the app).", "",
     "| | " + " | ".join(LABEL[m] for m in models) + " |", "|---|" + "---|" * len(models),
     "| First sound, whole line (median) | " + " | ".join(f2(stat(m, "whole", "first_audio_s"), " s") for m in models) + " |",
     "| First sound, 3-word pieces (median) | " + " | ".join(f2(stat(m, "chunked", "first_audio_s"), " s") for m in models) + " |",
     "| Speed, whole line (x real time) | " + " | ".join(f2(stat(m, "whole", "realtime_x"), "x") for m in models) + " |",
     "| Recogniser word error, whole (median) | " + " | ".join(f2(stat(m, "whole", "wer")) for m in models) + " |",
     "| Extra GPU memory | " + " | ".join(("CPU" if m == "kokoro" else f"{res[m]['gpu_mb'] / 1024:.1f} GB") for m in models) + " |",
     "| Load time | " + " | ".join(f2(res[m]["load_s"], " s") for m in models) + " |", "",
     *[f"- **{LABEL[m]}**: {res[m]['note']}" for m in models]]
(OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))

ffmpeg = shutil.which("ffmpeg")


def audio_src(m, line, mode):
    wav = OUT / m / f"{line['id']}.{mode}.wav"
    if not wav.exists():
        return None
    if ffmpeg:
        mp3 = subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(wav), "-ac", "1", "-b:a", "64k", "-f", "mp3", "-"], capture_output=True).stdout
        return "data:audio/mpeg;base64," + base64.b64encode(mp3).decode()
    return "data:audio/wav;base64," + base64.b64encode(wav.read_bytes()).decode()


rows = []
for line in LINES:
    cells = []
    for m in models:
        r = {x["mode"]: x for x in res[m]["rows"] if x["id"] == line["id"]}
        players = []
        for mode, label in (("whole", "whole"), ("chunked", "3-word pieces")):
            src = audio_src(m, line, mode)
            if src:
                first = r.get(mode, {}).get("first_audio_s")
                players.append(f'<div class="p"><span>{label}{f" · {first:.2f} s" if first is not None else ""}</span><audio controls preload="none" src="{src}"></audio></div>')
        cells.append(f'<div class="cell"><h4>{escape(LABEL[m])}</h4>{"".join(players)}</div>')
    rows.append(f'<section><h3>{escape(line["text"])}</h3><p class="mood">{escape(line["mood"])}</p><div class="grid">{"".join(cells)}</div></section>')

table = "\n".join(L[4:11])
html = f"""<title>Voice Bake-off</title>
<style>
:root {{ --bg: #f6f5f2; --card: #ffffff; --fg: #1d1c1a; --muted: #6b6760; --line: #e3e0d9; --accent: #3b6e8f; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #151617; --card: #1e2022; --fg: #ecebe8; --muted: #a09c95; --line: #2e3134; --accent: #8db8d4; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #151617; --card: #1e2022; --fg: #ecebe8; --muted: #a09c95; --line: #2e3134; --accent: #8db8d4; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; margin: 0; padding-inline: 16px; padding-block: 24px 48px; }}
main {{ max-width: 1200px; margin: 0 auto; display: grid; gap: 28px; }}
h1 {{ font-size: 1.6rem; margin: 0; }} h3 {{ margin: 0; font-size: 1.05rem; text-wrap: balance; }} h4 {{ margin: 0 0 6px; font-size: .9rem; color: var(--accent); }}
.mood {{ margin: 2px 0 10px; color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 10px; }}
.cell {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px; min-width: 0; }}
.p {{ display: grid; gap: 2px; margin-top: 6px; }} .p span {{ font-size: .75rem; color: var(--muted); font-variant-numeric: tabular-nums; }}
audio {{ width: 100%; height: 32px; }}
.table {{ overflow-x: auto; }} table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: .9rem; }}
td, th {{ border-bottom: 1px solid var(--line); padding: 6px 10px; text-align: left; white-space: nowrap; }}
</style>
<main>
<header><h1>Voice bake-off</h1><p class="mood">same ten lines, five voices · whole line vs. the app's 3-word pieces</p></header>
<div class="table"><table>{"".join("<tr>" + "".join(f"<td>{escape(c.strip())}</td>" for c in r.strip("|").split("|")) + "</tr>" for r in table.splitlines() if not r.startswith("|---"))}</table></div>
{"".join(rows)}
</main>"""
(OUT / "listen.html").write_text(html, encoding="utf-8")
print(f"wrote {OUT / 'report.md'} and {OUT / 'listen.html'} ({len(html) / 1e6:.1f} MB)")
