"""Shows or writes the talkme.* metadata in an adapter GGUF: which base model and system prompt it was trained with, the
listen threshold and carry-on bar it runs at, and its name in the app. The app reads these (src/main/gguf.js) instead of
going by the file name, so an adapter works under any name. llama-server ignores keys it does not know.

    uv run python adapter_meta.py ../models/lora/talkme-partner-v1-gemma4a4b.gguf          (show)
    uv run python adapter_meta.py FILE --base gemma4a4b --base-gguf ../models/base/gemma4/gemma-4-26B-A4B-it-heretic.Q4_K_M.gguf \
        --prompt partner --listen 0.6 --carry-on 0.9 --label "Partner v1" --recipe g26b4                         (write)

talkme.base is the app's and pipeline's name for the base (config.js / pipeline.py BASES); talkme.base_model is the base
GGUF's own general.name ("Gemma 4 26B A4B It Heretic"), which still finds the base if those names change.
"""
import argparse
import os
import sys
from pathlib import Path

import gguf
from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

S, F = gguf.GGUFValueType.STRING, gguf.GGUFValueType.FLOAT32


def show(path):
    r = gguf.GGUFReader(path)
    return {f.name: f.contents() for f in r.fields.values() if not f.name.startswith("GGUF.") and (f.name.startswith("talkme.") or f.name.startswith("general."))}


def base_name(base_gguf):
    """The general.name a base GGUF gives itself (only its header is read)."""
    f = gguf.GGUFReader(base_gguf).get_field("general.name")
    return f.contents() if f else None


def stamp(path, meta):
    """Rewrites `path` with the talkme.* keys in `meta` (base, base_model, prompt, label, recipe: text; listen_threshold,
    carry_on: numbers; None leaves a key out) added or replaced; tensors and other keys are copied as they are."""
    new = {f"talkme.{k}": MetadataDetails(F if isinstance(v, float) else S, v) for k, v in meta.items() if v is not None}
    r = gguf.GGUFReader(path)
    tmp = str(path) + ".stamp.tmp"
    w = gguf.GGUFWriter(tmp, arch=r.get_field("general.architecture").contents(), endianess=r.endianess)
    copy_with_new_metadata(r, w, new, [])
    del r  # the reader maps the file; let go of it before replacing it (Windows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--base", help="the base's key in config.js / pipeline.py BASES")
    ap.add_argument("--base-gguf", help="the base model GGUF, to record its own name")
    ap.add_argument("--prompt", choices=["partner", "assistant", "persona"])
    ap.add_argument("--listen", type=float, help="listen threshold it runs at")
    ap.add_argument("--carry-on", type=float, help="carry-on bar it runs at")
    ap.add_argument("--label", help="its name in the app's adapter list")
    ap.add_argument("--recipe", help="the run that made it (tune/recipes/<name>.json)")
    a = ap.parse_args()
    meta = {"base": a.base, "base_model": base_name(a.base_gguf) if a.base_gguf else None, "prompt": a.prompt,
            "listen_threshold": a.listen, "carry_on": a.carry_on, "label": a.label, "recipe": a.recipe}
    if any(v is not None for v in meta.values()):
        stamp(Path(a.file), meta)
    for k, v in show(a.file).items():
        print(f"{k:24} {v}")


if __name__ == "__main__":
    sys.exit(main())
