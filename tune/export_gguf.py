"""Convert a PEFT adapter from train_lora.py into a llama.cpp LoRA GGUF.

Uses llama.cpp's own convert_lora_to_gguf.py (vendor/llama-src, same build as the server binary), with the base HF
checkpoint for the architecture mapping. If the converter rejects a tensor, retrain with that module removed from
--targets (llama.cpp cannot apply adapters to tensors it splits or permutes during conversion). gpt-oss's expert
adapters (train_lora.py --experts, a file of their own) are added to the converter's GGUF here, in llama.cpp's layout.

    uv run --extra train python export_gguf.py --adapter out/adapter --hf ../models/hf/Qwen3.6-35B-A3B --out out/talkme-lora.gguf
"""
import argparse
import gc
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "vendor" / "llama.cpp" / "convert_lora_to_gguf.py"


def add_experts(path, experts_file, n_experts):
    """Adds ExpertLoRA's tensors to the adapter GGUF: llama.cpp applies a LoRA to a layer's experts (ffn_gate_exps,
    ffn_up_exps, ffn_down_exps: one 3-D tensor each, an expert per slice) through per-expert A and B, so the shared A and
    B are written once for every expert."""
    import numpy as np
    from safetensors.numpy import load_file
    sys.path.insert(0, str(CONVERTER.parent / "gguf-py"))
    import gguf
    extra = []
    for name, t in sorted(load_file(str(experts_file)).items()):
        li, proj, ab = re.search(r"layers\.(\d+)\.mlp\.experts\.(gate|up|down)\.lora_([AB])$", name).groups()
        extra.append((f"blk.{li}.ffn_{proj}_exps.weight.lora_{ab.lower()}", np.ascontiguousarray(np.broadcast_to(t, (n_experts, *t.shape)), dtype=np.float32)))
    def read(path):  # copies, so no memory map keeps the file open when it is replaced (Windows)
        r = gguf.GGUFReader(path)
        kv = [(f.name, f.contents(), f.types[0], f.types[-1] if f.types[0] == gguf.GGUFValueType.ARRAY else None)
              for f in r.fields.values() if not f.name.startswith("GGUF.")]
        return kv, [(t.name, np.array(t.data), t.tensor_type) for t in r.tensors]
    kv, tensors = read(path)
    gc.collect()
    tmp = Path(str(path) + ".tmp")
    arch = next(v for k, v, _, _ in kv if k == gguf.Keys.General.ARCHITECTURE)
    w = gguf.GGUFWriter(tmp, arch)
    for k, v, vtype, sub in kv:
        if k != gguf.Keys.General.ARCHITECTURE:
            w.add_key_value(k, v, vtype, sub_type=sub)
    tensors += [(name, arr, gguf.GGMLQuantizationType.F32) for name, arr in extra]
    for name, arr, kind in tensors:
        w.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, kind)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for _, arr, _ in tensors:
        w.write_tensor_data(arr)
    w.close()
    os.replace(tmp, path)
    print(f"added {len(extra)} expert adapter tensors ({n_experts} experts each)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outtype", default="f32", choices=["f32", "f16", "bf16"])
    a = ap.parse_args()
    if not CONVERTER.exists():
        sys.exit(f"missing {CONVERTER}: clone llama.cpp at the server build tag into vendor/llama.cpp")
    env = {**os.environ, "PYTHONPATH": str(CONVERTER.parent / "gguf-py") + os.pathsep + str(CONVERTER.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, str(CONVERTER), "--base", a.hf, "--outfile", a.out, "--outtype", a.outtype, a.adapter]
    cfg = json.load(open(Path(a.hf) / "config.json"))
    if cfg.get("text_config", {}).get("model_type") == "ministral3":
        # llama.cpp's Mistral3Model hands the work to an inner Ministral3Model that reads the base checkpoint itself, so
        # the converter's LoRA override never runs and it writes the whole base model (54 GB) as the "adapter". Register
        # the inner class for the architecture and run the converter as usual.
        boot = ("import runpy, sys; from conversion import ModelBase, ModelType, get_model_class; "
                "get_model_class('Mistral3ForConditionalGeneration'); from conversion.mistral3 import Mistral3Model; "
                "ModelBase._model_classes[ModelType.TEXT]['Mistral3ForConditionalGeneration'] = Mistral3Model.Ministral3Model; "
                "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
        cmd = [sys.executable, "-c", boot, *cmd[1:]]
    print("$", " ".join(cmd), flush=True)
    # the converter crashed once (access violation) on a Gemma 4 adapter and then worked on every retry: try three times
    for attempt in range(3):
        r = subprocess.run(cmd, env=env)
        if r.returncode == 0:
            break
        print(f"conversion failed ({r.returncode}), attempt {attempt + 1} of 3", flush=True)
    if r.returncode != 0:
        sys.exit(f"conversion failed ({r.returncode})")
    experts = Path(a.adapter) / "experts_lora.safetensors"
    if experts.exists():
        add_experts(a.out, experts, cfg["num_local_experts"])
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 2**20:.1f} MB)")
    print(f"test it with: llama-server -m <base.gguf> --lora {a.out} ...")


if __name__ == "__main__":
    main()
