"""Checks the trainer's layer-by-layer forward pass (train_lora.py + profiles.py) against llama.cpp on the same
prompt: streams every layer of the HF checkpoint through the GPU like phase A does, then compares the next-token
distribution with a running llama-server's (its GGUF quantization differs, so the top tokens should agree, not the
exact numbers). Run before training a new model family.
    uv run --extra train python check_forward.py --hf ../models/hf/gemma-4-31B-it-heretic --url http://127.0.0.1:8080"""
import argparse
import math

import requests
import torch

from common import SYSTEMS
from profiles import FORMATS
from train_lora import PREFIX, Shards, build_pieces, load_module

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True)
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--only", type=int, default=None, help="check just this prompt (0-based)")
a = ap.parse_args()

device = "cuda"
cfg, tcfg, P = build_pieces(a.hf, device)
fmt = FORMATS[P.family]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.hf)
# an ordinary chat turn and a micro-turn one: the model should continue both plausibly
prompts = [
    fmt.prompt("You are a helpful assistant.", [], "What is the capital of France? Answer in one word."),
    fmt.prompt(SYSTEMS["partner"], [("user", "<no voice>"), ("assistant", "<listen>")], "Hey, how's it going?"),
]
if P.family == "gemma4":  # past the thought block Gemma's template expects, the answer itself is the next token
    prompts.append(prompts[0] + "<|channel>thought\n<channel|>")
# a long call (~2,500 tokens): sliding-window attention only matters past its window (Gemma: 1,024 tokens)
from common import read_jsonl
_call = read_jsonl("out/calls_run12.jsonl")[0]
_hist = [x for t in _call["ticks"][:150] for x in (("user", t["heard"]), ("assistant", t["said"]))]
prompts.append(fmt.prompt(SYSTEMS["partner"], _hist, "Wait, what's twelve times twelve?"))
shards = Shards(a.hf)
if a.only is not None:
    prompts = [prompts[a.only]]
for text in prompts:
    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"], device=device)[None]
    with torch.device("meta"):
        emb = P.embed()
    emb = load_module(emb, shards, PREFIX + "embed_tokens.", device)
    with torch.no_grad():
        h = emb(ids)
    del emb
    ctx = P.context(h.shape[1], device)
    for li in range(tcfg.num_hidden_layers):
        with torch.device("meta"):
            layer = P.layer(li)
        layer = load_module(layer, shards, f"{PREFIX}layers.{li}.", device).eval()
        with torch.no_grad():
            h = P.run(layer, h, li, ctx)
        del layer
        torch.cuda.empty_cache()
    with torch.device("meta"):
        norm = P.norm()
        head = torch.nn.Linear(tcfg.hidden_size, tcfg.vocab_size, bias=False)
    norm = load_module(norm, shards, PREFIX + "norm.", device)
    head = load_module(head, shards, P.lm_head_key, device, rename={"weight": "weight"})
    with torch.no_grad():
        logits = P.logits(head, norm(h)[0, -1])
    del head, norm
    torch.cuda.empty_cache()
    probs = torch.softmax(logits, -1)
    top = torch.topk(probs, 5)
    ours = [(tok.decode([int(i)]), round(float(p), 3)) for p, i in zip(top.values, top.indices)]
    r = requests.post(a.url + "/completion", json={"prompt": text, "n_predict": 1, "n_probs": 5, "temperature": 0}).json()
    theirs = [(t["token"], round(math.exp(t["logprob"]), 3)) for t in r["completion_probabilities"][0]["top_logprobs"]]
    print(f"prompt ({ids.shape[1]} tokens) ends:", repr(text[-60:]))
    print("  trainer  :", ours)
    print("  llama.cpp:", theirs)
    print("  top token agrees:", ours[0][0].strip() == theirs[0][0].strip())
