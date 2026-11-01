"""What differs between base models: the chat format of a micro-turn call, and (for training) how to build and run
the model's layers one at a time. Everything else in the pipeline is model-agnostic.

    qwen3.6  Qwen3.6-35B-A3B and the HauhauCS rebuild: ChatML, MoE + delta-net layers, mrope positions
    gemma4   Gemma 4 31B (and Heretic's decensored build): <|turn> format, dense, sliding-window attention, tied
             embeddings, a final logit soft-cap
    mistral  Ministral 3 14B: [INST] format, replies closed by </s> with nothing between turns; dense, full attention,
             YaRN positions
    harmony  gpt-oss-20b: OpenAI's harmony format, replies in the final channel closed by <|return|>; mixture of
             experts, attention with sinks, sliding-window and full layers in turn

A turn is rendered identically in training, replays and the app (src/main/llama.js renderPrompt), and nothing already
sent is ever edited, so llama.cpp's prompt cache resumes every tick. Gemma's own template adds an empty thought block
("<|channel>thought\\n<channel|>") before a fresh answer but not in past turns; it is left out everywhere so all model
turns look the same.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Format:
    bos: str          # prepended once (Gemma needs <bos>; ChatML has none)
    system: str       # "{s}" -> the system block
    user: str         # "{h}" -> what was heard, up to and including the opening of the reply
    end: str          # closes a reply (part of the loss: the model learns to stop)
    sep: str          # after a reply

    def call(self, system, ticks):
        """The rendered call and the (start, end) character span of every reply including its end marker."""
        text = self.bos + self.system.format(s=system)
        spans = []
        for heard, said in ticks:
            text += self.user.format(h=heard)
            s = len(text)
            text += said + self.end
            spans.append((s, len(text)))
            text += self.sep
        return text, spans

    def prompt(self, system, history, heard=None, bos=True):
        """history: [(role, content)] alternating user/assistant; heard: this tick's input (None: no new turn).
        bos=False for text sent to llama-server's /completion, which puts the BOS token in front itself for models that
        want one (Gemma): with ours too the model saw two, where training had one."""
        text = (self.bos if bos else "") + self.system.format(s=system)
        for role, content in history:
            text += self.user.format(h=content) if role == "user" else content + self.end + self.sep
        if heard is not None:
            text += self.user.format(h=heard)
        return text


FORMATS = {
    "qwen3.6": Format(bos="", system="<|im_start|>system\n{s}<|im_end|>\n", user="<|im_start|>user\n{h}<|im_end|>\n<|im_start|>assistant\n",
                      end="<|im_end|>", sep="\n"),
    "gemma4": Format(bos="<bos>", system="<|turn>system\n{s}<turn|>\n", user="<|turn>user\n{h}<turn|>\n<|turn>model\n",
                     end="<turn|>", sep="\n"),
    "mistral": Format(bos="<s>", system="[SYSTEM_PROMPT]{s}[/SYSTEM_PROMPT]", user="[INST]{h}[/INST]", end="</s>", sep=""),
    # gpt-oss: OpenAI's harmony format. Our prompt is the developer message (harmony's place for instructions), after a
    # system message that asks for low reasoning and allows only the final channel; every reply opens straight in the
    # final channel, so the model never starts thinking aloud mid-call. Replies end with <|return|>, also in past turns
    # (harmony writes <|end|> there): llama.cpp stops on <|return|> but not on <|end|> for this tokenizer, and a past
    # turn is never rewritten (the prompt cache).
    "harmony": Format(bos="", system="<|start|>system<|message|>Reasoning: low\n\n# Valid channels: final.<|end|>"
                                     "<|start|>developer<|message|>{s}<|end|>",
                      user="<|start|>user<|message|>{h}<|end|><|start|>assistant<|channel|>final<|message|>",
                      end="<|return|>", sep=""),
}


def model_family(config_or_dir):
    """'qwen3.6', 'gemma4', 'mistral' or 'harmony' (gpt-oss) from an HF config (or a checkpoint dir)."""
    if isinstance(config_or_dir, (str, bytes)) or hasattr(config_or_dir, "__fspath__"):
        import json
        from pathlib import Path
        config_or_dir = json.load(open(Path(config_or_dir) / "config.json"))
    mt = config_or_dir.get("model_type") if isinstance(config_or_dir, dict) else getattr(config_or_dir, "model_type", "")
    if str(mt).startswith("gemma4"):
        return "gemma4"
    if mt in ("mistral3", "ministral3"):
        return "mistral"
    if mt == "gpt_oss":
        return "harmony"
    return "qwen3.6"


# ----------------------------------------------------------------------------------------------------------------------
# training: one family's layers, run one at a time (train_lora.py)
# ----------------------------------------------------------------------------------------------------------------------
class Qwen36:
    family = "qwen3.6"
    prefix = "model.language_model."
    targets = "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,shared_expert.gate_proj,shared_expert.up_proj,shared_expert.down_proj"
    lm_head_key = "lm_head."  # its own tensor

    def __init__(self, tcfg, device):
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
        self.M, self.tcfg = M, tcfg
        self.rotary = M.Qwen3_5MoeTextRotaryEmbedding(config=tcfg).to(device)

    def layer(self, li):
        return self.M.Qwen3_5MoeDecoderLayer(self.tcfg, li)

    def norm(self):
        return self.M.Qwen3_5MoeRMSNorm(self.tcfg.hidden_size, eps=self.tcfg.rms_norm_eps)

    def embed(self):
        import torch
        return torch.nn.Embedding(self.tcfg.vocab_size, self.tcfg.hidden_size, self.tcfg.pad_token_id)

    def context(self, T, device):
        """Whatever every layer of a T-token sequence needs (positions, masks), computed once per sequence."""
        import torch
        pos = torch.arange(T, device=device).view(1, 1, -1).expand(4, 1, -1)
        return {"text_pos": pos[0], "mrope_pos": pos[1:]}

    def run(self, layer, h, li, ctx):
        pe = self.rotary(h, ctx["mrope_pos"])
        return layer(h, position_embeddings=pe, attention_mask=None, position_ids=ctx["text_pos"], past_key_values=None)

    def logits(self, lm_head, hs):
        return lm_head(hs).float()


class Gemma4:
    family = "gemma4"
    prefix = "model.language_model."
    # attention and MLP of the trained layers (Gemma's full-attention layers share K and V: no v_proj there)
    targets = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lm_head_key = "model.language_model.embed_tokens."  # tied to the input embedding

    def __init__(self, tcfg, device):
        from transformers.models.gemma4 import modeling_gemma4 as M
        self.M, self.tcfg = M, tcfg
        self.rotary = M.Gemma4TextRotaryEmbedding(config=tcfg).to(device)
        self.softcap = getattr(tcfg, "final_logit_softcapping", None)

    def layer(self, li):
        return self.M.Gemma4TextDecoderLayer(self.tcfg, li)

    def norm(self):
        return self.M.Gemma4RMSNorm(self.tcfg.hidden_size, eps=self.tcfg.rms_norm_eps)

    def embed(self):
        # scaled by sqrt(hidden) as the model does; the scale is a buffer the checkpoint does not hold
        m = self.M.Gemma4TextScaledWordEmbedding(self.tcfg.vocab_size, self.tcfg.hidden_size, self.tcfg.pad_token_id,
                                                 embed_scale=self.tcfg.hidden_size ** 0.5)
        m.fixed_buffers = {"embed_scale": self.tcfg.hidden_size ** 0.5}
        return m

    def context(self, T, device):
        import torch
        from collections import UserDict
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        pos = torch.arange(T, device=device)[None]
        dummy = torch.empty(1, T, 1, device=device, dtype=torch.bfloat16)
        kw = {"config": self.tcfg, "inputs_embeds": dummy, "attention_mask": None, "past_key_values": None, "position_ids": pos}
        masks = {"full_attention": create_causal_mask(**kw), "sliding_attention": create_sliding_window_causal_mask(**kw)}
        return {"pos": pos, "masks": masks, "pe": {}, "shared": UserDict()}

    def run(self, layer, h, li, ctx):
        lt = self.tcfg.layer_types[li]
        if lt not in ctx["pe"]:
            ctx["pe"][lt] = self.rotary(h, ctx["pos"], lt)
        return layer(h, None, shared_kv_states=ctx["shared"], position_embeddings=ctx["pe"][lt], attention_mask=ctx["masks"][lt],
                     position_ids=ctx["pos"], past_key_values=None)

    def logits(self, lm_head, hs):
        import torch
        x = lm_head(hs).float()
        if self.softcap:
            x = torch.tanh(x / self.softcap) * self.softcap
        return x


class Ministral3:
    family = "mistral"
    prefix = "model.language_model."  # as train_lora.Shards reads it (the checkpoint says "language_model.model.")
    targets = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lm_head_key = "lm_head."  # its own tensor
    # the tokenizer file's pre-split pattern is wrong for some text (transformers warns); the fix matches Mistral's own
    tokenizer_kw = {"fix_mistral_regex": True}

    def __init__(self, tcfg, device):
        from transformers.models.ministral3 import modeling_ministral3 as M
        self.M, self.tcfg = M, tcfg
        self.rotary = M.Ministral3RotaryEmbedding(config=tcfg).to(device)

    def layer(self, li):
        return self.M.Ministral3DecoderLayer(self.tcfg, li)

    def norm(self):
        return self.M.Ministral3RMSNorm(self.tcfg.hidden_size, eps=self.tcfg.rms_norm_eps)

    def embed(self):
        import torch
        return torch.nn.Embedding(self.tcfg.vocab_size, self.tcfg.hidden_size, self.tcfg.pad_token_id)

    def context(self, T, device):
        # full causal attention (no sliding window): no mask, SDPA's causal flag; the positions also set its long-context
        # query scale, which is 1 below 16,384 tokens
        import torch
        return {"pos": torch.arange(T, device=device)[None]}

    def run(self, layer, h, li, ctx):
        if "pe" not in ctx:
            ctx["pe"] = self.rotary(h, ctx["pos"])
        return layer(h, attention_mask=None, position_ids=ctx["pos"], past_key_values=None, position_embeddings=ctx["pe"])

    def logits(self, lm_head, hs):
        return lm_head(hs).float()


def sink_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, sliding_window=None, s_aux=None,
                   chunk=512, **kwargs):
    """gpt-oss attention for training: what transformers' eager version computes (a learned "sink" logit per head in
    every softmax, every other layer a 128-token sliding window), a block of queries at a time. Eager attention holds the
    whole T x T score matrix of all 64 heads, ~16 GB at 8,000 tokens; PyTorch's fused kernels have no sinks. Each block
    is recomputed during backward (checkpoint), so only one block's scores exist at a time. The mask comes from the
    positions and the window; attention_mask is not used. Softmax in float32, as llama.cpp does it."""
    import torch
    B, H, T, D = query.shape
    rep = H // key.shape[1]
    k, v = key.repeat_interleave(rep, dim=1), value.repeat_interleave(rep, dim=1)  # = transformers' repeat_kv
    sink = s_aux.float().view(1, H, 1, 1)

    def block(q, kk, vv, s, lo):
        qpos = torch.arange(s, s + q.shape[2], device=q.device)[:, None]
        kpos = torch.arange(lo, lo + kk.shape[2], device=q.device)[None]
        ok = kpos <= qpos
        if sliding_window:
            ok &= kpos > qpos - sliding_window
        x = torch.matmul(q.float(), kk.float().transpose(2, 3)) * scaling
        x = x.masked_fill(~ok, float("-inf"))
        x = torch.cat([x, sink.expand(q.shape[0], H, q.shape[2], 1)], dim=-1)
        p = torch.softmax(x, dim=-1)[..., :-1]  # the sink takes its share of the weight, then is dropped
        return torch.matmul(p.to(vv.dtype), vv)

    out = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        lo = max(0, s - sliding_window + 1) if sliding_window else 0
        args = (query[:, :, s:e], k[:, :, lo:e], v[:, :, lo:e], s, lo)
        out.append(torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False) if torch.is_grad_enabled() else block(*args))
    return torch.cat(out, dim=2).transpose(1, 2).contiguous(), None


def expert_lora_class():
    import math
    import torch

    class ExpertLoRA(torch.nn.Module):
        """LoRA on gpt-oss's experts (the fused 3-D gate_up and down tensors PEFT cannot wrap): one adapter per projection
        (gate, up, down) shared by all 32 experts of the layer, so each learns from every token, not the 1 in 8 an expert
        sees. Computes what transformers' grouped_mm experts forward does (the kernel the trainer picks) plus the
        adapters; gate and up are the even and odd columns of gate_up, as the model's _apply_gate reads them. Exported to
        llama.cpp as the same A and B repeated for every expert (export_gguf.py)."""
        NAMES = ("gate", "up", "down")

        def __init__(self, experts, rank, alpha):
            super().__init__()
            self.base = experts
            hid, inter = experts.hidden_size, experts.intermediate_size
            self.scale = alpha / rank
            dev = experts.down_proj.device
            for n, (i, o) in zip(self.NAMES, ((hid, inter), (hid, inter), (inter, hid))):
                A = torch.empty(rank, i, device=dev)
                torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))  # as PEFT starts its A; B starts at zero
                setattr(self, f"{n}_A", torch.nn.Parameter(A))
                setattr(self, f"{n}_B", torch.nn.Parameter(torch.zeros(o, rank, device=dev)))

        def delta(self, n, x):  # in float32, and added in float32: a small change to a bf16 value would round away
            return x.float() @ getattr(self, f"{n}_A").T @ getattr(self, f"{n}_B").T * self.scale

        def forward(self, hidden_states, router_indices=None, routing_weights=None):
            from transformers.integrations.moe import _grouped_linear
            e, x = self.base, hidden_states
            k = router_indices.shape[-1]
            ids, perm = torch.sort(router_indices.reshape(-1))  # token-expert pairs, grouped by expert
            tok = perm // k
            counts = torch.histc(ids.float() if x.device.type == "cpu" else ids.int(), bins=e.num_experts, min=0, max=e.num_experts - 1)
            offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
            gate_up = _grouped_linear(x[tok], e.gate_up_proj, offsets, bias=e.gate_up_proj_bias[ids], is_transposed=True)
            # the adapters' change to gate and up, interleaved as gate_up holds them
            gate_up = gate_up.float() + torch.stack((self.delta("gate", x), self.delta("up", x)), dim=-1).flatten(-2)[tok]
            h = e._apply_gate(gate_up)
            o = _grouped_linear(h.to(x.dtype), e.down_proj, offsets, bias=e.down_proj_bias[ids], is_transposed=True).float() + self.delta("down", h)
            o = o * routing_weights.reshape(-1)[perm, None].float()
            return torch.zeros(x.shape, dtype=torch.float32, device=x.device).index_add_(0, tok, o).to(x.dtype)

    return ExpertLoRA


class GptOss:
    """gpt-oss-20b: harmony format; mixture of experts (32, 4 active), attention with sinks, sliding-window (128) and
    full attention layers in turn. The experts are fused 3-D tensors, not Linear layers: PEFT's LoRA sits on attention,
    and --experts adds ExpertLoRA on the experts. Trains from a BF16 checkpoint (OpenAI's own stores the experts in
    MXFP4)."""
    family = "harmony"
    prefix = "model.language_model."  # as train_lora.Shards reads it (the checkpoint says "model.")
    targets = "q_proj,k_proj,v_proj,o_proj"
    lm_head_key = "lm_head."  # its own tensor

    def __init__(self, tcfg, device):
        from transformers import AttentionInterface
        from transformers.models.gpt_oss import modeling_gpt_oss as M
        AttentionInterface.register("talkme_sinks", sink_attention)
        tcfg._attn_implementation = "talkme_sinks"
        self.M, self.tcfg = M, tcfg
        self.rotary = M.GptOssRotaryEmbedding(config=tcfg).to(device)

    def layer(self, li):
        return self.M.GptOssDecoderLayer(self.tcfg, li)

    def norm(self):
        return self.M.GptOssRMSNorm(self.tcfg.hidden_size, eps=self.tcfg.rms_norm_eps)

    def embed(self):
        import torch
        return torch.nn.Embedding(self.tcfg.vocab_size, self.tcfg.hidden_size, self.tcfg.pad_token_id)

    def context(self, T, device):
        import torch
        return {"pos": torch.arange(T, device=device)[None]}

    def run(self, layer, h, li, ctx):
        if "pe" not in ctx:
            ctx["pe"] = self.rotary(h, ctx["pos"])
        return layer(h, attention_mask=None, position_ids=ctx["pos"], past_key_values=None, position_embeddings=ctx["pe"])

    def logits(self, lm_head, hs):
        return lm_head(hs).float()

    def add_expert_lora(self, layer, rank, alpha):
        """Puts ExpertLoRA on the layer's experts; returns it."""
        layer.mlp.experts = expert_lora_class()(layer.mlp.experts, rank, alpha)
        return layer.mlp.experts


TRAIN = {"qwen3.6": Qwen36, "gemma4": Gemma4, "mistral": Ministral3, "harmony": GptOss}
