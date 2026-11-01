#!/usr/bin/env bash
# Weights for the TTS bake-off. Resumable: rerun to continue.
set -u
# CosyVoice code (Apache 2.0): git clone --depth 1 --recurse-submodules https://github.com/FunAudioLLM/CosyVoice.git tts_bench/cosyvoice/CosyVoice
cd "$(dirname "$0")/../models/tts_bench"
HF="uvx --from huggingface_hub hf"
$HF download unsloth/orpheus-3b-0.1-ft-GGUF orpheus-3b-0.1-ft-Q8_0.gguf --local-dir orpheus
$HF download hubertsiuzdak/snac_24khz --local-dir snac_24khz
$HF download ResembleAI/chatterbox-turbo --local-dir chatterbox-turbo
$HF download kyutai/tts-1.6b-en_fr --local-dir kyutai-tts-1.6b
$HF download kyutai/tts-voices --include "unmute-prod-website/*" "expresso/*" --local-dir kyutai-voices
$HF download FunAudioLLM/CosyVoice2-0.5B --local-dir CosyVoice2-0.5B
echo "downloads done"
uvx --from huggingface_hub hf download ResembleAI/chatterbox ve.safetensors t3_cfg.safetensors s3gen.safetensors tokenizer.json conds.pt README.md --local-dir chatterbox-original
uvx --from huggingface_hub hf download unsloth/orpheus-3b-0.1-ft-GGUF orpheus-3b-0.1-ft-Q6_K.gguf orpheus-3b-0.1-ft-Q4_K_M.gguf --local-dir orpheus
