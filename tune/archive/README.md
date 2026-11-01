# Old run scripts

The hand-written scripts behind runs 1-10 (Qwen3.6, October 2026), the voice-style adapters, the run 7 checks and the
GPU queues of 23-25 October. `pipeline.py` with `recipes/*.json` replaced them; they are kept as the record of how
those results in `docs/RESULTS.md` were made.

They will not run from here as they are: each one changes into its own folder and uses paths relative to `tune/`
(`out/`, `../vendor/llama`, `../models`), and most stop every running `llama-server`, the app's included.
