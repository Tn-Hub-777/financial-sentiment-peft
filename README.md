# Financial sentiment analysis: QLoRA SFT + DPO

This is a Colab-oriented, config-driven proof of concept testing whether QLoRA supervised fine-tuning and preference alignment improve three-class financial sentiment classification.

## Learning sequence

1. **Attention Is All You Need** — self-attention lets each token weight other tokens.
2. **Transformer architecture** — stacked attention/MLP blocks predict the next token; Qwen is a decoder-only transformer.
3. **Questions to ask of an LLM slide** — What data, objective, context length, tokenizer, model size, alignment method, evaluation and failure modes are being claimed?
4. **Context window** — the maximum tokens the model can condition on. This project caps SFT examples at 256 tokens and DPO sequences at 768 to fit a T4.
5. **Tokenizer algorithm** — subword tokenizers (Qwen uses a BPE-family tokenizer) turn text into IDs; labels must be evaluated through the same tokenizer/template as training.
6. **Base vs instruct models** — base models continue text; instruct models have been trained to follow chat-style requests. Qwen3 is prompted through its chat template.
7. **Open source vs open weight** — open weights permit downloading parameters; this does not necessarily mean training data, code, or license are fully open source.
8. **Adaptation ladder** — SFT teaches the requested task; full fine-tuning changes all weights; adapters add small trainable modules; LoRA is a low-rank adapter; QLoRA trains LoRA adapters over 4-bit frozen weights; DPO/RLHF uses preferences to align responses. This repository uses QLoRA then DPO (a preference-optimization alternative to the reward-model/PPO portion of classical RLHF).

## Setup (Google Colab T4)

```bash
pip install -r requirements-colab.txt
pytest -q
python -m src.data --config configs/base.yaml prepare
python -m src.train_sft --config configs/base.yaml
python -m src.evaluate --config configs/base.yaml --model base --split allagree
python -m src.evaluate --config configs/base.yaml --model sft --split allagree
python -m src.make_preferences --config configs/base.yaml
python -m src.train_dpo --config configs/base.yaml
python -m src.evaluate --config configs/base.yaml --model dpo --split allagree
```

## Recommended hardware workflow

Use **Colab** for all Qwen3-4B experiments and reported metrics. A T4 has 16 GB VRAM and provides one consistent fp16 environment for M0–M2. Save `models/sft_adapter`, `models/dpo_adapter`, `results/`, and checkpoints to Google Drive at the end of every run.

Use a 6 GB RTX 4050 only for local development and the final short-input demo. Debug the same code with `configs/dev_local.yaml`, which uses Qwen3-1.7B and a tiny data/run budget; those metrics are not research results. Download the final **adapter** from Colab, not a merged full-precision model, then launch the demo locally:

```bash
python -m src.demo --config configs/base.yaml --adapter path/to/dpo_adapter
```

The local demo loads Qwen3-4B in 4-bit plus the adapter. Keep inputs short; if Windows/bitsandbytes causes CUDA issues, run it in WSL2 or export to a 4-bit GGUF for Ollama/llama.cpp. Do not use DPO training on the 6 GB laptop.

Run the baseline *before* SFT, then evaluate SFT, then train/evaluate DPO. Every run writes CSV predictions and JSON metrics below `results/`; do not report a score above 95 macro-F1 without inspecting the leakage report.

## Experimental controls

- PhraseBank examples are removed from SFT via normalized exact matching and optional fuzzy matching.
- FinGPT inputs are deduplicated and group-split by input, preventing a text from appearing in both train and validation.
- SFT collapses graded labels to exactly `negative`, `neutral`, and `positive`.
- Main evaluation is PhraseBank `allagree`; other agreement subsets are robustness checks.
- DPO's personal-finance preference data is explicitly domain-mismatched; its duplicates, pairs-per-prompt cap, and chosen/rejected length bias are reported.
- Base, training, and evaluation prompts all go through `src.prompts.render_prompt` with `enable_thinking=False`.

See [docs/limitations.md](docs/limitations.md) for limitations and `configs/base.yaml` for every experimental setting.
