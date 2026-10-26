"""Run the final Colab adapter locally for a short financial-sentiment demo.

Example:
    python -m src.demo --config configs/base.yaml --adapter models/sft_adapter
"""
from __future__ import annotations
import argparse
import gradio as gr
import torch
from peft import PeftModel
from .config import load_config
from .models import base_model, tokenizer_for
from .parsing import parse_label
from .prompts import render_prompt

def main() -> None:
    p = argparse.ArgumentParser(description="Local 4-bit adapter demo")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--adapter", help="Adapter downloaded from your finished Colab run")
    p.add_argument("--share", action="store_true", help="Create a temporary public Gradio URL")
    args = p.parse_args()
    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise SystemExit("This demo expects an NVIDIA CUDA GPU. Use a GGUF/Ollama export for CPU inference.")
    tokenizer = tokenizer_for(cfg, "left")
    adapter = args.adapter or cfg["paths"]["dpo_adapter"]
    model = PeftModel.from_pretrained(base_model(cfg), adapter).eval()

    @torch.inference_mode()
    def classify(text: str):
        if not text.strip(): return "Enter a financial sentence.", ""
        prompt = render_prompt(tokenizer, text, cfg)
        batch = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**batch, do_sample=False, max_new_tokens=12, pad_token_id=tokenizer.pad_token_id)
        answer = tokenizer.decode(output[0, batch.input_ids.shape[1]:], skip_special_tokens=True).strip()
        parsed = parse_label(answer)
        return (parsed or "Could not parse a label"), answer

    ui = gr.Interface(classify, inputs=gr.Textbox(lines=4, label="Financial text"), outputs=[gr.Textbox(label="Predicted sentiment"), gr.Textbox(label="Raw model answer")], title="Financial Sentiment Analysis", description="Local 4-bit inference with the adapter produced in Colab.")
    ui.launch(share=args.share)

if __name__ == "__main__": main()
