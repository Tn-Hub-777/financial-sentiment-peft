from __future__ import annotations
from pathlib import Path
import json
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from .config import config_arg, load_config, seed_everything
from .models import base_model, print_versions, tokenizer_for
from .prompts import completion, render_prompt

def tokenize(dataset, tokenizer, cfg):
    max_len = cfg["sft"]["train"]["max_seq_length"]
    def fn(row):
        prompt = render_prompt(tokenizer, row["text"], cfg)
        answer = completion(row["label"], tokenizer)
        p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        a = tokenizer(answer, add_special_tokens=False)["input_ids"]
        ids = (p + a)[:max_len]
        labels = ([-100] * len(p) + a)[:max_len]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
    return dataset.map(fn, remove_columns=dataset.column_names)

def run(cfg: dict) -> None:
    seed_everything(cfg["seed"]); print_versions()
    tok = tokenizer_for(cfg, "right")
    ds = {s: load_dataset("json", data_files=str(Path(cfg["paths"]["processed"]) / f"sft_{s}.jsonl"), split="train") for s in ("train", "val")}
    train, val = tokenize(ds["train"], tok, cfg), tokenize(ds["val"], tok, cfg)
    model = prepare_model_for_kbit_training(base_model(cfg), use_gradient_checkpointing=cfg["sft"]["train"]["gradient_checkpointing"])
    l = cfg["sft"]["lora"]
    model = get_peft_model(model, LoraConfig(r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"], target_modules=l["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    t = cfg["sft"]["train"]
    args = TrainingArguments(output_dir=str(Path(cfg["paths"]["results"]) / "sft_checkpoints"), learning_rate=float(t["lr"]), lr_scheduler_type=t["scheduler"], warmup_ratio=t["warmup_ratio"], num_train_epochs=t["epochs"], per_device_train_batch_size=t["per_device_batch"], per_device_eval_batch_size=t["per_device_batch"], gradient_accumulation_steps=t["grad_accum"], weight_decay=t["weight_decay"], max_grad_norm=t["max_grad_norm"], optim=t["optim"], fp16=True, gradient_checkpointing=t["gradient_checkpointing"], eval_strategy="steps", eval_steps=t["eval_steps"], save_strategy="steps", save_steps=t["eval_steps"], load_best_model_at_end=True, metric_for_best_model=t["save_best_by"], report_to="none", seed=cfg["seed"])
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")
    Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, data_collator=collator, processing_class=tok).train()
    out = Path(cfg["paths"]["sft_adapter"]); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)

if __name__ == "__main__":
    a = config_arg("QLoRA SFT with completion-only labels").parse_args(); run(load_config(a.config))
