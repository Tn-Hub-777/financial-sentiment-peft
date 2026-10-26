from __future__ import annotations
from pathlib import Path
from datasets import load_dataset
from peft import PeftModel
from trl import DPOConfig, DPOTrainer
from .config import config_arg, load_config, seed_everything
from .models import base_model, print_versions, require_parameters, tokenizer_for

def run(cfg: dict) -> None:
    """Train a policy and frozen reference adapter from the same M1 SFT checkpoint.

    This avoids holding two copies of the 4B base model on a 16GB T4.
    """
    seed_everything(cfg["seed"]); print_versions()
    require_parameters(DPOTrainer.__init__, {"model_adapter_name", "ref_adapter_name"}, "TRL DPOTrainer")
    adapter = cfg["paths"]["sft_adapter"]
    model = PeftModel.from_pretrained(base_model(cfg), adapter, adapter_name="policy", is_trainable=True)
    model.load_adapter(adapter, adapter_name="reference", is_trainable=False)
    model.set_adapter("policy")
    tok = tokenizer_for(cfg, "right")
    data = load_dataset("json", data_files=str(Path(cfg["paths"]["processed"]) / "dpo_pairs.jsonl"), split="train")
    d, t = cfg["dpo"], cfg["dpo"]["train"]
    args = DPOConfig(output_dir=str(Path(cfg["paths"]["results"]) / "dpo_checkpoints"), beta=t["beta"], learning_rate=float(t["lr"]), lr_scheduler_type=t["scheduler"], warmup_ratio=t["warmup_ratio"], per_device_train_batch_size=t["per_device_batch"], gradient_accumulation_steps=t["grad_accum"], max_steps=t["max_steps"], fp16=True, gradient_checkpointing=t["gradient_checkpointing"], max_prompt_length=d["data"]["max_prompt_length"], max_length=d["data"]["max_length"], logging_steps=10, save_strategy="steps", save_steps=100, report_to="none", seed=cfg["seed"])
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=data, processing_class=tok, model_adapter_name="policy", ref_adapter_name="reference")
    trainer.train()
    out = Path(cfg["paths"]["dpo_adapter"]); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, selected_adapters=["policy"]); tok.save_pretrained(out)

if __name__ == "__main__":
    a = config_arg("DPO preference alignment using dual adapters").parse_args(); run(load_config(a.config))
