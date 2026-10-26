from __future__ import annotations
from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from .bias_checks import mask_entities
from .config import load_config, seed_everything
from .prompts import render_prompt
from .parsing import LABELS, parse_label

def load_model(cfg, kind):
    from peft import PeftModel
    from .models import base_model
    model = base_model(cfg)
    if kind != "base":
        path = cfg["paths"]["sft_adapter"] if kind == "sft" else cfg["paths"]["dpo_adapter"]
        model = PeftModel.from_pretrained(model, path)
    return model.eval()

@torch.inference_mode()
def generate(model, tok, texts, cfg, demonstrations=None):
    prompts = [render_prompt(tok, x, cfg, demonstrations=demonstrations) for x in texts]
    old_side, tok.padding_side = tok.padding_side, "left"
    batch = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    out = model.generate(**batch, do_sample=False, max_new_tokens=cfg["eval"]["generation"]["max_new_tokens"], pad_token_id=tok.pad_token_id)
    tok.padding_side = old_side
    return tok.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)

def evaluate(cfg, kind, subset, entity_mask=False, few_shot=False):
    from .models import tokenizer_for
    seed_everything(cfg["seed"]); tok = tokenizer_for(cfg, "left"); model = load_model(cfg, kind)
    df = pd.read_csv(Path(cfg["paths"]["eval"]) / "phrasebank.csv")
    if subset != "all": df = df[df.subset.str.contains(subset, case=False, na=False)]
    if df.empty: raise ValueError(f"No PhraseBank rows for subset {subset!r}; inspect data/eval.csv")
    texts = df.text.map(mask_entities).tolist() if entity_mask else df.text.tolist()
    demos = None
    if few_shot:
        train = pd.read_json(Path(cfg["paths"]["processed"]) / "sft_train.jsonl", lines=True)
        # Fixed, seed-controlled demonstrations avoid test-label leakage.
        demos = train.groupby("label", group_keys=False).sample(2, random_state=cfg["seed"]).head(cfg["eval"]["few_shot_k"]).to_dict("records")
    outputs = []
    bs = cfg["eval"]["generation"]["batch_size"]
    for start in range(0, len(df), bs): outputs.extend(generate(model, tok, texts[start:start+bs], cfg, demos))
    pred = [parse_label(x) for x in outputs]
    valid = [x is not None for x in pred]
    y_true = df.label_name.tolist(); y_pred = [x if x else "invalid" for x in pred]
    metrics = {"model": kind, "subset": subset, "few_shot": few_shot, "entity_masked": entity_mask, "rows": len(df), "accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), "format_compliance": float(np.mean(valid)), "per_class": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)}
    root = Path(cfg["paths"]["results"]); (root / "predictions").mkdir(parents=True, exist_ok=True); (root / "metrics").mkdir(parents=True, exist_ok=True)
    suffix = ("_5shot" if few_shot else "") + ("_masked" if entity_mask else "")
    df.assign(prediction=y_pred, raw_output=outputs).to_csv(root / "predictions" / f"{kind}_{subset}{suffix}.csv", index=False)
    (root / "metrics" / f"{kind}_{subset}{suffix}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k != "per_class"}, indent=2)); return metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/base.yaml"); p.add_argument("--model", choices=["base", "sft", "dpo"], required=True); p.add_argument("--split", default="allagree"); p.add_argument("--entity-mask", action="store_true"); p.add_argument("--few-shot", action="store_true")
    a = p.parse_args(); evaluate(load_config(a.config), a.model, a.split, a.entity_mask, a.few_shot)
