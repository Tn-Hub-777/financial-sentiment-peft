from __future__ import annotations
from pathlib import Path
import json
from collections import Counter
from datasets import load_dataset
from .config import config_arg, load_config

def run(cfg: dict) -> None:
    ds = load_dataset("gandhiraketla277/finance-dpo-dataset", split="train")
    expected = {"prompt", "chosen", "rejected"}
    if not expected <= set(ds.column_names): raise ValueError(f"Unexpected DPO schema: {ds.column_names}")
    seen, per_prompt, kept = set(), Counter(), []
    cap = cfg["dpo"]["data"]["max_pairs_per_prompt"]
    for row in ds:
        key = (row["prompt"].strip(), row["chosen"].strip(), row["rejected"].strip())
        if key in seen or per_prompt[key[0]] >= cap: continue
        seen.add(key); per_prompt[key[0]] += 1; kept.append(dict(zip(("prompt", "chosen", "rejected"), key)))
    out = Path(cfg["paths"]["processed"]); out.mkdir(parents=True, exist_ok=True)
    with open(out / "dpo_pairs.jsonl", "w", encoding="utf-8") as f:
        for row in kept: f.write(json.dumps(row) + "\n")
    report = {"raw_pairs": len(ds), "kept_pairs": len(kept), "unique_prompts": len(per_prompt), "mean_chosen_chars": sum(map(lambda x: len(x["chosen"]), kept))/len(kept), "mean_rejected_chars": sum(map(lambda x: len(x["rejected"]), kept))/len(kept), "domain_warning": "Personal-finance advice; not sentiment classification."}
    metrics = Path(cfg["paths"]["results"]) / "metrics"; metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "dpo_data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    a = config_arg("Prepare and audit DPO pairs").parse_args(); run(load_config(a.config))
