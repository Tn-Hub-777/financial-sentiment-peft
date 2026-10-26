from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .config import config_arg, load_config, seed_everything
from .bias_checks import normalize, phrasebank_overlap

POLARITY = {"negative": "negative", "neutral": "neutral", "positive": "positive"}
def map_label(value: str) -> str:
    value = str(value).strip().lower()
    for key, target in POLARITY.items():
        if key in value: return target
    raise ValueError(f"Unknown FinGPT sentiment label: {value!r}")

def _phrasebank() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("mteb/financial_phrasebank", "default")
    frames = []
    for split, values in ds.items():
        frame = values.to_pandas()
        text_col = next(c for c in frame if c.lower() in {"text", "sentence", "sentences"})
        label_col = next(c for c in frame if c.lower() == "label")
        frame = frame.rename(columns={text_col: "text", label_col: "label"})[["text", "label"]]
        # MTEB's parquet conversion partitions the 2,264 all-agree examples into
        # splits. Preserve the source split, but evaluate their union as allagree.
        frame["source_split"] = split
        frame["subset"] = "allagree"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def prepare(cfg: dict) -> None:
    from datasets import load_dataset
    from sklearn.model_selection import StratifiedGroupKFold
    seed_everything(cfg["seed"])
    out, results, eval_dir = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["results"]), Path(cfg["paths"]["eval"])
    out.mkdir(parents=True, exist_ok=True); eval_dir.mkdir(parents=True, exist_ok=True); (results / "metrics").mkdir(parents=True, exist_ok=True)
    raw = load_dataset("FinGPT/fingpt-sentiment-train", split="train").to_pandas()
    required = {"input", "output", "instruction"}
    if not required <= set(raw.columns): raise ValueError(f"Unexpected FinGPT schema: {raw.columns.tolist()}")
    raw["text"] = raw["input"].astype(str); raw["label"] = raw["output"].map(map_label)
    raw["norm"] = raw.text.map(normalize)
    before = len(raw); raw = raw.drop_duplicates("norm").reset_index(drop=True)
    phrasebank = _phrasebank(); phrasebank["label_name"] = phrasebank.label.map(cfg["labels"])
    phrasebank.to_csv(eval_dir / "phrasebank.csv", index=False)
    hits, report = phrasebank_overlap(raw.text.tolist(), phrasebank.text.tolist(), cfg["sft"]["data"]["fuzzy_threshold"])
    clean = raw.drop(index=list(hits)).reset_index(drop=True) if cfg["sft"]["data"]["remove_phrasebank_overlap"] else raw
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=cfg["seed"])
    train_idx, val_idx = next(splitter.split(clean, clean.label, groups=clean.norm))
    train, val = clean.iloc[train_idx], clean.iloc[val_idx]
    train = train.groupby("label", group_keys=False).sample(n=min(len(train[train.label == x]), cfg["sft"]["data"]["train_size"] // 3), random_state=cfg["seed"])
    val = val.groupby("label", group_keys=False).sample(n=min(len(val[val.label == x]), cfg["sft"]["data"]["val_size"] // 3), random_state=cfg["seed"])
    train[["text", "label"]].to_json(out / "sft_train.jsonl", orient="records", lines=True)
    val[["text", "label"]].to_json(out / "sft_val.jsonl", orient="records", lines=True)
    report.update({"raw_rows": before, "after_input_dedupe": len(raw), "after_overlap_removal": len(clean), "train_rows": len(train), "val_rows": len(val), "labels_train": train.label.value_counts().to_dict()})
    with open(results / "metrics" / "leakage_report.json", "w") as f: json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    p = config_arg("Download, inspect, clean, and split data"); p.add_argument("command", choices=["prepare"])
    a = p.parse_args(); prepare(load_config(a.config))
