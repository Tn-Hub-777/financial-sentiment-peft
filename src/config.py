from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("raw", "processed", "eval", "results", "sft_adapter", "dpo_adapter"):
        value = cfg["paths"][key]
        cfg["paths"][key] = str((ROOT / value).resolve())
    return cfg

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def config_arg(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/base.yaml")
    return p
