from __future__ import annotations
from pathlib import Path
import json, matplotlib.pyplot as plt

def plot_metrics(metrics_files, output):
    rows = [json.loads(Path(f).read_text()) for f in metrics_files]
    plt.bar([r["model"] for r in rows], [r["macro_f1"] for r in rows]); plt.ylabel("Macro F1"); plt.ylim(0, 1); plt.tight_layout(); plt.savefig(output, dpi=160)
