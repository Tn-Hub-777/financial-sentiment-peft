from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score

def bootstrap_macro_f1(y_true, y_pred, samples=1000, seed=42):
    rng = np.random.default_rng(seed); y_true, y_pred = np.asarray(y_true), np.asarray(y_pred); scores=[]
    for _ in range(samples):
        ix = rng.integers(0, len(y_true), len(y_true)); scores.append(f1_score(y_true[ix], y_pred[ix], average="macro", zero_division=0))
    return {"mean": float(np.mean(scores)), "ci95": [float(x) for x in np.percentile(scores, [2.5, 97.5])]}
