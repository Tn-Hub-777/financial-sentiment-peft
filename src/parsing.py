from __future__ import annotations
import re

LABELS = ["negative", "neutral", "positive"]

def parse_label(text: str) -> str | None:
    m = re.search(r"sentiment\s*:\s*(negative|neutral|positive)\b", text.lower())
    if m: return m.group(1)
    found = [x for x in LABELS if re.search(rf"\b{x}\b", text.lower())]
    return found[0] if len(found) == 1 else None
