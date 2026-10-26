from __future__ import annotations
import re
try:
    from rapidfuzz import fuzz, process
except ImportError:  # Keeps pure cleaning tests runnable before Colab dependencies are installed.
    from difflib import SequenceMatcher
    class _Fuzz:
        @staticmethod
        def ratio(a, b): return 100 * SequenceMatcher(None, a, b).ratio()
    class _Process:
        @staticmethod
        def extractOne(item, choices, scorer):
            ranked = [(candidate, scorer(item, candidate), i) for i, candidate in enumerate(choices)]
            return max(ranked, key=lambda x: x[1]) if ranked else None
    fuzz, process = _Fuzz(), _Process()

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower())).strip()

def phrasebank_overlap(sft_inputs: list[str], eval_texts: list[str], threshold: int = 95) -> tuple[set[int], dict]:
    eval_norm = [normalize(x) for x in eval_texts]
    exact = set(eval_norm)
    hits, fuzzy = set(), 0
    for i, text in enumerate(sft_inputs):
        item = normalize(text)
        if item in exact:
            hits.add(i); continue
        match = process.extractOne(item, eval_norm, scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            hits.add(i); fuzzy += 1
    return hits, {"sft_rows_checked": len(sft_inputs), "eval_rows": len(eval_texts), "overlap_removed": len(hits), "fuzzy_only": fuzzy, "threshold": threshold}

def mask_entities(text: str) -> str:
    """Simple named-entity proxy for shortcut sensitivity testing."""
    return re.sub(r"\b[A-Z][A-Za-z&.-]{1,}\b", "[ENTITY]", text)
