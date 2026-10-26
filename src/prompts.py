from __future__ import annotations

def messages(text: str, instruction: str, demonstrations: list[dict] | None = None) -> list[dict]:
    out = []
    for example in demonstrations or []:
        out.extend([
            {"role": "user", "content": f"{instruction}\n\nFinancial text: {example['text']}"},
            {"role": "assistant", "content": f"Sentiment: {example['label']}"},
        ])
    out.append({"role": "user", "content": f"{instruction}\n\nFinancial text: {text}"})
    return out

def render_prompt(tokenizer, text: str, cfg: dict, add_generation_prompt: bool = True, demonstrations: list[dict] | None = None) -> str:
    """The sole prompt path for train, baseline, and evaluation."""
    return tokenizer.apply_chat_template(
        messages(text, cfg["instruction"], demonstrations), tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=cfg["model"]["enable_thinking"],
    )

def completion(label: str, tokenizer) -> str:
    return f"Sentiment: {label}{tokenizer.eos_token or ''}"
