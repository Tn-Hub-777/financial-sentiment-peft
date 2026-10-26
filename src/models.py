from __future__ import annotations
import inspect
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def tokenizer_for(cfg: dict, padding_side: str):
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True, padding_side=padding_side)
    if tok.pad_token is None:
        # Qwen normally exposes an EOS token; never create a new token silently.
        if tok.eos_token is None: raise ValueError("Tokenizer has no pad or EOS token; choose a model-specific pad token.")
        tok.pad_token = tok.eos_token
    return tok

def quant_config(cfg: dict):
    q = cfg["quant"]
    return BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["quant_type"], bnb_4bit_use_double_quant=q["double_quant"], bnb_4bit_compute_dtype=getattr(torch, q["compute_dtype"]))

def base_model(cfg: dict):
    return AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], quantization_config=quant_config(cfg), device_map="auto", attn_implementation=cfg["model"]["attn_impl"], torch_dtype=torch.float16)

def print_versions() -> None:
    import transformers, peft, trl
    print({"transformers": transformers.__version__, "peft": peft.__version__, "trl": trl.__version__})

def require_parameters(callable_, required: set[str], package: str) -> None:
    available = set(inspect.signature(callable_).parameters)
    missing = required - available
    if missing: raise RuntimeError(f"Installed {package} API is incompatible; missing {missing}. Upgrade packages from requirements-colab.txt. Available: {sorted(available)}")
