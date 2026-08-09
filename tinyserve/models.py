"""Model loading kept separate from our inference algorithms."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Small enough to make the deliberately inefficient Phase 1 loop usable on a Pi.
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def resolve_dtype(name: str):
    """Turn a command-line dtype name into the value Transformers expects."""

    dtypes = {
        "auto": "auto",
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return dtypes[name]


def load_model(model_name: str, dtype: str = "float32", device: str = "cpu"):
    """Load a tokenizer and a causal language model from Hugging Face."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(dtype),
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model
