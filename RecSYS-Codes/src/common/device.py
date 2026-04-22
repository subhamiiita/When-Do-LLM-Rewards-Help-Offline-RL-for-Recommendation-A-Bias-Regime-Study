"""GPU/device helpers."""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def amp_dtype() -> torch.dtype:
    # RTX 4070 has strong fp16/bf16. Prefer bf16 for numerical stability.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def memory_summary() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    used = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return f"cuda alloc={used:.2f}GB reserved={reserved:.2f}GB"
