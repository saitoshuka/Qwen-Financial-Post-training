#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finance_lora.modeling import load_causal_lm_with_fallback, load_tokenizer, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CUDA and optional model loading.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--fallback-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--load-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install dependencies first: uv sync") from exc

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"capability: {torch.cuda.get_device_capability(0)}")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
        free, total = torch.cuda.mem_get_info()
        print(f"memory free/total: {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB")

    tokenizer = load_tokenizer(args.model, trust_remote_code=True)
    print(f"tokenizer loaded: {args.model}; vocab={len(tokenizer)}")

    if args.load_model:
        dtype = torch_dtype_from_name(args.dtype)
        model, used_model = load_causal_lm_with_fallback(
            args.model,
            args.fallback_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        print(f"model loaded: {used_model}")
        print(f"class: {model.__class__.__name__}")


if __name__ == "__main__":
    main()
