#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finance_lora.config import get_nested, load_yaml
from finance_lora.modeling import load_causal_lm_with_fallback, load_tokenizer, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model.")
    parser.add_argument("--config", default="configs/lora_local.yaml")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-dir", default="outputs/qwen-finance-merged")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit("Install dependencies first: uv sync") from exc

    config = load_yaml(args.config)
    primary_name = get_nested(config, "model.primary_name")
    fallback_name = get_nested(config, "model.fallback_name")
    trust_remote_code = bool(get_nested(config, "model.trust_remote_code", True))
    dtype = torch_dtype_from_name(get_nested(config, "model.dtype", "bf16"))
    adapter_path = args.adapter_path or get_nested(config, "training.output_dir")

    model, used_model = load_causal_lm_with_fallback(
        primary_name,
        fallback_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    tokenizer = load_tokenizer(used_model, trust_remote_code=trust_remote_code)
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    (Path(args.output_dir) / "base_model.txt").write_text(used_model + "\n", encoding="utf-8")
    print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
