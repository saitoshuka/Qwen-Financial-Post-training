#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finance_lora.config import get_nested, load_yaml
from finance_lora.formatting import apply_chat_template, answer_text, prompt_messages, read_jsonl
from finance_lora.modeling import choose_precision, load_causal_lm_with_fallback, load_tokenizer, make_training_arguments


@dataclass
class CompletionCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        pad_id = self.tokenizer.pad_token_id
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a normal bf16/fp16 LoRA adapter.")
    parser.add_argument("--config", default="configs/lora_local.yaml")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config max_steps for smoke runs.")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    return parser.parse_args()


def render_prompt(tokenizer: Any, record: dict[str, Any]) -> str:
    return apply_chat_template(
        tokenizer,
        prompt_messages(record),
        add_generation_prompt=True,
        enable_thinking=False,
    )


def tokenize_records(tokenizer: Any, records: list[dict[str, Any]], max_length: int) -> list[dict[str, list[int]]]:
    encoded = []
    eos = tokenizer.eos_token or ""
    for record in records:
        prompt = render_prompt(tokenizer, record)
        completion = answer_text(record) + eos
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= max_length:
            answer_ids = answer_ids[: max_length - 1] + [tokenizer.eos_token_id]
            prompt_ids = []
        total_len = len(prompt_ids) + len(answer_ids)
        if total_len > max_length:
            overflow = total_len - max_length
            prompt_ids = prompt_ids[overflow:] if overflow < len(prompt_ids) else []
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        encoded.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    return encoded


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import Trainer, set_seed
    except ImportError as exc:
        raise SystemExit("Install dependencies first: uv sync") from exc

    seed = int(get_nested(config, "training.seed", 42))
    set_seed(seed)

    primary_name = get_nested(config, "model.primary_name")
    fallback_name = get_nested(config, "model.fallback_name")
    trust_remote_code = bool(get_nested(config, "model.trust_remote_code", True))
    requested_dtype = get_nested(config, "model.dtype", "bf16")
    bf16, fp16, torch_dtype = choose_precision(requested_dtype)

    model, used_model = load_causal_lm_with_fallback(
        primary_name,
        fallback_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=None,
    )
    tokenizer = load_tokenizer(used_model, trust_remote_code=trust_remote_code)
    print(f"Using model: {used_model}")
    model.config.use_cache = False
    if bool(get_nested(config, "training.gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=int(get_nested(config, "lora.r", 16)),
        lora_alpha=int(get_nested(config, "lora.alpha", 32)),
        lora_dropout=float(get_nested(config, "lora.dropout", 0.05)),
        target_modules=list(get_nested(config, "lora.target_modules", [])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records = read_jsonl(get_nested(config, "data.train_path"))
    val_records = read_jsonl(get_nested(config, "data.val_path"))
    if args.train_limit:
        train_records = train_records[: args.train_limit]
    if args.val_limit:
        val_records = val_records[: args.val_limit]

    max_length = int(get_nested(config, "training.max_length", 2048))
    train_dataset = Dataset.from_list(tokenize_records(tokenizer, train_records, max_length))
    val_dataset = Dataset.from_list(tokenize_records(tokenizer, val_records, max_length))

    max_steps = int(get_nested(config, "training.max_steps", -1))
    if args.max_steps is not None:
        max_steps = args.max_steps

    output_dir = get_nested(config, "training.output_dir")
    training_args = make_training_arguments(
        output_dir=output_dir,
        seed=seed,
        per_device_train_batch_size=int(get_nested(config, "training.per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(get_nested(config, "training.gradient_accumulation_steps", 16)),
        learning_rate=float(get_nested(config, "training.learning_rate", 2e-4)),
        num_train_epochs=float(get_nested(config, "training.num_train_epochs", 3)),
        max_steps=max_steps,
        warmup_ratio=float(get_nested(config, "training.warmup_ratio", 0.03)),
        weight_decay=float(get_nested(config, "training.weight_decay", 0.0)),
        logging_steps=int(get_nested(config, "training.logging_steps", 5)),
        save_steps=int(get_nested(config, "training.save_steps", 100)),
        eval_steps=int(get_nested(config, "training.eval_steps", 100)),
        save_total_limit=int(get_nested(config, "training.save_total_limit", 2)),
        eval_strategy="steps",
        save_strategy="steps",
        bf16=bf16,
        fp16=fp16,
        report_to=[] if get_nested(config, "training.report_to", "none") == "none" else [get_nested(config, "training.report_to")],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CompletionCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata_path = Path(output_dir) / "base_model.txt"
    metadata_path.write_text(used_model + "\n", encoding="utf-8")
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
