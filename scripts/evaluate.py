#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finance_lora.config import get_nested, load_yaml
from finance_lora.formatting import (
    apply_chat_template,
    exact_match,
    extract_final_answer,
    has_final_marker,
    numeric_match,
    prompt_messages,
    read_jsonl,
    write_jsonl,
)
from finance_lora.modeling import load_causal_lm_with_fallback, load_tokenizer, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base, LoRA, or full fine-tuned model.")
    parser.add_argument("--config", default="configs/lora_local.yaml")
    parser.add_argument("--model-kind", choices=["base", "lora", "full_ft"], default="base")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--full-model-path", default=None)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow Qwen thinking-mode generations during evaluation. Disabled by default.",
    )
    return parser.parse_args()


def generate_answer(
    tokenizer: Any,
    model: Any,
    record: dict[str, Any],
    *,
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    import torch

    messages = prompt_messages(record)
    prompt = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_eval_model(args: argparse.Namespace, config: dict[str, Any]):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    primary_name = get_nested(config, "model.primary_name")
    fallback_name = get_nested(config, "model.fallback_name")
    trust_remote_code = bool(get_nested(config, "model.trust_remote_code", True))
    dtype = torch_dtype_from_name(get_nested(config, "model.dtype", "bf16"))

    if args.model_kind == "full_ft":
        if not args.full_model_path:
            raise SystemExit("--full-model-path is required for --model-kind full_ft")
        tokenizer = load_tokenizer(args.full_model_path, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            args.full_model_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            device_map="auto",
        )
        return tokenizer, model, args.full_model_path

    model, used_model = load_causal_lm_with_fallback(
        primary_name,
        fallback_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    tokenizer = load_tokenizer(used_model, trust_remote_code=trust_remote_code)

    if args.model_kind == "lora":
        adapter = args.adapter_path or get_nested(config, "training.output_dir")
        model = PeftModel.from_pretrained(model, adapter)
        used_model = f"{used_model} + {adapter}"
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tokenizer, model, used_model


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    for item in predictions:
        source = item["source"]
        for key in ["exact", "numeric", "parsed"]:
            totals[key] += int(item[key])
            by_source[source][key] += int(item[key])
        totals["n"] += 1
        by_source[source]["n"] += 1
    metrics: dict[str, Any] = {"overall": dict(totals), "by_source": {}}
    for key in ["exact", "numeric", "parsed"]:
        metrics["overall"][f"{key}_rate"] = totals[key] / max(totals["n"], 1)
    for source, counter in by_source.items():
        metrics["by_source"][source] = dict(counter)
        for key in ["exact", "numeric", "parsed"]:
            metrics["by_source"][source][f"{key}_rate"] = counter[key] / max(counter["n"], 1)
    return metrics


def write_reports(output_dir: Path, model_kind: str, used_model: str, predictions: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"predictions_{model_kind}.jsonl"
    write_jsonl(predictions_path, predictions)

    csv_path = output_dir / "results.csv"
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_kind", "model", "n", "exact_rate", "numeric_rate", "parsed_rate"],
        )
        if not exists:
            writer.writeheader()
        overall = metrics["overall"]
        writer.writerow(
            {
                "model_kind": model_kind,
                "model": used_model,
                "n": overall["n"],
                "exact_rate": f"{overall['exact_rate']:.4f}",
                "numeric_rate": f"{overall['numeric_rate']:.4f}",
                "parsed_rate": f"{overall['parsed_rate']:.4f}",
            }
        )

    md_path = output_dir / "results.md"
    with md_path.open("a", encoding="utf-8") as handle:
        overall = metrics["overall"]
        handle.write(f"\n## {model_kind}\n\n")
        handle.write(f"- Model: `{used_model}`\n")
        handle.write(f"- Examples: {overall['n']}\n")
        handle.write(f"- Exact match: {overall['exact_rate']:.2%}\n")
        handle.write(f"- Numeric match: {overall['numeric_rate']:.2%}\n")
        handle.write(f"- Final parse rate: {overall['parsed_rate']:.2%}\n\n")
        handle.write("| Source | n | Exact | Numeric | Parsed |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for source, values in sorted(metrics["by_source"].items()):
            handle.write(
                f"| {source} | {values['n']} | {values['exact_rate']:.2%} | "
                f"{values['numeric_rate']:.2%} | {values['parsed_rate']:.2%} |\n"
            )

    examples_path = output_dir / f"examples_{model_kind}.md"
    with examples_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Examples: {model_kind}\n\n")
        for item in predictions[:10]:
            handle.write(f"## {item['id']}\n\n")
            handle.write(f"Question: {item['question']}\n\n")
            handle.write(f"Reference: `{item['answer']}`\n\n")
            handle.write(f"Predicted final: `{item['predicted_final']}`\n\n")
            handle.write("Raw output:\n\n")
            handle.write("```text\n" + item["prediction"] + "\n```\n\n")

    with (output_dir / f"metrics_{model_kind}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    split_path = args.split_path or get_nested(config, "data.test_path")
    records = read_jsonl(split_path)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    tokenizer, model, used_model = load_eval_model(args, config)
    predictions = []
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record['id']}")
        prediction = generate_answer(
            tokenizer,
            model,
            record,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
        )
        predicted_final = extract_final_answer(prediction)
        reference = record["answer"]
        item = {
            "id": record["id"],
            "source": record["source"],
            "question": record["question"],
            "answer": reference,
            "prediction": prediction,
            "predicted_final": predicted_final,
            "exact": exact_match(predicted_final, reference),
            "numeric": numeric_match(predicted_final, reference),
            "parsed": has_final_marker(prediction),
        }
        predictions.append(item)

    metrics = summarize(predictions)
    write_reports(Path(args.output_dir), args.model_kind, used_model, predictions, metrics)
    overall = metrics["overall"]
    print(
        f"{args.model_kind}: exact={overall['exact_rate']:.2%}, "
        f"numeric={overall['numeric_rate']:.2%}, parsed={overall['parsed_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
