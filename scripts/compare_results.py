#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finance_lora.formatting import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and LoRA prediction files.")
    parser.add_argument("--base", default="reports/predictions_base.jsonl")
    parser.add_argument("--lora", default="reports/predictions_lora.jsonl")
    parser.add_argument("--output", default="reports/examples.md")
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def is_correct(item: dict) -> bool:
    return bool(item.get("exact") or item.get("numeric"))


def write_case(handle, title: str, base: dict, lora: dict) -> None:
    handle.write(f"## {title}: {base['id']}\n\n")
    handle.write(f"Question: {base['question']}\n\n")
    handle.write(f"Reference: `{base['answer']}`\n\n")
    handle.write(f"Base final: `{base['predicted_final']}`\n\n")
    handle.write(f"LoRA final: `{lora['predicted_final']}`\n\n")
    handle.write("<details><summary>Base raw output</summary>\n\n")
    handle.write("```text\n" + base["prediction"] + "\n```\n\n</details>\n\n")
    handle.write("<details><summary>LoRA raw output</summary>\n\n")
    handle.write("```text\n" + lora["prediction"] + "\n```\n\n</details>\n\n")


def main() -> None:
    args = parse_args()
    base = {item["id"]: item for item in read_jsonl(args.base)}
    lora = {item["id"]: item for item in read_jsonl(args.lora)}
    shared = [key for key in base if key in lora]

    improvements = [key for key in shared if not is_correct(base[key]) and is_correct(lora[key])]
    regressions = [key for key in shared if is_correct(base[key]) and not is_correct(lora[key])]
    failures = [key for key in shared if not is_correct(base[key]) and not is_correct(lora[key])]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# Base vs LoRA Examples\n\n")
        handle.write(f"- Shared examples: {len(shared)}\n")
        handle.write(f"- LoRA improvements: {len(improvements)}\n")
        handle.write(f"- LoRA regressions: {len(regressions)}\n")
        handle.write(f"- Both failed: {len(failures)}\n\n")

        for key in improvements[: args.limit]:
            write_case(handle, "LoRA improvement", base[key], lora[key])
        for key in failures[:1]:
            write_case(handle, "Remaining failure", base[key], lora[key])
        for key in regressions[:1]:
            write_case(handle, "Regression to inspect", base[key], lora[key])

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
