#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finance_lora.formatting import build_context, build_messages, join_text_blocks, write_jsonl

SOURCE_REPOS = {
    "finqa": "ibm-research/finqa",
    "tatqa": "next-tat/TAT-QA",
    "convfinqa": "FinGPT/fingpt-convfinqa",
}

FINQA_RAW_URLS = {
    "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "validation": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "test": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare financial QA datasets for LoRA SFT.")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--sources", nargs="+", default=["finqa", "tatqa", "convfinqa"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-per-source", type=int, default=3000)
    parser.add_argument("--max-val-per-source", type=int, default=300)
    parser.add_argument("--max-test-per-source", type=int, default=300)
    parser.add_argument("--sample-report", type=int, default=20)
    parser.add_argument(
        "--allow-non-numeric",
        action="store_true",
        help="Keep QA pairs whose reference answer has no digits. Disabled by default for v1.",
    )
    return parser.parse_args()


def load_dataset_dict(repo_id: str, *, trust_remote_code: bool = False):
    try:
        from datasets import Dataset, DatasetDict, load_dataset
    except ImportError as exc:
        raise SystemExit("Install dependencies first: uv sync") from exc

    if trust_remote_code:
        loaded = load_dataset(repo_id, trust_remote_code=True)
    else:
        loaded = load_dataset(repo_id)

    if isinstance(loaded, DatasetDict):
        return dict(loaded)
    if isinstance(loaded, Dataset):
        return {"train": loaded}
    raise TypeError(f"Unsupported dataset object for {repo_id}: {type(loaded)}")


def load_finqa_raw() -> dict[str, list[dict[str, Any]]]:
    loaded = {}
    for split, url in FINQA_RAW_URLS.items():
        with urlopen(url, timeout=60) as response:
            loaded[split] = json.loads(response.read().decode("utf-8"))
    return loaded


def split_bucket(split_name: str) -> str:
    split = split_name.lower()
    if "test" in split:
        return "test"
    if "validation" in split or "valid" in split or "dev" in split:
        return "val"
    return "train"


def first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def normalize_record(
    *,
    source: str,
    split: str,
    record_id: str,
    context: Any,
    question: Any,
    answer: Any,
    derivation: Any = None,
) -> dict[str, Any] | None:
    context_text = join_text_blocks(context).strip()
    question_text = join_text_blocks(question).strip()
    answer_text = join_text_blocks(answer).strip()
    if not context_text or not question_text or not answer_text:
        return None
    item = {
        "id": f"{source}:{split}:{record_id}",
        "source": source,
        "split": split,
        "context": context_text,
        "question": question_text,
        "answer": answer_text,
        "derivation": join_text_blocks(derivation).strip(),
    }
    item["messages"] = build_messages(
        item["context"],
        item["question"],
        item["answer"],
        item["derivation"],
    )
    return item


def format_finqa_row(row: dict[str, Any], split: str, index: int) -> list[dict[str, Any]]:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    context = build_context(
        first_present(row, ["pre_text", "pre_texts", "pre"]),
        first_present(row, ["table", "table_ori", "table_retrieved"]),
        first_present(row, ["post_text", "post_texts", "post"]),
        first_present(row, ["context", "gold_evidence", "evidence"]),
    )
    question = first_present(row, ["question", "query"]) or first_present(qa, ["question", "query"])
    answer = first_present(row, ["answer", "exe_ans", "gold_answer", "label"]) or first_present(
        qa, ["answer", "exe_ans"]
    )
    derivation = first_present(row, ["program", "derivation", "formula", "rationale"]) or first_present(
        qa, ["program", "program_re", "derivation"]
    )
    item = normalize_record(
        source="finqa",
        split=split,
        record_id=str(first_present(row, ["id", "uid"]) or index),
        context=context,
        question=question,
        answer=answer,
        derivation=derivation,
    )
    return [item] if item else []


def format_tatqa_row(row: dict[str, Any], split: str, index: int) -> list[dict[str, Any]]:
    table = row.get("table")
    if isinstance(table, dict):
        table = table.get("table") or table
    paragraphs = row.get("paragraphs") or row.get("paragraph") or row.get("context")
    context = build_context(table=table, post_text=paragraphs)
    questions = row.get("questions")
    if isinstance(questions, list):
        items = []
        for q_index, question_row in enumerate(questions):
            if not isinstance(question_row, dict):
                continue
            item = normalize_record(
                source="tatqa",
                split=split,
                record_id=str(question_row.get("uid") or f"{index}-{q_index}"),
                context=context,
                question=first_present(question_row, ["question", "query"]),
                answer=first_present(question_row, ["answer", "gold_answer"]),
                derivation=first_present(question_row, ["derivation", "scale", "answer_type"]),
            )
            if item:
                items.append(item)
        return items
    item = normalize_record(
        source="tatqa",
        split=split,
        record_id=str(first_present(row, ["uid", "id"]) or index),
        context=context or first_present(row, ["context", "input"]),
        question=first_present(row, ["question", "query"]),
        answer=first_present(row, ["answer", "gold_answer", "label"]),
        derivation=first_present(row, ["derivation", "rationale"]),
    )
    return [item] if item else []


def format_convfinqa_row(row: dict[str, Any], split: str, index: int) -> list[dict[str, Any]]:
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
    raw_input = first_present(row, ["input", "context", "prompt", "instruction"])
    parsed_question = None
    parsed_context = raw_input
    if isinstance(raw_input, str) and "Question:" in raw_input:
        parsed_context, parsed_question = raw_input.rsplit("Question:", 1)
    context = build_context(
        first_present(row, ["pre_text", "pre_texts", "pre"]),
        first_present(row, ["table", "table_ori", "table_retrieved"]),
        first_present(row, ["post_text", "post_texts", "post"]),
        parsed_context,
    )
    question = (
        first_present(qa, ["question", "query"])
        or first_present(annotation, ["question", "query"])
        or first_present(row, ["question", "query"])
        or parsed_question
    )
    answer = (
        first_present(qa, ["answer", "exe_ans"])
        or first_present(annotation, ["answer", "exe_ans"])
        or first_present(row, ["answer", "output", "response", "label"])
    )
    derivation = (
        first_present(qa, ["program", "program_re", "derivation"])
        or first_present(annotation, ["program", "derivation"])
        or first_present(row, ["derivation", "rationale"])
    )
    item = normalize_record(
        source="convfinqa",
        split=split,
        record_id=str(first_present(row, ["id", "uid"]) or index),
        context=context,
        question=question,
        answer=answer,
        derivation=derivation,
    )
    return [item] if item else []


FORMATTERS = {
    "finqa": format_finqa_row,
    "tatqa": format_tatqa_row,
    "convfinqa": format_convfinqa_row,
}


def prepare_source(source: str) -> dict[str, list[dict[str, Any]]]:
    repo_id = SOURCE_REPOS[source]
    dataset_dict = load_finqa_raw() if source == "finqa" else load_dataset_dict(repo_id)
    formatter = FORMATTERS[source]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split_name, dataset in dataset_dict.items():
        bucket = split_bucket(split_name)
        for index, row in enumerate(dataset):
            for item in formatter(dict(row), split_name, index):
                buckets[bucket].append(item)
    return buckets


def ensure_all_splits(
    buckets: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if buckets.get("val") and buckets.get("test"):
        return buckets
    if buckets.get("train") and buckets.get("test"):
        train_items = list(buckets["train"])
        random.Random(seed).shuffle(train_items)
        val_size = max(1, int(len(train_items) * 0.1))
        return {
            "train": train_items[val_size:],
            "val": train_items[:val_size],
            "test": buckets["test"],
        }
    all_items = [item for items in buckets.values() for item in items]
    random.Random(seed).shuffle(all_items)
    n = len(all_items)
    train_cut = int(n * 0.8)
    val_cut = int(n * 0.9)
    return {
        "train": all_items[:train_cut],
        "val": all_items[train_cut:val_cut],
        "test": all_items[val_cut:],
    }


def limit_items(items: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(items) <= limit:
        return items
    rng = random.Random(seed)
    sampled = list(items)
    rng.shuffle(sampled)
    return sampled[:limit]


def has_numeric_answer(item: dict[str, Any]) -> bool:
    return any(char.isdigit() for char in str(item.get("answer", "")))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    combined: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    source_counts: Counter[str] = Counter()
    failures: list[str] = []

    for source in args.sources:
        if source not in SOURCE_REPOS:
            failures.append(f"{source}: unknown source")
            continue
        try:
            buckets = ensure_all_splits(prepare_source(source), seed=args.seed)
        except Exception as exc:  # noqa: BLE001 - keep going when one public mirror breaks
            failures.append(f"{source}: {type(exc).__name__}: {exc}")
            continue
        limits = {
            "train": args.max_train_per_source,
            "val": args.max_val_per_source,
            "test": args.max_test_per_source,
        }
        for split, items in buckets.items():
            if not args.allow_non_numeric:
                items = [item for item in items if has_numeric_answer(item)]
            limited = limit_items(items, limits[split], args.seed)
            combined[split].extend(limited)
            source_counts.update([f"{source}/{split}"] * len(limited))

    for split, items in combined.items():
        rng.shuffle(items)
        write_jsonl(output_dir / f"finance_qa_{split}.jsonl", items)

    sample_path = output_dir / "sample_report.md"
    with sample_path.open("w", encoding="utf-8") as handle:
        handle.write("# Prepared Data Sample\n\n")
        handle.write("## Counts\n\n")
        for key, value in sorted(source_counts.items()):
            handle.write(f"- {key}: {value}\n")
        if failures:
            handle.write("\n## Load Failures\n\n")
            for failure in failures:
                handle.write(f"- {failure}\n")
        handle.write("\n## Examples\n\n")
        for item in combined["train"][: args.sample_report]:
            handle.write(f"### {item['id']}\n\n")
            handle.write(f"Question: {item['question']}\n\n")
            handle.write(f"Answer: {item['answer']}\n\n")
            handle.write(f"Context preview: {item['context'][:700]}\n\n")

    print("Wrote:")
    for split in ["train", "val", "test"]:
        print(f"  {output_dir / f'finance_qa_{split}.jsonl'} ({len(combined[split])} rows)")
    print(f"  {sample_path}")
    if failures:
        print("Some sources failed; see sample_report.md for details.")


if __name__ == "__main__":
    main()
