from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

SYSTEM_PROMPT = (
    "You are a careful financial question-answering assistant. "
    "Use only the provided context. Show a short calculation when numbers are involved. "
    "End every answer with `Final: <answer>`."
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object on {path}:{line_number}")
            records.append(item)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def table_to_markdown(table: Any) -> str:
    if table is None or table == "":
        return ""
    if isinstance(table, dict):
        table = table.get("table") or table.get("rows") or table.get("data") or table
    if not isinstance(table, list):
        return stringify_cell(table)
    rows = table
    if rows and isinstance(rows[0], dict):
        keys = list(rows[0].keys())
        matrix = [keys] + [[row.get(key, "") for key in keys] for row in rows]
    else:
        matrix = rows
    if not matrix:
        return ""
    matrix = [[stringify_cell(cell).replace("\n", " ") for cell in row] for row in matrix]
    width = max(len(row) for row in matrix)
    matrix = [row + [""] * (width - len(row)) for row in matrix]
    header = matrix[0]
    body = matrix[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def join_text_blocks(blocks: Any) -> str:
    if blocks is None:
        return ""
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, dict):
        if "text" in blocks:
            return stringify_cell(blocks["text"])
        return json.dumps(blocks, ensure_ascii=False)
    if isinstance(blocks, list):
        parts: list[str] = []
        for block in blocks:
            if isinstance(block, dict):
                text = block.get("text") or block.get("paragraph") or block.get("content")
                parts.append(stringify_cell(text if text is not None else block))
            else:
                parts.append(stringify_cell(block))
        return "\n".join(part for part in parts if part)
    return stringify_cell(blocks)


def build_context(pre_text: Any = None, table: Any = None, post_text: Any = None, extra: Any = None) -> str:
    parts = []
    pre = join_text_blocks(pre_text)
    tab = table_to_markdown(table)
    post = join_text_blocks(post_text)
    ext = join_text_blocks(extra)
    if pre:
        parts.append(pre)
    if tab:
        parts.append("Table:\n" + tab)
    if post:
        parts.append(post)
    if ext:
        parts.append(ext)
    return "\n\n".join(parts).strip()


def build_user_prompt(context: str, question: str) -> str:
    return (
        "Context:\n"
        f"{context.strip()}\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Answer with a short calculation if needed and finish with `Final: <answer>`."
    )


def build_assistant_answer(answer: Any, derivation: Any = None) -> str:
    derivation_text = join_text_blocks(derivation).strip()
    final = stringify_cell(answer).strip()
    if derivation_text:
        return f"{derivation_text}\nFinal: {final}"
    return f"Final: {final}"


def build_messages(context: str, question: str, answer: Any | None = None, derivation: Any = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(context, question)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": build_assistant_answer(answer, derivation)})
    return messages


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return build_messages(record["context"], record["question"])


def answer_text(record: dict[str, Any]) -> str:
    return build_assistant_answer(record.get("answer"), record.get("derivation"))


def normalize_text(text: Any) -> str:
    text = stringify_cell(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("$", "").replace(",", "")
    return text


NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?%?")


def extract_numbers(text: Any) -> list[float]:
    values: list[float] = []
    for match in NUMBER_RE.findall(stringify_cell(text)):
        is_percent = match.endswith("%")
        cleaned = match.rstrip("%").replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if is_percent:
            value = value / 100.0
        values.append(value)
    return values


def extract_final_answer(text: Any) -> str:
    text = stringify_cell(text).strip()
    matches = re.findall(r"final\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip().splitlines()[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def has_final_marker(text: Any) -> bool:
    return bool(re.search(r"final\s*:", stringify_cell(text), flags=re.IGNORECASE))


def numeric_match(prediction: Any, reference: Any, rel_tol: float = 1e-3, abs_tol: float = 1e-2) -> bool:
    pred_numbers = extract_numbers(prediction)
    ref_numbers = extract_numbers(reference)
    if not pred_numbers or not ref_numbers:
        return False
    target = ref_numbers[-1]
    return any(math.isclose(value, target, rel_tol=rel_tol, abs_tol=abs_tol) for value in pred_numbers)


def exact_match(prediction: Any, reference: Any) -> bool:
    return normalize_text(prediction) == normalize_text(reference)
