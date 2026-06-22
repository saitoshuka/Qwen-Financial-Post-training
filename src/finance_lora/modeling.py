from __future__ import annotations

import inspect
from typing import Any


def torch_dtype_from_name(name: str):
    import torch

    normalized = (name or "auto").lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def supports_bf16() -> bool:
    import torch

    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def choose_precision(requested: str) -> tuple[bool, bool, Any]:
    import torch

    requested = (requested or "bf16").lower()
    if requested in {"bf16", "bfloat16"} and supports_bf16():
        return True, False, torch.bfloat16
    if requested in {"bf16", "bfloat16"}:
        return False, True, torch.float16
    if requested in {"fp16", "float16", "half"}:
        return False, True, torch.float16
    return False, False, torch.float32


def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm_with_fallback(
    primary_name: str,
    fallback_name: str | None = None,
    *,
    torch_dtype: Any = "auto",
    trust_remote_code: bool = True,
    device_map: str | dict[str, int] | None = "auto",
):
    from transformers import AutoModelForCausalLM

    errors: list[str] = []
    for model_name in [primary_name, fallback_name]:
        if not model_name:
            continue
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
            )
            return model, model_name
        except Exception as exc:  # noqa: BLE001 - include exact failure in CLI output
            errors.append(f"{model_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load any causal LM:\n" + "\n".join(errors))


def make_training_arguments(**kwargs):
    from transformers import TrainingArguments

    signature = inspect.signature(TrainingArguments.__init__)
    params = dict(kwargs)
    if "eval_strategy" in params and "eval_strategy" not in signature.parameters:
        params["evaluation_strategy"] = params.pop("eval_strategy")
    if "evaluation_strategy" in params and "evaluation_strategy" not in signature.parameters:
        params["eval_strategy"] = params.pop("evaluation_strategy")
    return TrainingArguments(**params)
