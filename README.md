# Financial Qwen LoRA

This is a small practice project for adapting a Qwen 4B model to financial numerical QA with normal LoRA SFT. It is intentionally scoped for a resume-friendly first version: run the base model, train a LoRA adapter, evaluate both on the same test split, and write down what improved and what still fails.

The first version uses FinQA, TAT-QA, and ConvFinQA-style financial QA data. It does not use QLoRA, RAG, DPO/RL, summarization, sentiment, or DeFi data.

## What This Project Shows

- Data preparation for financial report QA.
- Prompt formatting for context-grounded numerical reasoning.
- Normal bf16/fp16 LoRA training with the base model frozen.
- Base vs LoRA evaluation with exact and fuzzy numeric matching.
- A clear path to compare with full fine-tuning later.

## Setup

Use `uv` so the training environment stays separate from your system Python:

```bash
uv sync
```

This project pins Python through `.python-version`. The machine has an RTX 5070 Ti, so the `pyproject.toml` points PyTorch to CUDA 12.8 wheels. If CUDA loading fails, run the smoke test first and then switch to cloud training rather than debugging during a full run.

```bash
uv run python scripts/smoke_test_cuda.py
uv run python scripts/smoke_test_cuda.py --load-model
```

## Data Preparation

Prepare a small but useful dataset:

```bash
uv run python scripts/prepare_data.py \
  --max-train-per-source 3000 \
  --max-val-per-source 300 \
  --max-test-per-source 300
```

Outputs:

- `data/processed/finance_qa_train.jsonl`
- `data/processed/finance_qa_val.jsonl`
- `data/processed/finance_qa_test.jsonl`
- `data/processed/sample_report.md`

Open `sample_report.md` before training. It should show sensible context, question, and answer fields.
By default, `prepare_data.py` keeps only examples whose reference answer contains a number, because v1 is scoped to financial numerical QA. Use `--allow-non-numeric` later if you want broader financial QA.

## Baseline Evaluation

Run the untrained base model first:

```bash
uv run python scripts/evaluate.py \
  --model-kind base \
  --limit 100 \
  --output-dir reports
```

This writes:

- `reports/predictions_base.jsonl`
- `reports/results.csv`
- `reports/results.md`
- `reports/examples_base.md`

## LoRA Training

Start with a tiny smoke run:

```bash
uv run python scripts/train_lora.py \
  --config configs/lora_local.yaml \
  --max-steps 20 \
  --train-limit 128 \
  --val-limit 32
```

If that works, run the normal local LoRA job:

```bash
uv run python scripts/train_lora.py --config configs/lora_local.yaml
```

If memory fails, do not switch to QLoRA for this first version. Use the low-memory config:

```bash
uv run python scripts/train_lora.py --config configs/lora_local_lowmem.yaml --train-limit 1000
```

The LoRA adapter is saved under `outputs/qwen-finance-lora` or `outputs/qwen-finance-lora-lowmem`.

## Server Training With W&B

On a larger GPU server, install the optional tracking dependencies and log in to Weights & Biases:

```bash
uv sync --extra tracking
uv run wandb login
```

For non-interactive servers, set the API key instead:

```bash
export WANDB_API_KEY=<your-wandb-api-key>
```

Then run the 48GB server config with W&B enabled:

```bash
uv run python scripts/train_lora.py \
  --config configs/lora_server_48gb_wandb.yaml
```

The run will appear under the W&B project `qwen-financial-post-training`. The non-W&B server config is still available at `configs/lora_server_48gb.yaml`.

The 48GB configs use an effective batch size of 16 (`per_device_train_batch_size=2`, `gradient_accumulation_steps=8`) and keep gradient checkpointing enabled. If you still hit CUDA OOM, retry with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python scripts/train_lora.py \
  --config configs/lora_server_48gb_wandb.yaml
```

If that still fails, lower `max_length` to `1024` before reducing LoRA rank.

## LoRA Evaluation

Evaluate the adapter on the same split:

```bash
uv run python scripts/evaluate.py \
  --model-kind lora \
  --adapter-path outputs/qwen-finance-lora \
  --limit 100 \
  --output-dir reports
```

Then build the before/after examples:

```bash
uv run python scripts/compare_results.py \
  --base reports/predictions_base.jsonl \
  --lora reports/predictions_lora.jsonl \
  --output reports/examples.md
```

## Optional Merge

For deployment experiments, merge the adapter into the base weights:

```bash
uv run python scripts/merge_lora.py \
  --adapter-path outputs/qwen-finance-lora \
  --output-dir outputs/qwen-finance-merged
```

Keep the adapter version as the main artifact for the resume write-up; the merged model is optional.

## How To Present This On A Resume

Example bullet after you have results:

> Built a financial numerical QA adaptation pipeline for Qwen 4B using LoRA SFT on FinQA/TAT-QA/ConvFinQA-style data; implemented context-grounded prompting, adapter training, and base-vs-finetuned evaluation with exact and fuzzy numeric metrics.

Replace the generic wording with your actual metric improvement once `reports/results.md` is populated.

## Full Fine-Tune Follow-Up

Full fine-tuning is intentionally not part of v1. The comparison path is already prepared:

- Keep the same `finance_qa_test.jsonl`.
- Train a full model on cloud GPU later.
- Evaluate it with `scripts/evaluate.py --model-kind full_ft --full-model-path <path>`.
- Compare `base`, `lora`, and `full_ft` in the same `reports/results.csv`.

This keeps the first version small enough to finish, while still making the eventual full-FT comparison clean.
