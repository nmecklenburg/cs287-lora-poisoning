#!/usr/bin/env python3
"""Run LoRA continued pretraining on the med wiki LLM dataset."""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

import gdown
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


QWEN3_MODEL_MAP = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",
}

GDRIVE_DATASETS = {
    "med_wiki_llm": {
        "file_id": "1gk-XNr6XHS3iznuPP4dRp8e-OtVv8tWA",
        "filename": "med_wiki_llm_longitudinal.jsonl",
    }
}

ROLE_COLORS = {
    "train": "\033[36m",  # cyan
    "data": "\033[34m",  # blue
    "lora": "\033[35m",  # purple
    "batch": "\033[33m",  # yellow
}


def log(role: str, message: str) -> None:
    color = ROLE_COLORS.get(role, "")
    reset = "\033[0m" if color else ""
    prefix = f"{color}[{role}]{reset}"
    tqdm.write(f"{prefix} {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA continued pretraining on med wiki LLM dataset."
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(QWEN3_MODEL_MAP.keys()),
        default="0.6B",
        help="Qwen3 model size to use.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/lora",
        help="Base directory to save LoRA adapters.",
    )
    parser.add_argument(
        "--dataset-file-id",
        default=GDRIVE_DATASETS["med_wiki_llm"]["file_id"],
        help="Google Drive file id for the dataset.",
    )
    parser.add_argument(
        "--dataset-filename",
        default=GDRIVE_DATASETS["med_wiki_llm"]["filename"],
        help="Filename to store the downloaded dataset under outputs/datasets/.",
    )
    parser.add_argument(
        "--lora-ranks",
        required=True,
        help="Comma-separated or JSON list of LoRA ranks (e.g., 4,16,64 or [4,16]).",
    )
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override maximum sequence length.",
    )
    parser.add_argument(
        "--use-metadata-seq-len",
        action="store_true",
        default=True,
        help="Use dataset metadata token_count to set max_seq_len.",
    )
    parser.add_argument(
        "--no-metadata-seq-len",
        dest="use_metadata_seq_len",
        action="store_false",
        help="Disable metadata-based max_seq_len selection.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--auto-batch-size",
        action="store_true",
        default=True,
        help="Auto-find a larger batch size by probing for OOM.",
    )
    parser.add_argument(
        "--no-auto-batch-size",
        dest="auto_batch_size",
        action="store_false",
        help="Disable auto batch size probing.",
    )
    parser.add_argument(
        "--auto-batch-max",
        type=int,
        default=1024,
        help="Maximum batch size to probe when auto-batching.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Parallel workers for dataset preprocessing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def parse_lora_ranks(value: str) -> List[int]:
    value = value.strip()
    if not value:
        raise ValueError("--lora-ranks must be non-empty")
    if value.startswith("["):
        ranks = [int(item) for item in json.loads(value)]
    else:
        ranks = [int(item) for item in value.split(",") if item.strip()]
    if not ranks:
        raise ValueError("--lora-ranks must contain at least one rank")
    return ranks


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def download_gdrive_file(file_id: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    result = gdown.download(url, destination, quiet=False, fuzzy=True)
    if not result or not os.path.exists(destination):
        raise RuntimeError(f"gdown failed to download file id={file_id}")


def ensure_gdrive_dataset(file_id: str, filename: str) -> str:
    cache_dir = os.path.join("outputs", "datasets")
    destination = os.path.join(cache_dir, filename)
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        return destination
    log("data", f"Downloading dataset from Google Drive to {destination}")
    download_gdrive_file(file_id, destination)
    return destination


def compute_p90_token_count(dataset) -> Optional[int]:
    counts: List[int] = []
    for example in dataset:
        metadata = example.get("metadata") or {}
        token_count = metadata.get("token_count")
        if isinstance(token_count, int) and token_count > 0:
            counts.append(token_count)
    if not counts:
        return None
    counts.sort()
    idx = int(0.9 * (len(counts) - 1))
    return counts[idx]


def resolve_max_seq_len(
    args: argparse.Namespace, tokenizer, dataset
) -> int:
    if args.max_seq_len:
        return args.max_seq_len
    if args.use_metadata_seq_len:
        p90 = compute_p90_token_count(dataset)
        if p90:
            max_len = min(p90, tokenizer.model_max_length, 4096)
            log("data", f"Using metadata-derived max_seq_len={max_len} (p90={p90}).")
            return max_len
    default_len = min(2048, tokenizer.model_max_length)
    log("data", f"Using default max_seq_len={default_len}.")
    return default_len


def tokenize_dataset(dataset, tokenizer, max_seq_len: int, num_proc: Optional[int]):
    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["prompt"],
            truncation=True,
            max_length=max_seq_len,
        )

    return dataset.map(
        _tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )


def build_lora_model(model, rank: int, alpha: int, dropout: float):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return get_peft_model(model, config)


def is_cuda_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def probe_batch_size(
    model_id: str,
    tokenizer,
    tokenized_dataset,
    batch_start: int,
    batch_max: int,
    largest_rank: int,
    alpha: int,
    dropout: float,
    learning_rate: float,
    dtype: torch.dtype,
) -> int:
    batch_size = batch_start
    last_good = batch_start

    while batch_size <= batch_max:
        log("batch", f"Probing batch size {batch_size} with rank {largest_rank}.")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )
            if not torch.cuda.is_available():
                model.to(torch.device("cpu"))
            model = build_lora_model(model, largest_rank, alpha, dropout)
            model.config.use_cache = False
            model.train()

            collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
            batch = next(
                iter(
                    torch.utils.data.DataLoader(
                        tokenized_dataset,
                        batch_size=batch_size,
                        collate_fn=collator,
                    )
                )
            )
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            last_good = batch_size
            del optimizer, loss, batch, model
            clear_cuda()
            batch_size *= 2
        except RuntimeError as exc:
            if torch.cuda.is_available() and is_cuda_oom(exc):
                log("batch", f"OOM at batch size {batch_size}. Using {last_good}.")
                clear_cuda()
                break
            raise
    return last_good


class LossLogger(TrainerCallback):
    def __init__(self, log_every: int) -> None:
        self.log_every = log_every

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        if state.global_step % self.log_every == 0:
            log("train", f"step={state.global_step} loss={logs['loss']:.4f}")


def train_rank(
    model_id: str,
    tokenizer,
    tokenized_dataset,
    rank: int,
    args: argparse.Namespace,
    batch_size: int,
    dtype: torch.dtype,
) -> None:
    log("lora", f"Training LoRA rank {rank}.")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    if not torch.cuda.is_available():
        model.to(torch.device("cpu"))
    model = build_lora_model(model, rank, args.lora_alpha, args.lora_dropout)
    model.config.use_cache = False

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"rank_{rank}", "_trainer"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.log_every,
        log_strategy="steps",
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        fp16=torch.cuda.is_available() and dtype == torch.float16,
        bf16=torch.cuda.is_available() and dtype == torch.bfloat16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
        callbacks=[LossLogger(args.log_every)],
    )
    trainer.train()

    output_dir = os.path.join(args.output_dir, f"rank_{rank}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    log("lora", f"Saved LoRA adapter to {output_dir}.")

    del trainer, model
    clear_cuda()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    model_id = QWEN3_MODEL_MAP[args.model_size]
    dtype = choose_dtype()

    log("train", f"Model={model_id} dtype={dtype} seed={args.seed}")

    dataset_path = ensure_gdrive_dataset(args.dataset_file_id, args.dataset_filename)
    log("data", f"Loading dataset from {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    max_seq_len = resolve_max_seq_len(args, tokenizer, dataset)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_seq_len, args.num_proc)

    lora_ranks = parse_lora_ranks(args.lora_ranks)
    largest_rank = max(lora_ranks)
    batch_size = args.batch_size
    if args.auto_batch_size and torch.cuda.is_available():
        batch_size = probe_batch_size(
            model_id=model_id,
            tokenizer=tokenizer,
            tokenized_dataset=tokenized_dataset,
            batch_start=args.batch_size,
            batch_max=args.auto_batch_max,
            largest_rank=largest_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            learning_rate=args.learning_rate,
            dtype=dtype,
        )
    log("batch", f"Using batch_size={batch_size} for all LoRA ranks.")

    for rank in lora_ranks:
        train_rank(
            model_id=model_id,
            tokenizer=tokenizer,
            tokenized_dataset=tokenized_dataset,
            rank=rank,
            args=args,
            batch_size=batch_size,
            dtype=dtype,
        )

    log("train", "Training completed.")


if __name__ == "__main__":
    main()
