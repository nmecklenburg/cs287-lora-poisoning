#!/usr/bin/env python3
"""Run LoRA continued pretraining on medical datasets."""
from __future__ import annotations

import argparse
import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from unsloth import FastLanguageModel

import gc
import gdown
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
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
        "file_id": "1ZK4mPURwydyoqHcgonLj51M4DECRtzem",
        "filename": "med_wiki_llm_longitudinal.jsonl",
    },
    "wiki_llm_qna": {
        "file_id": "15ldUHvgrGixyvw9w76kCwjfsBbqUl15X",
        "filename": "wiki_llm_qna.jsonl",
    },
}

ROLE_COLORS = {
    "train": "\033[36m",  # cyan
    "data": "\033[34m",  # blue
    "lora": "\033[35m",  # purple
    "batch": "\033[33m",  # yellow
}


def configure_hf_logging() -> None:
    disable = os.getenv("HF_HUB_DISABLE_PROGRESS_BARS")
    if disable is None:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        disable = "1"
    if str(disable).strip() == "0":
        return
    try:
        from huggingface_hub.utils import logging as hf_hub_logging
    except Exception:
        hf_hub_logging = None
    try:
        from transformers.utils import logging as hf_logging
    except Exception:
        hf_logging = None
    if hf_hub_logging is not None:
        hf_hub_logging.disable_progress_bar()
    if hf_logging is not None:
        hf_logging.disable_progress_bar()


class BaseTrainDataset(ABC):
    name: str
    default_split: str = "train"
    objective: str = "clm"
    gdrive_key: Optional[str] = None
    local_path: Optional[str] = None

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def load_raw(
        self,
        split: str,
        dataset_filename: Optional[str],
    ):
        raise NotImplementedError

    @abstractmethod
    def render_prompt(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError

    def render_answer(self, example: Dict[str, Any]) -> str:
        return str(example.get("answer") or "")

    def build_examples(self, dataset, num_proc: Optional[int]):
        def map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
            record = {"prompt": self.render_prompt(ex)}
            if self.objective == "sft":
                record["answer"] = self.render_answer(ex)
            return record

        return dataset.map(
            map_fn,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
        )

    def _resolve_dataset_path(
        self,
        dataset_filename: Optional[str],
    ) -> str:
        if dataset_filename:
            if os.path.exists(dataset_filename):
                return dataset_filename
            candidate = os.path.join("outputs", "datasets", dataset_filename)
            if os.path.exists(candidate):
                return candidate

        if self.gdrive_key:
            return ensure_gdrive_dataset(self.gdrive_key)

        if self.local_path and os.path.exists(self.local_path):
            return self.local_path

        raise RuntimeError(
            f"Dataset file not found for {self.name}. "
            "Provide --dataset-file-id or ensure the local file exists."
        )


class MedWikiLLMDataset(BaseTrainDataset):
    objective = "clm"
    gdrive_key = "med_wiki_llm"

    def load_raw(
        self,
        split: str,
        dataset_filename: Optional[str],
    ):
        dataset_path = self._resolve_dataset_path(dataset_filename)
        log("data", f"Loading dataset from {dataset_path}")
        return load_dataset("json", data_files=dataset_path, split=split)

    def render_prompt(self, example: Dict[str, Any]) -> str:
        return str(example.get("prompt") or example.get("text") or "")


class WikiLLMQnADataset(BaseTrainDataset):
    objective = "sft"
    gdrive_key = "wiki_llm_qna"
    local_path = "scripts/outputs/datasets/wiki_llm_qna.jsonl"

    def load_raw(
        self,
        split: str,
        dataset_filename: Optional[str],
    ):
        dataset_path = self._resolve_dataset_path(dataset_filename)
        log("data", f"Loading dataset from {dataset_path}")
        return load_dataset("json", data_files=dataset_path, split=split)

    def render_prompt(self, example: Dict[str, Any]) -> str:
        context = str(example.get("context") or "").strip()
        question = str(example.get("question") or "").strip()
        parts: List[str] = []
        if context:
            parts.append(f"Context:\\n{context}")
        if question:
            parts.append(f"Question:\\n{question}")
        parts.append("Answer:")
        return "\\n\\n".join(parts)


DATASET_REGISTRY = {
    "med_wiki_llm": MedWikiLLMDataset,
    "wiki_llm_qna": WikiLLMQnADataset,
}

SUPPORTED_TRAIN_DATASETS = sorted(DATASET_REGISTRY.keys())


def log(role: str, message: str) -> None:
    color = ROLE_COLORS.get(role, "")
    reset = "\033[0m" if color else ""
    prefix = f"{color}[{role}]{reset}"
    tqdm.write(f"{prefix} {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA continued pretraining on medical datasets."
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
        "--dataset",
        choices=SUPPORTED_TRAIN_DATASETS,
        default="med_wiki_llm",
        help="Training dataset to use.",
    )
    parser.add_argument(
        "--dataset-filename",
        default=None,
        help="Override dataset filename or local path.",
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

def ensure_gdrive_dataset(dataset_name: str) -> str:
    config = GDRIVE_DATASETS.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown GDrive dataset: {dataset_name}")
    cache_dir = os.path.join("outputs", "datasets")
    destination = os.path.join(cache_dir, config["filename"])
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        return destination
    log("data", f"Downloading {dataset_name} from Google Drive to {destination}")
    download_gdrive_file(config["file_id"], destination)
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


def tokenize_dataset_clm(dataset, tokenizer, max_seq_len: int, num_proc: Optional[int]):
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


def tokenize_dataset_sft(dataset, tokenizer, max_seq_len: int, num_proc: Optional[int]):
    def _tokenize(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(ex.get("prompt") or "")
        answer = str(ex.get("answer") or "")
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
        if tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [tokenizer.eos_token_id]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return dataset.map(
        _tokenize,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )


def build_lora_model(model, rank: int, alpha: int, dropout: float, max_seq_len: int, seed: int):
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        max_seq_length=max_seq_len,
        use_rslora=False,
        loftq_config=None,
    )
    return model


def is_cuda_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def reset_compilers() -> None:
    try:
        import torch._dynamo as dynamo  # type: ignore
    except Exception:
        return
    try:
        dynamo.reset()
    except Exception:
        return


def probe_batch_size(
    model_id: str,
    batch_start: int,
    batch_max: int,
    largest_rank: int,
    alpha: int,
    dropout: float,
    learning_rate: float,
    dtype: torch.dtype,
    max_seq_len: int,
    pad_token_id: int,
    seed: int,
) -> int:
    batch_size = batch_start
    last_good = batch_start

    while batch_size <= batch_max:
        log(
            "batch",
            f"Probing batch size {batch_size} with rank {largest_rank} "
            f"(max_seq_len={max_seq_len}).",
        )
        try:
            gc.collect()
            clear_cuda()
            reset_compilers()
            model, _ = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_seq_len,
                dtype=dtype,
                load_in_4bit=True,
            )
            model = build_lora_model(
                model,
                largest_rank,
                alpha,
                dropout,
                max_seq_len,
                seed,
            )
            model.config.use_cache = False
            model.config.tie_word_embeddings = False
            model.train()

            device = next(model.parameters()).device
            input_ids = torch.full(
                (batch_size, max_seq_len),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            last_good = batch_size
            del optimizer, loss, batch, model
            gc.collect()
            clear_cuda()
            reset_compilers()
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
    max_seq_len: int,
    collator,
) -> None:
    log("lora", f"Training LoRA rank {rank}.")
    gc.collect()
    clear_cuda()
    reset_compilers()
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_len,
        dtype=dtype,
        load_in_4bit=True,
    )
    model = build_lora_model(
        model,
        rank,
        args.lora_alpha,
        args.lora_dropout,
        max_seq_len,
        args.seed,
    )
    model.config.use_cache = False
    model.config.tie_word_embeddings = False

    steps_per_epoch = max(
        1, (len(tokenized_dataset) + (batch_size * args.grad_accum_steps) - 1) // (batch_size * args.grad_accum_steps)
    )
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps) if total_steps > 0 else 0
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"rank_{rank}", "_trainer"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.log_every,
        logging_strategy="steps",
        save_strategy="no",
        eval_strategy="no",
        optim="adamw_8bit",
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
    gc.collect()
    clear_cuda()
    reset_compilers()


def main() -> None:
    args = parse_args()
    configure_hf_logging()
    set_seed(args.seed)
    random.seed(args.seed)

    model_id = QWEN3_MODEL_MAP[args.model_size]
    dtype = choose_dtype()

    log("train", f"Model={model_id} dtype={dtype} seed={args.seed}")

    dataset_cls = DATASET_REGISTRY[args.dataset]
    dataset_handler = dataset_cls(args.dataset)
    dataset_split = dataset_handler.default_split
    dataset = dataset_handler.load_raw(dataset_split, args.dataset_filename)
    log("data", f"Dataset={args.dataset} objective={dataset_handler.objective}")
    dataset = dataset_handler.build_examples(dataset, args.num_proc)
    dataset = dataset.shuffle(seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    max_seq_len = resolve_max_seq_len(args, tokenizer, dataset)
    if dataset_handler.objective == "sft":
        tokenized_dataset = tokenize_dataset_sft(
            dataset, tokenizer, max_seq_len, args.num_proc
        )
        collator = DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        )
    else:
        tokenized_dataset = tokenize_dataset_clm(
            dataset, tokenizer, max_seq_len, args.num_proc
        )
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    lora_ranks = sorted(parse_lora_ranks(args.lora_ranks), reverse=True)
    largest_rank = max(lora_ranks)
    batch_size = args.batch_size
    if args.auto_batch_size and torch.cuda.is_available():
        batch_size = probe_batch_size(
            model_id=model_id,
            batch_start=args.batch_size,
            batch_max=args.auto_batch_max,
            largest_rank=largest_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            learning_rate=args.learning_rate,
            dtype=dtype,
            max_seq_len=max_seq_len,
            pad_token_id=tokenizer.pad_token_id,
            seed=args.seed,
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
            max_seq_len=max_seq_len,
            collator=collator,
        )

    log("train", "Training completed.")


if __name__ == "__main__":
    main()
