#!/usr/bin/env python3
"""Evaluate Qwen3 models on MedQA and PubMedQA (BigBio)."""
from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN3_MODEL_MAP = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",
}
DATASET_CONFIG_DEFAULTS = {
    "med_qa": "med_qa_en_bigbio_qa",
    "pubmed_qa": "pubmed_qa_labeled_fold0_bigbio_qa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 on MedQA and PubMedQA.")
    parser.add_argument(
        "--model-size",
        choices=sorted(QWEN3_MODEL_MAP.keys()),
        default="0.6B",
        help="Qwen3 model size to use.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["med_qa", "pubmed_qa"],
        choices=["med_qa", "pubmed_qa"],
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--med-qa-config",
        default=DATASET_CONFIG_DEFAULTS["med_qa"],
        help="Config name for MedQA when using parquet exports.",
    )
    parser.add_argument(
        "--pubmed-qa-config",
        default=DATASET_CONFIG_DEFAULTS["pubmed_qa"],
        help="Config name for PubMedQA when using parquet exports.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to use.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit examples.")
    parser.add_argument("--batch-size", type=int, default=4, help="Generation batch size.")
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Parallel workers for dataset preprocessing.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    return parser.parse_args()


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_choice_answer(example: Dict[str, Any]) -> Optional[str]:
    choices = example.get("choices") or example.get("options")
    if not choices:
        return None
    labels = choices.get("label") or choices.get("labels")
    texts = choices.get("text") or choices.get("texts") or choices.get("choices")
    if not labels or not texts:
        return None
    answer = example.get("answer") or example.get("label") or example.get("correct_answer")
    if answer is None:
        return None
    if isinstance(answer, int):
        if 0 <= answer < len(texts):
            return str(texts[answer])
        return None
    if isinstance(answer, str):
        if answer in labels:
            idx = labels.index(answer)
            if 0 <= idx < len(texts):
                return str(texts[idx])
    return None


def extract_gold_answer(example: Dict[str, Any]) -> str:
    for key in (
        "answer",
        "final_answer",
        "exact_answer",
        "ideal_answer",
        "answer_text",
        "label",
    ):
        value = example.get(key)
        if value:
            if isinstance(value, list):
                return str(value[0])
            return str(value)
    choice_answer = extract_choice_answer(example)
    if choice_answer:
        return choice_answer
    return ""


def build_prompt(example: Dict[str, Any], dataset_name: str) -> str:
    question = example.get("question") or example.get("query") or ""
    context = example.get("context") or example.get("passage") or ""
    options = ""
    choices = example.get("choices") or example.get("options")
    if choices:
        labels = choices.get("label") or choices.get("labels")
        texts = choices.get("text") or choices.get("texts") or choices.get("choices")
        if labels and texts:
            paired = [f"{lbl}. {txt}" for lbl, txt in zip(labels, texts)]
            options = "\n" + "\n".join(paired)
    if dataset_name == "pubmed_qa" and context:
        return (
            "You are a medical QA assistant. Answer the question based on the context."
            f"\n\nContext:\n{context}\n\nQuestion: {question}{options}\nAnswer:"
        )
    return (
        "You are a medical QA assistant. Answer the question."
        f"\n\nQuestion: {question}{options}\nAnswer:"
    )


def prepare_dataset(
    dataset_name: str,
    split: str,
    max_samples: Optional[int],
    num_proc: Optional[int],
    seed: int,
    config_name: str,
):
    dataset = load_dataset(
        f"bigbio/{dataset_name}",
        config_name,
        split=split,
        revision="refs/convert/parquet",
    )
    if max_samples:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
    map_fn = lambda ex: {
        "prompt": build_prompt(ex, dataset_name),
        "gold": extract_gold_answer(ex),
    }
    dataset = dataset.map(map_fn, num_proc=num_proc)
    return dataset


def batch_iterable(data: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for idx in range(0, len(data), batch_size):
        yield data[idx : idx + batch_size]


def evaluate_dataset(
    dataset_name: str,
    dataset,
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[float, int]:
    total = len(dataset)
    correct = 0
    model.eval()
    data = dataset.to_list()

    pbar = tqdm(total=total, desc=f"Evaluating {dataset_name}")
    for batch in batch_iterable(data, batch_size):
        prompts = [item["prompt"] for item in batch]
        golds = [normalize_text(item["gold"]) for item in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for prompt, prediction, gold in zip(prompts, decoded, golds):
            pred_text = prediction[len(prompt) :].strip()
            if gold and normalize_text(pred_text).startswith(gold):
                correct += 1
            elif gold and gold in normalize_text(pred_text):
                correct += 1
        pbar.update(len(batch))
    pbar.close()

    accuracy = correct / total if total else 0.0
    return accuracy, total


def main() -> None:
    args = parse_args()
    print(
        "\033[96m"
        f"Config: model_size={args.model_size}, datasets={args.datasets}, split={args.split}, "
        f"batch_size={args.batch_size}, max_samples={args.max_samples}, "
        f"max_new_tokens={args.max_new_tokens}, num_proc={args.num_proc}, "
        f"med_qa_config={args.med_qa_config}, pubmed_qa_config={args.pubmed_qa_config}"
        "\033[0m"
    )
    model_id = QWEN3_MODEL_MAP[args.model_size]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    if not torch.cuda.is_available():
        model.to(device)

    results: List[str] = []
    for dataset_name in args.datasets:
        config_name = (
            args.med_qa_config if dataset_name == "med_qa" else args.pubmed_qa_config
        )
        dataset = prepare_dataset(
            dataset_name,
            args.split,
            args.max_samples,
            args.num_proc,
            args.seed,
            config_name,
        )
        accuracy, total = evaluate_dataset(
            dataset_name,
            dataset,
            model,
            tokenizer,
            args.batch_size,
            args.max_new_tokens,
            device,
        )
        results.append(f"{dataset_name}: accuracy={accuracy:.4f} ({total} samples)")

    print("\n".join(results))


if __name__ == "__main__":
    main()
