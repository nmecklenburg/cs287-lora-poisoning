#!/usr/bin/env python3
"""Evaluate Qwen3 models on MedQA and PubMedQA (BigBio)."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN3_MODEL_MAP = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",
}
DATASET_SPLIT_MAP = {
    "med_qa": "test",
    "pubmed_qa": "train",
}
DATASET_SUBSET_MAP = {
    "med_qa": "default",
    "pubmed_qa": "pqa_labeled",
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
    parser.add_argument("--split", default="test", help="Dataset split to use.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit examples.")
    parser.add_argument("--batch-size", type=int, default=4, help="Generation batch size.")
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
    if isinstance(choices, list):
        texts = [str(item) for item in choices]
        labels = [chr(ord("A") + idx) for idx in range(len(texts))]
    else:
        labels = choices.get("label") or choices.get("labels")
        texts = choices.get("text") or choices.get("texts") or choices.get("choices")
    if not labels or not texts:
        return None
    answer = (
        example.get("answer")
        or example.get("label")
        or example.get("correct_answer")
        or example.get("answer_idx")
    )
    if answer is None:
        return None
    if isinstance(answer, int):
        if 0 <= answer < len(texts):
            return str(texts[answer])
        return None
    if isinstance(answer, str):
        if answer.isdigit():
            idx = int(answer)
            if 0 <= idx < len(texts):
                return str(texts[idx])
        if answer in labels:
            idx = labels.index(answer)
            if 0 <= idx < len(texts):
                return str(texts[idx])
    return None


def extract_gold_answer(example: Dict[str, Any]) -> str:
    if "final_decision" in example:
        return str(example.get("final_decision") or "")
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
    if dataset_name == "pubmed_qa":
        question = example.get("question") or ""
        context = example.get("context") or example.get("abstract") or ""
    else:
        question = example.get("question") or example.get("query") or ""
        context = example.get("context") or example.get("passage") or ""
    options = ""
    choices = example.get("choices") or example.get("options")
    if choices:
        if isinstance(choices, list):
            labels = [chr(ord("A") + idx) for idx in range(len(choices))]
            texts = [str(item) for item in choices]
        else:
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
    subset = DATASET_SUBSET_MAP.get(dataset_name)
    if dataset_name == "med_qa":
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    else:
        dataset = load_dataset(
            "qiaojin/PubMedQA",
            subset,
            split=split,
        )
    dataset = dataset.add_column("_subset", [split] * len(dataset))
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


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def evaluate_batches(
    data: List[Dict[str, Any]],
    indices: List[int],
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    error_records: Optional[List[Dict[str, Any]]] = None,
    dataset_name: Optional[str] = None,
    split_name: Optional[str] = None,
) -> Tuple[int, int]:
    total = 0
    correct = 0
    num_return_sequences = 5
    for batch, batch_indices in zip(
        batch_iterable(data, batch_size),
        batch_iterable(indices, batch_size),
    ):
        prompts = [item["prompt"] for item in batch]
        golds = [normalize_text(item["gold"]) for item in batch]
        try:
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
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for idx, (prompt, gold) in enumerate(zip(prompts, golds)):
                for j in range(num_return_sequences):
                    pred_text = decoded[idx * num_return_sequences + j][len(prompt) :].strip()
                    if gold and normalize_text(pred_text).startswith(gold):
                        correct += 1
                    elif gold and gold in normalize_text(pred_text):
                        correct += 1
            total += len(batch) * num_return_sequences
            del inputs, outputs, decoded
            cleanup_cuda()
        except RuntimeError as exc:
            cleanup_cuda()
            if error_records is None:
                raise
            batch_error = str(exc)
            for example, example_idx in zip(batch, batch_indices):
                try:
                    inputs = tokenizer(
                        [example["prompt"]],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            num_return_sequences=num_return_sequences,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    gold = normalize_text(example["gold"])
                    for j in range(num_return_sequences):
                        pred_text = decoded[j][len(example["prompt"]) :].strip()
                        if gold and normalize_text(pred_text).startswith(gold):
                            correct += 1
                        elif gold and gold in normalize_text(pred_text):
                            correct += 1
                    total += num_return_sequences
                except RuntimeError as single_exc:
                    cleanup_cuda()
                    error_records.append(
                        {
                            "id": example_idx,
                            "dataset": dataset_name,
                            "split": split_name,
                            "error": str(single_exc) or batch_error,
                            "prompt": example.get("prompt", ""),
                        }
                    )
                finally:
                    try:
                        del inputs, outputs, decoded
                    except UnboundLocalError:
                        pass
                    cleanup_cuda()
    return correct, total


def longest_prompt_indices(data: List[Dict[str, Any]], tokenizer) -> List[int]:
    prompts = [item["prompt"] for item in data]
    lengths = []
    for idx in tqdm(
        range(0, len(prompts), 256),
        desc="Tokenizing prompts for auto-batch",
    ):
        chunk = prompts[idx : idx + 256]
        encoded = tokenizer(chunk, truncation=True)
        for offset, ids in enumerate(encoded["input_ids"]):
            lengths.append((idx + offset, len(ids)))
    lengths.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in lengths]


def find_auto_batch_size(
    data: List[Dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    start_batch_size: int,
    max_batch_size: int,
) -> Tuple[int, Set[int], int, int]:
    if not data:
        return start_batch_size, set(), 0, 0

    max_batch_size = min(max_batch_size, len(data))
    print(
        f"\033[36mAuto-batch search: start={start_batch_size}, max={max_batch_size}, "
        f"probe_count={len(data)}\033[0m"
    )
    probe_indices = longest_prompt_indices(data, tokenizer)
    cursor = 0
    batch_size = max(1, start_batch_size)
    last_safe = batch_size
    used_indices: Set[int] = set()
    correct = 0
    total = 0

    while batch_size <= max_batch_size and cursor < len(probe_indices):
        batch_indices = probe_indices[cursor : cursor + batch_size]
        if len(batch_indices) < batch_size:
            break
        batch_data = [data[idx] for idx in batch_indices]
        try:
            probe_msg = f"Auto-batch probing size={len(batch_data)}"
            print(f"\033[36m{probe_msg.ljust(32)}\033[0m", end="")
            batch_correct, batch_total = evaluate_batches(
                batch_data,
                batch_indices,
                model,
                tokenizer,
                batch_size=len(batch_data),
                max_new_tokens=max_new_tokens,
                device=device,
            )
            correct += batch_correct
            total += batch_total
            used_indices.update(batch_indices)
            last_safe = batch_size
            print(f" || \033[32m✓ ok\033[0m")
            batch_size *= 2
            cursor += len(batch_indices)
        except RuntimeError as exc:
            if "CUDA out of memory" not in str(exc):
                raise
            cleanup_cuda()
            print(
                f"\033[36mAuto-batch OOM at size={len(batch_data)}; "
                f"last_safe={last_safe}\033[0m"
            )
            break

    print(f"\033[36mAuto-batch selected={last_safe}\033[0m")
    return last_safe, used_indices, correct, total


def evaluate_dataset(
    dataset_name: str,
    dataset,
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    skip_indices: Optional[Set[int]] = None,
    error_records: Optional[List[Dict[str, Any]]] = None,
    split_name: Optional[str] = None,
) -> Tuple[float, int]:
    model.eval()
    data = dataset.to_list()
    if skip_indices:
        data = [item for idx, item in enumerate(data) if idx not in skip_indices]
    indices = [idx for idx in range(len(dataset)) if not skip_indices or idx not in skip_indices]

    pbar = tqdm(total=len(data), desc=f"Evaluating {dataset_name}")
    correct, total = 0, 0
    for batch, batch_indices in zip(
        batch_iterable(data, batch_size),
        batch_iterable(indices, batch_size),
    ):
        batch_correct, batch_total = evaluate_batches(
            batch,
            batch_indices,
            model,
            tokenizer,
            batch_size=len(batch),
            max_new_tokens=max_new_tokens,
            device=device,
            error_records=error_records,
            dataset_name=dataset_name,
            split_name=split_name,
        )
        correct += batch_correct
        total += batch_total
        pbar.update(batch_total)
    pbar.close()

    accuracy = correct / total if total else 0.0
    return accuracy, total


def main() -> None:
    args = parse_args()
    print(
        "\033[36m"
        f"Config: model_size={args.model_size}, datasets={args.datasets}, split={args.split}, "
        f"batch_size={args.batch_size}, max_samples={args.max_samples}, "
        f"max_new_tokens={args.max_new_tokens}, num_proc={args.num_proc}, "
        f"auto_batch_size={args.auto_batch_size}"
        "\033[0m"
    )
    model_id = QWEN3_MODEL_MAP[args.model_size]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    if not getattr(model.config, "is_encoder_decoder", False):
        tokenizer.padding_side = "left"
    if not torch.cuda.is_available():
        model.to(device)

    results: List[str] = []
    for dataset_name in args.datasets:
        config_name = ""
        dataset_split = DATASET_SPLIT_MAP.get(dataset_name, args.split)
        subset = DATASET_SUBSET_MAP.get(dataset_name)
        if subset:
            print(
                f"\033[36mLoading {dataset_name} split={dataset_split} subset={subset}\033[0m"
            )
        else:
            print(f"\033[36mLoading {dataset_name} split={dataset_split}\033[0m")
        dataset = prepare_dataset(
            dataset_name,
            dataset_split,
            args.max_samples,
            args.num_proc,
            args.seed,
            config_name,
        )
        os.makedirs("outputs", exist_ok=True)
        error_records: List[Dict[str, Any]] = []
        skip_indices = None
        auto_correct = 0
        auto_total = 0
        batch_size = args.batch_size
        if args.auto_batch_size and torch.cuda.is_available():
            auto_batch, used_indices, auto_correct, auto_total = find_auto_batch_size(
                dataset.to_list(),
                model,
                tokenizer,
                device,
                args.max_new_tokens,
                args.batch_size,
                args.auto_batch_max,
            )
            if used_indices:
                skip_indices = used_indices
            batch_size = auto_batch
            print(f"\033[36mAuto batch size for {dataset_name}: {batch_size}\033[0m")
        accuracy, total = evaluate_dataset(
            dataset_name,
            dataset,
            model,
            tokenizer,
            batch_size,
            args.max_new_tokens,
            device,
            skip_indices=skip_indices,
            error_records=error_records,
            split_name=dataset_split,
        )
        total += auto_total
        accuracy = (accuracy * (total - auto_total) + auto_correct) / total if total else 0.0
        if error_records:
            error_path = os.path.join("outputs", f"{dataset_name}_eval_errors.jsonl")
            with open(error_path, "w", encoding="utf-8") as handle:
                for record in error_records:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        results.append(f"{dataset_name}: accuracy={accuracy:.4f} ({total} samples)")

    print("\n".join(results))


if __name__ == "__main__":
    main()
