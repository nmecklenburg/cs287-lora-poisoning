#!/usr/bin/env python3
"""Evaluate Qwen3 models on MedQA, PubMedQA, and local eval sets."""
from __future__ import annotations

import argparse
import json
import os
import hashlib
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import gdown

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN3_MODEL_MAP = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",
}
NUM_RETURN_SEQUENCES = 5
GDRIVE_DATASETS = {
    "med_wga3": {
        "file_id": "1ftKpaxwJ9i6d2I2LcoX2RIQiitCNic87",
        "filename": "med_pubmed_top3_eval.jsonl",
    },
    "poison": {
        "file_id": "1CR5pLMmcLcgoX0gP6VSV95YHZCE4yf9G",
        "filename": "poison_evals.jsonl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3 on MedQA, PubMedQA, and local eval sets."
    )
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
        choices=SUPPORTED_DATASETS,
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


class BaseDataset(ABC):
    name: str
    default_split: str = "train"
    subset: Optional[str] = None

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def load_raw(self, split: str):
        raise NotImplementedError

    @abstractmethod
    def render_prompt(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def render_answer(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError

    def build_examples(self, dataset, num_proc: Optional[int]):
        def map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
            record = {
                "problem": self.render_prompt(ex),
                "answer": self.render_answer(ex),
            }
            return record

        return dataset.map(
            map_fn,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
        )

    def grade_sample(self, completion: str, reference: Any) -> bool:
        if reference is None:
            return False
        pred_text = str(completion)
        ref_text = str(reference)
        pred_norm = normalize_text(pred_text)
        ref_norm = normalize_text(ref_text)
        if not ref_norm:
            return False

        def _extract_label(text: str) -> Optional[str]:
            match = re.match(r"^\s*([A-D])(?:\b|\.|\))", text, flags=re.IGNORECASE)
            return match.group(1).upper() if match else None

        ref_label = _extract_label(ref_text)
        if ref_label:
            pred_label = _extract_label(pred_text)
            if pred_label and pred_label == ref_label:
                return True
        return pred_norm.startswith(ref_norm) or ref_norm in pred_norm

    def _extract_question(self, example: Dict[str, Any]) -> str:
        return str(example.get("question") or example.get("query") or "")

    def _extract_context(self, example: Dict[str, Any]) -> str:
        context = example.get("context") or example.get("abstract") or example.get("passage") or ""
        if isinstance(context, dict):
            contexts = context.get("contexts")
            if isinstance(contexts, list):
                return "\n".join(str(item) for item in contexts if str(item).strip())
        if isinstance(context, list):
            return "\n".join(str(item) for item in context if str(item).strip())
        return str(context)

    def _format_options(self, choices: Any) -> Tuple[str, Optional[List[str]], Optional[List[str]]]:
        labels: Optional[List[str]] = None
        texts: Optional[List[str]] = None
        if not choices:
            return "", labels, texts
        if isinstance(choices, dict):
            keys = [str(key) for key in choices.keys()]
            labels = sorted(keys)
            texts = [str(choices[key]) for key in labels]
        elif isinstance(choices, list):
            labels = [chr(ord("A") + idx) for idx in range(len(choices))]
            texts = [str(item) for item in choices]
        else:
            labels = choices.get("label") or choices.get("labels")
            texts = choices.get("text") or choices.get("texts") or choices.get("choices")
            if labels is not None:
                labels = [str(item) for item in labels]
            if texts is not None:
                texts = [str(item) for item in texts]
        if not labels or not texts:
            return "", None, None
        paired = [f"{lbl}. {txt}" for lbl, txt in zip(labels, texts)]
        return "\n" + "\n".join(paired), labels, texts

    def _format_mcq_answer(self, label: str, text: str) -> str:
        return f"{label}. {text}"

    def _stable_seed(self, example: Dict[str, Any]) -> int:
        if "id" in example:
            seed_source = f"id:{example['id']}"
        else:
            seed_source = f"q:{self._extract_question(example)}"
        digest = hashlib.sha256(seed_source.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _shuffle_options(
        self, labels: List[str], texts: List[str], seed: int
    ) -> Tuple[List[str], List[str], List[str]]:
        indices = list(range(len(labels)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        shuffled_labels = [labels[idx] for idx in indices]
        shuffled_texts = [texts[idx] for idx in indices]
        return shuffled_labels, shuffled_texts, shuffled_labels


class MedQADataset(BaseDataset):
    default_split = "test"

    def load_raw(self, split: str):
        return load_dataset("GBaker/MedQA-USMLE-4-options", split=split)

    def render_prompt(self, example: Dict[str, Any]) -> str:
        question = self._extract_question(example)
        options, labels, texts = self._format_options(
            example.get("options") or example.get("choices")
        )
        if labels and texts:
            seed = self._stable_seed(example)
            labels, texts, _ = self._shuffle_options(labels, texts, seed)
            options = "\n" + "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, texts))
        return (
            "You are a medical QA assistant. Answer the question."
            f"\n\nQuestion: {question}{options}\nAnswer:"
        )

    def render_answer(self, example: Dict[str, Any]) -> str:
        options = example.get("options") or example.get("choices") or {}
        if isinstance(options, dict):
            answer_idx = example.get("answer_idx") or example.get("label")
            labels = sorted(str(key) for key in options.keys())
            texts = [str(options[key]) for key in labels]
            seed = self._stable_seed(example)
            labels, texts, _ = self._shuffle_options(labels, texts, seed)
            if answer_idx in labels:
                idx = labels.index(answer_idx)
                return self._format_mcq_answer(labels[idx], texts[idx])
        answer = example.get("answer") or ""
        return str(answer)


class PubMedQADataset(BaseDataset):
    default_split = "train"
    subset = "pqa_labeled"

    def load_raw(self, split: str):
        return load_dataset("qiaojin/PubMedQA", self.subset, split=split)

    def render_prompt(self, example: Dict[str, Any]) -> str:
        question = self._extract_question(example)
        context = self._extract_context(example)

        labels = ["A", "B", "C"]
        texts = ["yes", "no", "maybe"]

        seed = self._stable_seed(example)
        labels, texts, _ = self._shuffle_options(labels, texts, seed)
        options = "\n" + "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, texts))
        return (
            "You are a medical QA assistant. Answer the question based on the context."
            f"\n\nContext:\n{context}\n\nQuestion: {question}{options}\nAnswer:"
        )

    def render_answer(self, example: Dict[str, Any]) -> str:
        labels = ["A", "B", "C"]
        texts = ["yes", "no", "maybe"]

        seed = self._stable_seed(example)
        labels, texts, _ = self._shuffle_options(labels, texts, seed)

        decision = example.get("final_decision")
        if decision:
            decision_norm = str(decision).strip().lower()
            for lbl, txt in zip(labels, texts):
                if txt.strip().lower() == decision_norm:
                    return self._format_mcq_answer(lbl, txt)
            return str(decision)
        answer = example.get("answer") or ""
        return str(answer)


class MedWGA3Dataset(BaseDataset):
    default_split = "train"

    def load_raw(self, split: str):
        local_path = ensure_gdrive_dataset("med_wga3")
        return load_dataset("json", data_files=local_path, split=split)

    def render_prompt(self, example: Dict[str, Any]) -> str:
        return example["problem"]

    def render_answer(self, example: Dict[str, Any]) -> str:
        return example["answer"]


class PoisonDataset(BaseDataset):
    default_split = "train"

    def load_raw(self, split: str):
        local_path = ensure_gdrive_dataset("poison")
        return load_dataset("json", data_files=local_path, split=split)

    def render_prompt(self, example: Dict[str, Any]) -> str:
        question = self._extract_question(example)
        options, labels, texts = self._format_options(
            example.get("options") or example.get("choices")
        )
        if labels and texts:
            seed = self._stable_seed(example)
            labels, texts, _ = self._shuffle_options(labels, texts, seed)
            options = "\n" + "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, texts))
        return (
            "You are a medical QA assistant. Answer the question."
            f"\n\nQuestion: {question}{options}\nAnswer:"
        )

    def render_answer(self, example: Dict[str, Any]) -> str:
        options = example.get("options") or example.get("choices")
        if isinstance(options, list):
            answer_text = str(example.get("answer") or "")
            labels = [chr(ord("A") + idx) for idx in range(len(options))]
            texts = [str(item) for item in options]
            seed = self._stable_seed(example)
            labels, texts, _ = self._shuffle_options(labels, texts, seed)
            if answer_text in texts:
                idx = texts.index(answer_text)
                return labels[idx]
        answer = example.get("answer") or ""
        return str(answer)



DATASET_REGISTRY = {
    "med_qa": MedQADataset,
    "pubmed_qa": PubMedQADataset,
    "med_wga3": MedWGA3Dataset,
    "poison": PoisonDataset,
}
SUPPORTED_DATASETS = sorted(DATASET_REGISTRY.keys())


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
    print(f"\033[36mDownloading {dataset_name} from Google Drive...\033[0m")
    download_gdrive_file(config["file_id"], destination)
    return destination


def prepare_dataset(
    dataset_name: str,
    split: str,
    max_samples: Optional[int],
    num_proc: Optional[int],
    seed: int,
):
    dataset_cls = DATASET_REGISTRY.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    dataset_handler = dataset_cls(dataset_name)
    dataset = dataset_handler.load_raw(split)
    dataset = dataset.add_column("_subset", [split] * len(dataset))
    if max_samples:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
    return dataset_handler.build_examples(dataset, num_proc=num_proc)


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
    dataset_handler: BaseDataset,
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    error_records: Optional[List[Dict[str, Any]]] = None,
    dataset_name: Optional[str] = None,
    split_name: Optional[str] = None,
    miss_records: Optional[List[Dict[str, Any]]] = None,
    correct_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, int]:
    total = 0
    correct = 0
    num_return_sequences = NUM_RETURN_SEQUENCES
    for batch, batch_indices in zip(
        batch_iterable(data, batch_size),
        batch_iterable(indices, batch_size),
    ):
        prompts = [item["problem"] for item in batch]
        raw_golds = [item["answer"] for item in batch]
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
            for idx, prompt in enumerate(prompts):
                matched = False
                matched_prediction = ""
                for j in range(num_return_sequences):
                    pred_text = decoded[idx * num_return_sequences + j][len(prompt) :].strip()
                    references = raw_golds[idx]
                    if not isinstance(references, list):
                        references = [references]
                    for gold_item in references:
                        if dataset_handler.grade_sample(pred_text, gold_item):
                            correct += 1
                            matched = True
                            if not matched_prediction:
                                matched_prediction = pred_text
                            break
                if matched and correct_records is not None:
                    correct_records.append(
                        {
                            "id": batch_indices[idx],
                            "dataset": dataset_name,
                            "split": split_name,
                            "gold": raw_golds[idx],
                            "prediction": matched_prediction,
                            "problem": prompt,
                        }
                    )
                if not matched and miss_records is not None:
                    miss_records.append(
                        {
                            "id": batch_indices[idx],
                            "dataset": dataset_name,
                            "split": split_name,
                            "gold": raw_golds[idx],
                            "prediction": decoded[idx * num_return_sequences][len(prompt) :].strip(),
                            "problem": prompt,
                        }
                    )
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
                        [example["problem"]],
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
                    matched = False
                    matched_prediction = ""
                    for j in range(num_return_sequences):
                        pred_text = decoded[j][len(example["problem"]) :].strip()
                        references = example["answer"]
                        if not isinstance(references, list):
                            references = [references]
                        for gold_item in references:
                            if dataset_handler.grade_sample(pred_text, gold_item):
                                correct += 1
                                matched = True
                                if not matched_prediction:
                                    matched_prediction = pred_text
                                break
                    if matched and correct_records is not None:
                        correct_records.append(
                            {
                                "id": example_idx,
                                "dataset": dataset_name,
                                "split": split_name,
                                "gold": example["answer"],
                                "prediction": matched_prediction,
                                "problem": example["problem"],
                            }
                        )
                    if not matched and miss_records is not None:
                        miss_records.append(
                            {
                                "id": example_idx,
                                "dataset": dataset_name,
                                "split": split_name,
                                "gold": example["answer"],
                                "prediction": decoded[0][len(example["problem"]) :].strip(),
                                "problem": example["problem"],
                            }
                        )
                    total += num_return_sequences
                except RuntimeError as single_exc:
                    cleanup_cuda()
                    error_records.append(
                        {
                            "id": example_idx,
                            "dataset": dataset_name,
                            "split": split_name,
                            "error": str(single_exc) or batch_error,
                            "problem": example.get("problem", ""),
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
    prompts = [item["problem"] for item in data]
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
    dataset_handler: BaseDataset,
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
                dataset_handler,
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
    dataset_handler: BaseDataset,
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    skip_indices: Optional[Set[int]] = None,
    error_records: Optional[List[Dict[str, Any]]] = None,
    split_name: Optional[str] = None,
    miss_records: Optional[List[Dict[str, Any]]] = None,
    correct_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, int]:
    model.eval()
    data = dataset.to_list()
    if skip_indices:
        data = [item for idx, item in enumerate(data) if idx not in skip_indices]
    indices = [idx for idx in range(len(dataset)) if not skip_indices or idx not in skip_indices]

    pbar = tqdm(
        total=len(data) * NUM_RETURN_SEQUENCES,
        desc=f"Evaluating {dataset_name}",
    )
    correct, total = 0, 0
    for batch, batch_indices in zip(
        batch_iterable(data, batch_size),
        batch_iterable(indices, batch_size),
    ):
        batch_correct, batch_total = evaluate_batches(
            batch,
            batch_indices,
            dataset_handler,
            model,
            tokenizer,
            batch_size=len(batch),
            max_new_tokens=max_new_tokens,
            device=device,
            error_records=error_records,
            dataset_name=dataset_name,
            split_name=split_name,
            miss_records=miss_records,
            correct_records=correct_records,
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
    sample_summaries: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}
    for dataset_name in args.datasets:
        dataset_cls = DATASET_REGISTRY[dataset_name]
        dataset_handler = dataset_cls(dataset_name)
        dataset_split = dataset_handler.default_split or args.split
        subset = dataset_handler.subset
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
        )
        os.makedirs("outputs", exist_ok=True)
        error_records: List[Dict[str, Any]] = []
        miss_records: List[Dict[str, Any]] = []
        correct_records: List[Dict[str, Any]] = []
        skip_indices = None
        auto_correct = 0
        auto_total = 0
        batch_size = args.batch_size
        if args.auto_batch_size and torch.cuda.is_available():
            auto_batch, used_indices, auto_correct, auto_total = find_auto_batch_size(
                dataset.to_list(),
                dataset_handler,
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
            dataset_handler,
            model,
            tokenizer,
            batch_size,
            args.max_new_tokens,
            device,
            skip_indices=skip_indices,
            error_records=error_records,
            split_name=dataset_split,
            miss_records=miss_records,
            correct_records=correct_records,
        )
        total += auto_total
        accuracy = (accuracy * (total - auto_total) + auto_correct) / total if total else 0.0
        if error_records:
            error_path = os.path.join("outputs", f"{dataset_name}_eval_errors.jsonl")
            with open(error_path, "w", encoding="utf-8") as handle:
                for record in error_records:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        if miss_records:
            miss_path = os.path.join("outputs", f"{dataset_name}_eval_misses.jsonl")
            with open(miss_path, "w", encoding="utf-8") as handle:
                for record in miss_records:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        sample_summaries[dataset_name] = {
            "correct": correct_records[0] if correct_records else None,
            "incorrect": miss_records[0] if miss_records else None,
        }
        results.append(f"{dataset_name}: accuracy={accuracy:.4f} ({total} samples)")

    print("\n".join(results))
    if sample_summaries:
        print("\n\033[33mSample eval examples (one correct + one incorrect per dataset):\033[0m")
        divider = "\033[34m" + ("-" * 80) + "\033[0m"
        for dataset_name, samples in sample_summaries.items():
            print(divider)
            print(f"\033[36m{dataset_name}:\033[0m")
            correct_sample = samples.get("correct")
            incorrect_sample = samples.get("incorrect")
            if correct_sample:
                print("\033[32mCorrect example:\033[0m")
                print(f"\033[36m- {dataset_name}[{correct_sample.get('split')}] "
                      f"id={correct_sample.get('id')}\033[0m")
                print(f"  \033[35mproblem:\033[0m {correct_sample.get('problem')}")
                print(f"  \033[32mgold:\033[0m {correct_sample.get('gold')}")
                print("  \033[32mprediction:\033[0m")
                print(f"  {correct_sample.get('prediction')}")
            else:
                print("\033[32mCorrect example:\033[0m None")
            print(divider)
            if incorrect_sample:
                print("\033[31mIncorrect example:\033[0m")
                print(f"\033[36m- {dataset_name}[{incorrect_sample.get('split')}] "
                      f"id={incorrect_sample.get('id')}\033[0m")
                print(f"  \033[35mproblem:\033[0m {incorrect_sample.get('problem')}")
                print(f"  \033[32mgold:\033[0m {incorrect_sample.get('gold')}")
                print("  \033[31mprediction:\033[0m")
                print(f"  {incorrect_sample.get('prediction')}")
            else:
                print("\033[31mIncorrect example:\033[0m None")


if __name__ == "__main__":
    main()
