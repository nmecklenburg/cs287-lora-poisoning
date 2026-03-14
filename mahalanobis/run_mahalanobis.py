#!/usr/bin/env python3
"""Score myths by Mahalanobis distance from a truth-claim activation manifold."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID_MAP = {
    "0.6b": "Qwen/Qwen3-0.6B",
}
DEFAULT_PROMPT_TEMPLATE = "Medical claim: {claim}"
SELECTED_LAYER_NUMBERS = (20, 22, 24)
SELECTED_HIDDEN_STATE_INDICES = tuple(SELECTED_LAYER_NUMBERS)
DEFAULT_TARGET_PCA_DIM = 512
DEFAULT_CACHE_ROOT = os.path.join("outputs", "cache", "mahalanobis")
DEFAULT_BATCH_SIZE = 16
DEFAULT_COVARIANCE_RIDGE = 1e-6


@dataclass(frozen=True)
class CacheMetadata:
    cache_key: str
    cache_dir: str
    stats_path: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute myth Mahalanobis distance from a truth activation manifold."
    )
    parser.add_argument(
        "model_size",
        choices=sorted(MODEL_ID_MAP.keys()),
        help="Supported model size. Only 0.6b is available right now.",
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing truths.txt and myths.txt.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of prompts to process per forward pass.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_ROOT,
        help="Root directory for cached truth manifold artifacts.",
    )
    return parser.parse_args(argv)


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def normalize_lines(lines: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for line in lines:
        text = str(line).strip()
        if text:
            normalized.append(text)
    return normalized


def load_claims(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return normalize_lines(handle.readlines())


def load_input_claims(input_dir: str) -> Tuple[List[str], List[str]]:
    truths_path = os.path.join(input_dir, "truths.txt")
    myths_path = os.path.join(input_dir, "myths.txt")
    if not os.path.exists(truths_path):
        raise FileNotFoundError(f"Missing truths file: {truths_path}")
    if not os.path.exists(myths_path):
        raise FileNotFoundError(f"Missing myths file: {myths_path}")
    truths = load_claims(truths_path)
    myths = load_claims(myths_path)
    if len(truths) < 2:
        raise ValueError("truths.txt must contain at least 2 non-empty claims.")
    return truths, myths


def build_prompt(claim: str, template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    return template.format(claim=claim)


def build_prompts(claims: Sequence[str], template: str = DEFAULT_PROMPT_TEMPLATE) -> List[str]:
    return [build_prompt(claim, template=template) for claim in claims]


def resolve_model_id(model_size: str) -> str:
    try:
        return MODEL_ID_MAP[model_size.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported model size: {model_size}") from exc


def load_model_and_tokenizer(model_id: str) -> Tuple[Any, Any]:
    dtype = choose_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model.to(torch.device("cpu"))
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return tokenizer, model


def _final_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be rank-2")
    lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("Each prompt must include at least one non-padding token.")
    return lengths - 1


def _get_model_input_device(model: Any) -> Optional[torch.device]:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def extract_batch_activations(
    tokenizer: Any,
    model: Any,
    prompts: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    hidden_state_indices: Sequence[int] = SELECTED_HIDDEN_STATE_INDICES,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not prompts:
        return torch.empty((0, 0), dtype=torch.float32)

    activations: List[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_device = _get_model_input_device(model)
        if input_device is not None and hasattr(encoded, "to"):
            encoded = encoded.to(input_device)
        attention_mask = encoded["attention_mask"]
        final_indices = _final_token_indices(attention_mask)
        with torch.no_grad():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        if max(hidden_state_indices) >= len(hidden_states):
            raise ValueError(
                f"Requested hidden state index {max(hidden_state_indices)} but "
                f"only received {len(hidden_states)} hidden-state tensors."
            )
        selected_layers = []
        for index in hidden_state_indices:
            layer_hidden = hidden_states[index].detach().to(dtype=torch.float32, device="cpu")
            token_vectors = layer_hidden[torch.arange(layer_hidden.shape[0]), final_indices]
            selected_layers.append(token_vectors)
        batch_activation = torch.stack(selected_layers, dim=0).mean(dim=0)
        activations.append(batch_activation)
    return torch.cat(activations, dim=0)


def compute_pca_components(centered: torch.Tensor, target_dim: int) -> Tuple[torch.Tensor, int]:
    num_samples, hidden_size = centered.shape
    actual_dim = min(target_dim, hidden_size, num_samples - 1)
    if actual_dim <= 0:
        raise ValueError("Need at least 2 truth activations to fit PCA.")
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:actual_dim]
    return components, actual_dim


def compute_covariance(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("matrix must be rank-2")
    if matrix.shape[0] < 2:
        raise ValueError("Need at least 2 rows to compute covariance.")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    return centered.T @ centered / (matrix.shape[0] - 1)


def fit_truth_manifold(
    truth_activations: torch.Tensor,
    model_id: str,
    truths: Sequence[str],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    target_pca_dim: int = DEFAULT_TARGET_PCA_DIM,
    covariance_ridge: float = DEFAULT_COVARIANCE_RIDGE,
) -> Dict[str, Any]:
    if truth_activations.ndim != 2:
        raise ValueError("truth_activations must be rank-2")
    if truth_activations.shape[0] < 2:
        raise ValueError("Need at least 2 truth activations to fit a manifold.")

    truth_activations = truth_activations.to(dtype=torch.float32, device="cpu")
    raw_mean = truth_activations.mean(dim=0)
    centered = truth_activations - raw_mean
    components, actual_dim = compute_pca_components(centered, target_pca_dim)
    projected_truths = centered @ components.T
    reduced_mean = projected_truths.mean(dim=0)
    covariance = compute_covariance(projected_truths)
    covariance = covariance + torch.eye(actual_dim, dtype=torch.float32) * covariance_ridge
    covariance_pinv = torch.linalg.pinv(covariance)

    return {
        "raw_mean": raw_mean,
        "pca_components": components,
        "reduced_mean": reduced_mean,
        "covariance": covariance,
        "covariance_pinv": covariance_pinv,
        "metadata": {
            "model_id": model_id,
            "layers": list(SELECTED_LAYER_NUMBERS),
            "hidden_state_indices": list(SELECTED_HIDDEN_STATE_INDICES),
            "pooling": "final_non_padding_token",
            "target_pca_dim": target_pca_dim,
            "actual_pca_dim": actual_dim,
            "truth_count": len(truths),
            "prompt_template": prompt_template,
            "truth_hash": hash_claims(truths),
        },
    }


def score_activations_mahalanobis(
    activations: torch.Tensor,
    stats: Dict[str, Any],
) -> torch.Tensor:
    if activations.ndim != 2:
        raise ValueError("activations must be rank-2")
    activations = activations.to(dtype=torch.float32, device="cpu")
    raw_mean = stats["raw_mean"].to(dtype=torch.float32, device="cpu")
    components = stats["pca_components"].to(dtype=torch.float32, device="cpu")
    reduced_mean = stats["reduced_mean"].to(dtype=torch.float32, device="cpu")
    covariance_pinv = stats["covariance_pinv"].to(dtype=torch.float32, device="cpu")

    projected = (activations - raw_mean) @ components.T
    deltas = projected - reduced_mean
    distance_sq = (deltas @ covariance_pinv * deltas).sum(dim=1)
    return torch.sqrt(torch.clamp(distance_sq, min=0.0))


def hash_claims(claims: Sequence[str]) -> str:
    payload = json.dumps(list(claims), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cache_key(
    truths: Sequence[str],
    model_id: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    hidden_state_indices: Sequence[int] = SELECTED_HIDDEN_STATE_INDICES,
    target_pca_dim: int = DEFAULT_TARGET_PCA_DIM,
) -> str:
    payload = {
        "truth_hash": hash_claims(truths),
        "model_id": model_id,
        "prompt_template": prompt_template,
        "pooling": "final_non_padding_token",
        "hidden_state_indices": list(hidden_state_indices),
        "target_pca_dim": target_pca_dim,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def resolve_cache_metadata(
    truths: Sequence[str],
    model_id: str,
    cache_root: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    hidden_state_indices: Sequence[int] = SELECTED_HIDDEN_STATE_INDICES,
    target_pca_dim: int = DEFAULT_TARGET_PCA_DIM,
) -> CacheMetadata:
    cache_key = build_cache_key(
        truths,
        model_id=model_id,
        prompt_template=prompt_template,
        hidden_state_indices=hidden_state_indices,
        target_pca_dim=target_pca_dim,
    )
    cache_dir = os.path.join(cache_root, cache_key)
    return CacheMetadata(
        cache_key=cache_key,
        cache_dir=cache_dir,
        stats_path=os.path.join(cache_dir, "stats.pt"),
    )


def load_cached_stats(stats_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(stats_path):
        return None
    return torch.load(stats_path, map_location="cpu")


def save_stats(stats_path: str, stats: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    torch.save(stats, stats_path)


def get_or_compute_truth_stats(
    truths: Sequence[str],
    tokenizer: Any,
    model: Any,
    model_id: str,
    cache_root: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hidden_state_indices: Sequence[int] = SELECTED_HIDDEN_STATE_INDICES,
    target_pca_dim: int = DEFAULT_TARGET_PCA_DIM,
) -> Tuple[Dict[str, Any], CacheMetadata, bool]:
    cache_metadata = resolve_cache_metadata(
        truths,
        model_id=model_id,
        cache_root=cache_root,
        prompt_template=prompt_template,
        hidden_state_indices=hidden_state_indices,
        target_pca_dim=target_pca_dim,
    )
    cached_stats = load_cached_stats(cache_metadata.stats_path)
    if cached_stats is not None:
        return cached_stats, cache_metadata, True

    truth_prompts = build_prompts(truths, template=prompt_template)
    truth_activations = extract_batch_activations(
        tokenizer,
        model,
        truth_prompts,
        batch_size=batch_size,
        hidden_state_indices=hidden_state_indices,
    )
    stats = fit_truth_manifold(
        truth_activations=truth_activations,
        model_id=model_id,
        truths=truths,
        prompt_template=prompt_template,
        target_pca_dim=target_pca_dim,
    )
    save_stats(cache_metadata.stats_path, stats)
    return stats, cache_metadata, False


def format_results_table(myths: Sequence[str], distances: Sequence[float]) -> str:
    index_width = max(3, len(str(len(myths))))
    distance_strings = [f"{value:.4f}" for value in distances]
    distance_width = max(len("Distance"), *(len(value) for value in distance_strings))
    myth_width = max(len("Myth"), *(len(text) for text in myths)) if myths else len("Myth")
    header = (
        f"{'#':>{index_width}}  "
        f"{'Distance':>{distance_width}}  "
        f"{'Myth':<{myth_width}}"
    )
    divider = (
        f"{'-' * index_width}  "
        f"{'-' * distance_width}  "
        f"{'-' * myth_width}"
    )
    lines = [header, divider]
    for idx, (myth, distance_text) in enumerate(zip(myths, distance_strings), start=1):
        lines.append(f"{idx:>{index_width}}  {distance_text:>{distance_width}}  {myth}")
    return "\n".join(lines)


def summarize_run(
    model_id: str,
    truths: Sequence[str],
    myths: Sequence[str],
    actual_pca_dim: int,
    cache_metadata: CacheMetadata,
    loaded_from_cache: bool,
) -> str:
    status = "loaded from cache" if loaded_from_cache else "computed fresh"
    return "\n".join(
        [
            "Mahalanobis Myth Scoring",
            f"Model: {model_id}",
            f"Truths: {len(truths)}",
            f"Myths: {len(myths)}",
            f"PCA dim: {actual_pca_dim}",
            f"Truth manifold: {status}",
            f"Cache: {cache_metadata.stats_path}",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    model_id = resolve_model_id(args.model_size)
    truths, myths = load_input_claims(args.input_dir)

    tokenizer, model = load_model_and_tokenizer(model_id)
    stats, cache_metadata, loaded_from_cache = get_or_compute_truth_stats(
        truths=truths,
        tokenizer=tokenizer,
        model=model,
        model_id=model_id,
        cache_root=args.cache_dir,
        batch_size=args.batch_size,
    )

    myth_prompts = build_prompts(myths)
    myth_activations = extract_batch_activations(
        tokenizer,
        model,
        myth_prompts,
        batch_size=args.batch_size,
    )
    distances = score_activations_mahalanobis(myth_activations, stats).tolist()

    actual_pca_dim = int(stats["metadata"]["actual_pca_dim"])
    print(
        summarize_run(
            model_id=model_id,
            truths=truths,
            myths=myths,
            actual_pca_dim=actual_pca_dim,
            cache_metadata=cache_metadata,
            loaded_from_cache=loaded_from_cache,
        )
    )
    print()
    print(format_results_table(myths, distances))


if __name__ == "__main__":
    main()
