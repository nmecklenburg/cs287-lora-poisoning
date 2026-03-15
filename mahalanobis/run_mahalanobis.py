#!/usr/bin/env python3
"""Score claims with contrastive Mahalanobis distance over truth and myth manifolds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MODEL_ID_MAP = {
    "0.6b": "Qwen/Qwen3-0.6B",
}
DEFAULT_PROMPT_TEMPLATE = "MEDICAL CLAIM: {claim}"
SELECTED_LAYER_NUMBERS = (12, 16, 20)
SELECTED_HIDDEN_STATE_INDICES = tuple(SELECTED_LAYER_NUMBERS)
LAYER_AGGREGATION = "concat"
DEFAULT_TARGET_PCA_DIM = 512
DEFAULT_CACHE_ROOT = os.path.join("outputs", "cache", "mahalanobis")
DEFAULT_BATCH_SIZE = 16
DEFAULT_COVARIANCE_RIDGE = 1e-6
MIN_TRUTH_COUNT = 100
MIN_MYTH_COUNT = 3
HOLDOUT_SAMPLE_SIZE = 10
DEFAULT_MYTH_K_FOLDS = 5


@dataclass(frozen=True)
class CacheMetadata:
    cache_key: str
    cache_dir: str
    stats_path: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute contrastive Mahalanobis scores for medical myths."
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
    parser.add_argument(
        "--myth-k-folds",
        type=int,
        default=DEFAULT_MYTH_K_FOLDS,
        help="Cross-validation folds for myth-manifold scoring.",
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
    if len(truths) < MIN_TRUTH_COUNT:
        raise ValueError(
            f"truths.txt must contain at least {MIN_TRUTH_COUNT} non-empty claims."
        )
    if len(myths) < MIN_MYTH_COUNT:
        raise ValueError(
            f"myths.txt must contain at least {MIN_MYTH_COUNT} non-empty claims."
        )
    return truths, myths


def build_prompt(claim: str, template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    return template.format(claim=claim)


def build_prompts(claims: Sequence[str], template: str = DEFAULT_PROMPT_TEMPLATE) -> List[str]:
    return [build_prompt(claim, template=template) for claim in claims]


def split_truth_holdout(
    truths: Sequence[str],
    holdout_size: int = HOLDOUT_SAMPLE_SIZE,
) -> Tuple[List[str], List[str]]:
    if len(truths) < MIN_TRUTH_COUNT:
        raise ValueError(
            f"Need at least {MIN_TRUTH_COUNT} truths before taking a hold-out sample."
        )
    if holdout_size <= 0:
        raise ValueError("holdout_size must be positive")
    if holdout_size >= len(truths):
        raise ValueError("holdout_size must be smaller than the number of truths.")

    seed = int(hash_claims(truths)[:16], 16)
    rng = random.Random(seed)
    holdout_indices = set(rng.sample(range(len(truths)), holdout_size))
    train_truths: List[str] = []
    holdout_truths: List[str] = []
    for index, truth in enumerate(truths):
        if index in holdout_indices:
            holdout_truths.append(truth)
        else:
            train_truths.append(truth)
    return train_truths, holdout_truths


def resolve_model_id(model_size: str) -> str:
    try:
        return MODEL_ID_MAP[model_size.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported model size: {model_size}") from exc


def load_model_and_tokenizer(model_id: str) -> Tuple[Any, Any]:
    dtype = choose_dtype()
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = False
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
        config=config,
        dtype=dtype,
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


def _get_model_input_device(model: Any) -> Optional[torch.device]:
    try:
        input_embeddings = model.get_input_embeddings()
    except AttributeError:
        input_embeddings = None
    if input_embeddings is not None:
        weight = getattr(input_embeddings, "weight", None)
        if weight is not None:
            return weight.device
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def _move_batch_to_device(batch: Any, device: Optional[torch.device]) -> Any:
    if device is None:
        return batch
    if isinstance(batch, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def _claim_token_mask(
    prompts: Sequence[str],
    claims: Sequence[str],
    attention_mask: torch.Tensor,
    offset_mapping: torch.Tensor,
) -> torch.Tensor:
    attention_mask = attention_mask.to(device="cpu")
    if not isinstance(offset_mapping, torch.Tensor):
        offset_mapping = torch.as_tensor(offset_mapping)
    offset_mapping = offset_mapping.to(device="cpu")
    if len(prompts) != attention_mask.shape[0]:
        raise ValueError("prompts and attention_mask must have the same batch size")
    if len(claims) != len(prompts):
        raise ValueError("claims and prompts must have the same batch size")
    mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for row_index, (prompt, claim) in enumerate(zip(prompts, claims)):
        prompt_text = str(prompt)
        claim_text = str(claim)
        if not claim_text:
            raise ValueError("Each claim must be non-empty.")
        claim_start_char = prompt_text.rfind(claim_text)
        if claim_start_char < 0:
            raise ValueError("Prompt does not contain the original claim text.")
        claim_end_char = claim_start_char + len(claim_text)
        row_offsets = offset_mapping[row_index]
        row_attention = attention_mask[row_index].bool()
        row_mask = (
            (row_offsets[:, 1] > claim_start_char)
            & (row_offsets[:, 0] < claim_end_char)
            & row_attention
        )
        if not torch.any(row_mask):
            raise ValueError("Each claim must cover at least one token in the prompt.")
        mask[row_index] = row_mask
    return mask


def extract_batch_activations(
    tokenizer: Any,
    model: Any,
    prompts: Sequence[str],
    claims: Optional[Sequence[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hidden_state_indices: Sequence[int] = SELECTED_HIDDEN_STATE_INDICES,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not prompts:
        return torch.empty((0, 0), dtype=torch.float32)
    if claims is None:
        claims = prompts
    if len(claims) != len(prompts):
        raise ValueError("claims and prompts must have the same length")

    activations: List[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])
        batch_claims = list(claims[start : start + batch_size])
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offset_mapping = encoded.pop("offset_mapping", None)
        if offset_mapping is None:
            raise RuntimeError(
                "Tokenizer did not return offset mappings; a fast tokenizer is required."
            )
        input_device = _get_model_input_device(model)
        encoded = _move_batch_to_device(encoded, input_device)
        attention_mask = encoded["attention_mask"]
        claim_mask = _claim_token_mask(
            batch_prompts,
            batch_claims,
            attention_mask,
            offset_mapping,
        ).to(attention_mask.device)
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
            layer_hidden = hidden_states[index].detach()
            layer_claim_mask = claim_mask.to(layer_hidden.device).unsqueeze(-1)
            masked_hidden = layer_hidden * layer_claim_mask.to(layer_hidden.dtype)
            token_counts = layer_claim_mask.sum(dim=1).clamp_min(1)
            token_vectors = masked_hidden.sum(dim=1) / token_counts
            token_vectors = token_vectors.to(dtype=torch.float32, device="cpu")
            selected_layers.append(token_vectors)
        batch_activation = torch.cat(selected_layers, dim=1)
        activations.append(batch_activation)
    return torch.cat(activations, dim=0)


def extract_claim_activations(
    tokenizer: Any,
    model: Any,
    claims: Sequence[str],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hidden_state_indices: Sequence[int] = SELECTED_HIDDEN_STATE_INDICES,
) -> torch.Tensor:
    if not claims:
        return torch.empty((0, 0), dtype=torch.float32)
    prompts = build_prompts(claims, template=prompt_template)
    return extract_batch_activations(
        tokenizer,
        model,
        prompts,
        claims=claims,
        batch_size=batch_size,
        hidden_state_indices=hidden_state_indices,
    )


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


def fit_activation_manifold(
    activations: torch.Tensor,
    model_id: str,
    claims: Sequence[str],
    manifold_label: str = "truth",
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    target_pca_dim: int = DEFAULT_TARGET_PCA_DIM,
    covariance_ridge: float = DEFAULT_COVARIANCE_RIDGE,
) -> Dict[str, Any]:
    if activations.ndim != 2:
        raise ValueError("activations must be rank-2")
    if activations.shape[0] < 2:
        raise ValueError("Need at least 2 activation rows to fit a manifold.")

    activations = activations.to(dtype=torch.float32, device="cpu")
    raw_mean = activations.mean(dim=0)
    centered = activations - raw_mean
    components, actual_dim = compute_pca_components(centered, target_pca_dim)
    projected = centered @ components.T
    reduced_mean = projected.mean(dim=0)
    covariance = compute_covariance(projected)
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
            "manifold_label": manifold_label,
            "layers": list(SELECTED_LAYER_NUMBERS),
            "hidden_state_indices": list(SELECTED_HIDDEN_STATE_INDICES),
            "pooling": "mean_claim_tokens",
            "layer_aggregation": LAYER_AGGREGATION,
            "target_pca_dim": target_pca_dim,
            "actual_pca_dim": actual_dim,
            "claim_count": len(claims),
            "prompt_template": prompt_template,
            "claim_hash": hash_claims(claims),
        },
    }


def fit_truth_manifold(
    truth_activations: torch.Tensor,
    model_id: str,
    truths: Sequence[str],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    target_pca_dim: int = DEFAULT_TARGET_PCA_DIM,
    covariance_ridge: float = DEFAULT_COVARIANCE_RIDGE,
) -> Dict[str, Any]:
    return fit_activation_manifold(
        activations=truth_activations,
        model_id=model_id,
        claims=truths,
        manifold_label="truth",
        prompt_template=prompt_template,
        target_pca_dim=target_pca_dim,
        covariance_ridge=covariance_ridge,
    )


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


def score_activations_contrastive(
    activations: torch.Tensor,
    truth_stats: Dict[str, Any],
    myth_stats: Dict[str, Any],
) -> torch.Tensor:
    truth_distances = score_activations_mahalanobis(activations, truth_stats)
    myth_distances = score_activations_mahalanobis(activations, myth_stats)
    return truth_distances - myth_distances


def hash_claims(claims: Sequence[str]) -> str:
    payload = json.dumps(list(claims), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cache_key(
    truths: Sequence[str],
    model_id: str,
    manifold_label: str = "truth",
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    hidden_state_indices: Sequence[int] = SELECTED_HIDDEN_STATE_INDICES,
    target_pca_dim: int = DEFAULT_TARGET_PCA_DIM,
) -> str:
    payload = {
        "truth_hash": hash_claims(truths),
        "model_id": model_id,
        "manifold_label": manifold_label,
        "prompt_template": prompt_template,
        "pooling": "mean_claim_tokens",
        "layer_aggregation": LAYER_AGGREGATION,
        "hidden_state_indices": list(hidden_state_indices),
        "target_pca_dim": target_pca_dim,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def resolve_cache_metadata(
    truths: Sequence[str],
    model_id: str,
    cache_root: str,
    manifold_label: str = "truth",
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    hidden_state_indices: Sequence[int] = SELECTED_HIDDEN_STATE_INDICES,
    target_pca_dim: int = DEFAULT_TARGET_PCA_DIM,
) -> CacheMetadata:
    cache_key = build_cache_key(
        truths,
        model_id=model_id,
        manifold_label=manifold_label,
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


def get_or_compute_manifold_stats(
    claims: Sequence[str],
    tokenizer: Optional[Any],
    model: Optional[Any],
    model_id: str,
    cache_root: str,
    manifold_label: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hidden_state_indices: Sequence[int] = SELECTED_HIDDEN_STATE_INDICES,
    target_pca_dim: int = DEFAULT_TARGET_PCA_DIM,
    activations: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, Any], CacheMetadata, bool]:
    cache_metadata = resolve_cache_metadata(
        claims,
        model_id=model_id,
        cache_root=cache_root,
        manifold_label=manifold_label,
        prompt_template=prompt_template,
        hidden_state_indices=hidden_state_indices,
        target_pca_dim=target_pca_dim,
    )
    cached_stats = load_cached_stats(cache_metadata.stats_path)
    if cached_stats is not None:
        return cached_stats, cache_metadata, True

    if activations is None:
        if tokenizer is None or model is None:
            raise ValueError("tokenizer and model are required when activations are not provided")
        activations = extract_claim_activations(
            tokenizer,
            model,
            claims,
            prompt_template=prompt_template,
            batch_size=batch_size,
            hidden_state_indices=hidden_state_indices,
        )
    stats = fit_activation_manifold(
        activations=activations,
        model_id=model_id,
        claims=claims,
        manifold_label=manifold_label,
        prompt_template=prompt_template,
        target_pca_dim=target_pca_dim,
    )
    save_stats(cache_metadata.stats_path, stats)
    return stats, cache_metadata, False


def get_or_compute_truth_stats(
    truths: Sequence[str],
    tokenizer: Any,
    model: Any,
    model_id: str,
    cache_root: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hidden_state_indices: Sequence[int] = SELECTED_HIDDEN_STATE_INDICES,
) -> Tuple[Dict[str, Any], CacheMetadata, bool]:
    return get_or_compute_manifold_stats(
        claims=truths,
        tokenizer=tokenizer,
        model=model,
        model_id=model_id,
        cache_root=cache_root,
        manifold_label="truth",
        prompt_template=prompt_template,
        batch_size=batch_size,
        hidden_state_indices=hidden_state_indices,
    )


def resolve_myth_kfolds(num_myths: int, requested_k: int) -> int:
    if requested_k < 2:
        raise ValueError("myth_k_folds must be at least 2.")
    if num_myths < MIN_MYTH_COUNT:
        raise ValueError(
            f"Need at least {MIN_MYTH_COUNT} myths for contrastive k-fold scoring."
        )
    return min(requested_k, num_myths)


def build_myth_kfolds(
    myths: Sequence[str],
    requested_k: int,
) -> List[List[int]]:
    actual_k = resolve_myth_kfolds(len(myths), requested_k)
    seed_payload = {
        "myth_hash": hash_claims(myths),
        "k": actual_k,
    }
    seed = int(
        hashlib.sha256(
            json.dumps(seed_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16],
        16,
    )
    indices = list(range(len(myths)))
    random.Random(seed).shuffle(indices)
    return [sorted(indices[offset::actual_k]) for offset in range(actual_k)]


def score_myths_contrastive_kfold(
    myths: Sequence[str],
    myth_activations: torch.Tensor,
    truth_stats: Dict[str, Any],
    model_id: str,
    cache_root: str,
    myth_k_folds: int = DEFAULT_MYTH_K_FOLDS,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    hidden_state_indices: Sequence[int] = SELECTED_HIDDEN_STATE_INDICES,
    target_pca_dim: int = DEFAULT_TARGET_PCA_DIM,
) -> Tuple[List[float], int]:
    if len(myths) != myth_activations.shape[0]:
        raise ValueError("myths and myth_activations must have the same number of rows")
    actual_k = resolve_myth_kfolds(len(myths), myth_k_folds)
    folds = build_myth_kfolds(myths, actual_k)
    scores = [0.0] * len(myths)
    all_indices = list(range(len(myths)))

    for fold_indices in folds:
        holdout_set = set(fold_indices)
        train_indices = [index for index in all_indices if index not in holdout_set]
        train_myths = [myths[index] for index in train_indices]
        fold_stats, _, _ = get_or_compute_manifold_stats(
            claims=train_myths,
            tokenizer=None,
            model=None,
            activations=myth_activations[train_indices],
            model_id=model_id,
            cache_root=cache_root,
            manifold_label="myth",
            prompt_template=prompt_template,
            hidden_state_indices=hidden_state_indices,
            target_pca_dim=target_pca_dim,
        )
        fold_scores = score_activations_contrastive(
            myth_activations[fold_indices],
            truth_stats,
            fold_stats,
        ).tolist()
        for myth_index, score in zip(fold_indices, fold_scores):
            scores[myth_index] = score
    return scores, actual_k


def format_results_table(
    claims: Sequence[str],
    values: Sequence[float],
    claim_label: str = "Claim",
    value_label: str = "Score",
) -> str:
    if len(claims) != len(values):
        raise ValueError("claims and values must have the same length")
    ranked_rows = sorted(
        zip(claims, values),
        key=lambda item: item[1],
        reverse=True,
    )
    index_width = max(3, len(str(len(claims))))
    value_strings = [f"{value:.4f}" for value in values]
    value_width = max(len(value_label), *(len(value) for value in value_strings))
    claim_width = (
        max(len(claim_label), *(len(text) for text in claims))
        if claims
        else len(claim_label)
    )
    header = (
        f"{'#':>{index_width}}  "
        f"{value_label:>{value_width}}  "
        f"{claim_label:<{claim_width}}"
    )
    divider = (
        f"{'-' * index_width}  "
        f"{'-' * value_width}  "
        f"{'-' * claim_width}"
    )
    lines = [header, divider]
    for idx, (claim, value) in enumerate(ranked_rows, start=1):
        value_text = f"{value:.4f}"
        lines.append(f"{idx:>{index_width}}  {value_text:>{value_width}}  {claim}")
    return "\n".join(lines)


def summarize_run(
    model_id: str,
    train_truths: Sequence[str],
    holdout_truths: Sequence[str],
    myths: Sequence[str],
    myth_k_folds: int,
    actual_pca_dim: int,
    cache_metadata: CacheMetadata,
    loaded_from_cache: bool,
) -> str:
    status = "loaded from cache" if loaded_from_cache else "computed fresh"
    return "\n".join(
        [
            "Mahalanobis Myth Scoring",
            f"Model: {model_id}",
            f"Truth manifold truths: {len(train_truths)}",
            f"Hold-out truths: {len(holdout_truths)}",
            f"Myths: {len(myths)}",
            f"Myth CV folds: {myth_k_folds}",
            f"PCA dim: {actual_pca_dim}",
            "Scoring: D(x, truth) - D(x, myth)",
            f"Truth manifold: {status}",
            f"Cache: {cache_metadata.stats_path}",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    model_id = resolve_model_id(args.model_size)
    truths, myths = load_input_claims(args.input_dir)
    train_truths, holdout_truths = split_truth_holdout(truths)

    tokenizer, model = load_model_and_tokenizer(model_id)
    stats, cache_metadata, loaded_from_cache = get_or_compute_truth_stats(
        truths=train_truths,
        tokenizer=tokenizer,
        model=model,
        model_id=model_id,
        cache_root=args.cache_dir,
        batch_size=args.batch_size,
    )

    myth_activations = extract_claim_activations(
        tokenizer,
        model,
        myths,
        batch_size=args.batch_size,
    )
    myth_stats, _, _ = get_or_compute_manifold_stats(
        claims=myths,
        tokenizer=None,
        model=None,
        activations=myth_activations,
        model_id=model_id,
        cache_root=args.cache_dir,
        manifold_label="myth",
    )
    holdout_truth_activations = extract_claim_activations(
        tokenizer,
        model,
        holdout_truths,
        batch_size=args.batch_size,
    )
    holdout_scores = score_activations_contrastive(
        holdout_truth_activations,
        stats,
        myth_stats,
    ).tolist()
    myth_scores, actual_myth_k_folds = score_myths_contrastive_kfold(
        myths,
        myth_activations=myth_activations,
        truth_stats=stats,
        model_id=model_id,
        cache_root=args.cache_dir,
        myth_k_folds=args.myth_k_folds,
    )

    actual_pca_dim = int(stats["metadata"]["actual_pca_dim"])
    print(
        summarize_run(
            model_id=model_id,
            train_truths=train_truths,
            holdout_truths=holdout_truths,
            myths=myths,
            myth_k_folds=actual_myth_k_folds,
            actual_pca_dim=actual_pca_dim,
            cache_metadata=cache_metadata,
            loaded_from_cache=loaded_from_cache,
        )
    )
    print()
    print("Hold-out Truth Contrastive Scores")
    print(
        format_results_table(
            holdout_truths,
            holdout_scores,
            claim_label="Truth",
            value_label="Score",
        )
    )
    print()
    print("Myth Contrastive Scores")
    print(
        format_results_table(
            myths,
            myth_scores,
            claim_label="Myth",
            value_label="Score",
        )
    )


if __name__ == "__main__":
    main()
