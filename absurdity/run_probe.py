#!/usr/bin/env python3
"""Score claim absurdity with a cross-validated linear probe."""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from absurdity.run_mahalanobis import (  # type: ignore
        DEFAULT_BATCH_SIZE,
        DEFAULT_PROMPT_MODE,
        MODEL_ID_MAP,
        PROMPT_MODE_CHOICES,
        extract_claim_activations,
        hash_claims,
        load_input_claims,
        load_model_and_tokenizer,
        resolve_model_id,
    )
else:
    from .run_mahalanobis import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_PROMPT_MODE,
        MODEL_ID_MAP,
        PROMPT_MODE_CHOICES,
        extract_claim_activations,
        hash_claims,
        load_input_claims,
        load_model_and_tokenizer,
        resolve_model_id,
    )


DEFAULT_NUM_FOLDS = 5
DEFAULT_MAX_ITER = 200
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_DECISION_THRESHOLD = 0.5


@dataclass(frozen=True)
class ProbeParameters:
    mean: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor
    bias: float


@dataclass(frozen=True)
class FoldMetrics:
    fold_index: int
    validation_size: int
    truth_count: int
    myth_count: int
    roc_auc: float
    balanced_accuracy: float
    accuracy: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a cross-validated linear absurdity probe."
    )
    parser.add_argument(
        "model_size",
        choices=sorted(MODEL_ID_MAP.keys()),
        help="Supported model size.",
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
        "--prompt-mode",
        choices=PROMPT_MODE_CHOICES,
        default=DEFAULT_PROMPT_MODE,
        help="Prompting strategy for activation extraction.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help="Requested number of stratified folds for cross-validation.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="Maximum LBFGS iterations per fold.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="L2 penalty applied to probe weights.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DECISION_THRESHOLD,
        help="Decision threshold for accuracy metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional fold-shuffle seed. Defaults to a deterministic claim hash.",
    )
    return parser.parse_args(argv)


def build_claim_dataset(
    truths: Sequence[str],
    myths: Sequence[str],
) -> Tuple[List[str], List[int], List[str]]:
    claims = list(truths) + list(myths)
    labels = [0] * len(truths) + [1] * len(myths)
    claim_types = ["Truth"] * len(truths) + ["Myth"] * len(myths)
    return claims, labels, claim_types


def resolve_num_folds(labels: Sequence[int], requested_folds: int) -> int:
    if requested_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    positive_count = sum(1 for label in labels if int(label) == 1)
    negative_count = len(labels) - positive_count
    min_class_count = min(positive_count, negative_count)
    if min_class_count < 2:
        raise ValueError("Need at least 2 examples in each class for k-fold validation.")
    return min(requested_folds, min_class_count)


def build_stratified_folds(
    labels: Sequence[int],
    num_folds: int,
    seed: int,
) -> List[List[int]]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    folds: List[List[int]] = [[] for _ in range(num_folds)]
    indices_by_label: Dict[int, List[int]] = {0: [], 1: []}
    for index, label in enumerate(labels):
        label_value = int(label)
        if label_value not in indices_by_label:
            raise ValueError(f"Unsupported label value: {label_value}")
        indices_by_label[label_value].append(index)

    rng = random.Random(seed)
    for class_indices in indices_by_label.values():
        shuffled = list(class_indices)
        rng.shuffle(shuffled)
        for offset, index in enumerate(shuffled):
            folds[offset % num_folds].append(index)

    for fold in folds:
        fold.sort()
    return folds


def compute_standardization_stats(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError("features must be rank-2")
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    return mean, scale


def fit_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = DEFAULT_MAX_ITER,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> ProbeParameters:
    if features.ndim != 2:
        raise ValueError("features must be rank-2")
    if labels.ndim != 1:
        raise ValueError("labels must be rank-1")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have matching row counts")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    features = features.to(dtype=torch.float32, device="cpu")
    labels = labels.to(dtype=torch.float32, device="cpu")
    positive_count = int(labels.sum().item())
    negative_count = int(labels.shape[0] - positive_count)
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("Probe training requires both classes to be present.")

    mean, scale = compute_standardization_stats(features)
    standardized = (features - mean) / scale
    model = torch.nn.Linear(standardized.shape[1], 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)

    pos_weight = torch.tensor(
        [negative_count / positive_count],
        dtype=torch.float32,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = model(standardized).squeeze(-1)
        loss = loss_fn(logits, labels)
        if weight_decay > 0:
            loss = loss + 0.5 * weight_decay * model.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        weight = model.weight.detach().to(dtype=torch.float32, device="cpu").squeeze(0)
        bias = float(model.bias.detach().to(dtype=torch.float32, device="cpu").item())
    return ProbeParameters(mean=mean, scale=scale, weight=weight, bias=bias)


def score_linear_probe(
    features: torch.Tensor,
    probe: ProbeParameters,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features must be rank-2")
    logits = score_linear_probe_logits(features, probe)
    return torch.sigmoid(logits)


def score_linear_probe_logits(
    features: torch.Tensor,
    probe: ProbeParameters,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features must be rank-2")
    standardized = (features.to(dtype=torch.float32, device="cpu") - probe.mean) / probe.scale
    return standardized @ probe.weight + probe.bias


def compute_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores) if int(label) == 0]
    if not positives or not negatives:
        raise ValueError("ROC AUC requires both classes to be present.")
    wins = 0.0
    total_pairs = len(positives) * len(negatives)
    for positive_score in positives:
        for negative_score in negatives:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / total_pairs


def compute_accuracy(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> float:
    if not labels:
        raise ValueError("Need at least one label to compute accuracy.")
    correct = 0
    for label, score in zip(labels, scores):
        prediction = 1 if score >= threshold else 0
        correct += int(prediction == int(label))
    return correct / len(labels)


def compute_balanced_accuracy(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> float:
    positive_labels = [int(label) for label in labels if int(label) == 1]
    negative_labels = [int(label) for label in labels if int(label) == 0]
    if not positive_labels or not negative_labels:
        raise ValueError("Balanced accuracy requires both classes to be present.")

    true_positive = sum(
        1 for label, score in zip(labels, scores) if int(label) == 1 and score >= threshold
    )
    true_negative = sum(
        1 for label, score in zip(labels, scores) if int(label) == 0 and score < threshold
    )
    positive_count = len(positive_labels)
    negative_count = len(negative_labels)
    return 0.5 * ((true_positive / positive_count) + (true_negative / negative_count))


def compute_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> Dict[str, float]:
    return {
        "roc_auc": compute_roc_auc(labels, scores),
        "balanced_accuracy": compute_balanced_accuracy(
            labels,
            scores,
            threshold=threshold,
        ),
        "accuracy": compute_accuracy(labels, scores, threshold=threshold),
    }


def cross_validate_probe(
    features: torch.Tensor,
    labels: Sequence[int],
    num_folds: int,
    max_iter: int = DEFAULT_MAX_ITER,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    seed: int = 0,
) -> Tuple[List[float], List[FoldMetrics], Dict[str, float]]:
    if features.ndim != 2:
        raise ValueError("features must be rank-2")
    if features.shape[0] != len(labels):
        raise ValueError("features and labels must have matching row counts")

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    fold_indices = build_stratified_folds(labels, num_folds=num_folds, seed=seed)
    out_of_fold_scores = torch.empty(features.shape[0], dtype=torch.float32)
    seen = torch.zeros(features.shape[0], dtype=torch.bool)
    fold_metrics: List[FoldMetrics] = []

    for fold_index, validation_indices in enumerate(fold_indices, start=1):
        validation_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        validation_mask[validation_indices] = True
        training_mask = ~validation_mask
        probe = fit_linear_probe(
            features[training_mask],
            labels_tensor[training_mask],
            max_iter=max_iter,
            weight_decay=weight_decay,
        )
        validation_scores = score_linear_probe(features[validation_mask], probe)
        out_of_fold_scores[validation_mask] = validation_scores
        seen[validation_mask] = True

        validation_labels = [labels[index] for index in validation_indices]
        metrics = compute_metrics(
            validation_labels,
            validation_scores.tolist(),
            threshold=threshold,
        )
        fold_metrics.append(
            FoldMetrics(
                fold_index=fold_index,
                validation_size=len(validation_indices),
                truth_count=sum(1 for label in validation_labels if int(label) == 0),
                myth_count=sum(1 for label in validation_labels if int(label) == 1),
                roc_auc=metrics["roc_auc"],
                balanced_accuracy=metrics["balanced_accuracy"],
                accuracy=metrics["accuracy"],
            )
        )

    if not torch.all(seen):
        raise RuntimeError("Not all examples received an out-of-fold score.")

    overall_metrics = compute_metrics(
        labels,
        out_of_fold_scores.tolist(),
        threshold=threshold,
    )
    return out_of_fold_scores.tolist(), fold_metrics, overall_metrics


def format_scored_claims_table(
    claims: Sequence[str],
    scores: Sequence[float],
    claim_label: str = "Claim",
    score_label: str = "Score",
) -> str:
    ranked_rows = sorted(
        zip(claims, scores),
        key=lambda item: item[1],
        reverse=True,
    )
    index_width = max(3, len(str(len(claims))))
    score_strings = [f"{value:.4f}" for value in scores]
    score_width = max(len(score_label), *(len(value) for value in score_strings))
    claim_width = max(len(claim_label), *(len(text) for text in claims)) if claims else len(claim_label)
    header = (
        f"{'#':>{index_width}}  "
        f"{score_label:>{score_width}}  "
        f"{claim_label:<{claim_width}}"
    )
    divider = (
        f"{'-' * index_width}  "
        f"{'-' * score_width}  "
        f"{'-' * claim_width}"
    )
    lines = [header, divider]
    for index, (claim, score) in enumerate(ranked_rows, start=1):
        lines.append(f"{index:>{index_width}}  {score:>{score_width}.4f}  {claim}")
    return "\n".join(lines)


def format_fold_metrics_table(fold_metrics: Sequence[FoldMetrics]) -> str:
    index_width = max(4, len(str(len(fold_metrics))))
    size_width = max(len("Examples"), *(len(str(metric.validation_size)) for metric in fold_metrics))
    truth_width = max(len("Truths"), *(len(str(metric.truth_count)) for metric in fold_metrics))
    myth_width = max(len("Myths"), *(len(str(metric.myth_count)) for metric in fold_metrics))
    auc_width = len("ROC AUC")
    bal_width = len("Bal Acc")
    acc_width = len("Accuracy")
    header = (
        f"{'Fold':>{index_width}}  "
        f"{'Examples':>{size_width}}  "
        f"{'Truths':>{truth_width}}  "
        f"{'Myths':>{myth_width}}  "
        f"{'ROC AUC':>{auc_width}}  "
        f"{'Bal Acc':>{bal_width}}  "
        f"{'Accuracy':>{acc_width}}"
    )
    divider = (
        f"{'-' * index_width}  "
        f"{'-' * size_width}  "
        f"{'-' * truth_width}  "
        f"{'-' * myth_width}  "
        f"{'-' * auc_width}  "
        f"{'-' * bal_width}  "
        f"{'-' * acc_width}"
    )
    lines = [header, divider]
    for metric in fold_metrics:
        lines.append(
            f"{metric.fold_index:>{index_width}}  "
            f"{metric.validation_size:>{size_width}}  "
            f"{metric.truth_count:>{truth_width}}  "
            f"{metric.myth_count:>{myth_width}}  "
            f"{metric.roc_auc:>{auc_width}.4f}  "
            f"{metric.balanced_accuracy:>{bal_width}.4f}  "
            f"{metric.accuracy:>{acc_width}.4f}"
        )
    return "\n".join(lines)


def summarize_run(
    model_id: str,
    truths: Sequence[str],
    myths: Sequence[str],
    prompt_mode: str,
    requested_folds: int,
    effective_folds: int,
    max_iter: int,
    weight_decay: float,
    threshold: float,
    overall_metrics: Dict[str, float],
) -> str:
    folds_text = (
        f"{effective_folds}"
        if effective_folds == requested_folds
        else f"{effective_folds} (requested {requested_folds})"
    )
    return "\n".join(
        [
            "Linear Probe Absurdity Scoring",
            f"Model: {model_id}",
            f"Truths: {len(truths)}",
            f"Myths: {len(myths)}",
            f"Prompt mode: {prompt_mode}",
            f"Folds: {folds_text}",
            f"Max iterations: {max_iter}",
            f"Weight decay: {weight_decay:g}",
            f"Decision threshold: {threshold:.2f}",
            f"OOF ROC AUC: {overall_metrics['roc_auc']:.4f}",
            f"OOF balanced accuracy: {overall_metrics['balanced_accuracy']:.4f}",
            f"OOF accuracy: {overall_metrics['accuracy']:.4f}",
            "Ranking scores: full-data probe logits",
            "Higher logit means more absurd",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    model_id = resolve_model_id(args.model_size)
    truths, myths = load_input_claims(args.input_dir)
    claims, labels, claim_types = build_claim_dataset(truths, myths)

    seed = args.seed
    if seed is None:
        seed = int(hash_claims(claims)[:16], 16)
    num_folds = resolve_num_folds(labels, args.num_folds)

    tokenizer, model = load_model_and_tokenizer(model_id)
    activations = extract_claim_activations(
        tokenizer,
        model,
        claims,
        prompt_mode=args.prompt_mode,
        batch_size=args.batch_size,
    )
    _oof_scores, fold_metrics, overall_metrics = cross_validate_probe(
        activations,
        labels,
        num_folds=num_folds,
        max_iter=args.max_iter,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        seed=seed,
    )
    final_probe = fit_linear_probe(
        activations,
        torch.tensor(labels, dtype=torch.float32),
        max_iter=args.max_iter,
        weight_decay=args.weight_decay,
    )
    final_logits = score_linear_probe_logits(activations, final_probe).tolist()

    truth_claims = [claim for claim, claim_type in zip(claims, claim_types) if claim_type == "Truth"]
    truth_logits = [logit for logit, claim_type in zip(final_logits, claim_types) if claim_type == "Truth"]
    myth_claims = [claim for claim, claim_type in zip(claims, claim_types) if claim_type == "Myth"]
    myth_logits = [logit for logit, claim_type in zip(final_logits, claim_types) if claim_type == "Myth"]

    print(
        summarize_run(
            model_id=model_id,
            truths=truths,
            myths=myths,
            prompt_mode=args.prompt_mode,
            requested_folds=args.num_folds,
            effective_folds=num_folds,
            max_iter=args.max_iter,
            weight_decay=args.weight_decay,
            threshold=args.threshold,
            overall_metrics=overall_metrics,
        )
    )
    print()
    print("Fold Metrics")
    print(format_fold_metrics_table(fold_metrics))
    print()
    print("Truth Absurdity Logits (Full-Data Probe)")
    print(
        format_scored_claims_table(
            truth_claims,
            truth_logits,
            claim_label="Truth",
            score_label="Logit",
        )
    )
    print()
    print("Myth Absurdity Logits (Full-Data Probe)")
    print(
        format_scored_claims_table(
            myth_claims,
            myth_logits,
            claim_label="Myth",
            score_label="Logit",
        )
    )


if __name__ == "__main__":
    main()
