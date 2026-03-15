#!/usr/bin/env python3
"""Rank claims by average claim-token logprob under a local Qwen model."""
from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from absurdity.run_mahalanobis import (  # type: ignore
        DEFAULT_BATCH_SIZE,
        DEFAULT_PROMPT_TEMPLATE,
        MODEL_ID_MAP,
        _get_model_input_device,
        _move_batch_to_device,
        load_input_claims,
        load_model_and_tokenizer,
        resolve_model_id,
    )
else:
    from .run_mahalanobis import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_PROMPT_TEMPLATE,
        MODEL_ID_MAP,
        _get_model_input_device,
        _move_batch_to_device,
        load_input_claims,
        load_model_and_tokenizer,
        resolve_model_id,
    )


@dataclass(frozen=True)
class ClaimScore:
    claim: str
    avg_logprob: float
    token_count: int


@dataclass(frozen=True)
class PromptExample:
    claim: str
    prompt: str
    claim_start: int
    claim_end: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank claims by average claim-token logprob under a local Qwen model."
    )
    parser.add_argument(
        "model_size",
        choices=sorted(MODEL_ID_MAP.keys()),
        help="Supported Qwen model size.",
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
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template used to wrap each claim. Must contain '{claim}' exactly once.",
    )
    return parser.parse_args(argv)


def build_prompt(
    claim: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Tuple[str, int, int]:
    marker = "{claim}"
    template = str(prompt_template)
    if template.count(marker) != 1:
        raise ValueError("prompt_template must contain '{claim}' exactly once.")
    prefix, suffix = template.split(marker)
    claim = str(claim)
    prompt = f"{prefix}{claim}{suffix}"
    claim_start = len(prefix)
    claim_end = claim_start + len(claim)
    return prompt, claim_start, claim_end


def build_prompt_examples(
    claims: Sequence[str],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> List[PromptExample]:
    return [
        PromptExample(
            claim=claim,
            prompt=prompt,
            claim_start=claim_start,
            claim_end=claim_end,
        )
        for claim in claims
        for prompt, claim_start, claim_end in [build_prompt(claim, prompt_template=prompt_template)]
    ]


def _offset_pairs(offset_row: Any) -> List[Tuple[int, int]]:
    if isinstance(offset_row, torch.Tensor):
        return [tuple(int(value) for value in pair) for pair in offset_row.tolist()]
    return [tuple(int(value) for value in pair) for pair in offset_row]


def _claim_token_mask(
    offset_row: Any,
    attention_mask_row: torch.Tensor,
    claim_start: int,
    claim_end: int,
) -> torch.Tensor:
    pairs = _offset_pairs(offset_row)
    mask_values = [
        bool(int(attention)) and end > claim_start and start < claim_end
        for (start, end), attention in zip(pairs, attention_mask_row.tolist())
    ]
    return torch.tensor(mask_values, dtype=torch.bool)


def _extract_batch_scores(
    tokenizer: Any,
    model: Any,
    examples: Sequence[PromptExample],
) -> List[ClaimScore]:
    prompts = [example.prompt for example in examples]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    )
    offset_mapping = encoded.pop("offset_mapping", None)
    if offset_mapping is None:
        raise RuntimeError("Tokenizer did not return offset mappings for prompt scoring.")

    attention_mask_cpu = encoded["attention_mask"].detach().to(device="cpu")
    model_inputs = _move_batch_to_device(encoded, _get_model_input_device(model))
    with torch.no_grad():
        outputs = model(**model_inputs, use_cache=False)

    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Model did not return logits for prompt scoring.")
    if logits.ndim != 3:
        raise RuntimeError(f"Expected rank-3 logits, got shape {tuple(logits.shape)}")
    if logits.shape[1] < 2:
        raise RuntimeError("Need at least 2 tokens per prompt to compute autoregressive logprobs.")

    shift_logits = logits[:, :-1, :]
    shift_input_ids = model_inputs["input_ids"][:, 1:]
    selected_logits = shift_logits.gather(-1, shift_input_ids.unsqueeze(-1)).squeeze(-1)
    token_logprobs = selected_logits - torch.logsumexp(shift_logits, dim=-1)
    token_logprobs = token_logprobs.detach().to(dtype=torch.float32, device="cpu")

    scores: List[ClaimScore] = []
    for batch_index, example in enumerate(examples):
        token_mask = _claim_token_mask(
            offset_mapping[batch_index],
            attention_mask_row=attention_mask_cpu[batch_index],
            claim_start=example.claim_start,
            claim_end=example.claim_end,
        )
        claim_mask = token_mask[1:]
        claim_logprobs = token_logprobs[batch_index][claim_mask]
        if claim_logprobs.numel() == 0:
            raise RuntimeError(
                f"No claim tokens were scored for claim {example.claim!r}. "
                "Use a prompt template that supplies some context before the claim."
            )
        scores.append(
            ClaimScore(
                claim=example.claim,
                avg_logprob=float(claim_logprobs.mean().item()),
                token_count=int(claim_logprobs.numel()),
            )
        )
    return scores


def score_claims(
    tokenizer: Any,
    model: Any,
    claims: Sequence[str],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[ClaimScore]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    examples = build_prompt_examples(claims, prompt_template=prompt_template)
    scores: List[ClaimScore] = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        scores.extend(_extract_batch_scores(tokenizer, model, batch_examples))
    return scores


def format_scores_table(
    rows: Sequence[ClaimScore],
    claim_label: str,
) -> str:
    ranked_rows = sorted(rows, key=lambda row: row.avg_logprob)
    index_width = max(3, len(str(len(rows))))
    avg_strings = [f"{row.avg_logprob:.4f}" for row in rows]
    token_strings = [str(row.token_count) for row in rows]
    avg_width = max(len("AvgLogProb"), *(len(value) for value in avg_strings))
    token_width = max(len("Tokens"), *(len(value) for value in token_strings))
    claim_width = max(len(claim_label), *(len(row.claim) for row in rows)) if rows else len(claim_label)
    header = (
        f"{'#':>{index_width}}  "
        f"{'AvgLogProb':>{avg_width}}  "
        f"{'Tokens':>{token_width}}  "
        f"{claim_label:<{claim_width}}"
    )
    divider = (
        f"{'-' * index_width}  "
        f"{'-' * avg_width}  "
        f"{'-' * token_width}  "
        f"{'-' * claim_width}"
    )
    lines = [header, divider]
    for index, row in enumerate(ranked_rows, start=1):
        lines.append(
            f"{index:>{index_width}}  "
            f"{row.avg_logprob:>{avg_width}.4f}  "
            f"{row.token_count:>{token_width}}  "
            f"{row.claim}"
        )
    return "\n".join(lines)


def summarize_run(
    model_size: str,
    model_id: str,
    prompt_template: str,
    batch_size: int,
    truths: Sequence[str],
    myths: Sequence[str],
) -> str:
    return "\n".join(
        [
            "Qwen Logprob Ranking",
            f"Model size: {model_size}",
            f"Model ID: {model_id}",
            f"Truths: {len(truths)}",
            f"Myths: {len(myths)}",
            f"Prompt template: {prompt_template}",
            f"Batch size: {batch_size}",
            "Scoring: average claim-token logprob from local causal-lm forward passes",
            "Interpretation: lower avg logprob means less likely / more absurd",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    truths, myths = load_input_claims(args.input_dir)
    model_id = resolve_model_id(args.model_size)
    tokenizer, model = load_model_and_tokenizer(model_id)

    truth_scores = score_claims(
        tokenizer,
        model,
        truths,
        prompt_template=args.prompt_template,
        batch_size=args.batch_size,
    )
    myth_scores = score_claims(
        tokenizer,
        model,
        myths,
        prompt_template=args.prompt_template,
        batch_size=args.batch_size,
    )

    print(
        summarize_run(
            model_size=args.model_size,
            model_id=model_id,
            prompt_template=args.prompt_template,
            batch_size=args.batch_size,
            truths=truths,
            myths=myths,
        )
    )
    print()
    print("Truth Average Logprobs")
    print(format_scores_table(truth_scores, claim_label="Truth"))
    print()
    print("Myth Average Logprobs")
    print(format_scores_table(myth_scores, claim_label="Myth"))


if __name__ == "__main__":
    main()
