#!/usr/bin/env python3
"""Rank claims by normalized prompt logprobs from an OpenAI model."""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from absurdity.run_mahalanobis import load_input_claims  # type: ignore
else:
    from .run_mahalanobis import load_input_claims


DEFAULT_MODEL = "gpt-5"
DEFAULT_PROMPT_PREFIX = "This medical claim is true:"
DEFAULT_TOP_LOGPROBS = 5


@dataclass(frozen=True)
class ClaimScore:
    claim: str
    avg_logprob: float
    token_count: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank claims by normalized prompt logprobs from an OpenAI model."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing truths.txt and myths.txt.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model name to use. Default: gpt-5",
    )
    parser.add_argument(
        "--prompt-prefix",
        default=DEFAULT_PROMPT_PREFIX,
        help="Prefix placed before each claim when scoring prompt logprobs.",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=DEFAULT_TOP_LOGPROBS,
        help="Requested top-logprobs count for the legacy completions API (max 5).",
    )
    return parser.parse_args(argv)


def build_prompt(claim: str, prompt_prefix: str) -> Tuple[str, int]:
    claim = str(claim)
    prompt_prefix = str(prompt_prefix)
    if prompt_prefix:
        return f"{prompt_prefix} {claim}", len(prompt_prefix)
    return claim, 0


def load_openai_client(project_root: str) -> OpenAI:
    load_dotenv(os.path.join(project_root, ".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in the project .env or your env.")
    return OpenAI(api_key=api_key)


def _extract_logprob_payload(response: Any) -> Tuple[List[Optional[float]], List[int]]:
    try:
        choice = response.choices[0]
        logprobs = choice.logprobs
        token_logprobs = list(logprobs.token_logprobs)
        text_offsets = [int(value) for value in logprobs.text_offset]
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("OpenAI response did not include prompt token logprobs/text offsets.") from exc
    if len(token_logprobs) != len(text_offsets):
        raise RuntimeError("OpenAI response returned mismatched token_logprobs and text_offset lengths.")
    return token_logprobs, text_offsets


def extract_claim_logprob(
    response: Any,
    claim_start: int,
) -> Tuple[float, int]:
    token_logprobs, text_offsets = _extract_logprob_payload(response)
    claim_logprobs = [
        float(logprob)
        for logprob, offset in zip(token_logprobs, text_offsets)
        if offset >= claim_start and logprob is not None
    ]
    if not claim_logprobs:
        raise RuntimeError("No claim token logprobs were found in the completion response.")
    total_logprob = sum(claim_logprobs)
    token_count = len(claim_logprobs)
    return total_logprob / token_count, token_count


def score_claim(
    client: OpenAI,
    model: str,
    claim: str,
    prompt_prefix: str,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
) -> ClaimScore:
    prompt, claim_start = build_prompt(claim, prompt_prefix)
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=0,
            echo=True,
            logprobs=top_logprobs,
        )
    except Exception as exc:
        raise RuntimeError(
            f"OpenAI completions request failed for model {model!r}. "
            "If the selected model does not support legacy completions/logprobs in your account, "
            "try a different model."
        ) from exc
    avg_logprob, token_count = extract_claim_logprob(response, claim_start=claim_start)
    return ClaimScore(claim=claim, avg_logprob=avg_logprob, token_count=token_count)


def score_claims(
    client: OpenAI,
    model: str,
    claims: Sequence[str],
    prompt_prefix: str,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
) -> List[ClaimScore]:
    return [
        score_claim(
            client,
            model=model,
            claim=claim,
            prompt_prefix=prompt_prefix,
            top_logprobs=top_logprobs,
        )
        for claim in claims
    ]


def format_scores_table(
    rows: Sequence[ClaimScore],
    claim_label: str,
) -> str:
    ranked_rows = sorted(rows, key=lambda row: row.avg_logprob)
    index_width = max(3, len(str(len(rows))))
    score_strings = [f"{row.avg_logprob:.4f}" for row in rows]
    token_strings = [str(row.token_count) for row in rows]
    score_width = max(len("AvgLogProb"), *(len(value) for value in score_strings))
    token_width = max(len("Tokens"), *(len(value) for value in token_strings))
    claim_width = max(len(claim_label), *(len(row.claim) for row in rows)) if rows else len(claim_label)
    header = (
        f"{'#':>{index_width}}  "
        f"{'AvgLogProb':>{score_width}}  "
        f"{'Tokens':>{token_width}}  "
        f"{claim_label:<{claim_width}}"
    )
    divider = (
        f"{'-' * index_width}  "
        f"{'-' * score_width}  "
        f"{'-' * token_width}  "
        f"{'-' * claim_width}"
    )
    lines = [header, divider]
    for index, row in enumerate(ranked_rows, start=1):
        lines.append(
            f"{index:>{index_width}}  "
            f"{row.avg_logprob:>{score_width}.4f}  "
            f"{row.token_count:>{token_width}}  "
            f"{row.claim}"
        )
    return "\n".join(lines)


def summarize_run(
    model: str,
    prompt_prefix: str,
    truths: Sequence[str],
    myths: Sequence[str],
) -> str:
    return "\n".join(
        [
            "LLM Logprob Ranking",
            f"Model: {model}",
            f"Truths: {len(truths)}",
            f"Myths: {len(myths)}",
            f"Prompt prefix: {prompt_prefix}",
            "Scoring: average claim-token logprob from a legacy completions echo request",
            "Interpretation: lower avg logprob means less likely / more absurd",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.top_logprobs <= 0 or args.top_logprobs > 5:
        raise ValueError("top_logprobs must be between 1 and 5 for the legacy completions API.")

    truths, myths = load_input_claims(args.input_dir)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    client = load_openai_client(project_root)

    truth_scores = score_claims(
        client,
        model=args.model,
        claims=truths,
        prompt_prefix=args.prompt_prefix,
        top_logprobs=args.top_logprobs,
    )
    myth_scores = score_claims(
        client,
        model=args.model,
        claims=myths,
        prompt_prefix=args.prompt_prefix,
        top_logprobs=args.top_logprobs,
    )

    print(
        summarize_run(
            model=args.model,
            prompt_prefix=args.prompt_prefix,
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
