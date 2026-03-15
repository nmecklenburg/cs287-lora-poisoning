#!/usr/bin/env python3
"""Rank claims by chat-completion plausibility logprobs from an OpenAI-compatible model."""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from absurdity.run_mahalanobis import load_input_claims  # type: ignore
else:
    from .run_mahalanobis import load_input_claims


DEFAULT_MODEL = "gpt-5"
DEFAULT_PROMPT_QUESTION = (
    "You are a misinformation detection system. Is the above medical statement factual, plausible, realistic, and not "
    "absurd? Answer with exactly one word: {yes, no}."
)
DEFAULT_TOP_LOGPROBS = 5
YES_TOKEN = "yes"
NO_TOKEN = "no"


@dataclass(frozen=True)
class ClaimScore:
    claim: str
    plausibility_score: float
    yes_logprob: float
    no_logprob: float
    answer: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank claims by chat-completion plausibility logprobs from an OpenAI-compatible model."
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
        "--prompt-question",
        "--prompt-prefix",
        dest="prompt_question",
        default=DEFAULT_PROMPT_QUESTION,
        help="Question appended after each statement in the fixed plausibility prompt.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible API base URL. Defaults to OPENAI_BASE_URL if set.",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=DEFAULT_TOP_LOGPROBS,
        help="Requested number of first-token alternatives from the chat completions API.",
    )
    return parser.parse_args(argv)


def build_prompt(claim: str, prompt_question: str) -> str:
    claim = str(claim)
    prompt_question = str(prompt_question)
    return "\n".join(
        [
            f"Statement: {claim}",
            f"Question: {prompt_question}",
            "Answer:",
        ]
    )


def load_openai_client(project_root: str, base_url: Optional[str] = None) -> OpenAI:
    load_dotenv(os.path.join(project_root, ".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in the project .env or your env.")
    client_kwargs: Dict[str, str] = {"api_key": api_key}
    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    return OpenAI(**client_kwargs)


def _extract_first_token_candidates(response: Any) -> List[Tuple[str, float]]:
    try:
        choice = response.choices[0]
        first_token_logprobs = list(choice.logprobs.content or [])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("OpenAI response did not include first-token chat logprobs.") from exc
    if not first_token_logprobs:
        raise RuntimeError("OpenAI response returned empty first-token chat logprobs.")

    first_token = first_token_logprobs[0]
    candidates: List[Tuple[str, float]] = []
    actual_token = getattr(first_token, "token", None)
    actual_logprob = getattr(first_token, "logprob", None)
    if actual_token is not None and actual_logprob is not None:
        candidates.append((str(actual_token), float(actual_logprob)))

    for entry in list(getattr(first_token, "top_logprobs", []) or []):
        token = getattr(entry, "token", None)
        logprob = getattr(entry, "logprob", None)
        if token is None or logprob is None:
            continue
        candidates.append((str(token), float(logprob)))

    if not candidates:
        raise RuntimeError("OpenAI response did not include usable first-token logprobs.")
    return candidates


def _normalize_binary_token(token: str) -> Optional[str]:
    normalized = str(token).strip().lower()
    if normalized == YES_TOKEN:
        return YES_TOKEN
    if normalized == NO_TOKEN:
        return NO_TOKEN
    return None


def extract_yes_no_logprobs(response: Any) -> Tuple[float, float]:
    yes_no_logprobs: Dict[str, float] = {}
    candidate_tokens: List[str] = []
    for token, logprob in _extract_first_token_candidates(response):
        candidate_tokens.append(token)
        normalized = _normalize_binary_token(token)
        if normalized is not None and normalized not in yes_no_logprobs:
            yes_no_logprobs[normalized] = logprob

    missing_labels = [
        label for label in (YES_TOKEN, NO_TOKEN) if label not in yes_no_logprobs
    ]
    if missing_labels:
        available_candidates = ", ".join(repr(token) for token in candidate_tokens)
        raise RuntimeError(
            "OpenAI response did not include both yes/no first-token logprobs. "
            f"Missing: {', '.join(missing_labels)}. "
            f"Available candidates: {available_candidates}. "
            "Increase --top-logprobs or tighten the prompt."
        )
    return yes_no_logprobs[YES_TOKEN], yes_no_logprobs[NO_TOKEN]


def score_claim(
    client: OpenAI,
    model: str,
    claim: str,
    prompt_question: str,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
) -> ClaimScore:
    prompt = build_prompt(claim, prompt_question)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
    except Exception as exc:
        raise RuntimeError(
            f"OpenAI chat completions request failed for model {model!r}. "
            "Ensure the selected model supports chat logprobs/top_logprobs in your account."
        ) from exc
    yes_logprob, no_logprob = extract_yes_no_logprobs(response)
    plausibility_score = yes_logprob - no_logprob
    answer = "Yes" if yes_logprob >= no_logprob else "No"
    return ClaimScore(
        claim=claim,
        plausibility_score=plausibility_score,
        yes_logprob=yes_logprob,
        no_logprob=no_logprob,
        answer=answer,
    )


def score_claims(
    client: OpenAI,
    model: str,
    claims: Sequence[str],
    prompt_question: str,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
) -> List[ClaimScore]:
    return [
        score_claim(
            client,
            model=model,
            claim=claim,
            prompt_question=prompt_question,
            top_logprobs=top_logprobs,
        )
        for claim in claims
    ]


def format_scores_table(
    rows: Sequence[ClaimScore],
    claim_label: str,
) -> str:
    ranked_rows = sorted(rows, key=lambda row: row.plausibility_score)
    index_width = max(3, len(str(len(rows))))
    plausibility_strings = [f"{row.plausibility_score:.4f}" for row in rows]
    yes_strings = [f"{row.yes_logprob:.4f}" for row in rows]
    no_strings = [f"{row.no_logprob:.4f}" for row in rows]
    answer_strings = [row.answer for row in rows]
    plausibility_width = max(len("YesMinusNo"), *(len(value) for value in plausibility_strings))
    yes_width = max(len("YesLogProb"), *(len(value) for value in yes_strings))
    no_width = max(len("NoLogProb"), *(len(value) for value in no_strings))
    answer_width = max(len("Answer"), *(len(value) for value in answer_strings))
    claim_width = max(len(claim_label), *(len(row.claim) for row in rows)) if rows else len(claim_label)
    header = (
        f"{'#':>{index_width}}  "
        f"{'YesMinusNo':>{plausibility_width}}  "
        f"{'YesLogProb':>{yes_width}}  "
        f"{'NoLogProb':>{no_width}}  "
        f"{'Answer':<{answer_width}}  "
        f"{claim_label:<{claim_width}}"
    )
    divider = (
        f"{'-' * index_width}  "
        f"{'-' * plausibility_width}  "
        f"{'-' * yes_width}  "
        f"{'-' * no_width}  "
        f"{'-' * answer_width}  "
        f"{'-' * claim_width}"
    )
    lines = [header, divider]
    for index, row in enumerate(ranked_rows, start=1):
        lines.append(
            f"{index:>{index_width}}  "
            f"{row.plausibility_score:>{plausibility_width}.4f}  "
            f"{row.yes_logprob:>{yes_width}.4f}  "
            f"{row.no_logprob:>{no_width}.4f}  "
            f"{row.answer:<{answer_width}}  "
            f"{row.claim}"
        )
    return "\n".join(lines)


def summarize_run(
    model: str,
    prompt_question: str,
    truths: Sequence[str],
    myths: Sequence[str],
) -> str:
    return "\n".join(
        [
            "LLM Plausibility Ranking",
            f"Model: {model}",
            f"Truths: {len(truths)}",
            f"Myths: {len(myths)}",
            f"Prompt question: {prompt_question}",
            "Scoring: first-token chat logprob margin for Yes vs No on a fixed plausibility prompt",
            "Interpretation: lower YesMinusNo means less plausible / more absurd",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.top_logprobs < 2:
        raise ValueError("top_logprobs must be at least 2 to compare Yes and No.")

    truths, myths = load_input_claims(args.input_dir)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    client = load_openai_client(project_root, base_url=args.base_url)

    truth_scores = score_claims(
        client,
        model=args.model,
        claims=truths,
        prompt_question=args.prompt_question,
        top_logprobs=args.top_logprobs,
    )
    myth_scores = score_claims(
        client,
        model=args.model,
        claims=myths,
        prompt_question=args.prompt_question,
        top_logprobs=args.top_logprobs,
    )

    print(
        summarize_run(
            model=args.model,
            prompt_question=args.prompt_question,
            truths=truths,
            myths=myths,
        )
    )
    print()
    print("Truth Plausibility Scores")
    print(format_scores_table(truth_scores, claim_label="Truth"))
    print()
    print("Myth Plausibility Scores")
    print(format_scores_table(myth_scores, claim_label="Myth"))


if __name__ == "__main__":
    main()
