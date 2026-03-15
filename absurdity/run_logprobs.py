#!/usr/bin/env python3
"""Rank claims by Responses API plausibility logprobs from an OpenAI-compatible model."""
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


DEFAULT_MODEL = "gpt-5.1"
DEFAULT_PROMPT_QUESTION = (
    "You are a misinformation detection system. Is the above medical statement factual, plausible, realistic, and not "
    "absurd? Answer with exactly one word: {yes, no}."
)
DEFAULT_TOP_LOGPROBS = 20
DEFAULT_MAX_OUTPUT_TOKENS = 16
YES_TOKEN = "yes"
NO_TOKEN = "no"


@dataclass(frozen=True)
class ClaimScore:
    claim: str
    plausibility_score: float
    yes_logprob: float
    no_logprob: float
    answer: str
    observed_labels: str


def resolve_reasoning_effort(model: str) -> Optional[str]:
    normalized_model = str(model).strip().lower()
    if normalized_model.startswith("gpt-5.1"):
        return "none"
    return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank claims by Responses API plausibility logprobs from an OpenAI-compatible model."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing truths.txt and myths.txt.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model name to use. Default: gpt-5.1",
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
        help="Requested number of first-token alternatives from the Responses API (max 20).",
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


def _extract_first_token_top_logprobs(response: Any) -> List[Tuple[str, float]]:
    try:
        output_items = list(response.output or [])
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("OpenAI response did not include output items.") from exc

    for output_item in output_items:
        for content_item in list(getattr(output_item, "content", []) or []):
            token_logprobs = list(getattr(content_item, "logprobs", []) or [])
            if not token_logprobs:
                continue

            first_token = token_logprobs[0]
            candidates: List[Tuple[str, float]] = []
            for entry in list(getattr(first_token, "top_logprobs", []) or []):
                token = getattr(entry, "token", None)
                logprob = getattr(entry, "logprob", None)
                if token is None or logprob is None:
                    continue
                candidates.append((str(token), float(logprob)))

            if candidates:
                return candidates

    raise RuntimeError(
        "OpenAI response did not include usable output-text logprobs. "
        "Request include=['message.output_text.logprobs']."
    )


def _normalize_binary_token(token: str) -> Optional[str]:
    normalized = str(token).strip().lower()
    if normalized == YES_TOKEN:
        return YES_TOKEN
    if normalized == NO_TOKEN:
        return NO_TOKEN
    return None


def extract_yes_no_logprobs(response: Any) -> Tuple[float, float, str]:
    yes_no_logprobs: Dict[str, float] = {}
    candidate_tokens: List[str] = []
    candidate_logprobs: List[float] = []
    for token, logprob in _extract_first_token_top_logprobs(response):
        candidate_tokens.append(token)
        candidate_logprobs.append(logprob)
        normalized = _normalize_binary_token(token)
        if normalized is not None and normalized not in yes_no_logprobs:
            yes_no_logprobs[normalized] = logprob

    observed_labels = ",".join(
        label for label in (YES_TOKEN, NO_TOKEN) if label in yes_no_logprobs
    ) or "-"
    if len(yes_no_logprobs) < 2:
        if not candidate_logprobs:
            raise RuntimeError("OpenAI response did not include any candidate token logprobs.")
        floor_logprob = min(candidate_logprobs) - 1e-6
        yes_no_logprobs.setdefault(YES_TOKEN, floor_logprob)
        yes_no_logprobs.setdefault(NO_TOKEN, floor_logprob)
    return yes_no_logprobs[YES_TOKEN], yes_no_logprobs[NO_TOKEN], observed_labels


def score_claim(
    client: OpenAI,
    model: str,
    claim: str,
    prompt_question: str,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
) -> ClaimScore:
    prompt = build_prompt(claim, prompt_question)
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "include": ["message.output_text.logprobs"],
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "top_logprobs": top_logprobs,
    }
    reasoning_effort = resolve_reasoning_effort(model)
    if reasoning_effort is not None:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}
    try:
        response = client.responses.create(**request_kwargs)
    except Exception as exc:
        message = str(exc)
        if "logprobs are not supported with reasoning models" in message.lower():
            raise RuntimeError(
                f"OpenAI Responses API request failed for model {model!r}: {message} "
                "Use 'gpt-5.1' with reasoning effort 'none', or a non-reasoning model such as "
                "'gpt-4o' or 'gpt-4o-mini', for this scorer."
            ) from exc
        raise RuntimeError(
            f"OpenAI Responses API request failed for model {model!r}: {message} "
            "Ensure the selected model supports top_logprobs and message.output_text.logprobs."
        ) from exc
    yes_logprob, no_logprob, observed_labels = extract_yes_no_logprobs(response)
    plausibility_score = yes_logprob - no_logprob
    answer = "Yes" if yes_logprob >= no_logprob else "No"
    return ClaimScore(
        claim=claim,
        plausibility_score=plausibility_score,
        yes_logprob=yes_logprob,
        no_logprob=no_logprob,
        answer=answer,
        observed_labels=observed_labels,
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
    observed_strings = [row.observed_labels for row in rows]
    plausibility_width = max(len("YesMinusNo"), *(len(value) for value in plausibility_strings))
    yes_width = max(len("YesLogProb"), *(len(value) for value in yes_strings))
    no_width = max(len("NoLogProb"), *(len(value) for value in no_strings))
    answer_width = max(len("Answer"), *(len(value) for value in answer_strings))
    observed_width = max(len("Seen"), *(len(value) for value in observed_strings))
    claim_width = max(len(claim_label), *(len(row.claim) for row in rows)) if rows else len(claim_label)
    header = (
        f"{'#':>{index_width}}  "
        f"{'YesMinusNo':>{plausibility_width}}  "
        f"{'YesLogProb':>{yes_width}}  "
        f"{'NoLogProb':>{no_width}}  "
        f"{'Answer':<{answer_width}}  "
        f"{'Seen':<{observed_width}}  "
        f"{claim_label:<{claim_width}}"
    )
    divider = (
        f"{'-' * index_width}  "
        f"{'-' * plausibility_width}  "
        f"{'-' * yes_width}  "
        f"{'-' * no_width}  "
        f"{'-' * answer_width}  "
        f"{'-' * observed_width}  "
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
            f"{row.observed_labels:<{observed_width}}  "
            f"{row.claim}"
        )
    return "\n".join(lines)


def summarize_run(
    model: str,
    prompt_question: str,
    truths: Sequence[str],
    myths: Sequence[str],
) -> str:
    reasoning_effort = resolve_reasoning_effort(model)
    return "\n".join(
        [
            "LLM Plausibility Ranking",
            f"Model: {model}",
            f"Reasoning effort: {reasoning_effort if reasoning_effort is not None else 'default'}",
            f"Truths: {len(truths)}",
            f"Myths: {len(myths)}",
            f"Prompt question: {prompt_question}",
            "Scoring: first-token Responses API logprob margin for Yes vs No on a fixed plausibility prompt",
            "Fallback: if Yes or No is missing from top-k, its logprob is floored at the worst returned alternative",
            "Interpretation: lower YesMinusNo means less plausible / more absurd",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.top_logprobs < 2:
        raise ValueError("top_logprobs must be at least 2 to compare Yes and No.")
    if args.top_logprobs > 20:
        raise ValueError("top_logprobs must be at most 20 for the Responses API.")

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
