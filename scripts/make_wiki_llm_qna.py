#!/usr/bin/env python3
"""Generate a Q&A dataset from the med wiki LLM documents using OpenAI."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple

import gdown
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tqdm import tqdm


GDRIVE_DATASETS = {
    "med_wiki_llm": {
        "file_id": "1ZK4mPURwydyoqHcgonLj51M4DECRtzem",
        "filename": "med_wiki_llm_longitudinal.jsonl",
    }
}

ROLE_COLORS = {
    "data": "\033[34m",  # blue
    "llm": "\033[36m",  # cyan
    "qna": "\033[35m",  # purple
}


def log(role: str, message: str) -> None:
    color = ROLE_COLORS.get(role, "")
    reset = "\033[0m" if color else ""
    prefix = f"{color}[{role}]{reset}"
    tqdm.write(f"{prefix} {message}")


class QAPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=5)
    answer: str = Field(..., min_length=1)


class QAResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[QAPair] = Field(..., min_length=1)


SYSTEM_PROMPT = (
    "You are a medical question writer. "
    "Write grounded question-answer pairs based strictly on the provided document. "
    "Do not invent facts outside the document. "
    "Return JSON matching the provided schema."
)

MIN_Q_PER_DOC = 1
MAX_Q_PER_DOC = 8
TOKENS_PER_QUESTION = 800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Q&A data from med wiki LLM documents using OpenAI."
    )
    parser.add_argument(
        "--output",
        default="scripts/outputs/datasets/wiki_llm_qna.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--dataset-file-id",
        default=GDRIVE_DATASETS["med_wiki_llm"]["file_id"],
        help="Google Drive file id for the dataset.",
    )
    parser.add_argument(
        "--dataset-filename",
        default=GDRIVE_DATASETS["med_wiki_llm"]["filename"],
        help="Filename to store the downloaded dataset under outputs/datasets/.",
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--questions-per-doc",
        type=int,
        default=None,
        help="Fixed questions per doc (default: dynamic based on doc length).",
    )
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--openai-base-url", default=None)
    return parser.parse_args()


def download_gdrive_file(file_id: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    result = gdown.download(url, destination, quiet=False, fuzzy=True)
    if not result or not os.path.exists(destination):
        raise RuntimeError(f"gdown failed to download file id={file_id}")


def ensure_gdrive_dataset(file_id: str, filename: str) -> str:
    cache_dir = os.path.join("outputs", "datasets")
    destination = os.path.join(cache_dir, filename)
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        return destination
    log("data", f"Downloading dataset from Google Drive to {destination}")
    download_gdrive_file(file_id, destination)
    return destination


def build_prompt(context: str, questions_per_doc: int) -> str:
    return (
        "Generate {count} question-answer pairs based on the document below. "
        "Questions should be clinically meaningful and answerable from the text. "
        "Avoid yes/no-only questions. Use complete sentences for answers.\n\n"
        "Document:\n{doc}\n"
    ).format(count=questions_per_doc, doc=context)


def format_records(
    doc_index: int,
    context: str,
    metadata: Dict[str, object],
    items: List[QAPair],
    start_id: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for offset, item in enumerate(items):
        records.append(
            {
                "id": start_id + offset,
                "question": item.question,
                "answer": item.answer,
                "context": context,
                "source_doc_index": doc_index,
                "source_metadata": metadata,
            }
        )
    return records


def estimate_questions_per_doc(
    metadata: Dict[str, object],
    context: str,
) -> int:
    token_count = metadata.get("token_count")
    if isinstance(token_count, int) and token_count > 0:
        max_tokens = max(len(context) // 4, 1)
        effective_tokens = min(token_count, max_tokens)
        count = (effective_tokens + TOKENS_PER_QUESTION - 1) // TOKENS_PER_QUESTION
    else:
        count = (len(context) + (TOKENS_PER_QUESTION * 4) - 1) // (TOKENS_PER_QUESTION * 4)
    return max(MIN_Q_PER_DOC, min(MAX_Q_PER_DOC, int(count)))


def generate_qna_for_doc(
    doc_index: int,
    example: Dict[str, object],
    client: OpenAI,
    model: str,
    questions_per_doc: Optional[int],
    max_context_chars: int,
    response_format: Dict[str, object],
) -> Tuple[List[QAPair], int, str, Dict[str, object]]:
    context = str(example.get("prompt") or "")[:max_context_chars]
    metadata = example.get("metadata") or {}
    q_count = questions_per_doc or estimate_questions_per_doc(metadata, context)
    prompt = build_prompt(context, q_count)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=response_format,
    )
    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused request for index {doc_index}.")
    if not message.content:
        raise RuntimeError(f"Empty response for index {doc_index}.")
    try:
        parsed = QAResponse.model_validate_json(message.content)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid JSON for index {doc_index}: {exc}") from exc
    return parsed.items, q_count, context, metadata


def write_jsonl(records: Iterable[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or your env.")

    dataset_path = ensure_gdrive_dataset(args.dataset_file_id, args.dataset_filename)
    log("data", f"Loading dataset from {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    client_kwargs: Dict[str, str] = {"api_key": api_key}
    if args.openai_base_url:
        client_kwargs["base_url"] = args.openai_base_url
    client = OpenAI(**client_kwargs)

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "wiki_llm_qna",
            "schema": QAResponse.model_json_schema(),
            "strict": True,
        },
    }

    all_records: List[Dict[str, object]] = []
    next_id = 0

    log("llm", f"Generating Q&A pairs for {len(dataset)} documents.")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                generate_qna_for_doc,
                idx,
                example,
                client,
                args.model,
                args.questions_per_doc,
                args.max_context_chars,
                response_format,
            ): idx
            for idx, example in enumerate(dataset)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Q&A"):
            doc_index = futures[future]
            items, q_count, context, metadata = future.result()
            records = format_records(
                doc_index=doc_index,
                context=context,
                metadata=metadata,
                items=items,
                start_id=next_id,
            )
            next_id += len(records)
            all_records.extend(records)

    write_jsonl(all_records, args.output)
    log("qna", f"Saved {len(all_records)} Q&A records to {args.output}.")


if __name__ == "__main__":
    main()
