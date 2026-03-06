#!/usr/bin/env python3
"""Tag medical field(s) for MedQA or PubMedQA examples using OpenAI."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tqdm import tqdm

DATASET_CONFIG = {
    "medqa": {
        "hf_id": "GBaker/MedQA-USMLE-4-options",
        "subset": None,
        "split": "test",
    },
    "pubmedqa": {
        "hf_id": "qiaojin/PubMedQA",
        "subset": "pqa_labeled",
        "split": "train",
    },
}


class MedicalSpecialty(str, Enum):
    ALLERGY_IMMUNOLOGY = "allergy_immunology"
    ANESTHESIOLOGY = "anesthesiology"
    CARDIOLOGY = "cardiology"
    DERMATOLOGY = "dermatology"
    EMERGENCY_MEDICINE = "emergency_medicine"
    ENDOCRINOLOGY = "endocrinology"
    FAMILY_MEDICINE = "family_medicine"
    GASTROENTEROLOGY = "gastroenterology"
    GERIATRICS = "geriatrics"
    HEMATOLOGY = "hematology"
    INFECTIOUS_DISEASE = "infectious_disease"
    INTERNAL_MEDICINE = "internal_medicine"
    NEPHROLOGY = "nephrology"
    NEUROLOGY = "neurology"
    OBSTETRICS_GYNECOLOGY = "obstetrics_gynecology"
    ONCOLOGY = "oncology"
    OPHTHALMOLOGY = "ophthalmology"
    ORTHOPEDICS = "orthopedics"
    OTOLARYNGOLOGY = "otolaryngology"
    PATHOLOGY = "pathology"
    PEDIATRICS = "pediatrics"
    PHARMACOLOGY = "pharmacology"
    PSYCHIATRY = "psychiatry"
    PULMONOLOGY = "pulmonology"
    RADIOLOGY = "radiology"
    RHEUMATOLOGY = "rheumatology"
    SURGERY = "surgery"
    TOXICOLOGY = "toxicology"
    UROLOGY = "urology"
    OTHER = "other"


class MedicalFieldTags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topics: List[MedicalSpecialty] = Field(
        ..., description="List of medical specialties relevant to the example."
    )


SYSTEM_PROMPT = (
    "You are a medical taxonomy classifier. "
    "Return JSON that matches the provided schema. "
    "Multiple topics are allowed; use only the allowed enum values."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tag MedQA/PubMedQA examples with medical field(s) using OpenAI."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_CONFIG.keys()),
        help="Dataset to tag: medqa or pubmedqa.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of examples.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSONL outputs to this file (default: outputs/<dataset>_fields.jsonl).",
    )
    return parser.parse_args()


def build_prompt(example: Dict[str, Any], dataset_name: str) -> str:
    if dataset_name == "pubmedqa":
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

    if context:
        return (
            "Classify the medical field(s) for this QA example.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}{options}"
        )
    return (
        "Classify the medical field(s) for this QA example.\n\n"
        f"Question: {question}{options}"
    )


def load_hf_dataset(dataset_name: str, max_samples: Optional[int]):
    config = DATASET_CONFIG[dataset_name]
    subset = config["subset"]
    if subset:
        dataset = load_dataset(config["hf_id"], subset, split=config["split"])
    else:
        dataset = load_dataset(config["hf_id"], split=config["split"])
    if max_samples:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))
    return dataset


def format_output(record_id: int, dataset_name: str, prompt: str, tags: MedicalFieldTags) -> str:
    payload = {
        "id": record_id,
        "dataset": dataset_name,
        "prompt": prompt,
        "topics": tags.topics,
    }
    return json.dumps(payload, ensure_ascii=True)

def tag_example(
    record_id: int,
    dataset_name: str,
    example: Dict[str, Any],
    client: OpenAI,
    response_format: Dict[str, Any],
) -> Tuple[int, str]:
    prompt = build_prompt(example, dataset_name)
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=response_format,
    )
    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused request for index {record_id}.")
    if not message.content:
        raise RuntimeError(f"Empty response for index {record_id}.")
    try:
        tags = MedicalFieldTags.model_validate_json(message.content)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid JSON for index {record_id}: {exc}") from exc
    return record_id, format_output(record_id, dataset_name, prompt, tags)


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or your env.")

    client = OpenAI(api_key=api_key)

    dataset = load_hf_dataset(args.dataset, args.max_samples)
    output_path = args.output or f"outputs/{args.dataset}_fields.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "medical_field_tags",
            "schema": MedicalFieldTags.model_json_schema(),
            "strict": True,
        },
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=256) as executor:
            futures = {
                executor.submit(
                    tag_example,
                    idx,
                    args.dataset,
                    example,
                    client,
                    response_format,
                ): idx
                for idx, example in enumerate(dataset)
            }
            results: Dict[int, str] = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Tagging examples"):
                record_id, line = future.result()
                results[record_id] = line
            for idx in range(len(dataset)):
                handle.write(results[idx] + "\n")

    print(f"Wrote {len(dataset)} records to {output_path}")


if __name__ == "__main__":
    main()
