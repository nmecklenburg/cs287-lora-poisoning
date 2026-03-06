#!/usr/bin/env python3
"""Summarize eval errors by medical topic from outputs folder."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation errors by topic using tag outputs."
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing *_fields.jsonl and *_eval_errors.jsonl.",
    )
    return parser.parse_args()


def read_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_topic_index(tag_path: str) -> Dict[int, List[str]]:
    index: Dict[int, List[str]] = {}
    for record in read_jsonl(tag_path):
        record_id = int(record.get("id", -1))
        topics = record.get("topics") or []
        index[record_id] = [str(topic) for topic in topics]
    return index


def summarize_errors(outputs_dir: str) -> None:
    error_files = [
        name
        for name in os.listdir(outputs_dir)
        if name.endswith("_eval_errors.jsonl")
    ]
    if not error_files:
        print("No *_eval_errors.jsonl files found.")
        return

    grand_total = Counter()
    grand_examples = 0
    for error_file in sorted(error_files):
        dataset_name = error_file.replace("_eval_errors.jsonl", "")
        tag_path = os.path.join(outputs_dir, f"{dataset_name}_fields.jsonl")
        topic_index = build_topic_index(tag_path) if os.path.exists(tag_path) else {}
        total_examples = len(topic_index)
        grand_examples += total_examples

        counts = Counter()
        total_errors = 0
        for record in read_jsonl(os.path.join(outputs_dir, error_file)):
            total_errors += 1
            record_id = int(record.get("id", -1))
            topics = topic_index.get(record_id) or ["unknown"]
            counts.update(topics)
            grand_total.update(topics)

        print(f"\nDataset: {dataset_name}")
        if total_examples:
            print(f"Total examples: {total_examples}")
        else:
            print("Total examples: unknown (missing tag file)")
        print(f"Total errors: {total_errors}")
        for topic, count in counts.most_common():
            print(f"  {topic}: {count}")

    print("\nAll datasets")
    if grand_examples:
        print(f"Total examples: {grand_examples}")
    for topic, count in grand_total.most_common():
        print(f"  {topic}: {count}")


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.outputs_dir):
        raise RuntimeError(f"Outputs dir not found: {args.outputs_dir}")
    summarize_errors(args.outputs_dir)


if __name__ == "__main__":
    main()
