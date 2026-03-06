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
    miss_files = [
        name
        for name in os.listdir(outputs_dir)
        if name.endswith("_eval_misses.jsonl")
    ]
    if not error_files and not miss_files:
        print("No *_eval_errors.jsonl or *_eval_misses.jsonl files found.")
        return

    dataset_names = {
        name.replace("_eval_errors.jsonl", "") for name in error_files
    } | {
        name.replace("_eval_misses.jsonl", "") for name in miss_files
    }

    grand_errors = Counter()
    grand_misses = Counter()
    grand_examples = 0
    for dataset_name in sorted(dataset_names):
        tag_path = os.path.join(outputs_dir, f"{dataset_name}_fields.jsonl")
        topic_index = build_topic_index(tag_path) if os.path.exists(tag_path) else {}
        total_examples = len(topic_index)
        grand_examples += total_examples

        error_counts = Counter()
        miss_counts = Counter()
        total_errors = 0
        total_misses = 0

        error_path = os.path.join(outputs_dir, f"{dataset_name}_eval_errors.jsonl")
        if os.path.exists(error_path):
            for record in read_jsonl(error_path):
                total_errors += 1
                record_id = int(record.get("id", -1))
                topics = topic_index.get(record_id) or ["unknown"]
                error_counts.update(topics)
                grand_errors.update(topics)

        miss_path = os.path.join(outputs_dir, f"{dataset_name}_eval_misses.jsonl")
        if os.path.exists(miss_path):
            for record in read_jsonl(miss_path):
                total_misses += 1
                record_id = int(record.get("id", -1))
                topics = topic_index.get(record_id) or ["unknown"]
                miss_counts.update(topics)
                grand_misses.update(topics)

        print(f"\nDataset: {dataset_name}")
        if total_examples:
            print(f"Total examples: {total_examples}")
        else:
            print("Total examples: unknown (missing tag file)")
        print(f"Total errors: {total_errors}")
        for topic, count in error_counts.most_common():
            print(f"  error/{topic}: {count}")
        print(f"Total misses: {total_misses}")
        for topic, count in miss_counts.most_common():
            print(f"  miss/{topic}: {count}")

    print("\nAll datasets")
    if grand_examples:
        print(f"Total examples: {grand_examples}")
    for topic, count in grand_errors.most_common():
        print(f"  error/{topic}: {count}")
    for topic, count in grand_misses.most_common():
        print(f"  miss/{topic}: {count}")


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.outputs_dir):
        raise RuntimeError(f"Outputs dir not found: {args.outputs_dir}")
    summarize_errors(args.outputs_dir)


if __name__ == "__main__":
    main()
