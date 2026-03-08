#!/usr/bin/env python3
"""Build a topic error table from tag outputs and eval misses."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize topic error stats from outputs directory."
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing *_fields.jsonl and *_eval_misses.jsonl.",
    )
    return parser.parse_args()


def read_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir
    if not os.path.isdir(outputs_dir):
        raise RuntimeError(f"Outputs dir not found: {outputs_dir}")

    tag_files = [
        name for name in os.listdir(outputs_dir) if name.endswith("_fields.jsonl")
    ]
    miss_files = [
        name for name in os.listdir(outputs_dir) if name.endswith("_eval_misses.jsonl")
    ]
    if not tag_files:
        raise RuntimeError("No *_fields.jsonl files found.")
    if not miss_files:
        raise RuntimeError("No *_eval_misses.jsonl files found.")

    # Load topics per dataset and compute example counts.
    topics_by_dataset: Dict[str, Dict[int, List[str]]] = {}
    unweighted_examples = defaultdict(int)
    weighted_examples = defaultdict(float)

    for name in tag_files:
        dataset = name.replace("_fields.jsonl", "")
        path = os.path.join(outputs_dir, name)
        topics_by_id: Dict[int, List[str]] = {}
        for record in read_jsonl(path):
            record_id = int(record.get("id", -1))
            topics = [str(t) for t in (record.get("topics") or [])]
            topics_by_id[record_id] = topics
            if not topics:
                continue
            for t in topics:
                unweighted_examples[t] += 1
            w = 1.0 / len(topics)
            for t in topics:
                weighted_examples[t] += w
        topics_by_dataset[dataset] = topics_by_id

    # Load misses and compute error counts.
    unweighted_errors = defaultdict(int)
    weighted_errors = defaultdict(float)
    for name in miss_files:
        dataset = name.replace("_eval_misses.jsonl", "")
        path = os.path.join(outputs_dir, name)
        topics_by_id = topics_by_dataset.get(dataset, {})
        for record in read_jsonl(path):
            record_id = int(record.get("id", -1))
            topics = topics_by_id.get(record_id, [])
            if not topics:
                continue
            for t in topics:
                unweighted_errors[t] += 1
            w = 1.0 / len(topics)
            for t in topics:
                weighted_errors[t] += w

    topics = set(unweighted_examples) | set(weighted_examples)

    rows = []
    for topic in topics:
        ue = unweighted_errors.get(topic, 0)
        we = weighted_errors.get(topic, 0.0)
        ut = unweighted_examples.get(topic, 0)
        wt = weighted_examples.get(topic, 0.0)
        ur = (ue / ut) if ut else 0.0
        wr = (we / wt) if wt else 0.0
        rows.append((topic, ue, we, ut, wt, ur, wr))

    rows.sort(key=lambda r: r[6], reverse=True)

    header = (
        "topic",
        "err_unw",
        "err_w",
        "total_unw",
        "total_w",
        "rel_unw",
        "rel_w",
    )
    print(
        f"{header[0]:<28} {header[1]:>8} {header[2]:>9} "
        f"{header[3]:>10} {header[4]:>9} {header[5]:>8} {header[6]:>8}"
    )
    for topic, ue, we, ut, wt, ur, wr in rows:
        print(
            f"{topic:<28} {ue:>8d} {we:>9.2f} {ut:>10d} {wt:>9.2f} "
            f"{ur:>8.3f} {wr:>8.3f}"
        )


if __name__ == "__main__":
    main()
