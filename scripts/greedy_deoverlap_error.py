#!/usr/bin/env python3
"""Greedy de-overlap of topic error rates by removing assigned examples."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy de-overlap topic errors from outputs directory."
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing *_fields.jsonl and *_eval_misses.jsonl.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum greedy iterations to display.",
    )
    parser.add_argument(
        "--remove-misses-only",
        action="store_true",
        help="Remove only missed examples for the selected topic (keep non-miss examples).",
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

    # Load topics per example keyed by (dataset, id)
    topics_by_key: Dict[Tuple[str, int], List[str]] = {}
    for name in tag_files:
        dataset = name.replace("_fields.jsonl", "")
        path = os.path.join(outputs_dir, name)
        for record in read_jsonl(path):
            record_id = int(record.get("id", -1))
            topics = [str(t) for t in (record.get("topics") or [])]
            topics_by_key[(dataset, record_id)] = topics

    # Load misses set
    miss_keys: Set[Tuple[str, int]] = set()
    for name in miss_files:
        dataset = name.replace("_eval_misses.jsonl", "")
        path = os.path.join(outputs_dir, name)
        for record in read_jsonl(path):
            record_id = int(record.get("id", -1))
            miss_keys.add((dataset, record_id))

    remaining_keys = {k for k, v in topics_by_key.items() if v}
    step = 0

    header = (
        "step",
        "topic",
        "err_unw",
        "err_w",
        "total_unw",
        "total_w",
        "rel_unw",
        "rel_w",
        "removed",
    )
    print(
        f"{header[0]:>4} {header[1]:<28} {header[2]:>8} {header[3]:>9} "
        f"{header[4]:>10} {header[5]:>9} {header[6]:>8} {header[7]:>8} {header[8]:>8}"
    )

    while remaining_keys and step < args.max_steps:
        step += 1
        unweighted_examples = defaultdict(int)
        weighted_examples = defaultdict(float)
        unweighted_errors = defaultdict(int)
        weighted_errors = defaultdict(float)

        for key in remaining_keys:
            topics = topics_by_key.get(key) or []
            if not topics:
                continue
            n = len(topics)
            w = 1.0 / n
            for t in topics:
                unweighted_examples[t] += 1
                weighted_examples[t] += w
            if key in miss_keys:
                for t in topics:
                    unweighted_errors[t] += 1
                    weighted_errors[t] += w

        # compute best topic by weighted relative error
        best_topic = None
        best_rel = -1.0
        for t, total_w in weighted_examples.items():
            if total_w <= 0:
                continue
            rel = weighted_errors[t] / total_w
            if rel > best_rel:
                best_rel = rel
                best_topic = t

        if best_topic is None:
            break

        ue = unweighted_errors.get(best_topic, 0)
        we = weighted_errors.get(best_topic, 0.0)
        ut = unweighted_examples.get(best_topic, 0)
        wt = weighted_examples.get(best_topic, 0.0)
        ur = (ue / ut) if ut else 0.0
        wr = (we / wt) if wt else 0.0

        if args.remove_misses_only:
            to_remove = {
                k
                for k in remaining_keys
                if k in miss_keys and best_topic in topics_by_key.get(k, [])
            }
        else:
            to_remove = {k for k in remaining_keys if best_topic in topics_by_key.get(k, [])}
        remaining_keys -= to_remove

        print(
            f"{step:>4d} {best_topic:<28} {ue:>8d} {we:>9.2f} {ut:>10d} "
            f"{wt:>9.2f} {ur:>8.3f} {wr:>8.3f} {len(to_remove):>8d}"
        )


if __name__ == "__main__":
    main()
