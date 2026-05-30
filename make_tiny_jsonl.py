#!/usr/bin/env python3
"""
Create small balanced JSONL files for cheap SFT/RL hyperparameter checks.

The input is an existing prepared JSONL file from prepare_data.py. The output
keeps whole conversation records intact and can be fed directly to train_sft.py,
train_rl.py, evaluate.py, or the preflight scripts.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reward import compute_ground_truth


LABELS = ("clockwise", "counterclockwise", "neither")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pipeline", type=int, choices=[1, 2], default=None)
    parser.add_argument(
        "--per_class",
        type=int,
        default=200,
        help="Maximum examples to keep for each ground-truth class.",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=0,
        help="Optional cap after class sampling and shuffling. 0 means no cap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl(args.input)
    if args.pipeline is not None:
        rows = [row for row in rows if row.get("pipeline") == args.pipeline]
    rng.shuffle(rows)

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[compute_ground_truth(row["meta"])].append(row)

    sampled: list[dict[str, Any]] = []
    for label in LABELS:
        sampled.extend(by_class[label][: args.per_class])
    rng.shuffle(sampled)
    if args.max_records > 0:
        sampled = sampled[: args.max_records]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for row in sampled:
            f.write(json.dumps(row) + "\n")

    counts = Counter(compute_ground_truth(row["meta"]) for row in sampled)
    print(f"Wrote {len(sampled)} records to {args.output}")
    print(f"Pipeline: {args.pipeline or 'all'}")
    print(f"Class counts: {dict(counts)}")


if __name__ == "__main__":
    main()
