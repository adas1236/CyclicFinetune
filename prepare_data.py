"""
Convert the geographic-questions parquet file into JSONL training data
formatted as multi-turn conversations with tool calls.

Expected parquet columns:
  - question_id (int)
  - question (str)
  - location_names (sequence of str)
  - geometries (sequence of dicts with "type" and "coordinates" keys)
  - answer (str)
  - roles (dict with keys "center", "b", "c" — each value is an int index
           into the location_names / geometries arrays)

Each row becomes a conversation:

Pipeline 1 (internal computation):
  user      → question
  assistant → tool_call: geocode(place_names)
  tool      → coordinates
  assistant → "The answer is clockwise/counterclockwise." (with reasoning)

Pipeline 2 (tool-assisted computation):
  user      → question
  assistant → tool_call: geocode(place_names)
  tool      → coordinates
  assistant → tool_call: cyclic_order(center, B, C)
  tool      → "clockwise" / "counterclockwise"
  assistant → "The answer is clockwise/counterclockwise."

Usage:
    python prepare_data.py \
        --input data.parquet \
        --output ./data \
        --pipeline both \
        --val_fraction 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


def _to_native(obj):
    """Recursively convert numpy types to plain Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

from tools import (
    CYCLIC_ORDER_SCHEMA,
    GEOCODE_SCHEMA,
    build_geocode_result,
    compute_cyclic_order,
    representative_point,
)


# ---------------------------------------------------------------------------
# Role extraction from the dataset
# ---------------------------------------------------------------------------


def _extract_roles(
    roles_field: dict | list,
    location_names: list[str],
) -> tuple[int, int, int] | None:
    """
    Extract (center_idx, b_idx, c_idx) from the `roles` column.

    Accepted formats:
      - dict with keys "center", "b", "c" — values are int indices into
        location_names, OR string location names that get resolved to indices.
      - list/tuple of three elements [center, b, c] — same int-or-str rule.

    Returns None if the roles can't be parsed.
    """
    if isinstance(roles_field, dict):
        raw_center = roles_field.get("center")
        raw_b = roles_field.get("b")
        raw_c = roles_field.get("c")
    elif isinstance(roles_field, (list, tuple)) and len(roles_field) == 3:
        raw_center, raw_b, raw_c = roles_field
    else:
        return None

    def _resolve(val: int | str) -> int | None:
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            # Match by name
            lower_names = [n.lower() for n in location_names]
            if val.lower() in lower_names:
                return lower_names.index(val.lower())
        return None

    idx_a = _resolve(raw_center)
    idx_b = _resolve(raw_b)
    idx_c = _resolve(raw_c)

    if any(v is None for v in (idx_a, idx_b, idx_c)):
        return None
    return (idx_a, idx_b, idx_c)


# ---------------------------------------------------------------------------
# Conversation builders
# ---------------------------------------------------------------------------


def _system_prompt(pipeline: int) -> str:
    """Return the system prompt for the given pipeline."""
    base = (
        "You are a geographic reasoning assistant. When asked about spatial "
        "relationships between places, first use the geocode tool to look up "
        "their coordinates."
    )
    if pipeline == 1:
        return base + (
            " Then reason about the coordinates to determine the answer. "
            "Think step by step: compute the vectors from the center to each "
            "point, then determine the sign of the cross product to decide "
            "clockwise vs counterclockwise."
        )
    else:
        return base + (
            " Then use the cyclic_order tool to determine whether the "
            "arrangement is clockwise or counterclockwise."
        )


def _tool_list(pipeline: int) -> list[dict]:
    if pipeline == 1:
        return [GEOCODE_SCHEMA]
    else:
        return [GEOCODE_SCHEMA, CYCLIC_ORDER_SCHEMA]


def build_conversation_pipeline1(
    question: str,
    location_names: list[str],
    geometries: list[dict],
    answer: str,
    roles: tuple[int, int, int],
) -> list[dict]:
    """
    Build a multi-turn conversation for pipeline 1 (internal computation).
    """
    idx_a, idx_b, idx_c = roles
    geocode_result = build_geocode_result(location_names, geometries)

    # Compute representative points for the reasoning trace
    pts = {name: representative_point(geom) for name, geom in zip(location_names, geometries)}
    a_name, b_name, c_name = location_names[idx_a], location_names[idx_b], location_names[idx_c]
    a_pt, b_pt, c_pt = pts[a_name], pts[b_name], pts[c_name]

    a_pt = (round(a_pt[0], 1), round(a_pt[1], 1))
    b_pt = (round(b_pt[0], 1), round(b_pt[1], 1))
    c_pt = (round(c_pt[0], 1), round(c_pt[1], 1))

    # Build reasoning trace
    bx, by = round(b_pt[0] - a_pt[0], 1), round(b_pt[1] - a_pt[1], 1)
    cx, cy = round(c_pt[0] - a_pt[0], 1), round(c_pt[1] - a_pt[1], 1)
    det = round(bx * cy - by * cx, 2)

    reasoning = (
        f"Let me compute this from the coordinates.\n\n"
        f"Center (A): {a_name} at ({a_pt[0]}, {a_pt[1]})\n"
        f"Point B: {b_name} at ({b_pt[0]}, {b_pt[1]})\n"
        f"Point C: {c_name} at ({c_pt[0]}, {c_pt[1]})\n\n"
        f"Vector A→B = ({bx}, {by})\n"
        f"Vector A→C = ({cx}, {cy})\n\n"
        f"Cross product (determinant) = ({bx})({cy}) - ({by})({cx}) = {det}\n\n"
        f"Since the determinant is {'positive' if det > 0 else 'negative' if det < 0 else 'zero'}, "
        f"the arrangement is **{answer}**."
    )

    messages = [
        {"role": "system", "content": _system_prompt(1)},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "geocode",
                        "arguments": json.dumps({"place_names": location_names}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "geocode",
            "content": json.dumps(geocode_result),
        },
        {"role": "assistant", "content": reasoning},
    ]
    return messages


def build_conversation_pipeline2(
    question: str,
    location_names: list[str],
    geometries: list[dict],
    answer: str,
    roles: tuple[int, int, int],
) -> list[dict]:
    """
    Build a multi-turn conversation for pipeline 2 (tool-assisted computation).
    """
    idx_a, idx_b, idx_c = roles
    geocode_result = build_geocode_result(location_names, geometries)

    pts = {name: representative_point(geom) for name, geom in zip(location_names, geometries)}
    a_name, b_name, c_name = location_names[idx_a], location_names[idx_b], location_names[idx_c]
    a_pt, b_pt, c_pt = pts[a_name], pts[b_name], pts[c_name]

    messages = [
        {"role": "system", "content": _system_prompt(2)},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "geocode",
                        "arguments": json.dumps({"place_names": location_names}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "geocode",
            "content": json.dumps(geocode_result),
        },
        {
            "role": "assistant",
            "content": (
                f"I have the coordinates. Let me determine the cyclic order of "
                f"{b_name} and {c_name} around {a_name}."
            ),
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "cyclic_order",
                        "arguments": json.dumps(
                            {
                                "center": list(a_pt),
                                "point_b": list(b_pt),
                                "point_c": list(c_pt),
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "cyclic_order",
            "content": json.dumps({"result": answer}),
        },
        {
            "role": "assistant",
            "content": (
                f"Based on the coordinates, moving from {b_name} to {c_name} "
                f"around {a_name} is **{answer}**."
            ),
        },
    ]
    return messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for geo fine-tuning")
    parser.add_argument("--input", required=True, help="Path to the parquet file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--pipeline",
        choices=["1", "2", "both"],
        default="both",
        help="Which pipeline(s) to generate data for",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.1, help="Fraction of data for validation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # Validate that the 'roles' column exists
    if "roles" not in df.columns:
        raise ValueError(
            "Parquet file must contain a 'roles' column. Each entry should be "
            "a dict with keys 'center', 'b', 'c' whose values are int indices "
            "into location_names (or location name strings)."
        )

    records = []
    skipped = 0

    for _, row in df.iterrows():
        question = row["question"]
        location_names = list(row["location_names"])
        geometries = list(row["geometries"])
        answer = row["answer"].strip().lower()
        qid = int(row["question_id"])

        roles = _extract_roles(row["roles"], location_names)
        if roles is None:
            skipped += 1
            continue

        if args.pipeline in ("1", "both"):
            conv = build_conversation_pipeline1(
                question, location_names, geometries, answer, roles
            )
            records.append(
                {
                    "question_id": qid,
                    "pipeline": 1,
                    "tools": _tool_list(1),
                    "messages": conv,
                    "expected_answer": answer,
                    # Store raw data for RL reward computation
                    "meta": {
                        "location_names": location_names,
                        "geometries": geometries,
                        "center_idx": roles[0],
                        "b_idx": roles[1],
                        "c_idx": roles[2],
                    },
                }
            )

        if args.pipeline in ("2", "both"):
            conv = build_conversation_pipeline2(
                question, location_names, geometries, answer, roles
            )
            records.append(
                {
                    "question_id": qid,
                    "pipeline": 2,
                    "tools": _tool_list(2),
                    "messages": conv,
                    "expected_answer": answer,
                    "meta": {
                        "location_names": location_names,
                        "geometries": geometries,
                        "center_idx": roles[0],
                        "b_idx": roles[1],
                        "c_idx": roles[2],
                    },
                }
            )

    print(f"Generated {len(records)} training conversations ({skipped} rows skipped)")

    # Shuffle and split
    random.shuffle(records)
    val_size = int(len(records) * args.val_fraction)
    val_records = records[:val_size]
    train_records = records[val_size:]

    for split, data in [("train", train_records), ("val", val_records)]:
        path = os.path.join(args.output, f"{split}.jsonl")
        with open(path, "w") as f:
            for rec in data:
                f.write(json.dumps(_to_native(rec)) + "\n")
        print(f"Wrote {len(data)} records to {path}")


if __name__ == "__main__":
    main()