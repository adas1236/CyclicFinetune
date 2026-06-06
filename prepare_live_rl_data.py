#!/usr/bin/env python3
"""Expand prepared pipeline-2 transcripts into turn-level live-RL examples."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tool_call_parsing import ToolCallParseError, decode_tool_arguments


TURN_TYPES = ("geocode", "cyclic_order", "final")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    if output.parent != Path("."):
        output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def n_points_for_record(record: dict[str, Any]) -> int:
    meta = record.get("meta") or {}
    geometries = meta.get("geometries")
    if isinstance(geometries, list):
        return len(geometries)
    location_names = meta.get("location_names")
    if isinstance(location_names, list):
        return len(location_names)
    raise ValueError("record meta must contain geometries or location_names")


def get_tool_call_target(
    message: dict[str, Any],
    *,
    expected_name: str,
    record_id: Any,
    assistant_turn_index: int,
) -> tuple[str, dict[str, Any]]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise ValueError(
            f"record {record_id} assistant turn {assistant_turn_index} must "
            f"contain exactly one {expected_name} tool call"
        )

    function = tool_calls[0].get("function") if isinstance(tool_calls[0], dict) else None
    if not isinstance(function, dict):
        raise ValueError(
            f"record {record_id} assistant turn {assistant_turn_index} has "
            "malformed tool call function"
        )

    name = function.get("name")
    if name != expected_name:
        raise ValueError(
            f"record {record_id} assistant turn {assistant_turn_index} expected "
            f"{expected_name!r}, found {name!r}"
        )

    try:
        arguments = decode_tool_arguments(function.get("arguments", {}))
    except ToolCallParseError as exc:
        raise ValueError(
            f"record {record_id} assistant turn {assistant_turn_index} has "
            f"invalid {expected_name} arguments: {exc}"
        ) from exc

    return name, arguments


def expected_answer_for_record(record: dict[str, Any]) -> str | None:
    expected = record.get("expected_answer")
    if expected is None:
        expected = (record.get("meta") or {}).get("answer")
    return str(expected).strip().lower() if expected is not None else None


def validate_prompt_prefix(
    *,
    record_id: Any,
    prompt_messages: list[dict[str, Any]],
    turn_type: str,
    assistant_turn_index: int,
    n_points: int,
) -> None:
    prior_tool_results = [
        message for message in prompt_messages if message.get("role") == "tool"
    ]
    if turn_type == "geocode":
        expected_tool_results = 0
    elif turn_type == "cyclic_order":
        expected_tool_results = assistant_turn_index
    else:
        expected_tool_results = n_points - 1

    if len(prior_tool_results) != expected_tool_results:
        raise ValueError(
            f"record {record_id} {turn_type} turn {assistant_turn_index} has "
            f"{len(prior_tool_results)} prior tool result(s), expected "
            f"{expected_tool_results}"
        )


def build_live_turn_examples(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("pipeline") != 2:
        raise ValueError(
            f"record {record.get('question_id')} has pipeline={record.get('pipeline')}; "
            "prepare_live_rl_data.py only accepts pipeline 2"
        )

    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"record {record.get('question_id')} missing messages list")

    n_points = n_points_for_record(record)
    if n_points < 3:
        raise ValueError(f"record {record.get('question_id')} has n_points={n_points}")

    assistant_indices = [
        idx for idx, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if len(assistant_indices) != n_points:
        raise ValueError(
            f"record {record.get('question_id')} has {len(assistant_indices)} "
            f"assistant turns, expected {n_points}"
        )

    examples: list[dict[str, Any]] = []
    expected_answer = expected_answer_for_record(record)
    record_id = record.get("question_id")

    for assistant_turn_index, message_index in enumerate(assistant_indices):
        target_message = messages[message_index]
        prompt_messages = messages[:message_index]

        if assistant_turn_index == 0:
            turn_type = "geocode"
            expected_tool_name, expected_arguments = get_tool_call_target(
                target_message,
                expected_name="geocode",
                record_id=record_id,
                assistant_turn_index=assistant_turn_index,
            )
        elif assistant_turn_index == len(assistant_indices) - 1:
            turn_type = "final"
            if target_message.get("tool_calls"):
                raise ValueError(
                    f"record {record_id} final assistant turn unexpectedly has tool calls"
                )
            expected_tool_name = None
            expected_arguments = None
        else:
            turn_type = "cyclic_order"
            expected_tool_name, expected_arguments = get_tool_call_target(
                target_message,
                expected_name="cyclic_order",
                record_id=record_id,
                assistant_turn_index=assistant_turn_index,
            )

        validate_prompt_prefix(
            record_id=record_id,
            prompt_messages=prompt_messages,
            turn_type=turn_type,
            assistant_turn_index=assistant_turn_index,
            n_points=n_points,
        )

        examples.append(
            {
                "source_question_id": record_id,
                "pipeline": 2,
                "n_points": n_points,
                "prompt_messages": prompt_messages,
                "tools": record.get("tools"),
                "target_message": target_message,
                "turn_type": turn_type,
                "assistant_turn_index": assistant_turn_index,
                "expected_tool_name": expected_tool_name,
                "expected_arguments": expected_arguments,
                "expected_answer": expected_answer if turn_type == "final" else None,
                "meta": record.get("meta"),
            }
        )

    counts = Counter(example["turn_type"] for example in examples)
    if counts["geocode"] != 1 or counts["cyclic_order"] != n_points - 2 or counts["final"] != 1:
        raise ValueError(
            f"record {record_id} expanded to bad turn counts: {dict(counts)}"
        )
    return examples


def selected_turn_types(args: argparse.Namespace) -> set[str]:
    selected = set()
    if args.include_geocode:
        selected.add("geocode")
    if args.include_cyclic:
        selected.add("cyclic_order")
    if args.include_final:
        selected.add("final")
    return selected or set(TURN_TYPES)


def select_records(
    records: list[dict[str, Any]],
    *,
    max_records_per_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_records_per_n <= 0:
        return records

    rng = random.Random(seed)
    by_n: dict[int, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        by_n[n_points_for_record(record)].append(idx)

    selected_indices = set()
    for indices in by_n.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        selected_indices.update(shuffled[:max_records_per_n])

    return [record for idx, record in enumerate(records) if idx in selected_indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand prepared pipeline-2 JSONL into live-turn examples."
    )
    parser.add_argument("--input", required=True, help="Prepared pipeline-2 JSONL")
    parser.add_argument("--output", required=True, help="Expanded turn-level JSONL")
    parser.add_argument("--include_geocode", action="store_true")
    parser.add_argument("--include_cyclic", action="store_true")
    parser.add_argument("--include_final", action="store_true")
    parser.add_argument(
        "--max_records_per_n",
        type=int,
        default=0,
        help="If >0, sample up to this many original records per n bucket.",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    selected_records = select_records(
        records,
        max_records_per_n=args.max_records_per_n,
        seed=args.seed,
    )
    turn_types = selected_turn_types(args)

    expanded: list[dict[str, Any]] = []
    original_by_n: Counter[int] = Counter()
    expanded_by_turn: Counter[str] = Counter()
    expanded_by_n: Counter[int] = Counter()

    for record in selected_records:
        n_points = n_points_for_record(record)
        original_by_n[n_points] += 1
        examples = build_live_turn_examples(record)
        for example in examples:
            if example["turn_type"] not in turn_types:
                continue
            expanded.append(example)
            expanded_by_turn[example["turn_type"]] += 1
            expanded_by_n[example["n_points"]] += 1

    write_jsonl(args.output, expanded)

    print(f"Read original records:     {len(records)}")
    print(f"Selected original records: {len(selected_records)}")
    print(f"Wrote expanded examples:   {len(expanded)}")
    print(f"Original records by n:     {dict(sorted(original_by_n.items()))}")
    print(f"Expanded examples by n:    {dict(sorted(expanded_by_n.items()))}")
    print(f"Expanded examples by turn: {dict(sorted(expanded_by_turn.items()))}")


if __name__ == "__main__":
    main()
