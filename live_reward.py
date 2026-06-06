"""Reward helpers for turn-level live tool-loop RL."""

from __future__ import annotations

import json
from typing import Any

from reward import compute_ground_truth, extract_answer
from tool_call_parsing import (
    ToolCallParseError,
    decode_tool_arguments,
    extract_tool_calls_from_completion,
    parse_lonlat_pair,
)


TOOL_REWARD_WEIGHTS = {
    "format_reward": 0.25,
    "tool_name_reward": 0.25,
    "argument_schema_reward": 0.25,
    "argument_value_reward": 1.00,
    "single_call_reward": 0.15,
    "no_final_answer_reward": 0.10,
}

FINAL_REWARD_WEIGHTS = {
    "parseable_answer_reward": 0.25,
    "correctness_reward": 1.00,
    "no_tool_call_reward": 0.25,
}

VALID_TURN_TYPES = {"geocode", "cyclic_order", "final"}
COORD_TOLERANCE = 1e-4


def _empty_tool_components() -> dict[str, Any]:
    return {
        "format_reward": 0.0,
        "tool_name_reward": 0.0,
        "argument_schema_reward": 0.0,
        "argument_value_reward": 0.0,
        "single_call_reward": 0.0,
        "no_final_answer_reward": 0.0,
        "total": 0.0,
        "tool_call_count": 0,
        "parsed_tool_name": None,
        "parse_error": None,
        "predicted_answer": None,
    }


def _empty_final_components() -> dict[str, Any]:
    return {
        "parseable_answer_reward": 0.0,
        "correctness_reward": 0.0,
        "no_tool_call_reward": 0.0,
        "total": 0.0,
        "tool_call_count": 0,
        "predicted_answer": None,
        "parse_error": None,
    }


def _sum_reward_components(components: dict[str, Any], weights: dict[str, float]) -> float:
    total = sum(float(components.get(name, 0.0)) for name in weights)
    components["total"] = total
    return total


def _function_from_call(call: dict[str, Any]) -> dict[str, Any] | None:
    function = call.get("function") if isinstance(call, dict) else None
    return function if isinstance(function, dict) else None


def _tool_name_from_call(call: dict[str, Any]) -> str | None:
    function = _function_from_call(call)
    name = function.get("name") if function else None
    return name if isinstance(name, str) else None


def _arguments_from_call(call: dict[str, Any]) -> dict[str, Any] | None:
    function = _function_from_call(call)
    if not function:
        return None
    return decode_tool_arguments(function.get("arguments", {}))


def _valid_geocode_schema(arguments: dict[str, Any]) -> bool:
    place_names = arguments.get("place_names")
    return isinstance(place_names, list) and all(
        isinstance(name, str) for name in place_names
    )


def _valid_cyclic_schema(arguments: dict[str, Any]) -> bool:
    try:
        parse_lonlat_pair(arguments.get("center"), "center")
        parse_lonlat_pair(arguments.get("point_b"), "point_b")
        parse_lonlat_pair(arguments.get("point_c"), "point_c")
    except ToolCallParseError:
        return False
    return True


def _coord_pair_matches(
    generated: Any,
    expected: Any,
    *,
    tolerance: float,
) -> bool:
    try:
        gen_lon, gen_lat = parse_lonlat_pair(generated, "generated")
        exp_lon, exp_lat = parse_lonlat_pair(expected, "expected")
    except ToolCallParseError:
        return False
    return abs(gen_lon - exp_lon) <= tolerance and abs(gen_lat - exp_lat) <= tolerance


def _geocode_arguments_match(
    generated: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    if not _valid_geocode_schema(generated):
        return False
    expected_place_names = expected.get("place_names")
    return generated.get("place_names") == expected_place_names


def _cyclic_arguments_match(
    generated: dict[str, Any],
    expected: dict[str, Any],
    *,
    tolerance: float,
) -> bool:
    if not _valid_cyclic_schema(generated):
        return False
    return all(
        _coord_pair_matches(generated.get(name), expected.get(name), tolerance=tolerance)
        for name in ("center", "point_b", "point_c")
    )


def argument_schema_valid(tool_name: str, arguments: dict[str, Any]) -> bool:
    if tool_name == "geocode":
        return _valid_geocode_schema(arguments)
    if tool_name == "cyclic_order":
        return _valid_cyclic_schema(arguments)
    return False


def argument_values_match(
    tool_name: str,
    generated: dict[str, Any],
    expected: dict[str, Any],
    *,
    tolerance: float = COORD_TOLERANCE,
) -> bool:
    if tool_name == "geocode":
        return _geocode_arguments_match(generated, expected)
    if tool_name == "cyclic_order":
        return _cyclic_arguments_match(generated, expected, tolerance=tolerance)
    return False


def score_tool_turn_components(
    completion: str,
    expected_tool_name: str,
    expected_arguments: dict[str, Any],
    *,
    tolerance: float = COORD_TOLERANCE,
) -> dict[str, Any]:
    components = _empty_tool_components()
    parsed = extract_tool_calls_from_completion(completion)
    components["parse_error"] = parsed.parse_error
    components["tool_call_count"] = len(parsed.tool_calls)

    predicted_answer = extract_answer(completion)
    components["predicted_answer"] = predicted_answer
    if predicted_answer is None:
        components["no_final_answer_reward"] = TOOL_REWARD_WEIGHTS[
            "no_final_answer_reward"
        ]

    if not parsed.tool_calls:
        _sum_reward_components(components, TOOL_REWARD_WEIGHTS)
        return components

    components["format_reward"] = TOOL_REWARD_WEIGHTS["format_reward"]
    if len(parsed.tool_calls) == 1:
        components["single_call_reward"] = TOOL_REWARD_WEIGHTS["single_call_reward"]

    call = parsed.tool_calls[0]
    parsed_tool_name = _tool_name_from_call(call)
    components["parsed_tool_name"] = parsed_tool_name
    if parsed_tool_name == expected_tool_name:
        components["tool_name_reward"] = TOOL_REWARD_WEIGHTS["tool_name_reward"]
    else:
        _sum_reward_components(components, TOOL_REWARD_WEIGHTS)
        return components

    try:
        arguments = _arguments_from_call(call)
    except ToolCallParseError as exc:
        components["parse_error"] = str(exc)
        _sum_reward_components(components, TOOL_REWARD_WEIGHTS)
        return components
    if arguments is None:
        _sum_reward_components(components, TOOL_REWARD_WEIGHTS)
        return components

    if argument_schema_valid(expected_tool_name, arguments):
        components["argument_schema_reward"] = TOOL_REWARD_WEIGHTS[
            "argument_schema_reward"
        ]
    if argument_values_match(
        expected_tool_name,
        arguments,
        expected_arguments,
        tolerance=tolerance,
    ):
        components["argument_value_reward"] = TOOL_REWARD_WEIGHTS[
            "argument_value_reward"
        ]

    _sum_reward_components(components, TOOL_REWARD_WEIGHTS)
    return components


def score_tool_turn(
    completion: str,
    expected_tool_name: str,
    expected_arguments: dict[str, Any],
) -> float:
    components = score_tool_turn_components(
        completion,
        expected_tool_name,
        expected_arguments,
    )
    return float(components["total"])


def score_final_turn_components(
    completion: str,
    expected_answer: str,
) -> dict[str, Any]:
    components = _empty_final_components()
    parsed = extract_tool_calls_from_completion(completion)
    components["parse_error"] = parsed.parse_error
    components["tool_call_count"] = len(parsed.tool_calls)

    predicted = extract_answer(completion)
    components["predicted_answer"] = predicted
    if predicted is not None:
        components["parseable_answer_reward"] = FINAL_REWARD_WEIGHTS[
            "parseable_answer_reward"
        ]
        if predicted == expected_answer:
            components["correctness_reward"] = FINAL_REWARD_WEIGHTS[
                "correctness_reward"
            ]

    if not parsed.tool_calls and parsed.parse_error is None:
        components["no_tool_call_reward"] = FINAL_REWARD_WEIGHTS[
            "no_tool_call_reward"
        ]

    _sum_reward_components(components, FINAL_REWARD_WEIGHTS)
    return components


def score_final_turn(completion: str, expected_answer: str) -> float:
    components = score_final_turn_components(completion, expected_answer)
    return float(components["total"])


def maybe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"", "null", "None"}:
            return None
        return json.loads(stripped)
    return value


def expected_answer_from_row(row: dict[str, Any]) -> str | None:
    expected_answer = row.get("expected_answer")
    if expected_answer:
        return str(expected_answer).strip().lower()

    meta = maybe_json_loads(row.get("meta_json") or row.get("meta"))
    if isinstance(meta, dict):
        return compute_ground_truth(meta)
    return None


def live_turn_reward_components(
    completion: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    turn_type = row.get("turn_type")
    if turn_type not in VALID_TURN_TYPES:
        raise ValueError(f"Unknown turn_type={turn_type!r}")

    if turn_type in {"geocode", "cyclic_order"}:
        expected_arguments = maybe_json_loads(
            row.get("expected_arguments_json", row.get("expected_arguments"))
        )
        if not isinstance(expected_arguments, dict):
            expected_arguments = {}
        expected_tool_name = row.get("expected_tool_name") or turn_type
        return score_tool_turn_components(
            completion,
            str(expected_tool_name),
            expected_arguments,
        )

    expected_answer = expected_answer_from_row(row)
    if expected_answer is None:
        raise ValueError("final turn reward requires expected_answer or meta")
    return score_final_turn_components(completion, expected_answer)


def live_turn_reward(completion: str, row: dict[str, Any]) -> float:
    components = live_turn_reward_components(completion, row)
    return float(components["total"])
