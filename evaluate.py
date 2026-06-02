#!/usr/bin/env python3
"""
Evaluate a fine-tuned model on the geographic cyclic-ordering task.

By default, this preserves the existing prefilled-tool evaluation behavior:
the prompt contains every record message except the final assistant answer.

Optional live-tool evaluation starts from only the system/user prompt, executes
model-generated tool calls in Python, and feeds tool results back to the model.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from reward import compute_ground_truth, extract_answer
from tools import (
    CYCLIC_ORDER_SCHEMA,
    GEOCODE_SCHEMA,
    compute_cyclic_order,
    compute_cyclic_order_earth,
    representative_point_lonlat,
)


@dataclass
class ParsedAssistantTurn:
    content: str
    tool_calls: list[dict]
    parse_error: str | None = None


class ToolExecutionError(ValueError):
    """Raised when a live model-generated tool call cannot be executed."""


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def apply_chat_template_with_tools(
    messages: list[dict],
    tools: list[dict] | None,
    tokenizer: AutoTokenizer,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        messages_copy = [dict(m) for m in messages]
        if tools and messages_copy and messages_copy[0]["role"] == "system":
            tool_desc = "\n\nAvailable tools:\n" + json.dumps(tools, indent=2)
            messages_copy[0]["content"] = (
                (messages_copy[0].get("content") or "") + tool_desc
            )
        return tokenizer.apply_chat_template(
            messages_copy, tokenize=False, add_generation_prompt=True,
        )


def build_prompt(
    record: dict,
    tokenizer: AutoTokenizer,
    *,
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
) -> str:
    if messages is None:
        messages = record["messages"][:-1]  # Everything except final assistant turn
    if tools is None:
        tools = record.get("tools", None)
    return apply_chat_template_with_tools(messages, tools, tokenizer)


def get_eval_tools(record: dict) -> list[dict]:
    return record.get("tools") or [GEOCODE_SCHEMA, CYCLIC_ORDER_SCHEMA]


def resolve_input_coord_order(args: argparse.Namespace) -> str:
    if args.input_coord_order is not None:
        return args.input_coord_order
    return "latlon" if args.earth else "lonlat"


def make_tool_call(name: str, arguments: dict[str, Any]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def parse_lonlat_pair(value: Any, arg_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ToolExecutionError(f"{arg_name} must be a two-number [longitude, latitude] pair")

    lon, lat = value
    if (
        isinstance(lon, bool)
        or isinstance(lat, bool)
        or not isinstance(lon, (int, float))
        or not isinstance(lat, (int, float))
    ):
        raise ToolExecutionError(f"{arg_name} must contain numeric longitude/latitude values")
    return (float(lon), float(lat))


def execute_geocode(
    arguments: dict,
    record: dict,
    *,
    input_coord_order: str,
) -> dict[str, dict[str, float]]:
    place_names = arguments.get("place_names")
    if not isinstance(place_names, list) or not all(isinstance(n, str) for n in place_names):
        raise ToolExecutionError("geocode requires place_names as a list of strings")

    meta = record["meta"]
    known_names = list(meta["location_names"])
    geometries = list(meta["geometries"])
    if len(known_names) != len(geometries):
        raise ToolExecutionError("record metadata has mismatched location_names/geometries")

    known: dict[str, dict[str, float]] = {}
    for name, geom in zip(known_names, geometries):
        lon, lat = representative_point_lonlat(
            geom,
            input_coord_order=input_coord_order,
        )
        known[name] = {"longitude": round(lon, 6), "latitude": round(lat, 6)}

    result = {}
    for name in place_names:
        if name not in known:
            raise ToolExecutionError(f"Unknown place name: {name}")
        result[name] = known[name]
    return result


def execute_cyclic_order(
    arguments: dict,
    *,
    earth: bool,
) -> dict[str, str]:
    missing = [key for key in ("center", "point_b", "point_c") if key not in arguments]
    if missing:
        raise ToolExecutionError(f"cyclic_order missing required argument(s): {', '.join(missing)}")

    center = parse_lonlat_pair(arguments["center"], "center")
    point_b = parse_lonlat_pair(arguments["point_b"], "point_b")
    point_c = parse_lonlat_pair(arguments["point_c"], "point_c")
    cyclic_order_fn = compute_cyclic_order_earth if earth else compute_cyclic_order
    return {"result": cyclic_order_fn(center, point_b, point_c)}


def decode_tool_arguments(raw_arguments: Any) -> dict:
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"Invalid JSON tool arguments: {exc.msg}") from exc
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        raise ToolExecutionError("Tool arguments must be a JSON object or JSON object string")

    if not isinstance(arguments, dict):
        raise ToolExecutionError("Tool arguments must decode to a JSON object")
    return arguments


def return_or_raise_tool_error(
    message: str,
    metrics: dict[str, int],
    tool_error_policy: str,
) -> dict[str, str]:
    if tool_error_policy == "return_error":
        metrics["tool_error_returns"] += 1
        return {"error": message}
    raise ToolExecutionError(message)


def execute_tool_call(
    tool_call: dict,
    record: dict,
    *,
    earth: bool,
    input_coord_order: str,
    tool_error_policy: str,
    metrics: dict[str, int],
) -> dict:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        metrics["invalid_tool_calls"] += 1
        return return_or_raise_tool_error(
            "Malformed tool call: expected function.name",
            metrics,
            tool_error_policy,
        )

    name = function["name"]
    if name not in {"geocode", "cyclic_order"}:
        metrics["unknown_tool_calls"] += 1
        return return_or_raise_tool_error(
            f"Unknown tool: {name}",
            metrics,
            tool_error_policy,
        )

    try:
        arguments = decode_tool_arguments(function.get("arguments", {}))
        if name == "geocode":
            metrics["geocode_calls"] += 1
            return execute_geocode(
                arguments,
                record,
                input_coord_order=input_coord_order,
            )
        metrics["cyclic_order_calls"] += 1
        return execute_cyclic_order(arguments, earth=earth)
    except ToolExecutionError as exc:
        metrics["invalid_tool_calls"] += 1
        return return_or_raise_tool_error(str(exc), metrics, tool_error_policy)


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def clean_parsed_content(text: str) -> str:
    text = re.sub(r"<\|[^>]*?\|>", "", text)
    text = text.replace("</s>", "").replace("<s>", "")
    return text.strip()


def normalize_tool_call(obj: Any) -> dict | None:
    if not isinstance(obj, dict):
        return None

    function = obj.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
    elif isinstance(obj.get("function_call"), dict):
        function_call = obj["function_call"]
        name = function_call.get("name")
        arguments = function_call.get("arguments", {})
    else:
        name = obj.get("name")
        arguments = obj.get("arguments", {})

    if not isinstance(name, str):
        return None

    if isinstance(arguments, str):
        argument_text = arguments
    else:
        argument_text = json.dumps(arguments, ensure_ascii=False)

    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": argument_text,
        },
    }


def tool_calls_from_json_obj(obj: Any) -> list[dict]:
    if isinstance(obj, list):
        calls = []
        for item in obj:
            calls.extend(tool_calls_from_json_obj(item))
        return calls

    if not isinstance(obj, dict):
        return []

    if isinstance(obj.get("tool_calls"), list):
        calls = []
        for item in obj["tool_calls"]:
            calls.extend(tool_calls_from_json_obj(item))
        return calls

    normalized = normalize_tool_call(obj)
    return [normalized] if normalized is not None else []


def parse_json_tool_calls(text: str) -> list[dict]:
    stripped = strip_markdown_fence(text)
    try:
        calls = tool_calls_from_json_obj(json.loads(stripped))
        if calls:
            return calls
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    calls: list[dict] = []
    idx = 0
    while idx < len(stripped):
        starts = [pos for pos in (stripped.find("{", idx), stripped.find("[", idx)) if pos >= 0]
        if not starts:
            break
        pos = min(starts)
        try:
            obj, end = decoder.raw_decode(stripped[pos:])
        except json.JSONDecodeError:
            idx = pos + 1
            continue

        parsed_calls = tool_calls_from_json_obj(obj)
        if parsed_calls:
            calls.extend(parsed_calls)
        idx = pos + max(end, 1)

    return calls


def extract_tool_calls_from_completion(text: str) -> ParsedAssistantTurn:
    tag_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    tag_matches = list(tag_pattern.finditer(text))
    if tag_matches:
        calls: list[dict] = []
        parse_errors = 0
        for match in tag_matches:
            parsed = parse_json_tool_calls(match.group(1))
            if parsed:
                calls.extend(parsed)
            else:
                parse_errors += 1

        content = clean_parsed_content(tag_pattern.sub("", text))
        if calls:
            error = f"{parse_errors} malformed tool call block(s)" if parse_errors else None
            return ParsedAssistantTurn(content=content, tool_calls=calls, parse_error=error)
        return ParsedAssistantTurn(
            content=content,
            tool_calls=[],
            parse_error="Malformed <tool_call> block",
        )

    calls = parse_json_tool_calls(text)
    if calls:
        return ParsedAssistantTurn(content="", tool_calls=calls)

    lower = text.lower()
    looks_like_tool_call = (
        "tool_call" in lower
        or "tool_calls" in lower
        or ("arguments" in lower and "name" in lower and "function" in lower)
    )
    return ParsedAssistantTurn(
        content=clean_parsed_content(text),
        tool_calls=[],
        parse_error="Could not parse tool-call-like text" if looks_like_tool_call else None,
    )


def materialize_prefilled_messages_from_meta(
    record: dict,
    *,
    earth: bool,
    input_coord_order: str,
) -> list[dict]:
    messages = [dict(record["messages"][0]), dict(record["messages"][1])]
    meta = record["meta"]
    location_names = list(meta["location_names"])
    geometries = list(meta["geometries"])

    if len(geometries) < 3 or len(location_names) != len(geometries):
        raise ValueError("record metadata must contain matching location_names/geometries length >= 3")

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [make_tool_call("geocode", {"place_names": location_names})],
        }
    )
    geocode_result = execute_geocode(
        {"place_names": location_names},
        record,
        input_coord_order=input_coord_order,
    )
    messages.append(
        {
            "role": "tool",
            "name": "geocode",
            "content": json.dumps(geocode_result),
        }
    )

    pts = [
        representative_point_lonlat(g, input_coord_order=input_coord_order)
        for g in geometries
    ]
    center_pt = pts[0]
    center_name = location_names[0]
    cyclic_order_fn = compute_cyclic_order_earth if earth else compute_cyclic_order

    for i in range(1, len(pts) - 1):
        b_pt = pts[i]
        c_pt = pts[i + 1]
        b_name = location_names[i]
        c_name = location_names[i + 1]
        if i == 1:
            preface = (
                f"I have the coordinates. Let me check the arc from "
                f"{b_name} to {c_name} around {center_name}."
            )
        else:
            preface = (
                f"Now checking the arc from {b_name} to {c_name} around "
                f"{center_name}."
            )

        messages.append(
            {
                "role": "assistant",
                "content": preface,
                "tool_calls": [
                    make_tool_call(
                        "cyclic_order",
                        {
                            "center": list(center_pt),
                            "point_b": list(b_pt),
                            "point_c": list(c_pt),
                        },
                    )
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": "cyclic_order",
                "content": json.dumps({"result": cyclic_order_fn(center_pt, b_pt, c_pt)}),
            }
        )

    return messages


def generate_one(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[0][prompt_len:]
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    clean = tokenizer.decode(generated, skip_special_tokens=True)
    return raw, clean


def new_live_metrics() -> dict[str, int]:
    return {
        "tool_parse_failures": 0,
        "invalid_tool_calls": 0,
        "unknown_tool_calls": 0,
        "geocode_calls": 0,
        "cyclic_order_calls": 0,
        "tool_error_returns": 0,
    }


def evaluate_record_live(
    record: dict,
    model,
    tokenizer: AutoTokenizer,
    *,
    tools: list[dict],
    earth: bool,
    input_coord_order: str,
    max_tool_turns: int,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    tool_error_policy: str,
) -> dict:
    messages = [dict(record["messages"][0]), dict(record["messages"][1])]
    metrics = new_live_metrics()

    for turn_idx in range(max_tool_turns):
        prompt = apply_chat_template_with_tools(messages, tools, tokenizer)
        raw_completion, clean_completion = generate_one(
            model,
            tokenizer,
            prompt,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        parsed = extract_tool_calls_from_completion(raw_completion)
        if parsed.parse_error:
            metrics["tool_parse_failures"] += 1

        if parsed.tool_calls:
            assistant_msg = {
                "role": "assistant",
                "tool_calls": parsed.tool_calls,
            }
            if parsed.content:
                assistant_msg["content"] = parsed.content
            messages.append(assistant_msg)

            try:
                for call in parsed.tool_calls:
                    tool_result = execute_tool_call(
                        call,
                        record,
                        earth=earth,
                        input_coord_order=input_coord_order,
                        tool_error_policy=tool_error_policy,
                        metrics=metrics,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "name": call["function"]["name"],
                            "content": json.dumps(tool_result),
                        }
                    )
            except ToolExecutionError as exc:
                return {
                    "completion": clean_completion,
                    "predicted": None,
                    "messages": messages,
                    "status": "tool_failure",
                    "turns": turn_idx + 1,
                    "metrics": metrics,
                    "error": str(exc),
                }
            continue

        final_message = {"role": "assistant", "content": clean_completion}
        if parsed.parse_error:
            return {
                "completion": clean_completion,
                "predicted": None,
                "messages": messages + [final_message],
                "status": "tool_parse_fail",
                "turns": turn_idx + 1,
                "metrics": metrics,
                "error": parsed.parse_error,
            }

        predicted = extract_answer(clean_completion)
        return {
            "completion": clean_completion,
            "predicted": predicted,
            "messages": messages + [final_message],
            "status": "ok" if predicted is not None else "parse_fail",
            "turns": turn_idx + 1,
            "metrics": metrics,
            "error": None,
        }

    return {
        "completion": "",
        "predicted": None,
        "messages": messages,
        "status": "max_tool_turns",
        "turns": max_tool_turns,
        "metrics": metrics,
        "error": "maximum live tool turns reached",
    }


def new_score_state() -> dict[str, Any]:
    return {
        "correct": 0,
        "total": 0,
        "parse_failures": 0,
        "per_class_total": {"clockwise": 0, "counterclockwise": 0, "neither": 0},
        "per_class_correct": {"clockwise": 0, "counterclockwise": 0, "neither": 0},
        "confusion_labels": ["clockwise", "counterclockwise", "neither", "parse_fail"],
        "confusion": {
            gt: {pred: 0 for pred in ["clockwise", "counterclockwise", "neither", "parse_fail"]}
            for gt in ("clockwise", "counterclockwise", "neither")
        },
        "per_n_total": {},
        "per_n_correct": {},
    }


def update_score(
    score: dict[str, Any],
    *,
    predicted: str | None,
    ground_truth: str,
    n_pts: int,
) -> None:
    if ground_truth in score["per_class_total"]:
        score["per_class_total"][ground_truth] += 1
    score["per_n_total"][n_pts] = score["per_n_total"].get(n_pts, 0) + 1

    if predicted is None:
        score["parse_failures"] += 1
        if ground_truth in score["confusion"]:
            score["confusion"][ground_truth]["parse_fail"] += 1
    elif predicted == ground_truth:
        score["correct"] += 1
        if ground_truth in score["per_class_correct"]:
            score["per_class_correct"][ground_truth] += 1
        if ground_truth in score["confusion"]:
            score["confusion"][ground_truth][predicted] += 1
        score["per_n_correct"][n_pts] = score["per_n_correct"].get(n_pts, 0) + 1
    elif ground_truth in score["confusion"] and predicted in score["confusion"][ground_truth]:
        score["confusion"][ground_truth][predicted] += 1

    score["total"] += 1


def prediction_row(
    record: dict,
    *,
    expected: str,
    predicted: str | None,
    status: str,
    completion: str,
    live_result: dict | None = None,
) -> dict:
    row = {
        "question_id": record.get("question_id"),
        "expected_answer": expected,
        "predicted": predicted,
        "status": status,
        "completion": completion,
    }
    if live_result is not None:
        row["messages"] = live_result["messages"]
        row["tool_metrics"] = live_result["metrics"]
        row["turns"] = live_result["turns"]
        row["error"] = live_result.get("error")
    return row


def print_results(
    score: dict[str, Any],
    *,
    pipeline: int | None,
    tool_mode: str,
    earth: bool,
    input_coord_order: str,
    live_counts: dict[str, int] | None = None,
    live_turns_sum: int = 0,
    live_cyclic_sum: int = 0,
) -> None:
    total = score["total"]
    correct = score["correct"]

    print("\n" + "=" * 50)
    print(f"Results (pipeline={pipeline or 'all'}):")
    print(f"  Tool mode:              {tool_mode}")
    print(f"  Earth backend:          {str(earth).lower()}")
    print(f"  Input coordinate order: {input_coord_order}")
    print(f"  Total:          {total}")
    print(f"  Correct:        {correct}")
    print(f"  Accuracy:       {correct / total:.1%}" if total > 0 else "  Accuracy: N/A")
    print(f"  Parse failures: {score['parse_failures']}")
    print("  Per-class accuracy:")
    for cls in ("clockwise", "counterclockwise", "neither"):
        n_cls = score["per_class_total"][cls]
        c_cls = score["per_class_correct"][cls]
        if n_cls > 0:
            print(f"    {cls:18s} {c_cls}/{n_cls} = {c_cls / n_cls:.1%}")
        else:
            print(f"    {cls:18s} 0/0 (no examples)")
    print("  Confusion matrix (rows=ground truth, columns=prediction):")
    print("    " + "".join(f"{label:>18s}" for label in score["confusion_labels"]))
    for gt in ("clockwise", "counterclockwise", "neither"):
        counts = "".join(
            f"{score['confusion'][gt][pred]:18d}"
            for pred in score["confusion_labels"]
        )
        print(f"    {gt:18s}{counts}")
    print("  Per-n accuracy (n = number of points = len(geometries)):")
    for n_pts in sorted(score["per_n_total"].keys()):
        n_cnt = score["per_n_total"][n_pts]
        c_cnt = score["per_n_correct"].get(n_pts, 0)
        print(f"    n={n_pts:<3d}            {c_cnt}/{n_cnt} = {c_cnt / n_cnt:.1%}")

    if live_counts is not None:
        print("  Live tool metrics:")
        for key in (
            "tool_parse_failures",
            "invalid_tool_calls",
            "unknown_tool_calls",
            "geocode_calls",
            "cyclic_order_calls",
            "tool_error_returns",
            "max_tool_turn_failures",
            "tool_failure_examples",
        ):
            print(f"    live/{key}: {live_counts.get(key, 0)}")
        avg_turns = live_turns_sum / total if total else 0.0
        avg_cyclic = live_cyclic_sum / total if total else 0.0
        print(f"    live/avg_turns: {avg_turns:.2f}")
        print(f"    live/avg_cyclic_order_calls_per_example: {avg_cyclic:.2f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model if model_name is a LoRA adapter",
    )
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument(
        "--pipeline",
        type=int,
        default=None,
        help="Filter to a specific pipeline (1 or 2)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Generation budget per completion. Bumped from 512 to 1024 to "
             "accommodate up to n-2=8 cyclic_order tool calls in pipeline 2 "
             "(n=10 worst case).",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help="Maximum prompt length before generation.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy decoding")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--live_tools",
        action="store_true",
        help="Execute model-generated tool calls on the fly instead of using the prefilled transcript.",
    )
    parser.add_argument(
        "--earth",
        action="store_true",
        help="Use the spherical Earth cyclic_order backend while keeping the public tool schema unchanged.",
    )
    parser.add_argument(
        "--input_coord_order",
        choices=["lonlat", "latlon"],
        default=None,
        help="Coordinate order used inside meta['geometries']; defaults to lonlat, or latlon with --earth.",
    )
    parser.add_argument(
        "--max_tool_turns",
        type=int,
        default=12,
        help="Maximum live assistant/tool iterations before marking the example as failed.",
    )
    parser.add_argument(
        "--tool_error_policy",
        choices=["return_error", "fail_example"],
        default="return_error",
        help="How live evaluation handles invalid tool calls.",
    )
    parser.add_argument(
        "--refresh_prefilled_tools",
        action="store_true",
        help="Regenerate prefilled tool messages from meta before scoring.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on examples for smoke tests.")
    parser.add_argument(
        "--save_predictions",
        type=str,
        default=None,
        help="Optional JSONL path for per-example predictions and live transcripts.",
    )
    args = parser.parse_args()

    if (args.live_tools or args.earth) and args.pipeline != 2:
        parser.error("--live_tools and --earth require --pipeline 2")
    if args.refresh_prefilled_tools and args.pipeline != 2:
        parser.error("--refresh_prefilled_tools requires --pipeline 2")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.max_tool_turns < 1:
        parser.error("--max_tool_turns must be positive")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")

    input_coord_order = resolve_input_coord_order(args)
    refresh_prefilled = args.refresh_prefilled_tools or args.earth
    tool_mode = "live" if args.live_tools else "prefilled"

    # ---- Load model ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if args.base_model:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, args.model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    model.eval()

    # ---- Load data ----
    records = load_jsonl(args.test_file)
    if args.pipeline:
        records = [r for r in records if r.get("pipeline") == args.pipeline]
    if args.limit is not None:
        records = records[: args.limit]

    print(f"Evaluating on {len(records)} examples (pipeline={args.pipeline or 'all'})")
    print(f"Tool mode: {tool_mode}")
    print(f"Earth backend: {str(args.earth).lower()}")
    print(f"Input coordinate order: {input_coord_order}")
    if args.live_tools and args.batch_size != 1:
        print("Live tool mode evaluates one example at a time; --batch_size is ignored.")

    score = new_score_state()
    predictions_f = open(args.save_predictions, "w") if args.save_predictions else None

    live_counts: dict[str, int] | None = None
    live_turns_sum = 0
    live_cyclic_sum = 0
    if args.live_tools:
        live_counts = {
            **new_live_metrics(),
            "max_tool_turn_failures": 0,
            "tool_failure_examples": 0,
        }

    try:
        if args.live_tools:
            pbar = tqdm(records, desc="Evaluating", unit="example")
            for record in pbar:
                live_result = evaluate_record_live(
                    record,
                    model,
                    tokenizer,
                    tools=get_eval_tools(record),
                    earth=args.earth,
                    input_coord_order=input_coord_order,
                    max_tool_turns=args.max_tool_turns,
                    max_prompt_length=args.max_prompt_length,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    tool_error_policy=args.tool_error_policy,
                )

                predicted = live_result["predicted"]
                ground_truth = compute_ground_truth(
                    record["meta"],
                    earth=args.earth,
                    input_coord_order=input_coord_order,
                )
                n_pts = len(record["meta"]["geometries"])
                update_score(score, predicted=predicted, ground_truth=ground_truth, n_pts=n_pts)

                assert live_counts is not None
                for key, value in live_result["metrics"].items():
                    live_counts[key] += value
                if live_result["status"] == "max_tool_turns":
                    live_counts["max_tool_turn_failures"] += 1
                if live_result["status"] == "tool_failure":
                    live_counts["tool_failure_examples"] += 1
                live_turns_sum += live_result["turns"]
                live_cyclic_sum += live_result["metrics"]["cyclic_order_calls"]

                if predictions_f is not None:
                    predictions_f.write(
                        json.dumps(
                            prediction_row(
                                record,
                                expected=ground_truth,
                                predicted=predicted,
                                status=live_result["status"],
                                completion=live_result["completion"],
                                live_result=live_result,
                            )
                        )
                        + "\n"
                    )

                total = score["total"]
                pbar.set_postfix(
                    acc=f"{score['correct'] / total:.1%}" if total else "n/a",
                    correct=f"{score['correct']}/{total}",
                    parse_fail=score["parse_failures"],
                )
        else:
            pbar = tqdm(
                range(0, len(records), args.batch_size),
                desc="Evaluating",
                unit="batch",
            )
            for i in pbar:
                batch = records[i : i + args.batch_size]
                prompts = []
                for record in batch:
                    if refresh_prefilled:
                        messages = materialize_prefilled_messages_from_meta(
                            record,
                            earth=args.earth,
                            input_coord_order=input_coord_order,
                        )
                        prompts.append(
                            build_prompt(
                                record,
                                tokenizer,
                                messages=messages,
                                tools=get_eval_tools(record),
                            )
                        )
                    else:
                        prompts.append(build_prompt(record, tokenizer))

                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_prompt_length,
                ).to(model.device)

                with torch.no_grad():
                    gen_kwargs = {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": args.temperature > 0,
                        "pad_token_id": tokenizer.pad_token_id,
                    }
                    if args.temperature > 0:
                        gen_kwargs["temperature"] = args.temperature

                    outputs = model.generate(**inputs, **gen_kwargs)

                for j, (output, record) in enumerate(zip(outputs, batch)):
                    prompt_len = inputs["input_ids"][j].shape[0]
                    completion = tokenizer.decode(
                        output[prompt_len:], skip_special_tokens=True
                    )

                    predicted = extract_answer(completion)
                    ground_truth = compute_ground_truth(
                        record["meta"],
                        earth=args.earth,
                        input_coord_order=input_coord_order,
                    )
                    n_pts = len(record["meta"]["geometries"])
                    update_score(score, predicted=predicted, ground_truth=ground_truth, n_pts=n_pts)

                    if predictions_f is not None:
                        predictions_f.write(
                            json.dumps(
                                prediction_row(
                                    record,
                                    expected=ground_truth,
                                    predicted=predicted,
                                    status="ok" if predicted is not None else "parse_fail",
                                    completion=completion,
                                )
                            )
                            + "\n"
                        )

                total = score["total"]
                pbar.set_postfix(
                    acc=f"{score['correct'] / total:.1%}" if total else "n/a",
                    correct=f"{score['correct']}/{total}",
                    parse_fail=score["parse_failures"],
                )
    finally:
        if predictions_f is not None:
            predictions_f.close()

    print_results(
        score,
        pipeline=args.pipeline,
        tool_mode=tool_mode,
        earth=args.earth,
        input_coord_order=input_coord_order,
        live_counts=live_counts,
        live_turns_sum=live_turns_sum,
        live_cyclic_sum=live_cyclic_sum,
    )


if __name__ == "__main__":
    main()
