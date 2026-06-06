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
import time
from dataclasses import dataclass
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from reward import compute_ground_truth, extract_answer
from tool_call_parsing import (
    ToolCallParseError,
    _match_braces,
    decode_tool_arguments,
    discover_tool_markers,
    extract_tool_calls_from_completion,
    parse_lonlat_pair,
)
from tools import (
    CYCLIC_ORDER_SCHEMA,
    GEOCODE_SCHEMA,
    compute_cyclic_order,
    compute_cyclic_order_earth,
    representative_point_lonlat,
)


@dataclass
class GenerationResult:
    raw: str
    clean: str
    prompt_tokens: int
    generated_tokens: int
    hit_token_cap: bool
    tokenization_seconds: float = 0.0
    generation_seconds: float = 0.0


@dataclass
class LiveEvalState:
    index: int
    record: dict
    tools: list[dict]
    messages: list[dict]
    metrics: dict[str, int]
    turns: int = 0
    status: str | None = None
    completion: str = ""
    predicted: str | None = None
    error: str | None = None
    prompt_tokens_total: int = 0
    generated_tokens_total: int = 0
    prompt_render_seconds: float = 0.0
    tokenization_seconds: float = 0.0
    generation_seconds: float = 0.0
    parsing_seconds: float = 0.0
    tool_execution_seconds: float = 0.0
    truncation_retries: int = 0
    truncation_retry_successes: int = 0


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
            result = execute_geocode(
                arguments,
                record,
                input_coord_order=input_coord_order,
            )
            metrics["valid_geocode_calls"] += 1
            return result
        metrics["cyclic_order_calls"] += 1
        result = execute_cyclic_order(arguments, earth=earth)
        metrics["valid_cyclic_order_calls"] += 1
        return result
    except (ToolExecutionError, ToolCallParseError) as exc:
        metrics["invalid_tool_calls"] += 1
        return return_or_raise_tool_error(str(exc), metrics, tool_error_policy)


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


def get_model_input_device(model) -> torch.device | str:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def cuda_synchronize_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_batch(
    model,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    stop_strings: list[str] | None = None,
) -> list[GenerationResult]:
    tokenization_start = time.perf_counter()
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    ).to(get_model_input_device(model))
    tokenization_seconds = time.perf_counter() - tokenization_start

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    if stop_strings:
        gen_kwargs["stop_strings"] = stop_strings
        gen_kwargs["tokenizer"] = tokenizer

    with torch.no_grad():
        cuda_synchronize_if_available()
        generation_start = time.perf_counter()
        try:
            outputs = model.generate(**inputs, **gen_kwargs)
        except TypeError:
            if not stop_strings:
                raise
            if not getattr(generate_batch, "_warned_stop_strings_failure", False):
                print(
                    "WARNING: model.generate rejected stop_strings; "
                    "continuing without tool-call stop strings."
                )
                setattr(generate_batch, "_warned_stop_strings_failure", True)
            fallback_kwargs = dict(gen_kwargs)
            fallback_kwargs.pop("stop_strings", None)
            fallback_kwargs.pop("tokenizer", None)
            outputs = model.generate(**inputs, **fallback_kwargs)
        cuda_synchronize_if_available()
        generation_seconds = time.perf_counter() - generation_start

    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[:, prompt_len:]
    per_row_tokenization = tokenization_seconds / len(prompts) if prompts else 0.0
    per_row_generation = generation_seconds / len(prompts) if prompts else 0.0
    pad_token_id = tokenizer.pad_token_id

    results: list[GenerationResult] = []
    for row_idx in range(len(prompts)):
        row = generated[row_idx]
        if pad_token_id is None:
            generated_tokens = int(row.numel())
        else:
            generated_tokens = int((row != pad_token_id).sum().item())
        raw = tokenizer.decode(row, skip_special_tokens=False)
        clean = tokenizer.decode(row, skip_special_tokens=True)
        results.append(
            GenerationResult(
                raw=raw,
                clean=clean,
                prompt_tokens=int(inputs["attention_mask"][row_idx].sum().item()),
                generated_tokens=generated_tokens,
                hit_token_cap=generated_tokens >= max_new_tokens,
                tokenization_seconds=per_row_tokenization,
                generation_seconds=per_row_generation,
            )
        )

    return results


def generate_one(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    result = generate_batch(
        model,
        tokenizer,
        [prompt],
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )[0]
    return result.raw, result.clean


def new_live_metrics() -> dict[str, int]:
    return {
        "tool_parse_failures": 0,
        "invalid_tool_calls": 0,
        "unknown_tool_calls": 0,
        "geocode_calls": 0,
        "cyclic_order_calls": 0,
        "valid_geocode_calls": 0,
        "valid_cyclic_order_calls": 0,
        "tool_error_returns": 0,
        "truncation_retries": 0,
        "truncation_retry_successes": 0,
    }


def stable_tools_key(tools: list[dict]) -> str:
    return json.dumps(tools, sort_keys=True, separators=(",", ":"))


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def init_live_state(index: int, record: dict) -> LiveEvalState:
    return LiveEvalState(
        index=index,
        record=record,
        tools=get_eval_tools(record),
        messages=[dict(record["messages"][0]), dict(record["messages"][1])],
        metrics=new_live_metrics(),
    )


def build_live_prompt(state: LiveEvalState, tokenizer: AutoTokenizer) -> str:
    start = time.perf_counter()
    prompt = apply_chat_template_with_tools(state.messages, state.tools, tokenizer)
    state.prompt_render_seconds += time.perf_counter() - start
    return prompt


def record_generation_timing(state: LiveEvalState, result: GenerationResult) -> None:
    state.prompt_tokens_total += result.prompt_tokens
    state.generated_tokens_total += result.generated_tokens
    state.tokenization_seconds += result.tokenization_seconds
    state.generation_seconds += result.generation_seconds


def has_unclosed_tool_marker(
    text: str,
    *,
    prefix_marker: str | None,
    suffix_marker: str | None,
) -> bool:
    checks = []
    if prefix_marker:
        checks.append((prefix_marker.strip() or prefix_marker, suffix_marker))
    checks.append(("<tool_call>", "</tool_call>"))

    for prefix, suffix in checks:
        if not prefix or prefix not in text:
            continue
        if not suffix:
            return True
        prefix_idx = text.rfind(prefix)
        suffix_idx = text.find(suffix, prefix_idx + len(prefix))
        if suffix_idx == -1:
            return True
    return False


def has_unclosed_json_object(text: str) -> bool:
    starts = [pos for pos in (text.rfind("{"), text.rfind("[")) if pos >= 0]
    if not starts:
        return False
    start = max(starts)
    if text[start] == "{":
        return _match_braces(text, start) == -1

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return False
    return True


def generation_has_parseable_step(result: GenerationResult) -> bool:
    parsed = extract_tool_calls_from_completion(result.raw)
    return bool(parsed.tool_calls) or extract_answer(result.clean) is not None


def looks_like_truncated_generation(
    result: GenerationResult,
    *,
    prefix_marker: str | None,
    suffix_marker: str | None,
) -> bool:
    if not result.hit_token_cap:
        return False
    if generation_has_parseable_step(result):
        return False

    raw = result.raw
    lower = raw.lower()
    if has_unclosed_tool_marker(
        raw,
        prefix_marker=prefix_marker,
        suffix_marker=suffix_marker,
    ):
        return True
    if has_unclosed_json_object(raw) and (
        "tool_call" in lower
        or "arguments" in lower
        or "function" in lower
        or '"name"' in lower
    ):
        return True
    if extract_tool_calls_from_completion(raw).parse_error:
        return True

    clean = result.clean.strip()
    return bool(clean) and clean[-1] not in ".!?)]}\"'"


def advance_live_state(
    state: LiveEvalState,
    result: GenerationResult,
    *,
    earth: bool,
    input_coord_order: str,
    max_tool_turns: int,
    tool_error_policy: str,
) -> None:
    state.turns += 1
    state.completion = result.clean

    parse_start = time.perf_counter()
    parsed = extract_tool_calls_from_completion(result.raw)
    state.parsing_seconds += time.perf_counter() - parse_start
    if parsed.parse_error:
        state.metrics["tool_parse_failures"] += 1

    if parsed.tool_calls:
        assistant_msg = {
            "role": "assistant",
            "tool_calls": parsed.tool_calls,
        }
        if parsed.content:
            assistant_msg["content"] = parsed.content
        state.messages.append(assistant_msg)

        try:
            for call in parsed.tool_calls:
                tool_start = time.perf_counter()
                tool_result = execute_tool_call(
                    call,
                    state.record,
                    earth=earth,
                    input_coord_order=input_coord_order,
                    tool_error_policy=tool_error_policy,
                    metrics=state.metrics,
                )
                state.tool_execution_seconds += time.perf_counter() - tool_start
                function = call.get("function") if isinstance(call, dict) else None
                tool_name = (
                    function.get("name")
                    if isinstance(function, dict) and isinstance(function.get("name"), str)
                    else "unknown"
                )
                state.messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(tool_result),
                    }
                )
        except ToolExecutionError as exc:
            state.status = "tool_failure"
            state.error = str(exc)
            state.predicted = None
            return

        if state.turns >= max_tool_turns:
            state.status = "max_tool_turns"
            state.error = "maximum live tool turns reached"
        return

    final_message = {"role": "assistant", "content": result.clean}
    state.messages.append(final_message)
    if parsed.parse_error:
        state.status = "tool_parse_fail"
        state.error = parsed.parse_error
        state.predicted = None
        return

    state.predicted = extract_answer(result.clean)
    state.status = "ok" if state.predicted is not None else "parse_fail"
    state.error = None


def finalize_live_state(state: LiveEvalState) -> dict:
    return {
        "completion": state.completion,
        "predicted": state.predicted,
        "messages": state.messages,
        "status": state.status or "unknown",
        "turns": state.turns,
        "metrics": state.metrics,
        "error": state.error,
        "n_points": len(state.record["meta"]["geometries"]),
        "timing": {
            "prompt_render_seconds": state.prompt_render_seconds,
            "tokenization_seconds": state.tokenization_seconds,
            "generation_seconds": state.generation_seconds,
            "parsing_seconds": state.parsing_seconds,
            "tool_execution_seconds": state.tool_execution_seconds,
            "prompt_tokens": state.prompt_tokens_total,
            "generated_tokens": state.generated_tokens_total,
            "truncation_retries": state.truncation_retries,
            "truncation_retry_successes": state.truncation_retry_successes,
        },
    }


def evaluate_records_live_batched(
    records: list[dict],
    model,
    tokenizer: AutoTokenizer,
    *,
    batch_size: int,
    earth: bool,
    input_coord_order: str,
    max_tool_turns: int,
    max_prompt_length: int,
    max_new_tokens: int,
    retry_max_new_tokens: int,
    temperature: float,
    tool_error_policy: str,
    stop_strings: list[str] | None,
    prefix_marker: str | None,
    suffix_marker: str | None,
) -> tuple[list[dict], dict[str, float | int]]:
    states = [init_live_state(i, record) for i, record in enumerate(records)]
    active = list(states)
    results_by_index: list[dict | None] = [None] * len(records)
    frontier_stats: dict[str, float | int] = {
        "frontier_batches": 0,
        "frontier_batch_seconds": 0.0,
    }

    pbar = tqdm(total=len(records), desc="Evaluating", unit="example")
    try:
        while active:
            next_active: list[LiveEvalState] = []
            grouped: dict[str, list[LiveEvalState]] = {}
            for state in active:
                grouped.setdefault(stable_tools_key(state.tools), []).append(state)

            for group in grouped.values():
                for batch in chunks(group, batch_size):
                    batch_start = time.perf_counter()
                    prompts = [build_live_prompt(state, tokenizer) for state in batch]
                    generation_results = generate_batch(
                        model,
                        tokenizer,
                        prompts,
                        max_prompt_length=max_prompt_length,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        stop_strings=stop_strings,
                    )
                    for state, result in zip(batch, generation_results):
                        record_generation_timing(state, result)

                    retry_items: list[tuple[int, LiveEvalState]] = []
                    if max_new_tokens < retry_max_new_tokens:
                        for result_idx, (state, result) in enumerate(
                            zip(batch, generation_results)
                        ):
                            if looks_like_truncated_generation(
                                result,
                                prefix_marker=prefix_marker,
                                suffix_marker=suffix_marker,
                            ):
                                state.truncation_retries += 1
                                state.metrics["truncation_retries"] += 1
                                retry_items.append((result_idx, state))

                    if retry_items:
                        retry_prompts = [prompts[result_idx] for result_idx, _ in retry_items]
                        retry_results = generate_batch(
                            model,
                            tokenizer,
                            retry_prompts,
                            max_prompt_length=max_prompt_length,
                            max_new_tokens=retry_max_new_tokens,
                            temperature=temperature,
                            stop_strings=stop_strings,
                        )
                        for retry_result, (result_idx, state) in zip(
                            retry_results,
                            retry_items,
                        ):
                            record_generation_timing(state, retry_result)
                            if generation_has_parseable_step(retry_result):
                                state.truncation_retry_successes += 1
                                state.metrics["truncation_retry_successes"] += 1
                            generation_results[result_idx] = retry_result

                    for state, result in zip(batch, generation_results):
                        advance_live_state(
                            state,
                            result,
                            earth=earth,
                            input_coord_order=input_coord_order,
                            max_tool_turns=max_tool_turns,
                            tool_error_policy=tool_error_policy,
                        )
                        if state.status is None:
                            next_active.append(state)
                        else:
                            results_by_index[state.index] = finalize_live_state(state)
                            pbar.update(1)

                    frontier_stats["frontier_batches"] = int(frontier_stats["frontier_batches"]) + 1
                    frontier_stats["frontier_batch_seconds"] = (
                        float(frontier_stats["frontier_batch_seconds"])
                        + time.perf_counter()
                        - batch_start
                    )

            active = next_active
    finally:
        pbar.close()

    missing = [i for i, result in enumerate(results_by_index) if result is None]
    if missing:
        raise RuntimeError(f"live evaluation did not finalize result(s): {missing}")
    return [result for result in results_by_index if result is not None], frontier_stats


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


ANSWER_LABELS = ("clockwise", "counterclockwise", "neither")
CONFUSION_LABELS = ("clockwise", "counterclockwise", "neither", "parse_fail")


def new_confusion_matrix() -> dict[str, dict[str, int]]:
    return {
        gt: {pred: 0 for pred in CONFUSION_LABELS}
        for gt in ANSWER_LABELS
    }


def update_confusion_matrix(
    matrix: dict[str, dict[str, int]],
    *,
    ground_truth: str,
    predicted: str | None,
) -> None:
    if ground_truth not in matrix:
        return

    prediction_bucket = "parse_fail" if predicted is None else predicted
    if prediction_bucket in matrix[ground_truth]:
        matrix[ground_truth][prediction_bucket] += 1


def print_confusion_matrix(
    matrix: dict[str, dict[str, int]],
    labels: list[str],
    *,
    indent: str,
) -> None:
    print(indent + "".join(f"{label:>18s}" for label in labels))
    for gt in ANSWER_LABELS:
        counts = "".join(
            f"{matrix[gt][pred]:18d}"
            for pred in labels
        )
        print(f"{indent}{gt:18s}{counts}")


def new_score_state() -> dict[str, Any]:
    return {
        "correct": 0,
        "total": 0,
        "parse_failures": 0,
        "per_class_total": {label: 0 for label in ANSWER_LABELS},
        "per_class_correct": {label: 0 for label in ANSWER_LABELS},
        "confusion_labels": list(CONFUSION_LABELS),
        "confusion": new_confusion_matrix(),
        "per_n_total": {},
        "per_n_correct": {},
        "per_n_confusion": {},
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
    per_n_confusion = score["per_n_confusion"].setdefault(
        n_pts,
        new_confusion_matrix(),
    )
    update_confusion_matrix(
        score["confusion"],
        ground_truth=ground_truth,
        predicted=predicted,
    )
    update_confusion_matrix(
        per_n_confusion,
        ground_truth=ground_truth,
        predicted=predicted,
    )

    if predicted is None:
        score["parse_failures"] += 1
    elif predicted == ground_truth:
        score["correct"] += 1
        if ground_truth in score["per_class_correct"]:
            score["per_class_correct"][ground_truth] += 1
        score["per_n_correct"][n_pts] = score["per_n_correct"].get(n_pts, 0) + 1

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
        row["timing"] = live_result.get("timing")
    return row


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def aggregate_live_timings(
    live_results: list[dict],
    *,
    live_total_seconds: float,
    frontier_stats: dict[str, float | int],
) -> dict[str, Any]:
    total_turns = sum(int(result.get("turns", 0)) for result in live_results)
    total_tool_calls = 0
    prompt_render_seconds = 0.0
    tokenization_seconds = 0.0
    generation_seconds = 0.0
    parsing_seconds = 0.0
    tool_execution_seconds = 0.0
    prompt_tokens = 0
    generated_tokens = 0
    truncation_retries = 0
    truncation_retry_successes = 0
    seconds_by_n: dict[int, list[float]] = {}

    for result in live_results:
        metrics = result.get("metrics", {})
        total_tool_calls += int(metrics.get("geocode_calls", 0))
        total_tool_calls += int(metrics.get("cyclic_order_calls", 0))
        timing = result.get("timing") or {}
        prompt_render_seconds += float(timing.get("prompt_render_seconds", 0.0))
        tokenization_seconds += float(timing.get("tokenization_seconds", 0.0))
        generation_seconds += float(timing.get("generation_seconds", 0.0))
        parsing_seconds += float(timing.get("parsing_seconds", 0.0))
        tool_execution_seconds += float(timing.get("tool_execution_seconds", 0.0))
        prompt_tokens += int(timing.get("prompt_tokens", 0))
        generated_tokens += int(timing.get("generated_tokens", 0))
        truncation_retries += int(timing.get("truncation_retries", 0))
        truncation_retry_successes += int(timing.get("truncation_retry_successes", 0))
        n_points = int(result.get("n_points", 0))
        state_seconds = (
            float(timing.get("prompt_render_seconds", 0.0))
            + float(timing.get("tokenization_seconds", 0.0))
            + float(timing.get("generation_seconds", 0.0))
            + float(timing.get("parsing_seconds", 0.0))
            + float(timing.get("tool_execution_seconds", 0.0))
        )
        seconds_by_n.setdefault(n_points, []).append(state_seconds)

    avg_seconds_by_n = {
        n_points: safe_div(sum(values), len(values))
        for n_points, values in seconds_by_n.items()
    }
    frontier_batches = int(frontier_stats.get("frontier_batches", 0))
    frontier_batch_seconds = float(frontier_stats.get("frontier_batch_seconds", 0.0))

    return {
        "total_turns": total_turns,
        "total_tool_calls": total_tool_calls,
        "avg_seconds_per_example": safe_div(live_total_seconds, len(live_results)),
        "avg_seconds_per_turn": safe_div(live_total_seconds, total_turns),
        "avg_seconds_per_tool_call": safe_div(live_total_seconds, total_tool_calls),
        "avg_seconds_by_n": avg_seconds_by_n,
        "avg_generated_tokens_per_turn": safe_div(generated_tokens, total_turns),
        "avg_prompt_tokens_per_turn": safe_div(prompt_tokens, total_turns),
        "avg_generation_seconds_per_turn": safe_div(generation_seconds, total_turns),
        "avg_tool_execution_seconds_per_tool_call": safe_div(
            tool_execution_seconds,
            total_tool_calls,
        ),
        "prompt_render_seconds": prompt_render_seconds,
        "tokenization_seconds": tokenization_seconds,
        "generation_seconds": generation_seconds,
        "parsing_seconds": parsing_seconds,
        "tool_execution_seconds": tool_execution_seconds,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "truncation_retries": truncation_retries,
        "truncation_retry_successes": truncation_retry_successes,
        "frontier_batches": frontier_batches,
        "frontier_batch_seconds": frontier_batch_seconds,
        "avg_seconds_per_frontier_batch": safe_div(
            frontier_batch_seconds,
            frontier_batches,
        ),
    }


def print_results(
    score: dict[str, Any],
    *,
    pipeline: int | None,
    tool_mode: str,
    earth: bool,
    input_coord_order: str,
    timing: dict[str, float] | None = None,
    live_counts: dict[str, int] | None = None,
    live_timings: dict[str, Any] | None = None,
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
    for cls in ANSWER_LABELS:
        n_cls = score["per_class_total"][cls]
        c_cls = score["per_class_correct"][cls]
        if n_cls > 0:
            print(f"    {cls:18s} {c_cls}/{n_cls} = {c_cls / n_cls:.1%}")
        else:
            print(f"    {cls:18s} 0/0 (no examples)")
    print("  Confusion matrix (rows=ground truth, columns=prediction):")
    print_confusion_matrix(
        score["confusion"],
        score["confusion_labels"],
        indent="    ",
    )
    print("  Per-n accuracy (n = number of points = len(geometries)):")
    for n_pts in sorted(score["per_n_total"].keys()):
        n_cnt = score["per_n_total"][n_pts]
        c_cnt = score["per_n_correct"].get(n_pts, 0)
        print(f"    n={n_pts:<3d}            {c_cnt}/{n_cnt} = {c_cnt / n_cnt:.1%}")
    print("  Per-n confusion matrices (rows=ground truth, columns=prediction):")
    for n_pts in sorted(score["per_n_total"].keys()):
        print(f"    n={n_pts}:")
        print_confusion_matrix(
            score["per_n_confusion"][n_pts],
            score["confusion_labels"],
            indent="      ",
        )

    if timing is not None:
        print("  Timing:")
        print(f"    timing/total_seconds: {timing.get('total_seconds', 0.0):.3f}")
        print(
            "    timing/avg_seconds_per_sample: "
            f"{timing.get('avg_seconds_per_sample', 0.0):.3f}"
        )
        print(
            "    timing/samples_per_second: "
            f"{timing.get('samples_per_second', 0.0):.3f}"
        )

    if live_counts is not None:
        print("  Live tool metrics:")
        for key in (
            "tool_parse_failures",
            "invalid_tool_calls",
            "unknown_tool_calls",
            "geocode_calls",
            "cyclic_order_calls",
            "valid_geocode_calls",
            "valid_cyclic_order_calls",
            "expected_cyclic_order_calls",
            "tool_error_returns",
            "truncation_retries",
            "truncation_retry_successes",
            "max_tool_turn_failures",
            "tool_failure_examples",
        ):
            print(f"    live/{key}: {live_counts.get(key, 0)}")
        avg_turns = live_turns_sum / total if total else 0.0
        avg_cyclic = live_cyclic_sum / total if total else 0.0
        avg_valid_cyclic = (
            live_counts.get("valid_cyclic_order_calls", 0) / total
            if total
            else 0.0
        )
        expected_cyclic = live_counts.get("expected_cyclic_order_calls", 0)
        valid_cyclic = live_counts.get("valid_cyclic_order_calls", 0)
        attempted_tool_calls = (
            live_counts.get("geocode_calls", 0)
            + live_counts.get("cyclic_order_calls", 0)
            + live_counts.get("unknown_tool_calls", 0)
        )
        print(f"    live/avg_turns: {avg_turns:.2f}")
        print(f"    live/avg_cyclic_order_calls_per_example: {avg_cyclic:.2f}")
        print(
            "    live/avg_valid_cyclic_order_calls_per_example: "
            f"{avg_valid_cyclic:.2f}"
        )
        print(
            "    live/valid_cyclic_order_call_ratio: "
            f"{safe_div(valid_cyclic, expected_cyclic):.1%}"
        )
        print(
            "    live/call_count_exact_match_rate: "
            f"{safe_div(live_counts.get('call_count_exact_match_examples', 0), total):.1%}"
        )
        print(
            "    live/geocode_once_rate: "
            f"{safe_div(live_counts.get('geocode_once_examples', 0), total):.1%}"
        )
        print(
            "    live/final_after_all_tools_rate: "
            f"{safe_div(live_counts.get('final_after_all_tools_examples', 0), total):.1%}"
        )
        print(
            "    live/invalid_tool_call_rate: "
            f"{safe_div(live_counts.get('invalid_tool_calls', 0), attempted_tool_calls):.1%}"
        )
        print(
            "    live/tool_parse_failure_rate: "
            f"{safe_div(live_counts.get('tool_parse_failures', 0), live_turns_sum):.1%}"
        )
    if live_timings is not None:
        print("  Live timing:")
        for key in (
            "total_turns",
            "total_tool_calls",
            "avg_seconds_per_example",
            "avg_seconds_per_turn",
            "avg_seconds_per_tool_call",
            "avg_generated_tokens_per_turn",
            "avg_prompt_tokens_per_turn",
            "avg_generation_seconds_per_turn",
            "avg_tool_execution_seconds_per_tool_call",
            "frontier_batches",
            "avg_seconds_per_frontier_batch",
            "truncation_retries",
            "truncation_retry_successes",
        ):
            value = live_timings.get(key, 0)
            if isinstance(value, float):
                print(f"    live/{key}: {value:.3f}")
            else:
                print(f"    live/{key}: {value}")
        avg_seconds_by_n = live_timings.get("avg_seconds_by_n", {})
        if avg_seconds_by_n:
            formatted = ", ".join(
                f"n={n}:{avg_seconds_by_n[n]:.3f}s"
                for n in sorted(avg_seconds_by_n)
            )
            print(f"    live/avg_seconds_by_n: {formatted}")
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
        "--live_max_new_tokens",
        type=int,
        default=None,
        help="Optional lower generation budget for live turns. If a live turn "
        "looks truncated, it is retried once with --max_new_tokens.",
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
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be positive")
    if args.live_max_new_tokens is not None and args.live_max_new_tokens < 1:
        parser.error("--live_max_new_tokens must be positive")

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
    live_max_new_tokens = (
        args.live_max_new_tokens
        if args.live_max_new_tokens is not None
        else args.max_new_tokens
    )
    prefix_marker: str | None = None
    suffix_marker: str | None = None
    stop_strings: list[str] | None = None
    if args.live_tools:
        marker_tools = get_eval_tools(records[0]) if records else [GEOCODE_SCHEMA, CYCLIC_ORDER_SCHEMA]
        discovered_markers = discover_tool_markers(tokenizer, marker_tools)
        if discovered_markers is not None:
            prefix_marker, suffix_marker = discovered_markers
            if suffix_marker:
                stop_strings = [suffix_marker]
            print("Live tool-call markers discovered from chat template.")
        else:
            prefix_marker, suffix_marker = "<tool_call>", "</tool_call>"
            stop_strings = [suffix_marker]
            print(
                "WARNING: chat template did not yield discoverable tool-call "
                "markers; falling back to Qwen-style </tool_call> stop string."
            )
        print(f"Live batch size: {args.batch_size}")
        print(f"Live max new tokens: {live_max_new_tokens}")
        print(f"Truncation retry cap: {args.max_new_tokens}")
        print(
            "Tool-call stop string: "
            f"{repr(stop_strings[0]) if stop_strings else 'unavailable'}"
        )

    score = new_score_state()
    predictions_f = open(args.save_predictions, "w") if args.save_predictions else None

    live_counts: dict[str, int] | None = None
    live_turns_sum = 0
    live_cyclic_sum = 0
    live_timings: dict[str, Any] | None = None
    if args.live_tools:
        live_counts = {
            **new_live_metrics(),
            "max_tool_turn_failures": 0,
            "tool_failure_examples": 0,
            "expected_cyclic_order_calls": 0,
            "call_count_exact_match_examples": 0,
            "geocode_once_examples": 0,
            "final_after_all_tools_examples": 0,
        }

    eval_start = time.perf_counter()
    try:
        if args.live_tools:
            live_results, frontier_stats = evaluate_records_live_batched(
                records,
                model,
                tokenizer,
                batch_size=args.batch_size,
                earth=args.earth,
                input_coord_order=input_coord_order,
                max_tool_turns=args.max_tool_turns,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=live_max_new_tokens,
                retry_max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                tool_error_policy=args.tool_error_policy,
                stop_strings=stop_strings,
                prefix_marker=prefix_marker,
                suffix_marker=suffix_marker,
            )
            live_elapsed = time.perf_counter() - eval_start
            live_timings = aggregate_live_timings(
                live_results,
                live_total_seconds=live_elapsed,
                frontier_stats=frontier_stats,
            )

            for record, live_result in zip(records, live_results):
                predicted = live_result["predicted"]
                ground_truth = compute_ground_truth(
                    record["meta"],
                    earth=args.earth,
                    input_coord_order=input_coord_order,
                )
                n_pts = len(record["meta"]["geometries"])
                update_score(score, predicted=predicted, ground_truth=ground_truth, n_pts=n_pts)

                assert live_counts is not None
                expected_cyclic_calls = max(n_pts - 2, 0)
                valid_geocode_calls = int(
                    live_result["metrics"].get("valid_geocode_calls", 0)
                )
                valid_cyclic_calls = int(
                    live_result["metrics"].get("valid_cyclic_order_calls", 0)
                )
                live_counts["expected_cyclic_order_calls"] += expected_cyclic_calls
                if valid_cyclic_calls == expected_cyclic_calls:
                    live_counts["call_count_exact_match_examples"] += 1
                if valid_geocode_calls == 1:
                    live_counts["geocode_once_examples"] += 1
                if (
                    live_result["status"] in {"ok", "parse_fail"}
                    and valid_geocode_calls == 1
                    and valid_cyclic_calls == expected_cyclic_calls
                ):
                    live_counts["final_after_all_tools_examples"] += 1

                for key, value in live_result["metrics"].items():
                    live_counts[key] = live_counts.get(key, 0) + value
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
                ).to(get_model_input_device(model))

                with torch.no_grad():
                    gen_kwargs = {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": args.temperature > 0,
                        "pad_token_id": tokenizer.pad_token_id,
                    }
                    if args.temperature > 0:
                        gen_kwargs["temperature"] = args.temperature

                    cuda_synchronize_if_available()
                    outputs = model.generate(**inputs, **gen_kwargs)
                    cuda_synchronize_if_available()

                for j, (output, record) in enumerate(zip(outputs, batch)):
                    prompt_len = inputs["input_ids"].shape[1]
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

    eval_total_seconds = time.perf_counter() - eval_start
    timing = {
        "total_seconds": eval_total_seconds,
        "avg_seconds_per_sample": safe_div(eval_total_seconds, len(records)),
        "samples_per_second": safe_div(len(records), eval_total_seconds),
    }

    print_results(
        score,
        pipeline=args.pipeline,
        tool_mode=tool_mode,
        earth=args.earth,
        input_coord_order=input_coord_order,
        timing=timing,
        live_counts=live_counts,
        live_timings=live_timings,
        live_turns_sum=live_turns_sum,
        live_cyclic_sum=live_cyclic_sum,
    )


if __name__ == "__main__":
    main()
