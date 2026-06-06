"""Shared parsing helpers for assistant tool-call completions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedAssistantTurn:
    content: str
    tool_calls: list[dict]
    parse_error: str | None = None


class ToolCallParseError(ValueError):
    """Raised when tool-call arguments are malformed."""


def parse_lonlat_pair(value: Any, arg_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ToolCallParseError(
            f"{arg_name} must be a two-number [longitude, latitude] pair"
        )

    lon, lat = value
    if (
        isinstance(lon, bool)
        or isinstance(lat, bool)
        or not isinstance(lon, (int, float))
        or not isinstance(lat, (int, float))
    ):
        raise ToolCallParseError(
            f"{arg_name} must contain numeric longitude/latitude values"
        )
    return (float(lon), float(lat))


def decode_tool_arguments(raw_arguments: Any) -> dict:
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(
                f"Invalid JSON tool arguments: {exc.msg}"
            ) from exc
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        raise ToolCallParseError(
            "Tool arguments must be a JSON object or JSON object string"
        )

    if not isinstance(arguments, dict):
        raise ToolCallParseError("Tool arguments must decode to a JSON object")
    return arguments


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


def _match_braces(text: str, start: int) -> int:
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
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def discover_tool_markers(tokenizer, tools: list[dict]) -> tuple[str, str] | None:
    """Infer the tokenizer's tool-call wrapper from its chat template."""
    sentinel_tool = "ZZPROBETOOL"
    sentinel_text = "ZZPROBETXT"

    probe_tool = [
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "probe0",
                    "type": "function",
                    "function": {
                        "name": sentinel_tool,
                        "arguments": json.dumps({"k": 1}),
                    },
                }
            ],
        },
    ]
    probe_text = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": sentinel_text},
    ]

    def render(msgs: list[dict]) -> str | None:
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            try:
                return tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                return None

    rendered_tool = render(probe_tool)
    rendered_text = render(probe_text)
    if rendered_tool is None or rendered_text is None:
        return None
    if sentinel_tool not in rendered_tool or sentinel_text not in rendered_text:
        return None

    pre_len = 0
    n = min(len(rendered_tool), len(rendered_text))
    while pre_len < n and rendered_tool[pre_len] == rendered_text[pre_len]:
        pre_len += 1

    suf_len = 0
    while (
        suf_len < len(rendered_tool) - pre_len
        and suf_len < len(rendered_text) - pre_len
        and rendered_tool[-1 - suf_len] == rendered_text[-1 - suf_len]
    ):
        suf_len += 1

    diverged = rendered_tool[pre_len : len(rendered_tool) - suf_len]
    sent_pos = diverged.find(sentinel_tool)
    if sent_pos == -1:
        return None
    json_start = diverged.rfind("{", 0, sent_pos)
    if json_start == -1:
        return None
    json_end = _match_braces(diverged, json_start)
    if json_end == -1:
        return None

    prefix = diverged[:json_start]
    suffix = diverged[json_end:]
    if not prefix and not suffix:
        return None
    return prefix, suffix


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
        starts = [
            pos
            for pos in (stripped.find("{", idx), stripped.find("[", idx))
            if pos >= 0
        ]
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
            error = (
                f"{parse_errors} malformed tool call block(s)"
                if parse_errors
                else None
            )
            return ParsedAssistantTurn(
                content=content,
                tool_calls=calls,
                parse_error=error,
            )
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
