"""
Local interactive test bench for the geographic cyclic-reasoning model.

Run with:
    uv run python chat.py --adapter checkpoints/rl-2 --pipeline 2

Two tabs:
  - Geometries: add / view / delete points, lines, polygons. Persisted to
    geometries.json so they survive restarts.
  - Chat: ask the model questions. The model's emitted tool calls are
    actually dispatched against the local geometry store, and shown live in
    a side panel.

The tool-call parser is *not* hardcoded for any model: at startup it asks
the loaded tokenizer to render a sentinel tool_call via apply_chat_template,
diffs that against a content-only render, and reads off the prefix/suffix
delimiters the chat template uses. That makes the parsing portable across
model families that have tool-call-aware chat templates.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from typing import Any, Generator

import gradio as gr
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from tools import (
    CYCLIC_ORDER_SCHEMA,
    GEOCODE_SCHEMA,
    compute_cyclic_order,
    representative_point,
)

# --------------------------------------------------------------------------- #
# Constants and small helpers
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent
GEOMETRIES_PATH = PROJECT_ROOT / "geometries.json"
os.environ["GRADIO_TEMP_DIR"] = str(PROJECT_ROOT / "gradio_tmp")
os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)

VALID_TYPES = ("point", "line", "polygon")


def _system_prompt(pipeline: int) -> str:
    """Mirrors prepare_data._system_prompt — kept inline so importing this
    module doesn't pull pandas/numpy in via prepare_data."""
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
    return base + (
        " Then use the cyclic_order tool to determine whether the "
        "arrangement is clockwise or counterclockwise."
    )


def _tool_list(pipeline: int) -> list[dict]:
    if pipeline == 1:
        return [GEOCODE_SCHEMA]
    return [GEOCODE_SCHEMA, CYCLIC_ORDER_SCHEMA]


# --------------------------------------------------------------------------- #
# Geometry store
# --------------------------------------------------------------------------- #


def load_store() -> dict[str, dict]:
    if not GEOMETRIES_PATH.exists():
        return {}
    try:
        with open(GEOMETRIES_PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, OSError):
        return {}


def save_store(store: dict[str, dict]) -> None:
    tmp = GEOMETRIES_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(store, f, indent=2)
    os.replace(tmp, GEOMETRIES_PATH)


def parse_coords(text: str) -> list[list[float]]:
    """Parse a coordinate string into a list of [x, y] pairs.

    Accepts:
      - JSON: [[1, 2], [3, 4]]
      - One pair per line: "1, 2\\n3, 4" (commas optional)
      - Semicolon-separated: "1 2; 3 4"
    """
    text = text.strip()
    if not text:
        raise ValueError("Coordinates are empty.")

    try:
        parsed = json.loads(text)
        if (
            isinstance(parsed, list)
            and parsed
            and isinstance(parsed[0], (list, tuple))
        ):
            return [[float(p[0]), float(p[1])] for p in parsed]
        if (
            isinstance(parsed, list)
            and len(parsed) == 2
            and all(isinstance(v, (int, float)) for v in parsed)
        ):
            return [[float(parsed[0]), float(parsed[1])]]
    except (json.JSONDecodeError, ValueError, TypeError, IndexError):
        pass

    pairs: list[list[float]] = []
    chunks = [c.strip() for c in text.replace(";", "\n").splitlines() if c.strip()]
    for chunk in chunks:
        cleaned = chunk.replace(",", " ").split()
        if len(cleaned) != 2:
            raise ValueError(
                f"Could not parse coordinate pair from {chunk!r} "
                f"(expected two numbers)."
            )
        try:
            pairs.append([float(cleaned[0]), float(cleaned[1])])
        except ValueError as e:
            raise ValueError(f"Non-numeric coordinate in {chunk!r}: {e}") from e
    if not pairs:
        raise ValueError("No coordinate pairs found.")
    return pairs


def infer_type(coords: list[list[float]]) -> str:
    n = len(coords)
    if n == 1:
        return "point"
    if n == 2:
        return "line"
    return "polygon"


def add_geometry(
    store: dict[str, dict],
    name: str,
    gtype: str,
    coords: list[list[float]],
) -> None:
    name = name.strip()
    if not name:
        raise ValueError("Name must be non-empty.")
    if name in store:
        raise ValueError(
            f"A geometry called {name!r} already exists. Delete it first."
        )

    if gtype == "auto":
        gtype = infer_type(coords)
    if gtype not in VALID_TYPES:
        raise ValueError(f"Type must be one of {VALID_TYPES} (got {gtype!r}).")

    if gtype == "point" and len(coords) != 1:
        raise ValueError(f"A point needs exactly 1 coordinate (got {len(coords)}).")
    if gtype == "line" and len(coords) < 2:
        raise ValueError(f"A line needs at least 2 coordinates (got {len(coords)}).")
    if gtype == "polygon" and len(coords) < 3:
        raise ValueError(f"A polygon needs at least 3 coordinates (got {len(coords)}).")

    store[name] = {"type": gtype, "coordinates": coords}
    save_store(store)


def delete_geometry(store: dict[str, dict], name: str) -> None:
    if name not in store:
        raise ValueError(f"No geometry named {name!r}.")
    del store[name]
    save_store(store)


def format_store(store: dict[str, dict]) -> str:
    if not store:
        return "_(no geometries yet — add some on the left)_"
    rows = ["| Name | Type | # pts | Centroid (x, y) |", "|---|---|---|---|"]
    for name, geom in store.items():
        try:
            cx, cy = representative_point(geom)
            centroid = f"({cx:.4g}, {cy:.4g})"
        except Exception as e:
            centroid = f"_error: {e}_"
        rows.append(
            f"| `{name}` | {geom.get('type', '?')} "
            f"| {len(geom.get('coordinates', []))} | {centroid} |"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Live tool dispatch (the real implementations the model's tool calls hit)
# --------------------------------------------------------------------------- #


def tool_geocode(place_names: list[str], store: dict[str, dict]) -> dict:
    out: dict[str, dict] = {}
    for name in place_names:
        if name in store:
            x, y = representative_point(store[name])
            out[name] = {"longitude": round(x, 6), "latitude": round(y, 6)}
            continue
        # Case-insensitive fallback
        match = next((k for k in store if k.lower() == name.lower()), None)
        if match is not None:
            x, y = representative_point(store[match])
            out[name] = {"longitude": round(x, 6), "latitude": round(y, 6)}
        else:
            out[name] = {"error": f"unknown place {name!r}"}
    return out


def tool_cyclic_order(
    center: list[float], point_b: list[float], point_c: list[float]
) -> dict:
    return {
        "result": compute_cyclic_order(
            (float(center[0]), float(center[1])),
            (float(point_b[0]), float(point_b[1])),
            (float(point_c[0]), float(point_c[1])),
        )
    }


def dispatch_tool(name: str, arguments: dict, store: dict[str, dict]) -> dict:
    if name == "geocode":
        return tool_geocode(arguments.get("place_names", []), store)
    if name == "cyclic_order":
        return tool_cyclic_order(
            arguments["center"], arguments["point_b"], arguments["point_c"]
        )
    raise ValueError(f"Unknown tool {name!r}")


# --------------------------------------------------------------------------- #
# Tool-call marker auto-discovery (chat-template-derived, not regex-hardcoded)
# --------------------------------------------------------------------------- #


def _match_braces(text: str, start: int) -> int:
    """Given an opening '{' at text[start], return index just past the
    matching '}'. Returns -1 on imbalance. Honours strings + escapes."""
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
    """Render a sentinel tool_call message via apply_chat_template, diff it
    against a content-only render to extract the prefix/suffix the chat
    template wraps tool-call JSON with. Returns (prefix, suffix), or None
    if the template can't be probed."""
    SENTINEL_TOOL = "ZZPROBETOOL"
    SENTINEL_TXT = "ZZPROBETXT"

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
                        "name": SENTINEL_TOOL,
                        "arguments": json.dumps({"k": 1}),
                    },
                }
            ],
        },
    ]
    probe_text = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": SENTINEL_TXT},
    ]

    def _render(msgs):
        try:
            return tokenizer.apply_chat_template(
                msgs, tools=tools, tokenize=False, add_generation_prompt=False
            )
        except (TypeError, Exception):
            try:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                return None

    rendered_tool = _render(probe_tool)
    rendered_text = _render(probe_text)
    if rendered_tool is None or rendered_text is None:
        return None
    if SENTINEL_TOOL not in rendered_tool or SENTINEL_TXT not in rendered_text:
        return None

    # Common prefix
    pre_len = 0
    n = min(len(rendered_tool), len(rendered_text))
    while pre_len < n and rendered_tool[pre_len] == rendered_text[pre_len]:
        pre_len += 1

    # Common suffix
    suf_len = 0
    while (
        suf_len < len(rendered_tool) - pre_len
        and suf_len < len(rendered_text) - pre_len
        and rendered_tool[-1 - suf_len] == rendered_text[-1 - suf_len]
    ):
        suf_len += 1

    diverged = rendered_tool[pre_len : len(rendered_tool) - suf_len]
    sent_pos = diverged.find(SENTINEL_TOOL)
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


# --------------------------------------------------------------------------- #
# Tool-call extraction from generated text
# --------------------------------------------------------------------------- #


def extract_tool_calls(
    text: str, prefix: str, suffix: str
) -> tuple[list[dict], str]:
    """Walk through `text` looking for `prefix` … JSON … `suffix` blocks.
    Returns (tool_calls, narration_text). `tool_calls` is a list of
    {"name": str, "arguments": dict}. `narration_text` is the assistant's
    plain-language content with the tool-call blocks removed."""
    p_strip = prefix.strip() or prefix
    s_strip = suffix.strip() or suffix

    tool_calls: list[dict] = []
    keep: list[str] = []
    pos = 0

    while pos < len(text):
        p_idx = text.find(p_strip, pos)
        if p_idx == -1:
            keep.append(text[pos:])
            break
        keep.append(text[pos:p_idx])

        # Skip past prefix (and any whitespace) to find opening brace
        scan = p_idx + len(p_strip)
        while scan < len(text) and text[scan] in " \t\n\r":
            scan += 1
        if scan >= len(text) or text[scan] != "{":
            # Looked like a prefix but no JSON follows — keep raw and move on.
            keep.append(text[p_idx : scan + 1])
            pos = scan + 1
            continue

        json_start = scan
        json_end = _match_braces(text, json_start)
        if json_end == -1:
            # Unclosed JSON — keep the rest verbatim and stop.
            keep.append(text[p_idx:])
            break

        json_str = text[json_start:json_end]
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            keep.append(text[p_idx:json_end])
            pos = json_end
            continue

        # Advance past the suffix marker if present.
        s_idx = text.find(s_strip, json_end) if s_strip else -1
        advance_to = (
            s_idx + len(s_strip)
            if s_idx != -1 and s_idx - json_end < 32
            else json_end
        )

        # Validate the JSON looks like a tool call.
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append({"name": obj["name"], "arguments": args})
        else:
            keep.append(text[p_idx:advance_to])

        pos = advance_to

    narration = "".join(keep).strip()
    return tool_calls, narration


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def load_model_and_tokenizer(
    base_model: str, adapter: str | None, use_4bit: bool
) -> tuple[Any, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        adapter or base_model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Streaming generation + tool-call loop
# --------------------------------------------------------------------------- #


def render_prompt(messages: list[dict], tools: list[dict], tokenizer) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
    except TypeError:
        msgs = [dict(m) for m in messages]
        if tools and msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] += "\n\nAvailable tools:\n" + json.dumps(
                tools, indent=2
            )
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )


def stream_one_turn(
    messages: list[dict],
    tools: list[dict],
    model,
    tokenizer,
    suffix_marker: str | None,
    max_new_tokens: int = 512,
) -> Generator[str, None, str]:
    """Yield incrementally accumulated assistant text. Returns the final
    full text once generation completes (via StopIteration value)."""
    prompt = render_prompt(messages, tools, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }
    stop_strings = [s for s in [suffix_marker] if s and s.strip()]
    if stop_strings:
        gen_kwargs["stop_strings"] = stop_strings
        gen_kwargs["tokenizer"] = tokenizer

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    accumulated = ""
    for chunk in streamer:
        accumulated += chunk
        yield accumulated
    thread.join()
    return accumulated


def run_conversation(
    user_message: str,
    history_messages: list[dict],
    tools: list[dict],
    model,
    tokenizer,
    prefix_marker: str,
    suffix_marker: str,
    store: dict[str, dict],
    max_iters: int = 6,
) -> Generator[tuple[list[dict], list[dict], list[dict]], None, None]:
    """Drive one user query through the tool-call loop.

    Yields (messages, chat_display, tool_log_display) tuples after every
    streamed chunk and every tool dispatch.
    """
    messages = history_messages + [{"role": "user", "content": user_message}]
    chat_display: list[dict] = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ""},
    ]
    tool_log: list[dict] = []
    yield messages, chat_display, tool_log

    for _ in range(max_iters):
        gen = stream_one_turn(messages, tools, model, tokenizer, suffix_marker)
        full_text = ""
        try:
            while True:
                full_text = next(gen)
                chat_display[-1]["content"] = full_text
                yield messages, chat_display, tool_log
        except StopIteration as stop:
            full_text = stop.value or full_text

        tool_calls, narration = extract_tool_calls(
            full_text, prefix_marker, suffix_marker
        )

        if not tool_calls:
            # Final assistant turn.
            messages = messages + [{"role": "assistant", "content": full_text}]
            chat_display[-1]["content"] = full_text
            yield messages, chat_display, tool_log
            return

        # Persist the assistant turn that issued the tool calls.
        assistant_msg: dict = {
            "role": "assistant",
            "content": narration or "",
            "tool_calls": [
                {
                    "id": f"call_{len(messages)}_{i}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for i, tc in enumerate(tool_calls)
            ],
        }
        messages = messages + [assistant_msg]

        for i, tc in enumerate(tool_calls):
            tool_log.append(
                {
                    "role": "user",
                    "content": (
                        f"**call** `{tc['name']}`\n"
                        f"```json\n{json.dumps(tc['arguments'], indent=2)}\n```"
                    ),
                }
            )
            yield messages, chat_display, tool_log

            try:
                result = dispatch_tool(tc["name"], tc["arguments"], store)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}

            messages = messages + [
                {
                    "role": "tool",
                    "name": tc["name"],
                    "tool_call_id": assistant_msg["tool_calls"][i]["id"],
                    "content": json.dumps(result),
                }
            ]
            tool_log.append(
                {
                    "role": "assistant",
                    "content": (
                        f"**result** `{tc['name']}`\n"
                        f"```json\n{json.dumps(result, indent=2)}\n```"
                    ),
                }
            )
            yield messages, chat_display, tool_log

        # Add a fresh empty assistant slot for the next streamed turn.
        chat_display.append({"role": "assistant", "content": ""})
        yield messages, chat_display, tool_log

    # Hit max_iters without a final answer.
    chat_display[-1]["content"] = (
        chat_display[-1]["content"]
        + "\n\n_(stopped: tool-call loop hit max iterations)_"
    )
    yield messages, chat_display, tool_log


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #


def build_ui(model, tokenizer, pipeline: int) -> gr.Blocks:
    tools = _tool_list(pipeline)
    sys_prompt = _system_prompt(pipeline)

    markers = discover_tool_markers(tokenizer, tools)
    if markers is None:
        print(
            "[chat] WARNING: chat template did not yield discoverable tool-call "
            "markers; falling back to <tool_call>/</tool_call>."
        )
        prefix_marker, suffix_marker = "<tool_call>", "</tool_call>"
    else:
        prefix_marker, suffix_marker = markers
        print(
            "[chat] Auto-discovered tool-call markers from chat template:\n"
            f"  prefix = {prefix_marker!r}\n"
            f"  suffix = {suffix_marker!r}"
        )

    with gr.Blocks(title="Geo Cyclic Reasoning — Test Bench") as demo:
        gr.Markdown(
            "# Geographic Cyclic-Reasoning Test Bench\n"
            f"Pipeline **{pipeline}** · geometries persisted to `geometries.json`"
        )

        # ------------------------- Geometries tab ------------------------- #
        with gr.Tab("Geometries"):
            with gr.Row():
                with gr.Column(scale=1):
                    name_in = gr.Textbox(label="Name", placeholder="e.g. Paris")
                    type_in = gr.Radio(
                        ["auto", "point", "line", "polygon"],
                        value="auto",
                        label="Type",
                        info="'auto' picks point/line/polygon by point count.",
                    )
                    coords_in = gr.Textbox(
                        label="Coordinates",
                        lines=6,
                        placeholder=(
                            "JSON, one pair per line, or semicolon-separated:\n"
                            "[[2.35, 48.85]]\n"
                            "or\n"
                            "0, 0\n2, 0\n2, 2\n0, 2"
                        ),
                    )
                    add_btn = gr.Button("Add geometry", variant="primary")
                    add_status = gr.Markdown()
                with gr.Column(scale=2):
                    geom_table = gr.Markdown(value=format_store(load_store()))
                    delete_dd = gr.Dropdown(
                        label="Delete",
                        choices=list(load_store().keys()),
                        value=None,
                    )
                    with gr.Row():
                        delete_btn = gr.Button("Delete selected")
                        refresh_btn = gr.Button("Refresh")

            def _on_add(name, gtype, coords_text):
                store = load_store()
                try:
                    coords = parse_coords(coords_text or "")
                    add_geometry(store, name, gtype, coords)
                    msg = f"Added `{name}` ({store[name]['type']})."
                except Exception as e:
                    msg = f"**Error:** {e}"
                store = load_store()
                return (
                    msg,
                    format_store(store),
                    gr.update(choices=list(store.keys()), value=None),
                )

            def _on_delete(name):
                store = load_store()
                if not name:
                    msg = "_(select a geometry first)_"
                else:
                    try:
                        delete_geometry(store, name)
                        msg = f"Deleted `{name}`."
                    except Exception as e:
                        msg = f"**Error:** {e}"
                store = load_store()
                return (
                    msg,
                    format_store(store),
                    gr.update(choices=list(store.keys()), value=None),
                )

            def _on_refresh():
                store = load_store()
                return (
                    format_store(store),
                    gr.update(choices=list(store.keys())),
                )

            add_btn.click(
                _on_add,
                inputs=[name_in, type_in, coords_in],
                outputs=[add_status, geom_table, delete_dd],
            )
            delete_btn.click(
                _on_delete,
                inputs=[delete_dd],
                outputs=[add_status, geom_table, delete_dd],
            )
            refresh_btn.click(
                _on_refresh, inputs=[], outputs=[geom_table, delete_dd]
            )

        # --------------------------- Chat tab ----------------------------- #
        with gr.Tab("Chat"):
            messages_state = gr.State(
                [{"role": "system", "content": sys_prompt}]
            )
            tool_log_state = gr.State([])

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Assistant",
                        height=500,
                    )
                with gr.Column(scale=2):
                    tool_log = gr.Chatbot(
                        label="Tool calls (live)",
                        height=500,
                    )

            with gr.Row():
                user_in = gr.Textbox(
                    label="Your question",
                    placeholder=(
                        "e.g. From Paris, are London and Berlin in clockwise "
                        "or counterclockwise order?"
                    ),
                    lines=2,
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear conversation")

            def _messages_to_display(messages: list[dict]) -> list[dict]:
                return [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages
                    if m["role"] in ("user", "assistant") and m.get("content")
                ]

            def _on_send(user_text, messages, prior_tool_log):
                if not user_text or not user_text.strip():
                    yield messages, gr.update(), prior_tool_log, ""
                    return
                store = load_store()
                prior_display = _messages_to_display(messages)

                final_messages = messages
                round_chat: list[dict] = []
                round_tools: list[dict] = []
                for new_messages, round_chat, round_tools in run_conversation(
                    user_text,
                    messages,
                    tools,
                    model,
                    tokenizer,
                    prefix_marker,
                    suffix_marker,
                    store,
                ):
                    final_messages = new_messages
                    yield (
                        new_messages,
                        prior_display + round_chat,
                        prior_tool_log + round_tools,
                        "",
                    )

                yield (
                    final_messages,
                    _messages_to_display(final_messages),
                    prior_tool_log + round_tools,
                    "",
                )

            def _on_clear():
                return (
                    [{"role": "system", "content": sys_prompt}],
                    [],
                    [],
                    [],
                    "",
                )

            send_btn.click(
                _on_send,
                inputs=[user_in, messages_state, tool_log_state],
                outputs=[messages_state, chatbot, tool_log, user_in],
            ).then(
                lambda log: log,
                inputs=[tool_log],
                outputs=[tool_log_state],
            )
            user_in.submit(
                _on_send,
                inputs=[user_in, messages_state, tool_log_state],
                outputs=[messages_state, chatbot, tool_log, user_in],
            ).then(
                lambda log: log,
                inputs=[tool_log],
                outputs=[tool_log_state],
            )
            clear_btn.click(
                _on_clear,
                inputs=[],
                outputs=[messages_state, tool_log_state, chatbot, tool_log, user_in],
            )

    return demo


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model id or path. If you pass an already-merged checkpoint here, "
        "leave --adapter unset.",
    )
    p.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT adapter directory (e.g. checkpoints/rl-2).",
    )
    p.add_argument(
        "--pipeline",
        type=int,
        default=2,
        choices=[1, 2],
        help="1: model reasons internally after geocode. 2: model also calls "
        "cyclic_order. Default 2 (more visible tool calls).",
    )
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Expose a Gradio share URL.")
    p.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantisation (uses bf16 — needs much more VRAM).",
    )
    args = p.parse_args()

    print(f"[chat] Loading base model: {args.base_model}")
    if args.adapter:
        print(f"[chat] With LoRA adapter: {args.adapter}")
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.adapter, use_4bit=not args.no_4bit
    )
    print(f"[chat] Model loaded on device: {model.device}")
    print(f"[chat] Geometries file: {GEOMETRIES_PATH}")

    demo = build_ui(model, tokenizer, args.pipeline)
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
