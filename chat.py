"""
Local interactive test bench for the geographic cyclic-reasoning model.

Run with:
    uv run python chat.py --adapter checkpoints/rl-2 --pipeline 2

Two tabs:
  - Geometries: add / view / delete points, lines, polygons. Coordinate order
    follows the Earth toggle ([latitude, longitude] when on, [longitude,
    latitude] / x-y when off); geometries persist to geometries.json across
    restarts (the raw stored numbers are never rewritten — only re-interpreted).
  - Chat: ask the model questions. The model's emitted tool calls are actually
    dispatched against the local geometry store and shown live in a side panel.
    An "Earth (spherical) backend" checkbox (present and kept in sync on both
    tabs) toggles cyclic_order between the planar cross-product and the spherical
    Earth determinant (and the matching system prompt / coordinate order)
    without restarting.

This mirrors evaluate.py's live tool loop: tool calls are parsed with the
shared tool_call_parsing.extract_tool_calls_from_completion, and the system
prompt / tool schemas come from prepare_data so the bench always matches what
the models were trained on (including multi-arc reasoning that issues one
cyclic_order call per consecutive waypoint pair). The tokenizer's chat template
is probed once at startup only to derive the per-turn generation stop string.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import dataclass
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

from prepare_data import _system_prompt, _tool_list
from tool_call_parsing import (
    decode_tool_arguments,
    discover_tool_markers,
    extract_tool_calls_from_completion,
    parse_lonlat_pair,
)
from tools import (
    DEFAULT_GEOCODE_USER_AGENT,
    compute_cyclic_order,
    compute_cyclic_order_earth,
    geocode_nominatim,
    representative_point_lonlat,
)

# --------------------------------------------------------------------------- #
# Constants and small helpers
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent
GEOMETRIES_PATH = PROJECT_ROOT / "geometries.json"
os.environ["GRADIO_TEMP_DIR"] = str(PROJECT_ROOT / "gradio_tmp")
os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)

VALID_TYPES = ("point", "line", "polygon")


@dataclass
class GeocodeConfig:
    """Settings for the live OpenStreetMap (Nominatim) geocode tool.

    Present (non-None) only when the user passes ``--geocode``; otherwise the
    geocode tool resolves names from the local store alone.
    """

    candidates: int = 5
    countrycodes: str | None = None
    viewbox: str | None = None
    user_agent: str = DEFAULT_GEOCODE_USER_AGENT
    cache_writeback: bool = True

# The system prompt (`_system_prompt`) and tool schemas (`_tool_list`) are
# imported from prepare_data so this test bench always matches what the models
# were trained on — including the multi-arc / "neither" instructions and the
# earth=... variant.


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


def format_store(store: dict[str, dict], input_coord_order: str = "latlon") -> str:
    if not store:
        return "_(no geometries yet — add some on the left)_"
    # Show the model-facing representative point (longitude, latitude),
    # regardless of the order coordinates were entered/stored in.
    rows = ["| Name | Type | # pts | Centroid (lon, lat) |", "|---|---|---|---|"]
    for name, geom in store.items():
        try:
            lon, lat = representative_point_lonlat(
                geom, input_coord_order=input_coord_order
            )
            centroid = f"({lon:.4g}, {lat:.4g})"
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


def tool_geocode(
    place_names: list[str],
    store: dict[str, dict],
    *,
    input_coord_order: str,
) -> dict:
    out: dict[str, dict] = {}
    for name in place_names:
        match = name if name in store else next(
            (k for k in store if k.lower() == name.lower()), None
        )
        if match is None:
            out[name] = {"error": f"unknown place {name!r}"}
            continue
        lon, lat = representative_point_lonlat(
            store[match], input_coord_order=input_coord_order
        )
        out[name] = {"longitude": round(lon, 6), "latitude": round(lat, 6)}
    return out


def _coords_in_store_order(
    lon: float, lat: float, input_coord_order: str
) -> list[float]:
    """Order a model-facing (lon, lat) pair to match how the store is read."""
    return [lat, lon] if input_coord_order == "latlon" else [lon, lat]


def tool_geocode_live(
    place_names: list[str],
    store: dict[str, dict],
    *,
    input_coord_order: str,
    config: GeocodeConfig,
) -> dict:
    """Live geocode via Nominatim, with the local store as an override + cache.

    Names already in the store (exact / case-insensitive) resolve from it
    directly and also seed the proximity disambiguation as fixed anchors, so the
    remaining names cluster around them. Newly resolved names are (optionally)
    written back into the store as point geometries so repeat lookups are cached
    and the user can correct them from the Geometries tab.
    """
    out: dict[str, dict] = {}
    fixed_points: dict[str, tuple[float, float]] = {}
    misses: list[str] = []

    for name in place_names:
        match = name if name in store else next(
            (k for k in store if k.lower() == name.lower()), None
        )
        if match is None:
            misses.append(name)
            continue
        lon, lat = representative_point_lonlat(
            store[match], input_coord_order=input_coord_order
        )
        out[name] = {"longitude": round(lon, 6), "latitude": round(lat, 6)}
        fixed_points[name] = (lon, lat)

    if misses:
        resolved = geocode_nominatim(
            misses,
            candidates=config.candidates,
            countrycodes=config.countrycodes,
            viewbox=config.viewbox,
            fixed_points=fixed_points,
            user_agent=config.user_agent,
        )
        for name, res in resolved.items():
            out[name] = res
            if config.cache_writeback and "error" not in res and name not in store:
                try:
                    coords = _coords_in_store_order(
                        res["longitude"], res["latitude"], input_coord_order
                    )
                    add_geometry(store, name, "point", [coords])
                except Exception:
                    pass  # caching is best-effort; never break a geocode call
    return out


def tool_cyclic_order(arguments: dict, *, earth: bool) -> dict:
    center = parse_lonlat_pair(arguments["center"], "center")
    point_b = parse_lonlat_pair(arguments["point_b"], "point_b")
    point_c = parse_lonlat_pair(arguments["point_c"], "point_c")
    cyclic_order_fn = compute_cyclic_order_earth if earth else compute_cyclic_order
    return {"result": cyclic_order_fn(center, point_b, point_c)}


def dispatch_tool(
    name: str,
    arguments: dict,
    store: dict[str, dict],
    *,
    earth: bool,
    input_coord_order: str,
    geocode_config: GeocodeConfig | None = None,
) -> dict:
    if name == "geocode":
        place_names = arguments.get("place_names", [])
        if geocode_config is not None:
            return tool_geocode_live(
                place_names,
                store,
                input_coord_order=input_coord_order,
                config=geocode_config,
            )
        return tool_geocode(
            place_names,
            store,
            input_coord_order=input_coord_order,
        )
    if name == "cyclic_order":
        return tool_cyclic_order(arguments, earth=earth)
    raise ValueError(f"Unknown tool {name!r}")


# Tool-call marker discovery (`discover_tool_markers`) and tool-call extraction
# (`extract_tool_calls_from_completion`) are imported from tool_call_parsing so
# this test bench parses model output exactly like evaluate.py does. Markers are
# discovered once at startup purely to derive the generation stop string.


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
    max_new_tokens: int = 1024,
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


def _call_name(call: dict) -> str:
    function = call.get("function") if isinstance(call, dict) else None
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return "unknown"


def run_conversation(
    user_message: str,
    history_messages: list[dict],
    tools: list[dict],
    model,
    tokenizer,
    suffix_marker: str | None,
    store: dict[str, dict],
    *,
    earth: bool,
    input_coord_order: str,
    geocode_config: GeocodeConfig | None = None,
    max_tool_turns: int = 12,
    max_new_tokens: int = 1024,
) -> Generator[tuple[list[dict], list[dict], list[dict]], None, None]:
    """Drive one user query through the tool-call loop.

    Yields (messages, chat_display, tool_log_display) tuples after every
    streamed chunk and every tool dispatch. Tool calls are parsed with the
    shared `extract_tool_calls_from_completion` so this matches evaluate.py.
    """
    messages = history_messages + [{"role": "user", "content": user_message}]
    chat_display: list[dict] = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ""},
    ]
    tool_log: list[dict] = []
    yield messages, chat_display, tool_log

    for _ in range(max_tool_turns):
        gen = stream_one_turn(
            messages, tools, model, tokenizer, suffix_marker, max_new_tokens
        )
        full_text = ""
        try:
            while True:
                full_text = next(gen)
                chat_display[-1]["content"] = full_text
                yield messages, chat_display, tool_log
        except StopIteration as stop:
            full_text = stop.value or full_text

        parsed = extract_tool_calls_from_completion(full_text)

        if not parsed.tool_calls:
            # Final assistant turn (no further tool calls).
            messages = messages + [{"role": "assistant", "content": full_text}]
            chat_display[-1]["content"] = full_text
            yield messages, chat_display, tool_log
            return

        # Persist the assistant turn that issued the tool calls. parsed.tool_calls
        # are already in OpenAI function-call format, so the chat template can
        # re-render them on the next turn.
        assistant_msg: dict = {"role": "assistant", "tool_calls": parsed.tool_calls}
        if parsed.content:
            assistant_msg["content"] = parsed.content
        messages = messages + [assistant_msg]

        # Show narration (not raw <tool_call> markup) in the main chat bubble.
        chat_display[-1]["content"] = parsed.content or full_text

        for call in parsed.tool_calls:
            name = _call_name(call)
            try:
                arguments = decode_tool_arguments(call["function"]["arguments"])
            except Exception as e:
                arguments = {}
                arg_error = f"{type(e).__name__}: {e}"
            else:
                arg_error = None

            tool_log.append(
                {
                    "role": "user",
                    "content": (
                        f"**call** `{name}`\n"
                        f"```json\n{json.dumps(arguments, indent=2)}\n```"
                    ),
                }
            )
            yield messages, chat_display, tool_log

            if arg_error is not None:
                result: dict = {"error": f"could not decode arguments: {arg_error}"}
            else:
                try:
                    result = dispatch_tool(
                        name,
                        arguments,
                        store,
                        earth=earth,
                        input_coord_order=input_coord_order,
                        geocode_config=geocode_config,
                    )
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}

            messages = messages + [
                {
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(result),
                }
            ]
            tool_log.append(
                {
                    "role": "assistant",
                    "content": (
                        f"**result** `{name}`\n"
                        f"```json\n{json.dumps(result, indent=2)}\n```"
                    ),
                }
            )
            yield messages, chat_display, tool_log

        # Add a fresh empty assistant slot for the next streamed turn.
        chat_display.append({"role": "assistant", "content": ""})
        yield messages, chat_display, tool_log

    # Hit max_tool_turns without a final answer.
    chat_display[-1]["content"] = (
        chat_display[-1]["content"]
        + "\n\n_(stopped: tool-call loop hit max tool turns)_"
    )
    yield messages, chat_display, tool_log


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #


def build_ui(
    model,
    tokenizer,
    pipeline: int,
    *,
    earth_default: bool,
    max_tool_turns: int,
    max_new_tokens: int,
    geocode_config: GeocodeConfig | None = None,
) -> gr.Blocks:
    tools = _tool_list(pipeline)

    def sys_prompt_for(earth: bool) -> str:
        return _system_prompt(pipeline, earth=earth)

    # Coordinate order is driven by the Earth toggle: Earth on reads geometries
    # as [latitude, longitude] (geographic); Earth off reads them as
    # [longitude, latitude] (x-y / planar).
    def coord_order_for(earth: bool) -> str:
        return "latlon" if earth else "lonlat"

    def coords_label(order: str) -> str:
        return f"Coordinates ([{order}] order)"

    def coords_placeholder(order: str) -> str:
        paris = "48.85, 2.35" if order == "latlon" else "2.35, 48.85"
        return (
            "JSON, one pair per line, or semicolon-separated.\n"
            f"Enter each pair as [{order}], e.g. Paris:\n"
            f"[[{paris}]]\n"
            "or a polygon, one vertex per line:\n"
            "0, 0\n2, 0\n2, 2\n0, 2"
        )

    # Markers are discovered once, purely to derive the generation stop string
    # (so each turn halts at the end of a tool call). Parsing of the generated
    # text is done by the shared extract_tool_calls_from_completion.
    markers = discover_tool_markers(tokenizer, tools)
    if markers is None:
        print(
            "[chat] WARNING: chat template did not yield discoverable tool-call "
            "markers; falling back to </tool_call> stop string."
        )
        suffix_marker = "</tool_call>"
    else:
        _prefix_marker, suffix_marker = markers
        print(
            "[chat] Auto-discovered tool-call markers from chat template:\n"
            f"  prefix = {_prefix_marker!r}\n"
            f"  suffix = {suffix_marker!r}"
        )

    if geocode_config is not None:
        _region = geocode_config.countrycodes or geocode_config.viewbox
        geocode_note = (
            " · **live OSM geocoding** on (store = override/cache"
            + (f", region `{_region}`" if _region else "")
            + ")"
        )
    else:
        geocode_note = ""

    with gr.Blocks(title="Geo Cyclic Reasoning — Test Bench") as demo:
        gr.Markdown(
            "# Geographic Cyclic-Reasoning Test Bench\n"
            f"Pipeline **{pipeline}** · coordinate order follows the **Earth** "
            "toggle (**[lat, lon]** when on, **[lon, lat]** / x–y when off) "
            "· geometries persisted to `geometries.json`"
            + geocode_note
        )

        # ------------------------- Geometries tab ------------------------- #
        _initial_order = coord_order_for(earth_default)
        with gr.Tab("Geometries"):
            with gr.Row():
                earth_geom = gr.Checkbox(
                    label="Earth (spherical) backend",
                    value=earth_default,
                    info=(
                        "Controls how coordinates below are read: on → "
                        "[latitude, longitude]; off → [longitude, latitude] "
                        "(x-y). Stays in sync with the Chat tab."
                    ),
                )
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
                        label=coords_label(_initial_order),
                        lines=6,
                        placeholder=coords_placeholder(_initial_order),
                    )
                    add_btn = gr.Button("Add geometry", variant="primary")
                    add_status = gr.Markdown()
                with gr.Column(scale=2):
                    geom_table = gr.Markdown(
                        value=format_store(load_store(), _initial_order)
                    )
                    delete_dd = gr.Dropdown(
                        label="Delete",
                        choices=list(load_store().keys()),
                        value=None,
                    )
                    with gr.Row():
                        delete_btn = gr.Button("Delete selected")
                        refresh_btn = gr.Button("Refresh")

            def _on_add(name, gtype, coords_text, earth):
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
                    format_store(store, coord_order_for(bool(earth))),
                    gr.update(choices=list(store.keys()), value=None),
                )

            def _on_delete(name, earth):
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
                    format_store(store, coord_order_for(bool(earth))),
                    gr.update(choices=list(store.keys()), value=None),
                )

            def _on_refresh(earth):
                store = load_store()
                return (
                    format_store(store, coord_order_for(bool(earth))),
                    gr.update(choices=list(store.keys())),
                )

            add_btn.click(
                _on_add,
                inputs=[name_in, type_in, coords_in, earth_geom],
                outputs=[add_status, geom_table, delete_dd],
            )
            delete_btn.click(
                _on_delete,
                inputs=[delete_dd, earth_geom],
                outputs=[add_status, geom_table, delete_dd],
            )
            refresh_btn.click(
                _on_refresh, inputs=[earth_geom], outputs=[geom_table, delete_dd]
            )

        # --------------------------- Chat tab ----------------------------- #
        with gr.Tab("Chat"):
            messages_state = gr.State(
                [{"role": "system", "content": sys_prompt_for(earth_default)}]
            )
            tool_log_state = gr.State([])

            with gr.Row():
                earth_chat = gr.Checkbox(
                    label="Earth (spherical) backend",
                    value=earth_default,
                    info=(
                        "On: cyclic_order uses the spherical Earth determinant "
                        "(and, in pipeline 1, the Earth system prompt) and "
                        "coordinates are read as [lat, lon]. Off: the planar "
                        "cross-product backend with [lon, lat] (x-y). Stays in "
                        "sync with the Geometries tab."
                    ),
                )

            with gr.Row():
                with gr.Column(scale=3):
                    # Gradio 6 Chatbot is messages-only ({"role","content"} dicts);
                    # the legacy `type`/"tuples" format was removed.
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

            def _on_send(user_text, messages, prior_tool_log, earth):
                if not user_text or not user_text.strip():
                    yield messages, gr.update(), prior_tool_log, ""
                    return
                store = load_store()
                # Keep the system prompt in sync with the current Earth toggle.
                messages = [
                    {"role": "system", "content": sys_prompt_for(bool(earth))}
                ] + list(messages[1:])
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
                    suffix_marker,
                    store,
                    earth=bool(earth),
                    input_coord_order=coord_order_for(bool(earth)),
                    geocode_config=geocode_config,
                    max_tool_turns=max_tool_turns,
                    max_new_tokens=max_new_tokens,
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

            def _on_clear(earth):
                return (
                    [{"role": "system", "content": sys_prompt_for(bool(earth))}],
                    [],
                    [],
                    [],
                    "",
                )

            send_btn.click(
                _on_send,
                inputs=[user_in, messages_state, tool_log_state, earth_chat],
                outputs=[messages_state, chatbot, tool_log, user_in],
            ).then(
                lambda log: log,
                inputs=[tool_log],
                outputs=[tool_log_state],
            )
            user_in.submit(
                _on_send,
                inputs=[user_in, messages_state, tool_log_state, earth_chat],
                outputs=[messages_state, chatbot, tool_log, user_in],
            ).then(
                lambda log: log,
                inputs=[tool_log],
                outputs=[tool_log_state],
            )
            clear_btn.click(
                _on_clear,
                inputs=[earth_chat],
                outputs=[messages_state, tool_log_state, chatbot, tool_log, user_in],
            )

        # ---- Keep the two Earth checkboxes (and the coordinate order they
        # imply) in sync. Wired with .input() so it fires only on real user
        # clicks, not on the programmatic mirror update — avoiding a feedback
        # loop. Both update the shared geometry table + coordinate input field.
        def _on_earth_toggle(earth):
            order = coord_order_for(bool(earth))
            return (
                bool(earth),
                format_store(load_store(), order),
                gr.update(
                    label=coords_label(order),
                    placeholder=coords_placeholder(order),
                ),
            )

        earth_chat.input(
            _on_earth_toggle,
            inputs=[earth_chat],
            outputs=[earth_geom, geom_table, coords_in],
        )
        earth_geom.input(
            _on_earth_toggle,
            inputs=[earth_geom],
            outputs=[earth_chat, geom_table, coords_in],
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
    p.add_argument(
        "--earth",
        action="store_true",
        help="Start with the Earth (spherical) backend toggle enabled. It can be "
        "toggled live from either tab. Earth on reads coordinates as [lat, lon]; "
        "off reads them as [lon, lat] (x-y).",
    )
    p.add_argument(
        "--no_earth",
        dest="earth",
        action="store_false",
        help="Start with the Earth backend toggle disabled (planar).",
    )
    p.set_defaults(earth=True)
    p.add_argument(
        "--max_tool_turns",
        type=int,
        default=12,
        help="Maximum assistant/tool iterations per query (handles n-2 cyclic "
        "calls for large n). Default 12.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Generation budget per assistant turn. Default 1024.",
    )
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Expose a Gradio share URL.")
    p.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantisation (uses bf16 — needs much more VRAM).",
    )
    p.add_argument(
        "--geocode",
        action="store_true",
        help="Enable live OpenStreetMap (Nominatim) geocoding for the geocode "
        "tool instead of looking up the local store only. Ambiguous names are "
        "disambiguated by geographic proximity to the other places in the same "
        "call; the local store still acts as an override + cache.",
    )
    p.add_argument(
        "--geocode_countrycodes",
        default=None,
        help="Comma-separated ISO country codes to bias/restrict live geocoding, "
        "e.g. 'us,ca'. (Nominatim countrycodes=.)",
    )
    p.add_argument(
        "--geocode_viewbox",
        default=None,
        help="Bounding box 'min_lon,min_lat,max_lon,max_lat' to restrict live "
        "geocoding to a region (implies bounded). (Nominatim viewbox=.)",
    )
    p.add_argument(
        "--geocode_candidates",
        type=int,
        default=5,
        help="Top-K Nominatim candidates per name considered during "
        "disambiguation. Default 5.",
    )
    p.add_argument(
        "--geocode_user_agent",
        default=None,
        help="Override the Nominatim User-Agent header (its usage policy "
        f"requires a real one). Default {DEFAULT_GEOCODE_USER_AGENT!r}.",
    )
    p.add_argument(
        "--no_geocode_cache",
        dest="geocode_cache",
        action="store_false",
        help="Do not write live-geocoded points back into geometries.json.",
    )
    p.set_defaults(geocode_cache=True)
    args = p.parse_args()

    print(f"[chat] Loading base model: {args.base_model}")
    if args.adapter:
        print(f"[chat] With LoRA adapter: {args.adapter}")
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.adapter, use_4bit=not args.no_4bit
    )
    print(f"[chat] Model loaded on device: {model.device}")
    print(f"[chat] Geometries file: {GEOMETRIES_PATH}")
    print(
        f"[chat] Pipeline {args.pipeline} · earth default "
        f"{'on (coords [lat, lon])' if args.earth else 'off (coords [lon, lat])'}"
    )

    geocode_config = None
    if args.geocode:
        geocode_config = GeocodeConfig(
            candidates=args.geocode_candidates,
            countrycodes=args.geocode_countrycodes,
            viewbox=args.geocode_viewbox,
            user_agent=args.geocode_user_agent or DEFAULT_GEOCODE_USER_AGENT,
            cache_writeback=args.geocode_cache,
        )
        region = args.geocode_countrycodes or args.geocode_viewbox or "none"
        print(
            f"[chat] Live OSM geocoding ON · candidates={geocode_config.candidates} "
            f"· region={region} · cache_writeback={geocode_config.cache_writeback}"
        )

    demo = build_ui(
        model,
        tokenizer,
        args.pipeline,
        earth_default=args.earth,
        max_tool_turns=args.max_tool_turns,
        max_new_tokens=args.max_new_tokens,
        geocode_config=geocode_config,
    )
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
