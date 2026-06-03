"""
Convert the geographic-questions parquet file into JSONL training data
formatted as multi-turn conversations with tool calls.

Expected parquet columns:
  - question_id (int)
  - question (str)
  - location_names (sequence of str, length n)
  - geometries (sequence of dicts with "type" and "coordinates" keys, length n)
      • index 0 is the CENTER
      • indices 1..n-1 are the path waypoints in order
  - answer (str): one of "clockwise", "counterclockwise", "neither"

Each row becomes a conversation. Let n = len(geometries). The path has
n-1 waypoints and n-2 pairwise arcs.

Pipeline 1 (internal computation, no cyclic_order tool):
  user      → question
  assistant → tool_call: geocode(place_names)
  tool      → coordinates
  assistant → reasoning trace covering all n-2 cross products + final answer

Pipeline 2 (tool-assisted computation, one cyclic_order call per assistant turn):
  user      → question
  assistant → tool_call: geocode(place_names)
  tool      → coordinates
  (repeat n-2 times:)
    assistant → tool_call: cyclic_order(center, B_i, B_{i+1})
    tool      → "clockwise" / "counterclockwise"
  assistant → final answer combining the n-2 pairwise results

The train/val split is taken from the input files (no internal shuffling/
splitting); each input parquet that is provided becomes a single output JSONL.

Usage:
    python prepare_data.py \
        --val_earth_input    ./data/parquet/earth_cyclic_order.parquet \
        --output ./data/jsonl \
        --pipeline 2

The train, val_balanced, and val_natural inputs are synthetic planar data.
If --val_earth_input is provided, it is written as val_earth.jsonl using the
spherical Earth backend. Raw earth geometry coordinates default to
[latitude, longitude], while model-facing tool calls remain [longitude, latitude].
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from tools import (
    CYCLIC_ORDER_SCHEMA,
    GEOCODE_SCHEMA,
    compute_cyclic_order,
    compute_cyclic_order_earth,
    lonlat_to_unit_vector,
    representative_point_lonlat,
)


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


# ---------------------------------------------------------------------------
# Conversation builders
# ---------------------------------------------------------------------------


def _system_prompt(pipeline: int, *, earth: bool = False) -> str:
    """Return the system prompt for the given pipeline."""
    base = (
        "You are a geographic reasoning assistant. When asked about spatial "
        "relationships between places, first use the geocode tool to look up "
        "their coordinates."
    )
    if pipeline == 1:
        if earth:
            return base + (
                " Then reason about the longitude/latitude coordinates on a "
                "spherical Earth to determine the answer. For each consecutive "
                "pair of waypoints (B_i, B_{i+1}), use the sign of the 3x3 "
                "determinant of the unit Earth vectors for A, B_i, and "
                "B_{i+1}. The overall path is clockwise if every arc is "
                "clockwise, counterclockwise if every arc is counterclockwise, "
                "and 'neither' if the arcs disagree."
            )
        else:
            return base + (
                " Then reason about the coordinates to determine the answer. "
                "For each consecutive pair of waypoints (B_i, B_{i+1}), compute "
                "the cross product of the vectors from the center A. The sign "
                "tells you whether that arc is clockwise or counterclockwise. "
                "The overall path is clockwise if every arc is clockwise, "
                "counterclockwise if every arc is counterclockwise, and "
                "'neither' if the arcs disagree."
            )
    else:
        return base + (
            " Then call the cyclic_order tool once for each consecutive pair "
            "of waypoints around the center. Combine the per-arc results: "
            "if every arc is clockwise, the answer is clockwise; if every "
            "arc is counterclockwise, the answer is counterclockwise; "
            "otherwise the answer is 'neither'."
        )


def _tool_list(pipeline: int) -> list[dict]:
    if pipeline == 1:
        return [GEOCODE_SCHEMA]
    else:
        return [GEOCODE_SCHEMA, CYCLIC_ORDER_SCHEMA]


def _fmt_pt(pt: tuple[float, float], digits: int = 2) -> str:
    return f"({pt[0]:.{digits}f}, {pt[1]:.{digits}f})"


def build_geocode_result_for_order(
    location_names: list[str],
    geometries: list[dict],
    *,
    input_coord_order: str,
) -> dict[str, dict[str, float]]:
    """
    Build a geocode result, normalized to model-facing longitude/latitude.
    """
    result = {}
    for name, geom in zip(location_names, geometries):
        lon, lat = representative_point_lonlat(
            geom,
            input_coord_order=input_coord_order,
        )
        result[name] = {"longitude": round(lon, 6), "latitude": round(lat, 6)}
    return result


def _det3_from_lonlat(
    center: tuple[float, float],
    point_b: tuple[float, float],
    point_c: tuple[float, float],
) -> float:
    a = lonlat_to_unit_vector(center)
    b = lonlat_to_unit_vector(point_b)
    c = lonlat_to_unit_vector(point_c)
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _combine_arcs(arc_labels: list[str]) -> str:
    if all(a == "clockwise" for a in arc_labels):
        return "clockwise"
    if all(a == "counterclockwise" for a in arc_labels):
        return "counterclockwise"
    return "neither"


def _cyclic_order_fn(earth: bool):
    return compute_cyclic_order_earth if earth else compute_cyclic_order


def _representative_points_lonlat(
    geometries: list[dict],
    *,
    input_coord_order: str,
) -> list[tuple[float, float]]:
    return [
        representative_point_lonlat(g, input_coord_order=input_coord_order)
        for g in geometries
    ]


def derive_answer_from_geometries(
    geometries: list[dict],
    *,
    earth: bool,
    input_coord_order: str,
) -> str:
    pts = _representative_points_lonlat(
        geometries,
        input_coord_order=input_coord_order,
    )
    center_pt = pts[0]
    cyclic_order = _cyclic_order_fn(earth)
    arc_labels = [
        cyclic_order(center_pt, pts[i], pts[i + 1])
        for i in range(1, len(pts) - 1)
    ]
    return _combine_arcs(arc_labels)


def build_conversation_pipeline1(
    question: str,
    location_names: list[str],
    geometries: list[dict],
    answer: str,
    *,
    earth: bool = False,
    input_coord_order: str = "lonlat",
) -> list[dict]:
    """
    Build a multi-turn conversation for pipeline 1 (internal computation).

    geometries[0] is the center; geometries[1:] are the ordered waypoints.
    """
    n = len(geometries)
    assert n >= 3, f"Need at least 3 geometries (1 center + 2 waypoints), got {n}"

    geocode_result = build_geocode_result_for_order(
        location_names,
        geometries,
        input_coord_order=input_coord_order,
    )

    # Use unrounded points for math (avoids sign-flip rounding edge cases);
    # display with 2-decimal precision for readability.
    pts = _representative_points_lonlat(
        geometries,
        input_coord_order=input_coord_order,
    )
    center_name = location_names[0]
    center_pt = pts[0]
    cyclic_order = _cyclic_order_fn(earth)

    if earth:
        lines = ["Let me compute this from the Earth coordinates.\n"]
    else:
        lines = ["Let me compute this from the coordinates.\n"]
    lines.append(f"Center (A): {center_name} at {_fmt_pt(center_pt)}")
    for i in range(1, n):
        lines.append(
            f"Waypoint B{i}: {location_names[i]} at {_fmt_pt(pts[i])}"
        )
    lines.append("")

    arc_labels: list[str] = []
    for i in range(1, n - 1):
        b_pt = pts[i]
        c_pt = pts[i + 1]
        arc = cyclic_order(center_pt, b_pt, c_pt)
        arc_labels.append(arc)
        if earth:
            det = _det3_from_lonlat(center_pt, b_pt, c_pt)
            lines.append(
                f"Arc B{i}→B{i+1} ({location_names[i]} → {location_names[i+1]}): "
                f"3x3 Earth determinant = {det:.6g} → {arc}."
            )
        else:
            bx, by = b_pt[0] - center_pt[0], b_pt[1] - center_pt[1]
            cx, cy = c_pt[0] - center_pt[0], c_pt[1] - center_pt[1]
            det = bx * cy - by * cx
            lines.append(
                f"Arc B{i}→B{i+1} ({location_names[i]} → {location_names[i+1]}): "
                f"vector A→B{i} = ({bx:.2f}, {by:.2f}), "
                f"vector A→B{i+1} = ({cx:.2f}, {cy:.2f}), "
                f"cross = ({bx:.2f})({cy:.2f}) - ({by:.2f})({cx:.2f}) = {det:.2f} → {arc}."
            )

    lines.append("")
    combined = _combine_arcs(arc_labels)

    lines.append(
        f"All {len(arc_labels)} arcs combined: the path is **{combined}**."
    )

    reasoning = "\n".join(lines)

    messages = [
        {"role": "system", "content": _system_prompt(1, earth=earth)},
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
    return messages, combined


def build_conversation_pipeline2(
    question: str,
    location_names: list[str],
    geometries: list[dict],
    answer: str,
    *,
    earth: bool = False,
    input_coord_order: str = "lonlat",
) -> list[dict]:
    """
    Build a multi-turn conversation for pipeline 2 (tool-assisted computation).

    One cyclic_order tool call per assistant turn. Final assistant message
    combines the n-2 pairwise results.
    """
    n = len(geometries)
    assert n >= 3, f"Need at least 3 geometries (1 center + 2 waypoints), got {n}"

    geocode_result = build_geocode_result_for_order(
        location_names,
        geometries,
        input_coord_order=input_coord_order,
    )
    pts = _representative_points_lonlat(
        geometries,
        input_coord_order=input_coord_order,
    )
    center_pt = pts[0]
    center_name = location_names[0]
    cyclic_order = _cyclic_order_fn(earth)

    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(2, earth=earth)},
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
    ]

    arc_labels: list[str] = []
    for i in range(1, n - 1):
        b_pt = pts[i]
        c_pt = pts[i + 1]
        b_name = location_names[i]
        c_name = location_names[i + 1]
        arc = cyclic_order(center_pt, b_pt, c_pt)
        arc_labels.append(arc)

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
                    {
                        "type": "function",
                        "function": {
                            "name": "cyclic_order",
                            "arguments": json.dumps(
                                {
                                    "center": list(center_pt),
                                    "point_b": list(b_pt),
                                    "point_c": list(c_pt),
                                }
                            ),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": "cyclic_order",
                "content": json.dumps({"result": arc}),
            }
        )

    combined = _combine_arcs(arc_labels)

    arc_summary = ", ".join(arc_labels)
    messages.append(
        {
            "role": "assistant",
            "content": (
                f"The {len(arc_labels)} pairwise arcs around {center_name} are: "
                f"{arc_summary}. Since "
                + (
                    "they all agree"
                    if combined != "neither"
                    else "they disagree"
                )
                + f", the overall path is **{combined}**."
            ),
        }
    )

    return messages, combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


REQUIRED_COLS = {"question_id", "question", "location_names", "geometries", "answer"}


def build_records(
    df: pd.DataFrame,
    pipeline: str,
    *,
    earth: bool = False,
    input_coord_order: str = "lonlat",
) -> tuple[list[dict], int, int]:
    """
    Build conversation records from a parquet DataFrame.

    Returns (records, answer_mismatch_count, skipped_count).

    In --earth mode, output answers are regenerated from geometries with the
    spherical backend instead of trusting the parquet `answer`.
    """
    records: list[dict] = []
    answer_mismatches = 0
    skipped = 0

    for _, row in df.iterrows():
        question = row["question"]
        location_names = list(row["location_names"])
        geometries = [dict(g) if not isinstance(g, dict) else g for g in row["geometries"]]
        # `geometries` items may carry numpy arrays for "coordinates"; downstream
        # code tolerates that, and _to_native fixes it before JSON serialization.
        answer = str(row["answer"]).strip().lower()
        qid = int(row["question_id"])

        if len(geometries) < 3 or len(location_names) != len(geometries):
            skipped += 1
            continue

        derived_answer = derive_answer_from_geometries(
            geometries,
            earth=earth,
            input_coord_order=input_coord_order,
        )
        if derived_answer != answer:
            answer_mismatches += 1

        output_answer = derived_answer if earth else answer

        meta = {
            "location_names": location_names,
            "geometries": geometries,
            "answer": output_answer,
        }

        if pipeline in ("1", "both"):
            conv, derived = build_conversation_pipeline1(
                question,
                location_names,
                geometries,
                output_answer,
                earth=earth,
                input_coord_order=input_coord_order,
            )
            if derived != derived_answer:
                raise ValueError(
                    f"pipeline 1 derived {derived!r}, expected {derived_answer!r}"
                )
            records.append(
                {
                    "question_id": qid,
                    "pipeline": 1,
                    "tools": _tool_list(1),
                    "messages": conv,
                    "expected_answer": output_answer,
                    "meta": meta,
                }
            )

        if pipeline in ("2", "both"):
            conv, derived = build_conversation_pipeline2(
                question,
                location_names,
                geometries,
                output_answer,
                earth=earth,
                input_coord_order=input_coord_order,
            )
            if derived != derived_answer:
                raise ValueError(
                    f"pipeline 2 derived {derived!r}, expected {derived_answer!r}"
                )
            records.append(
                {
                    "question_id": qid,
                    "pipeline": 2,
                    "tools": _tool_list(2),
                    "messages": conv,
                    "expected_answer": output_answer,
                    "meta": meta,
                }
            )

    return records, answer_mismatches, skipped


def process_split(
    split_name: str,
    input_path: str,
    output_path: str,
    pipeline: str,
    *,
    earth: bool,
    input_coord_order: str,
) -> None:
    """Read one parquet, build records, write one JSONL."""
    df = pd.read_parquet(input_path)
    print(f"[{split_name}] Loaded {len(df)} rows from {input_path}")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"[{split_name}] parquet missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns)}."
        )

    records, answer_mismatches, skipped = build_records(
        df,
        pipeline,
        earth=earth,
        input_coord_order=input_coord_order,
    )
    print(f"[{split_name}] Generated {len(records)} conversations")
    if skipped:
        print(
            f"  WARNING: skipped {skipped} rows with malformed geometries or "
            f"mismatched location_names/geometries lengths."
        )
    if answer_mismatches:
        if earth:
            print(
                f"  WARNING: {answer_mismatches} stored `answer` values differed "
                f"from the earth backend; output records use the regenerated labels."
            )
        else:
            print(
                f"  WARNING: {answer_mismatches} stored `answer` values differed "
                f"from the planar backend; output records keep the stored labels."
            )
    elif earth:
        print(
            "  All stored `answer` values matched the regenerated earth labels."
        )

    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(_to_native(rec)) + "\n")
    print(f"[{split_name}] Wrote {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for geo fine-tuning")
    parser.add_argument(
        "--train_input",
        default=None,
        help="Optional path to the train parquet file",
    )
    parser.add_argument(
        "--val_balanced_input",
        default=None,
        help="Optional path to the balanced validation parquet file",
    )
    parser.add_argument(
        "--val_natural_input",
        default=None,
        help="Optional path to the natural validation parquet file",
    )
    parser.add_argument(
        "--val_earth_input",
        default=None,
        help=(
            "Optional path to the earth validation parquet file. This split is "
            "formatted with the spherical Earth backend and written to "
            "val_earth.jsonl."
        ),
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--pipeline",
        choices=["1", "2", "both"],
        default="both",
        help="Which pipeline(s) to generate data for",
    )
    parser.add_argument(
        "--val_earth_coord_order",
        choices=["lonlat", "latlon"],
        default="latlon",
        help=(
            "Coordinate order used inside --val_earth_input geometries. "
            "Defaults to latlon."
        ),
    )
    args = parser.parse_args()

    splits = []
    if args.train_input:
        splits.append(("train", args.train_input, "train.jsonl", False, "lonlat"))
    if args.val_balanced_input:
        splits.append(
            (
                "val_balanced",
                args.val_balanced_input,
                "val_balanced.jsonl",
                False,
                "lonlat",
            )
        )
    if args.val_natural_input:
        splits.append(
            (
                "val_natural",
                args.val_natural_input,
                "val_natural.jsonl",
                False,
                "lonlat",
            )
        )
    if args.val_earth_input:
        splits.append(
            (
                "val_earth",
                args.val_earth_input,
                "val_earth.jsonl",
                True,
                args.val_earth_coord_order,
            )
        )
    if not splits:
        parser.error(
            "At least one input must be provided: --train_input, "
            "--val_balanced_input, --val_natural_input, or --val_earth_input."
        )

    os.makedirs(args.output, exist_ok=True)

    for split_name, input_path, output_filename, earth, input_coord_order in splits:
        output_path = os.path.join(args.output, args.pipeline, output_filename)
        process_split(
            split_name,
            input_path,
            output_path,
            args.pipeline,
            earth=earth,
            input_coord_order=input_coord_order,
        )


if __name__ == "__main__":
    main()
