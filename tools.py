"""
Tool definitions for the geographic cyclic-ordering pipeline.

Two tools are defined:
  1. geocode  — looks up coordinates for a list of place names
  2. cyclic_order — computes whether B→C is clockwise or counterclockwise around A

During *training*, these tools are simulated using ground-truth data from the parquet
file. During *inference*, `geocode` would call a real geocoding API (Nominatim,
Google Geocoding, etc.) and `cyclic_order` performs the deterministic computation.
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# JSON-schema descriptions (used in system prompts / chat templates)
# ---------------------------------------------------------------------------

GEOCODE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "geocode",
        "description": (
            "Look up geographic coordinates for a list of place names. "
            "Returns a mapping from each name to its representative point "
            "(latitude, longitude)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "place_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of place names to geocode.",
                }
            },
            "required": ["place_names"],
        },
    },
}

CYCLIC_ORDER_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "cyclic_order",
        "description": (
            "Given a center point A and two other points B and C (each as "
            "[longitude, latitude]), determine whether the arc from B to C "
            "around A is clockwise or counterclockwise. Returns one of "
            "'clockwise' or 'counterclockwise'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "center": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Center point [longitude, latitude].",
                },
                "point_b": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "First peripheral point [longitude, latitude].",
                },
                "point_c": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Second peripheral point [longitude, latitude].",
                },
            },
            "required": ["center", "point_b", "point_c"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def compute_cyclic_order(
    center: tuple[float, float],
    point_b: tuple[float, float],
    point_c: tuple[float, float],
) -> str:
    """
    Compute sgn(det(B - A, C - A)).

    Convention (standard math / screen-with-Y-up):
      positive determinant  → counterclockwise
      negative determinant  → clockwise
      zero                  → collinear (we default to 'clockwise' to avoid ambiguity)

    Parameters are (x, y) — for geographic data, use (longitude, latitude) so that
    the x-axis points east and y-axis points north, matching the standard orientation.
    """
    bx, by = point_b[0] - center[0], point_b[1] - center[1]
    cx, cy = point_c[0] - center[0], point_c[1] - center[1]
    det = bx * cy - by * cx
    if det > 0:
        return "counterclockwise"
    else:
        return "clockwise"


def representative_point(geometry: dict) -> tuple[float, float]:
    """
    Extract a single representative (x, y) point from a geometry dict.

    Handles:
      - point:   returns the coordinate directly
      - line:    returns the midpoint
      - polygon: returns the centroid (simple average of vertices)
    """
    gtype = geometry["type"].lower()
    coords = geometry["coordinates"]
    coords = [[float(v) for v in c] for c in coords]

    if gtype == "point":
        return (coords[0][0], coords[0][1])
    elif gtype == "line":
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    elif gtype == "polygon":
        # Exclude closing vertex if it duplicates the first
        ring = coords
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        xs = [c[0] for c in ring]
        ys = [c[1] for c in ring]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    else:
        # Fallback: average all coordinates
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


# ---------------------------------------------------------------------------
# Helpers for building simulated tool-call turns
# ---------------------------------------------------------------------------


def build_geocode_result(
    location_names: list[str],
    geometries: list[dict],
) -> dict[str, dict[str, float]]:
    """
    Build the JSON object a geocode tool would return, using ground-truth geometries.
    """
    result = {}
    for name, geom in zip(location_names, geometries):
        x, y = representative_point(geom)
        result[name] = {"longitude": round(x, 6), "latitude": round(y, 6)}
    return result
