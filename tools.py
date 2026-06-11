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

import json
import math
import time
import urllib.parse
import urllib.request
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


def lonlat_to_unit_vector(point: tuple[float, float]) -> tuple[float, float, float]:
    """
    Convert a [longitude, latitude] point in degrees to an Earth-centered unit vector.
    """
    lon_deg, lat_deg = point
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    cos_lat = math.cos(lat)
    return (
        cos_lat * math.cos(lon),
        cos_lat * math.sin(lon),
        math.sin(lat),
    )


def _cross3(
    u: tuple[float, float, float],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _dot3(
    u: tuple[float, float, float],
    v: tuple[float, float, float],
) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def compute_cyclic_order_earth(
    center: tuple[float, float],
    point_b: tuple[float, float],
    point_c: tuple[float, float],
    *,
    eps: float = 1e-12,
) -> str:
    """
    Compute spherical cyclic order using det([A, B, C]) on unit Earth vectors.

    Arguments use the same model-facing convention as cyclic_order:
    [longitude, latitude]. The sign convention matches compute_cyclic_order near
    the equator: east-to-north around [0, 0] is counterclockwise.
    """
    a = lonlat_to_unit_vector(center)
    b = lonlat_to_unit_vector(point_b)
    c = lonlat_to_unit_vector(point_c)
    det = _dot3(a, _cross3(b, c))
    if det > eps:
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


def representative_point_lonlat(
    geometry: dict,
    input_coord_order: str = "lonlat",
) -> tuple[float, float]:
    """
    Extract a representative point and normalize it to [longitude, latitude].
    """
    raw_x, raw_y = representative_point(geometry)
    if input_coord_order == "lonlat":
        return (raw_x, raw_y)
    if input_coord_order == "latlon":
        lat, lon = raw_x, raw_y
        return (lon, lat)
    raise ValueError(f"Unknown input_coord_order={input_coord_order!r}")


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


# ---------------------------------------------------------------------------
# Live geocoding (Nominatim / OpenStreetMap)
# ---------------------------------------------------------------------------
#
# A real implementation of `geocode` for inference/interactive use. Nominatim can
# return several candidates for one name (Paris, France vs Paris, Texas), but the
# tool must return exactly one coordinate per name. We disambiguate by geographic
# context: because a cyclic-order question's places cluster in one region, we pick
# the candidate per name that keeps the whole set mutually close.

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_GEOCODE_USER_AGENT = "geo-finetune-testbench/0.1"

# Nominatim's usage policy asks for at most one request per second.
_MIN_REQUEST_INTERVAL = 1.0
_last_request_time = 0.0


def _respect_rate_limit() -> None:
    global _last_request_time
    wait = _MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()


def _nominatim_search(
    query: str,
    *,
    candidates: int,
    countrycodes: str | None,
    viewbox: str | None,
    user_agent: str,
    timeout: float,
) -> list[dict[str, Any]]:
    """Query Nominatim for one place name; return a list of candidate dicts
    (``lon``, ``lat``, ``importance``, ``display_name``), highest importance
    first (the order Nominatim already returns)."""
    params = {"q": query, "format": "json", "limit": str(max(1, candidates))}
    if countrycodes:
        params["countrycodes"] = countrycodes
    if viewbox:
        params["viewbox"] = viewbox
        params["bounded"] = "1"
    url = NOMINATIM_URL + "?" + urllib.parse.urlencode(params)

    _respect_rate_limit()
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)

    out: list[dict[str, Any]] = []
    for item in data:
        try:
            lon = float(item["lon"])
            lat = float(item["lat"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(
            {
                "lon": lon,
                "lat": lat,
                "importance": float(item.get("importance", 0.0) or 0.0),
                "display_name": item.get("display_name", ""),
            }
        )
    return out


def _centroid_unit(
    vecs: list[tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    sx = sum(v[0] for v in vecs)
    sy = sum(v[1] for v in vecs)
    sz = sum(v[2] for v in vecs)
    norm = math.sqrt(sx * sx + sy * sy + sz * sz)
    if norm < 1e-12:
        return None
    return (sx / norm, sy / norm, sz / norm)


def _angular_distance(
    u: tuple[float, float, float],
    v: tuple[float, float, float],
) -> float:
    return math.acos(max(-1.0, min(1.0, _dot3(u, v))))


def _cluster_spread(vecs: list[tuple[float, float, float]]) -> float:
    """Sum of angular distances from each unit vector to the cluster centroid.
    Lower = geographically tighter. ``inf`` for a degenerate (antipodal) cluster."""
    centroid = _centroid_unit(vecs)
    if centroid is None:
        return float("inf")
    return sum(_angular_distance(v, centroid) for v in vecs)


def _greedy_assign(
    multi: dict[str, list[dict[str, Any]]],
    base_vecs: list[tuple[float, float, float]],
    seed: tuple[str, dict[str, Any]] | None,
) -> tuple[dict[str, dict[str, Any]], list[tuple[float, float, float]]]:
    """Greedily assign one candidate per name in ``multi``, growing a cluster from
    ``base_vecs`` (fixed anchors) plus an optional ``seed`` (name, candidate). At
    each step pick the (name, candidate) closest to the running centroid (ties by
    higher importance, then display name). Returns (assignment, all unit vectors).
    """
    assignment: dict[str, dict[str, Any]] = {}
    vecs = list(base_vecs)
    remaining = dict(multi)
    if seed is not None:
        sname, scand = seed
        assignment[sname] = scand
        vecs.append(lonlat_to_unit_vector((scand["lon"], scand["lat"])))
        remaining.pop(sname, None)

    while remaining:
        centroid = _centroid_unit(vecs)
        best_key = None
        best_name = None
        best_cand = None
        for name, cands in remaining.items():
            for cand in cands:
                vec = lonlat_to_unit_vector((cand["lon"], cand["lat"]))
                if centroid is None:  # nothing to be near yet: go by importance
                    key = (0.0, -cand["importance"], cand["display_name"])
                else:
                    key = (
                        _angular_distance(vec, centroid),
                        -cand["importance"],
                        cand["display_name"],
                    )
                if best_key is None or key < best_key:
                    best_key, best_name, best_cand = key, name, cand
        assignment[best_name] = best_cand
        vecs.append(
            lonlat_to_unit_vector((best_cand["lon"], best_cand["lat"]))
        )
        del remaining[best_name]

    return assignment, vecs


def _disambiguate_by_proximity(
    candidates_per_name: dict[str, list[dict[str, Any]]],
    fixed_points: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float] | None]:
    """Pick exactly one candidate per name so the chosen set is geographically
    coherent.

    ``fixed_points`` and any single-candidate names anchor the cluster. For the
    ambiguous (multi-candidate) names we run a **multi-start greedy**: try
    starting the cluster from each candidate, then keep the assignment whose
    final cluster is tightest (ties broken toward higher total importance). This
    prevents one famous-but-wrong candidate (e.g. Paris, France in a Texan
    question) from anchoring the whole set in the wrong region. Names with no
    candidates resolve to ``None``.
    """
    chosen: dict[str, tuple[float, float] | None] = {}
    base_vecs: list[tuple[float, float, float]] = []

    if fixed_points:
        for lon, lat in fixed_points.values():
            base_vecs.append(lonlat_to_unit_vector((lon, lat)))

    multi: dict[str, list[dict[str, Any]]] = {}
    for name, cands in candidates_per_name.items():
        if not cands:
            chosen[name] = None
        elif len(cands) == 1:
            coord = (cands[0]["lon"], cands[0]["lat"])
            chosen[name] = coord
            base_vecs.append(lonlat_to_unit_vector(coord))
        else:
            multi[name] = cands

    if not multi:
        return chosen

    starts: list[tuple[str, dict[str, Any]] | None] = []
    if base_vecs:
        starts.append(None)  # also try growing purely from the fixed anchors
    starts.extend(
        (name, cand) for name, cands in multi.items() for cand in cands
    )

    best_score = None
    best_assignment: dict[str, dict[str, Any]] = {}
    for seed in starts:
        assignment, vecs = _greedy_assign(multi, base_vecs, seed)
        total_importance = sum(c["importance"] for c in assignment.values())
        score = (_cluster_spread(vecs), -total_importance)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment

    for name, cand in best_assignment.items():
        chosen[name] = (cand["lon"], cand["lat"])
    return chosen


def geocode_nominatim(
    place_names: list[str],
    *,
    candidates: int = 5,
    countrycodes: str | None = None,
    viewbox: str | None = None,
    fixed_points: dict[str, tuple[float, float]] | None = None,
    user_agent: str = DEFAULT_GEOCODE_USER_AGENT,
    timeout: float = 10.0,
) -> dict[str, dict[str, float | str]]:
    """Geocode a list of names via Nominatim, returning exactly one coordinate
    per name as ``{name: {"longitude": x, "latitude": y}}`` (or
    ``{name: {"error": ...}}`` for names that fail or have no match).

    ``fixed_points`` are already-known ``name -> (lon, lat)`` anchors (e.g. names
    resolved from a local store) that seed the proximity cluster but are not
    re-resolved or returned. Never raises on a single bad name — failures are
    reported per-name so the model can react.
    """
    candidates_per_name: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, dict[str, float | str]] = {}

    for name in place_names:
        query = name.strip()
        if not query:
            errors[name] = {"error": "empty place name"}
            continue
        try:
            candidates_per_name[name] = _nominatim_search(
                query,
                candidates=candidates,
                countrycodes=countrycodes,
                viewbox=viewbox,
                user_agent=user_agent,
                timeout=timeout,
            )
        except Exception as e:  # network / HTTP / JSON errors
            errors[name] = {"error": f"geocoding failed: {type(e).__name__}: {e}"}

    chosen = _disambiguate_by_proximity(candidates_per_name, fixed_points)

    result: dict[str, dict[str, float | str]] = {}
    for name in place_names:
        if name in errors:
            result[name] = errors[name]
            continue
        coord = chosen.get(name)
        if coord is None:
            result[name] = {"error": f"no geocoding result for {name!r}"}
        else:
            lon, lat = coord
            result[name] = {"longitude": round(lon, 6), "latitude": round(lat, 6)}
    return result
