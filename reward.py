"""
Reward functions for GRPO training.

The cyclic-ordering problem with `n >= 3` points is fully determined by the
n-2 pairwise cross products around the center. The model's answer is one of
'clockwise', 'counterclockwise', or 'neither' (mixed). Ground truth is the
`answer` field stored in the meta dict (already computed during dataset
generation); a fallback recomputation from `meta["geometries"]` is provided
for backwards compatibility.
"""

from __future__ import annotations

import re

from tools import compute_cyclic_order, representative_point


_VALID_ANSWERS = ("clockwise", "counterclockwise", "neither")


def extract_answer(text: str) -> str | None:
    """
    Extract 'clockwise', 'counterclockwise', or 'neither' from model output.
    The last keyword to appear in the text wins (so a final answer overrides
    intermediate reasoning). Returns None if none are found.
    """
    text_lower = text.lower()

    ccw_positions = [m.start() for m in re.finditer(r"\bcounterclockwise\b", text_lower)]
    ccw_last = ccw_positions[-1] if ccw_positions else -1

    cw_only_positions = []
    for m in re.finditer(r"\bclockwise\b", text_lower):
        start = m.start()
        prefix = text_lower[max(0, start - 7) : start]
        if "counter" not in prefix:
            cw_only_positions.append(start)
    cw_last = cw_only_positions[-1] if cw_only_positions else -1

    neither_positions = [m.start() for m in re.finditer(r"\bneither\b", text_lower)]
    neither_last = neither_positions[-1] if neither_positions else -1

    candidates = [
        ("counterclockwise", ccw_last),
        ("clockwise", cw_last),
        ("neither", neither_last),
    ]
    candidates = [c for c in candidates if c[1] >= 0]
    if not candidates:
        return None

    return max(candidates, key=lambda c: c[1])[0]


def _combine_arcs(arc_labels: list[str]) -> str:
    if not arc_labels:
        raise ValueError("Need at least one pairwise arc to combine.")
    if all(a == "clockwise" for a in arc_labels):
        return "clockwise"
    if all(a == "counterclockwise" for a in arc_labels):
        return "counterclockwise"
    return "neither"


def compute_ground_truth(meta: dict) -> str:
    """
    Return the ground-truth answer for a record's metadata.

    Prefers `meta["answer"]` (written by `prepare_data.py`). Falls back to
    recomputing from `meta["geometries"]` (index 0 = center, rest in order)
    if `answer` is missing.
    """
    stored = meta.get("answer")
    if stored is not None:
        stored = str(stored).strip().lower()
        if stored in _VALID_ANSWERS:
            return stored

    geometries = meta["geometries"]
    if len(geometries) < 3:
        raise ValueError(
            f"meta['geometries'] must have length >= 3, got {len(geometries)}"
        )

    pts = [representative_point(g) for g in geometries]
    center = pts[0]
    arcs = [
        compute_cyclic_order(center, pts[i], pts[i + 1])
        for i in range(1, len(pts) - 1)
    ]
    return _combine_arcs(arcs)


def correctness_reward(completion: str, meta: dict) -> float:
    """
    Binary reward: 1.0 if the model's answer matches ground truth, 0.0 otherwise.
    """
    predicted = extract_answer(completion)
    if predicted is None:
        return 0.0

    ground_truth = compute_ground_truth(meta)
    return 1.0 if predicted == ground_truth else 0.0


def format_reward(completion: str, **kwargs) -> float:
    """
    Small reward for following the expected format.
    0.25 if the model produced a parseable answer (even if wrong).
    """
    predicted = extract_answer(completion)
    return 0.25 if predicted is not None else 0.0


def combined_reward(completion: str, meta: dict) -> float:
    """
    Combined reward = correctness (0 or 1) + format bonus (0 or 0.25).
    Max reward = 1.25, which gives a slight bonus for correct + well-formatted.
    """
    return correctness_reward(completion, meta) + format_reward(completion)
