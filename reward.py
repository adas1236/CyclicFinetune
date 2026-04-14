"""
Reward functions for GRPO training.

The cyclic ordering problem has a perfectly verifiable reward:
given three points, sgn(det(B-A, C-A)) determines the answer
deterministically. This makes it ideal for outcome-based RL.
"""

from __future__ import annotations

import re

from tools import compute_cyclic_order, representative_point


def extract_answer(text: str) -> str | None:
    """
    Extract 'clockwise' or 'counterclockwise' from model output.
    Returns None if neither is found.
    """
    text_lower = text.lower()

    # Look for the last occurrence (the final answer, not intermediate reasoning)
    # Find all matches
    cw_positions = [m.start() for m in re.finditer(r"\bcounterclockwise\b", text_lower)]
    ccw_last = cw_positions[-1] if cw_positions else -1

    # "clockwise" that is NOT part of "counterclockwise"
    cw_only_positions = []
    for m in re.finditer(r"\bclockwise\b", text_lower):
        # Check it's not preceded by "counter"
        start = m.start()
        prefix = text_lower[max(0, start - 7) : start]
        if "counter" not in prefix:
            cw_only_positions.append(start)
    cw_last = cw_only_positions[-1] if cw_only_positions else -1

    if ccw_last > cw_last:
        return "counterclockwise"
    elif cw_last > ccw_last:
        return "clockwise"
    return None


def compute_ground_truth(meta: dict) -> str:
    """
    Compute the ground-truth answer from the metadata stored in the JSONL.
    """
    location_names = meta["location_names"]
    geometries = meta["geometries"]
    idx_a, idx_b, idx_c = meta["center_idx"], meta["b_idx"], meta["c_idx"]

    a_pt = representative_point(geometries[idx_a])
    b_pt = representative_point(geometries[idx_b])
    c_pt = representative_point(geometries[idx_c])

    return compute_cyclic_order(a_pt, b_pt, c_pt)


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
