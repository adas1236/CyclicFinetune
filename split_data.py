#!/usr/bin/env python3
"""
Split prepared JSONL evaluation data into small, balanced shards.

The input is a formatted JSONL file from prepare_data.py. Records are kept
intact and written back as JSONL; only record ordering and shard boundaries are
changed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LABELS = ("clockwise", "counterclockwise", "neither")
DEFAULT_MAX_RECORDS = 1024
DEFAULT_SEED = 42


@dataclass(frozen=True)
class JsonlRecord:
    line_number: int
    raw_line: str
    n_points: int
    label: str


def parse_record(raw_line: str, line_number: int) -> JsonlRecord:
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc

    if not isinstance(record, dict):
        raise ValueError(f"line {line_number}: record must be a JSON object")

    meta = record.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"line {line_number}: missing object field meta")

    geometries = meta.get("geometries")
    if not isinstance(geometries, list):
        raise ValueError(f"line {line_number}: meta.geometries must be a list")
    n_points = len(geometries)

    label = record.get("expected_answer", meta.get("answer"))
    if not isinstance(label, str):
        raise ValueError(
            f"line {line_number}: missing string expected_answer or meta.answer"
        )
    label = label.strip().lower()
    if label not in LABELS:
        raise ValueError(
            f"line {line_number}: unsupported answer label {label!r}; "
            f"expected one of {', '.join(LABELS)}"
        )

    if not raw_line.endswith("\n"):
        raw_line += "\n"

    return JsonlRecord(
        line_number=line_number,
        raw_line=raw_line,
        n_points=n_points,
        label=label,
    )


def load_jsonl(path: Path) -> list[JsonlRecord]:
    records: list[JsonlRecord] = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                records.append(parse_record(line, line_number))
    return records


def allocate_uniform(
    available: dict[Any, int],
    total: int,
    ordered_keys: list[Any],
) -> dict[Any, int]:
    """
    Allocate integer quotas as evenly as possible across non-empty buckets.

    Buckets are capped by availability. Scarce buckets are exhausted before the
    excess is redistributed to the remaining buckets.
    """
    quotas = {key: 0 for key in ordered_keys}
    remaining = min(total, sum(max(0, available.get(key, 0)) for key in ordered_keys))

    while remaining > 0:
        progressed = False
        for key in ordered_keys:
            if remaining <= 0:
                break
            if quotas[key] >= available.get(key, 0):
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    return quotas


def build_queues(
    records: list[JsonlRecord],
    rng: random.Random,
) -> dict[tuple[int, str], deque[JsonlRecord]]:
    grouped: dict[tuple[int, str], list[JsonlRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.n_points, record.label)].append(record)

    queues: dict[tuple[int, str], deque[JsonlRecord]] = {}
    for key in sorted(grouped):
        rows = grouped[key]
        rng.shuffle(rows)
        queues[key] = deque(rows)
    return queues


def remaining_by_n(queues: dict[tuple[int, str], deque[JsonlRecord]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for (n_points, _label), queue in queues.items():
        counts[n_points] += len(queue)
    return dict(counts)


def remaining_labels_for_n(
    queues: dict[tuple[int, str], deque[JsonlRecord]],
    n_points: int,
) -> dict[str, int]:
    return {label: len(queues.get((n_points, label), ())) for label in LABELS}


def pop_balanced_records(
    queues: dict[tuple[int, str], deque[JsonlRecord]],
    quotas: dict[tuple[int, str], int],
) -> list[JsonlRecord]:
    selected: dict[tuple[int, str], deque[JsonlRecord]] = {}
    for key in sorted(quotas):
        quota = quotas[key]
        selected[key] = deque()
        source = queues[key]
        for _ in range(quota):
            selected[key].append(source.popleft())

    output: list[JsonlRecord] = []
    while True:
        progressed = False
        for key in sorted(selected):
            if not selected[key]:
                continue
            output.append(selected[key].popleft())
            progressed = True
        if not progressed:
            break
    return output


def build_shards(
    records: list[JsonlRecord],
    *,
    max_records: int,
    seed: int,
) -> list[list[JsonlRecord]]:
    rng = random.Random(seed)
    queues = build_queues(records, rng)
    n_order = sorted({record.n_points for record in records})

    shards: list[list[JsonlRecord]] = []
    remaining_total = len(records)
    while remaining_total > 0:
        shard_size = min(max_records, remaining_total)
        n_counts = remaining_by_n(queues)
        n_quotas = allocate_uniform(n_counts, shard_size, n_order)

        record_quotas: dict[tuple[int, str], int] = {}
        for n_points in n_order:
            n_quota = n_quotas.get(n_points, 0)
            if n_quota <= 0:
                continue
            label_counts = remaining_labels_for_n(queues, n_points)
            label_quotas = allocate_uniform(label_counts, n_quota, list(LABELS))
            for label, quota in label_quotas.items():
                if quota > 0:
                    record_quotas[(n_points, label)] = quota

        shard = pop_balanced_records(queues, record_quotas)
        if len(shard) != shard_size:
            raise RuntimeError(
                f"internal error: planned shard of {shard_size} records but "
                f"selected {len(shard)}"
            )
        shards.append(shard)
        remaining_total -= len(shard)

    return shards


def count_by_n(records: list[JsonlRecord]) -> Counter[int]:
    return Counter(record.n_points for record in records)


def count_by_n_label(records: list[JsonlRecord]) -> Counter[tuple[int, str]]:
    return Counter((record.n_points, record.label) for record in records)


def format_counter(counter: Counter[Any]) -> str:
    if not counter:
        return "{}"
    return ", ".join(f"{key}:{counter[key]}" for key in sorted(counter))


def format_n_label_counts(counter: Counter[tuple[int, str]]) -> str:
    parts: list[str] = []
    for n_points in sorted({key[0] for key in counter}):
        label_bits = [
            f"{label}={counter[(n_points, label)]}"
            for label in LABELS
            if counter[(n_points, label)] > 0
        ]
        parts.append(f"n={n_points}({', '.join(label_bits)})")
    return "; ".join(parts) if parts else "{}"


def print_summary(shards: list[list[JsonlRecord]]) -> None:
    for idx, shard in enumerate(shards, start=1):
        print(f"shard {idx:03d}: {len(shard)} records")
        print(f"  by n: {format_counter(count_by_n(shard))}")
        print(f"  by n,label: {format_n_label_counts(count_by_n_label(shard))}")


def default_output_dir(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_splits")


def output_paths(output_dir: Path, prefix: str, shard_count: int) -> list[Path]:
    return [
        output_dir / f"{prefix}_{idx:03d}.jsonl"
        for idx in range(1, shard_count + 1)
    ]


def check_output_paths(
    paths: list[Path],
    *,
    output_dir: Path,
    prefix: str,
    overwrite: bool,
) -> None:
    existing = set(path for path in paths if path.exists())
    if output_dir.exists():
        existing.update(
            path for path in output_dir.glob(f"{prefix}_*.jsonl") if path.is_file()
        )
    existing_paths = sorted(existing)
    if existing_paths and not overwrite:
        shown = "\n".join(f"  {path}" for path in existing_paths[:10])
        extra = (
            ""
            if len(existing_paths) <= 10
            else f"\n  ... and {len(existing_paths) - 10} more"
        )
        raise FileExistsError(
            "refusing to overwrite existing split file(s):\n"
            f"{shown}{extra}\nPass --overwrite to replace them."
        )


def remove_existing_splits(output_dir: Path, prefix: str) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob(f"{prefix}_*.jsonl"):
        if path.is_file():
            path.unlink()


def write_shards(
    shards: list[list[JsonlRecord]],
    *,
    output_dir: Path,
    prefix: str,
    overwrite: bool,
) -> list[Path]:
    paths = output_paths(output_dir, prefix, len(shards))
    check_output_paths(
        paths,
        output_dir=output_dir,
        prefix=prefix,
        overwrite=overwrite,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        remove_existing_splits(output_dir, prefix)

    for path, shard in zip(paths, shards):
        with path.open("w") as f:
            for record in shard:
                f.write(record.raw_line)
    return paths


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split prepared geo JSONL data into balanced evaluation shards."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .jsonl file")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for output shards. Defaults to <input_stem>_splits beside input.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the input filename stem.",
    )
    parser.add_argument(
        "--max_records",
        type=positive_int,
        default=DEFAULT_MAX_RECORDS,
        help=f"Maximum records per shard. Default: {DEFAULT_MAX_RECORDS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for deterministic within-bucket shuffling. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned shard distributions without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing split files with the same prefix.",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        parser.error(f"--input does not exist: {input_path}")
    if not input_path.is_file():
        parser.error(f"--input must be a file: {input_path}")

    output_dir = args.output_dir or default_output_dir(input_path)
    prefix = args.prefix or input_path.stem

    try:
        records = load_jsonl(input_path)
        shards = build_shards(
            records,
            max_records=args.max_records,
            seed=args.seed,
        )
        paths = output_paths(output_dir, prefix, len(shards))
        if not args.dry_run:
            written_paths = write_shards(
                shards,
                output_dir=output_dir,
                prefix=prefix,
                overwrite=args.overwrite,
            )
            paths = written_paths
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Input: {input_path}")
    print(f"Records: {len(records)}")
    print(f"Max records per shard: {args.max_records}")
    print(f"Shards: {len(shards)}")
    if args.dry_run:
        print(f"Dry run: would write to {output_dir}")
    else:
        print(f"Output directory: {output_dir}")
        print(f"First shard: {paths[0] if paths else 'n/a'}")
        print(f"Last shard: {paths[-1] if paths else 'n/a'}")
    print_summary(shards)


if __name__ == "__main__":
    main()
