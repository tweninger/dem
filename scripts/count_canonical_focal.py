#!/usr/bin/env python3
"""Print the most frequent canonical focal actors from a labeled CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from canonicalize_focal import DEFAULT_INPUT, DEFAULT_OUTPUT, normalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT, help="Labeled focal CSV")
    parser.add_argument("--map", type=Path, default=DEFAULT_OUTPUT, help="Canonical mapping CSV")
    parser.add_argument("--column", default="focal", help="Column containing semicolon-separated focal labels")
    parser.add_argument("--top", type=int, default=50, help="Number of canonical actors to retain")
    parser.add_argument("--catch-all", default="Other", help="Name for every actor outside the top N")
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            normalize_label(row["label"]): normalize_label(row["canonical"])
            for row in reader
            if normalize_label(row.get("label", "")) and normalize_label(row.get("canonical", ""))
        }


def main() -> None:
    args = parse_args()
    if args.top < 1:
        raise ValueError("--top must be at least 1")

    mapping = load_mapping(args.map)
    canonical_counts: Counter[str] = Counter()
    catch_all_count = 0

    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or args.column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(f"Column {args.column!r} was not found. Available columns: {available}")
        for row in reader:
            for value in str(row.get(args.column, "")).split(";"):
                label = normalize_label(value)
                canonical = mapping.get(label)
                if not label or label == "none" or canonical in {None, "none"}:
                    catch_all_count += 1
                else:
                    canonical_counts[canonical] += 1

    top = canonical_counts.most_common(args.top)
    result = Counter(dict(top))
    result[args.catch_all] = catch_all_count + sum(canonical_counts.values()) - sum(
        count for _, count in top
    )
    print(result)
    print(
        f"\nTop {args.top} account for {sum(count for _, count in top):,} focal slots; "
        f"{args.catch_all} accounts for {result[args.catch_all]:,}."
    )


if __name__ == "__main__":
    main()
