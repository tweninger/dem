#!/usr/bin/env python3
"""Interactively map focal-actor labels to a canonical name.

The input `focal` field may contain up to three semicolon-separated actors.  This
tool counts the *individual* actors, highest frequency first, and records a
mapping as soon as it is entered, so it is safe to stop with Ctrl+C and resume.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "dem_labeled_focus2.csv"
DEFAULT_OUTPUT = REPO_ROOT / "derived" / "focal_actor_canonical_map.csv"
SKIP_VALUES = {"", "none"}


def normalize_label(value: str) -> str:
    """Fold a label to lowercase ASCII letters, digits, and single spaces only."""
    text = str(value)
    # Parenthetical text is descriptive rather than part of the actor name.
    # Repeating handles simple nested parentheses as well.
    while True:
        without_parenthetical = re.sub(r"\([^()]*\)", "", text)
        if without_parenthetical == text:
            break
        text = without_parenthetical
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Whitespace separates words; punctuation is discarded so G-7, G7+, and
    # G7 resolve to the same label.
    ascii_text = re.sub(r"\s+", " ", ascii_text.lower())
    return " ".join(re.sub(r"[^a-z0-9 ]+", "", ascii_text).split())


def count_labels(path: Path, column: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(f"Column {column!r} was not found. Available columns: {available}")
        for row in reader:
            for item in str(row.get(column, "")).split(";"):
                label = normalize_label(item)
                if label.casefold() not in SKIP_VALUES:
                    counts[label] += 1
    return counts


def load_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "canonical"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: label, canonical")
        return {
            normalize_label(row["label"]): normalize_label(row["canonical"])
            for row in reader
            if normalize_label(row.get("label", ""))
        }


def append_mapping(path: Path, label: str, canonical: str, occurrences: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "canonical", "occurrences"])
        if new_file:
            writer.writeheader()
        writer.writerow({"label": label, "canonical": canonical, "occurrences": occurrences})
        handle.flush()
        os.fsync(handle.fileno())


def rewrite_normalized_map(path: Path) -> tuple[int, int, list[tuple[str, list[str], str]]]:
    """Normalize an existing map and resolve label collisions by frequency.

    When the same normalized source label has multiple targets, the target with
    the largest total number of occurrences wins.  This makes the result
    deterministic while retaining the decision that covered most posts.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[str, Counter[str]] = {}
    for row in rows:
        label = normalize_label(row.get("label", ""))
        canonical = normalize_label(row.get("canonical", ""))
        if not label or not canonical:
            continue
        try:
            occurrences = int(row.get("occurrences", "0"))
        except ValueError:
            occurrences = 0
        grouped.setdefault(label, Counter())[canonical] += occurrences

    conflicts = []
    normalized_rows = []
    for label, targets in grouped.items():
        canonical, occurrences = sorted(targets.items(), key=lambda item: (-item[1], item[0]))[0]
        if len(targets) > 1:
            conflicts.append((label, sorted(targets), canonical))
        normalized_rows.append({"label": label, "canonical": canonical, "occurrences": sum(targets.values())})
    normalized_rows.sort(key=lambda row: (-row["occurrences"], row["label"]))

    backup = path.with_name(f"{path.stem}_before_ascii_normalization{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "canonical", "occurrences"])
        writer.writeheader()
        writer.writerows(normalized_rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return len(rows), len(normalized_rows), conflicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT, help="Labeled focal CSV to inspect")
    parser.add_argument("--column", default="focal", help="Column containing semicolon-separated focal labels")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="CSV mapping to create or resume")
    parser.add_argument(
        "--rewrite-normalized-map",
        action="store_true",
        help="Normalize and deduplicate the existing map, then exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rewrite_normalized_map:
        before, after, conflicts = rewrite_normalized_map(args.out)
        print(f"Rewrote {args.out}: {before:,} rows → {after:,} normalized mappings.")
        if conflicts:
            print(f"Resolved {len(conflicts)} conflicting collisions using the highest occurrence total:")
            for label, targets, chosen in conflicts:
                print(f"  {label!r}: {', '.join(targets)} → {chosen!r}")
        return

    counts = count_labels(args.csv, args.column)
    mapping = load_mapping(args.out)
    pending = [(label, count) for label, count in counts.most_common() if label not in mapping]

    print(f"Found {sum(counts.values()):,} actor mentions across {len(counts):,} distinct labels.")
    print(f"Loaded {len(mapping):,} existing mappings; {len(pending):,} labels remain.")
    print("Press Enter to keep the displayed label as canonical. Ctrl+C stops safely.")

    completed = 0
    try:
        for label, occurrences in pending:
            while True:
                answer = input(
                    f"\n[{completed + 1:,}/{len(pending):,}] {label!r} "
                    f"({occurrences:,} mentions) → "
                )
                canonical = normalize_label(answer) or label
                if ";" not in canonical:
                    break
                print("A canonical label must be one actor (no semicolon). Try again.")
            append_mapping(args.out, label, canonical, occurrences)
            mapping[label] = canonical
            completed += 1
    except (KeyboardInterrupt, EOFError):
        print(f"\nStopped safely. Recorded {completed:,} mappings this session in {args.out}.")
        return

    print(f"Completed all labels. Mapping saved to {args.out}.")


if __name__ == "__main__":
    main()
