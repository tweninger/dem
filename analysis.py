#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from prompts import FOCAL_LABEL_COLUMN, TASK_LABEL_COLUMNS


NO_CATEGORY = "No Category"
BLANK = "[blank]"
NO_SUBSTANTIVE_COMBO = "[none]"
DEFAULT_LABEL_COLUMNS = [*TASK_LABEL_COLUMNS.values(), FOCAL_LABEL_COLUMN]


def normalize_value(value: str) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    if text.lower().startswith("no categor"):
        return NO_CATEGORY
    return text


def pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * numerator / denominator


def read_labeled_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def display_value(value: str) -> str:
    return value if value else BLANK


def detect_label_columns(df: pd.DataFrame, pred_prefix: str, explicit_columns: list[str] | None) -> list[str]:
    if explicit_columns:
        missing = [col for col in explicit_columns if col not in df.columns]
        if missing:
            raise SystemExit(f"Missing requested column(s): {', '.join(missing)}")
        return explicit_columns

    expected = [f"{pred_prefix}{col}" for col in DEFAULT_LABEL_COLUMNS]
    found = [col for col in expected if col in df.columns]
    if found:
        return found

    available_defaults = [col for col in DEFAULT_LABEL_COLUMNS if col in df.columns]
    if pred_prefix and available_defaults:
        raise SystemExit(
            "No label columns found for the requested prefix "
            f"'{pred_prefix}'. Unprefixed columns do exist: {', '.join(available_defaults)}"
        )

    raise SystemExit(
        "Could not find any default label columns. "
        f"Expected one or more of: {', '.join(expected)}"
    )


def build_normalized_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = df[columns].copy()
    for col in columns:
        normalized[col] = normalized[col].map(normalize_value)
    return normalized


def build_coverage_summary(normalized: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(normalized)
    rows: list[dict[str, object]] = []

    for col in normalized.columns:
        series = normalized[col]
        nonblank_count = int(series.ne("").sum())
        no_category_count = int(series.eq(NO_CATEGORY).sum())
        substantive_count = int(((series.ne("")) & (series.ne(NO_CATEGORY))).sum())
        rows.append(
            {
                "column": col,
                "rows": total_rows,
                "nonblank_count": nonblank_count,
                "nonblank_pct": pct(nonblank_count, total_rows),
                "blank_count": total_rows - nonblank_count,
                "blank_pct": pct(total_rows - nonblank_count, total_rows),
                "no_category_count": no_category_count,
                "no_category_pct": pct(no_category_count, total_rows),
                "substantive_count": substantive_count,
                "substantive_pct": pct(substantive_count, total_rows),
            }
        )

    return pd.DataFrame(rows)


def build_label_distribution(normalized: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(normalized)
    rows: list[dict[str, object]] = []

    for col in normalized.columns:
        series = normalized[col]
        nonblank_count = int(series.ne("").sum())
        counts = series.map(display_value).value_counts()

        for label, count in counts.items():
            rows.append(
                {
                    "column": col,
                    "label": label,
                    "count": int(count),
                    "pct_of_rows": pct(int(count), total_rows),
                    "pct_of_nonblank": pct(int(count), nonblank_count) if label != BLANK else 0.0,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["column", "count", "label"], ascending=[True, False, True], ignore_index=True)


def combo_signature(row: pd.Series, columns: list[str], include_no_category: bool) -> str:
    parts: list[str] = []
    for col in columns:
        value = row[col]
        if not value:
            parts.append(f"{col}={BLANK}")
            continue
        if not include_no_category and value == NO_CATEGORY:
            continue
        parts.append(f"{col}={value}")
    return " | ".join(parts) if parts else NO_SUBSTANTIVE_COMBO


def build_combo_distribution(
    normalized: pd.DataFrame,
    columns: list[str],
    include_no_category: bool,
) -> pd.DataFrame:
    total_rows = len(normalized)
    combos = normalized.apply(combo_signature, axis=1, columns=columns, include_no_category=include_no_category)
    counts = combos.value_counts()

    rows = [
        {
            "combo": combo,
            "count": int(count),
            "pct_of_rows": pct(int(count), total_rows),
        }
        for combo, count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(["count", "combo"], ascending=[False, True], ignore_index=True)


def write_summary_csvs(
    out_dir: str,
    coverage: pd.DataFrame,
    label_distribution: pd.DataFrame,
    combo_distribution: pd.DataFrame,
) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(path / "coverage_summary.csv", index=False)
    label_distribution.to_csv(path / "label_distribution.csv", index=False)
    combo_distribution.to_csv(path / "label_combinations.csv", index=False)


def print_coverage_summary(coverage: pd.DataFrame) -> None:
    print("Coverage")
    for _, row in coverage.iterrows():
        print(
            f"  {row['column']:10s} "
            f"nonblank={int(row['nonblank_count']):6d} ({float(row['nonblank_pct']):6.2f}%) "
            f"substantive={int(row['substantive_count']):6d} ({float(row['substantive_pct']):6.2f}%) "
            f"no_category={int(row['no_category_count']):6d} ({float(row['no_category_pct']):6.2f}%)"
        )
    print()


def print_label_distribution(label_distribution: pd.DataFrame) -> None:
    print("Label Distribution")
    if label_distribution.empty:
        print("  No nonblank labels found.\n")
        return

    for column, group in label_distribution.groupby("column", sort=False):
        print(f"  [{column}]")
        for _, row in group.iterrows():
            pct_nonblank = "n/a" if row["label"] == BLANK else f"{float(row['pct_of_nonblank']):6.2f}%"
            print(
                f"    {row['label']:45s} "
                f"n={int(row['count']):6d} "
                f"pct_rows={float(row['pct_of_rows']):6.2f}% "
                f"pct_nonblank={pct_nonblank:>7s}"
            )
        print()


def print_combo_distribution(combo_distribution: pd.DataFrame, top_n: int, include_no_category: bool) -> None:
    flavor = "including blanks and No Category" if include_no_category else "including blanks, excluding No Category"
    print(f"Row Label Combinations ({flavor})")
    if combo_distribution.empty:
        print("  No combinations found.\n")
        return

    for _, row in combo_distribution.head(top_n).iterrows():
        print(f"  {row['combo']:80s} n={int(row['count']):6d} pct_rows={float(row['pct_of_rows']):6.2f}%")

    hidden = len(combo_distribution) - min(top_n, len(combo_distribution))
    if hidden > 0:
        print(f"  ... {hidden} more combination(s) not shown")
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Labeled CSV to analyze")
    parser.add_argument("--pred-prefix", default="", help="Prefix used on prediction columns, e.g. TEST_")
    parser.add_argument("--cols", nargs="+", default=None, help="Optional explicit list of label columns to analyze")
    parser.add_argument("--top-combos", type=int, default=25, help="How many combinations to print")
    parser.add_argument(
        "--ignore-no-category-in-combos",
        action="store_true",
        help="Drop 'No Category' from row combinations instead of treating it as part of the output signature",
    )
    parser.add_argument("--out-dir", default=None, help="Optional directory for CSV summaries")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.top_combos <= 0:
        raise SystemExit("--top-combos must be greater than 0")

    df = read_labeled_csv(args.csv)
    columns = detect_label_columns(df, args.pred_prefix, args.cols)
    normalized = build_normalized_frame(df, columns)

    coverage = build_coverage_summary(normalized)
    label_distribution = build_label_distribution(normalized)
    combo_distribution = build_combo_distribution(
        normalized=normalized,
        columns=columns,
        include_no_category=not args.ignore_no_category_in_combos,
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {', '.join(columns)}\n")
    print_coverage_summary(coverage)
    print_label_distribution(label_distribution)
    print_combo_distribution(
        combo_distribution=combo_distribution,
        top_n=args.top_combos,
        include_no_category=not args.ignore_no_category_in_combos,
    )

    if args.out_dir:
        write_summary_csvs(args.out_dir, coverage, label_distribution, combo_distribution)
        print(f"Saved CSV summaries to {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
