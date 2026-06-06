#!/usr/bin/env python3

import argparse
import sys

import pandas as pd

from prompts import FOCAL_LABEL_COLUMN, TASK_LABEL_COLUMNS


FRAME_COLS = list(TASK_LABEL_COLUMNS.values())
FRAME_COL_ALIASES = {
    "aut": ["aut", "AUT_LABEL"],
    "dem": ["dem", "DEM_LABEL"],
    "west": ["west", "WEST_LABEL"],
}
FOCAL_COL = FOCAL_LABEL_COLUMN
FOCAL_ALIASES = [FOCAL_LABEL_COLUMN, "FOCAL_COUNTRY"]


def norm(value: str) -> str:
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    low = s.lower()

    mapping = {
        "no categor": "No Category",
        "no category": "No Category",
        "authoritarian - military/security": "Military/Security Promotion",
        "authoritarian - military / security": "Military/Security Promotion",
        "authoritarian - economic influence": "Economic Influence",
        "authoritarian - digital": "Digital Control and Surveillance",
        "authoritarian - legal tools for entrenchment": "Legal Entrenchment",
        "authoritarian - alliances": "Alliances",
        "authoritarian - ideological promotion": "Ideological Promotion",
        "democracy - values and rights": "Values and Rights",
        "democracy - elections": "Elections",
        "democracy - institutions": "Institutions",
        "democracy - civil society": "Civil Society",
        "declining west": "Declining West",
        "regime change / internal instability": "Western induced Regime Change/Internal Instability",
        "western induced regime change/internal instability": "Western induced Regime Change/Internal Instability",
        "hostile global order": "Hostile Global Order",
        "specific adversary framing": "Specific Adversary Framing",
        "wi - declining west": "Declining West",
        "wi - western induced regime change/internal instability": "Western induced Regime Change/Internal Instability",
        "wi - hostile global order": "Hostile Global Order",
        "wi - specific adversary framing": "Specific Adversary Framing",
        "oecd/west": "OECD",
        "other": "Other",
    }
    return mapping.get(low, s)


def is_yes(value: str) -> bool:
    return str(value).strip().upper() == "Y"


def first_present(columns: list[str], df: pd.DataFrame) -> str | None:
    for col in columns:
        if col in df.columns:
            return col
    return None


def print_accuracy(title: str, truth: pd.Series, pred: pd.Series) -> None:
    truth = truth.map(norm)
    pred = pred.map(norm)
    mask = truth.ne("")

    if mask.sum() == 0:
        print(f"{title:15s} n=0\n")
        return

    scoped = pd.DataFrame({"truth": truth[mask], "pred": pred[mask]})
    overall = float((scoped["truth"] == scoped["pred"]).mean())
    print(f"{title:15s} n={len(scoped):5d} acc={overall:.3f}")

    by_truth = (
        scoped.assign(correct=scoped["truth"] == scoped["pred"])
        .groupby("truth", dropna=False)
        .agg(n=("correct", "size"), acc=("correct", "mean"))
        .sort_values(["n", "truth"], ascending=[False, True])
    )

    print("By reference label:")
    for truth_value, row in by_truth.iterrows():
        print(f"  {truth_value:45s} n={int(row['n']):5d} acc={float(row['acc']):.3f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pred-prefix", default="TEST_")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    if "sample_from" not in df.columns or "label_value" not in df.columns:
        print("ERROR: csv must contain 'sample_from' and 'label_value' columns", file=sys.stderr)
        sys.exit(2)

    frame_ok = df.get("Label_value Correct?", pd.Series("", index=df.index)).map(is_yes)
    focal_ok = df.get("Focal Country Correct?", pd.Series("", index=df.index)).map(is_yes)

    for frame_col in FRAME_COLS:
        pred_source_col = first_present(FRAME_COL_ALIASES[frame_col], df)
        if pred_source_col is None:
            print(f"Skipping {frame_col}: missing any of {FRAME_COL_ALIASES[frame_col]}\n")
            continue

        pred_col = f"{args.pred_prefix}{pred_source_col}"
        if pred_col not in df.columns:
            print(f"Skipping {frame_col}: missing {pred_col}\n")
            continue

        sample_from = df["sample_from"].astype(str).str.strip()
        mask = frame_ok & sample_from.isin(FRAME_COL_ALIASES[frame_col])
        truth = df.loc[mask, "label_value"]
        pred = df.loc[mask, pred_col]
        print_accuracy(frame_col, truth, pred)

    focal_source_col = first_present(FOCAL_ALIASES, df)
    if focal_source_col is None:
        print(f"Skipping {FOCAL_COL}: missing any of {FOCAL_ALIASES}\n")
    else:
        focal_pred_col = f"{args.pred_prefix}{focal_source_col}"
        if focal_pred_col not in df.columns:
            print(f"Skipping {FOCAL_COL}: missing {focal_pred_col}\n")
        else:
            truth_source_col = first_present(FOCAL_ALIASES, df)
            truth = df.loc[focal_ok, truth_source_col]
            pred = df.loc[focal_ok, focal_pred_col]
            print_accuracy(FOCAL_COL, truth, pred)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
