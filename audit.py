#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# Label normalization
# ------------------------------------------------------------

def norm(x):
    if pd.isna(x):
        return ""

    s = str(x).strip().strip('"').strip("'")
    if not s:
        return ""

    low = s.lower().strip()

    mapping = {
        # Empty / no category
        "no categor": "No Category",
        "no category": "No Category",
        "nan": "",

        # AUT old -> new
        "authoritarian - military/security": "Military/Security Promotion",
        "authoritarian - military / security": "Military/Security Promotion",
        "military/security": "Military/Security Promotion",
        "military/security promotion": "Military/Security Promotion",
        "military security promotion": "Military/Security Promotion",

        "authoritarian - economic influence": "Economic Influence",
        "economic influence": "Economic Influence",

        "authoritarian - digital": "Digital Control and Surveillance",
        "digital": "Digital Control and Surveillance",
        "digital control": "Digital Control and Surveillance",
        "digital control and surveillance": "Digital Control and Surveillance",

        "authoritarian - legal tools for entrenchment": "Legal Entrenchment",
        "legal tools for entrenchment": "Legal Entrenchment",
        "legal entrenchment": "Legal Entrenchment",

        "authoritarian - alliances": "Alliances",
        "alliances": "Alliances",

        "authoritarian - ideological promotion": "Ideological Promotion",
        "ideological promotion": "Ideological Promotion",
        "authoritarian - ideology": "Ideological Promotion",

        # DEM old -> new
        "democracy - values and rights": "Values and Rights",
        "values and rights": "Values and Rights",

        "democracy - elections": "Elections",
        "elections": "Elections",

        "democracy - institutions": "Institutions",
        "institutions": "Institutions",

        "democracy - civil society": "Civil Society",
        "civil society": "Civil Society",

        # WEST old -> new
        "wi - declining west": "Declining West",
        "declining west": "Declining West",

        "wi - western induced regime change/internal instability": "Western induced Regime Change/Internal Instability",
        "western induced regime change/internal instability": "Western induced Regime Change/Internal Instability",
        "western-induced regime change/internal instability": "Western induced Regime Change/Internal Instability",

        "wi - hostile global order": "Hostile Global Order",
        "hostile global order": "Hostile Global Order",
        "hostile global order framing": "Hostile Global Order",

        "wi - specific adversary framing": "Specific Adversary Framing",
        "specific adversary framing": "Specific Adversary Framing",

        # Focal country / actor
        "china": "China",
        "prc": "China",

        "russia": "Russia",
        "russian federation": "Russia",
        "ussr": "Russia",
        "soviet union": "Russia",

        "oecd": "OECD",
        "usa": "OECD",
        "us": "OECD",
        "u.s.": "OECD",
        "united states": "OECD",
        "america": "OECD",
        "europe": "OECD",
        "eu": "OECD",
        "european union": "OECD",
        "nato": "OECD",
        "g7": "OECD",
        "west": "OECD",
        "the west": "OECD",
        "western countries": "OECD",

        "other": "Other",
    }

    if low in mapping:
        return mapping[low]

    if low.startswith("no categor"):
        return "No Category"

    return s


def is_nonempty_gold(x):
    return norm(x) not in {"", "nan"}


# ------------------------------------------------------------
# Prediction selection
# ------------------------------------------------------------

def pred_for_sample(row):
    """
    Uses sample_from to decide which V2 prediction should be compared
    against AUDIT_GOLD_LABEL_VALUE.
    """

    sample_from = str(row.get("sample_from", "")).strip().upper()

    if sample_from.startswith("AUT"):
        return row.get("AUT_LABEL_V2", "")

    if sample_from.startswith("DEM"):
        return row.get("DEM_LABEL_V2", "")

    if sample_from.startswith("WEST"):
        return row.get("WEST_LABEL_V2", "")

    # Fallback: use first non-No Category prediction.
    for col in ["AUT_LABEL_V2", "DEM_LABEL_V2", "WEST_LABEL_V2"]:
        val = norm(row.get(col, ""))
        if val and val != "No Category":
            return val

    return "No Category"


def task_from_sample(x):
    s = str(x).strip().upper()

    if s.startswith("AUT"):
        return "AUT"
    if s.startswith("DEM"):
        return "DEM"
    if s.startswith("WEST"):
        return "WEST"

    return "UNKNOWN"


# ------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------

def accuracy(y_true, y_pred):
    y_true = pd.Series(y_true).map(norm)
    y_pred = pd.Series(y_pred).map(norm)
    mask = y_true.map(is_nonempty_gold)

    if mask.sum() == 0:
        return np.nan, 0

    return (y_true[mask] == y_pred[mask]).mean(), int(mask.sum())


def print_acc(label, y_true, y_pred):
    acc, n = accuracy(y_true, y_pred)
    if n == 0:
        print(f"{label:35s} n=0")
    else:
        print(f"{label:35s} n={n:5d}  acc={acc:.3f}")


def print_confusion(title, y_true, y_pred):
    tmp = pd.DataFrame({
        "gold": pd.Series(y_true).map(norm),
        "pred": pd.Series(y_pred).map(norm),
    })

    tmp = tmp[tmp["gold"].map(is_nonempty_gold)]

    if len(tmp) == 0:
        print(f"\n{title}: no rows")
        return

    print(f"\n{title}")
    print(pd.crosstab(tmp["gold"], tmp["pred"], dropna=False))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Audit results CSV")
    ap.add_argument("--errors-out", default="", help="Optional CSV of error rows")
    args = ap.parse_args()

    df = pd.read_csv(
        args.csv,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        engine="python",
        on_bad_lines="skip",
    )

    required = [
        "sample_from",
        "AUDIT_GOLD_LABEL_VALUE",
        "AUT_LABEL_V2",
        "DEM_LABEL_V2",
        "WEST_LABEL_V2",
        "FOCAL_COUNTRY_V2",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "AUDIT_GOLD_FOCAL_COUNTRY" not in df.columns:
        df["AUDIT_GOLD_FOCAL_COUNTRY"] = ""

    df["TASK"] = df["sample_from"].map(task_from_sample)

    df["GOLD_LABEL_NORM"] = df["AUDIT_GOLD_LABEL_VALUE"].map(norm)
    df["PRED_LABEL_NORM"] = df.apply(pred_for_sample, axis=1).map(norm)

    df["GOLD_FOCAL_NORM"] = df["AUDIT_GOLD_FOCAL_COUNTRY"].map(norm)
    df["PRED_FOCAL_NORM"] = df["FOCAL_COUNTRY_V2"].map(norm)

    df["LABEL_CORRECT"] = df["GOLD_LABEL_NORM"] == df["PRED_LABEL_NORM"]

    focal_mask = df["GOLD_FOCAL_NORM"].map(is_nonempty_gold)
    df["FOCAL_CORRECT"] = np.nan
    df.loc[focal_mask, "FOCAL_CORRECT"] = (
        df.loc[focal_mask, "GOLD_FOCAL_NORM"]
        == df.loc[focal_mask, "PRED_FOCAL_NORM"]
    )

    print("\n==================== AUDIT ACCURACY ====================")
    print(f"Rows loaded: {len(df):,}")

    print_acc(
        "Overall label task",
        df["GOLD_LABEL_NORM"],
        df["PRED_LABEL_NORM"],
    )

    for task in ["AUT", "DEM", "WEST"]:
        sub = df[df["TASK"] == task]
        print_acc(
            f"{task} label task",
            sub["GOLD_LABEL_NORM"],
            sub["PRED_LABEL_NORM"],
        )

    print_acc(
        "Focal country task",
        df.loc[focal_mask, "GOLD_FOCAL_NORM"],
        df.loc[focal_mask, "PRED_FOCAL_NORM"],
    )

    print("\n==================== ORIGINAL VS V2 ====================")

    if "label_value" in df.columns:
        df["ORIGINAL_LABEL_NORM"] = df["label_value"].map(norm)
        print_acc(
            "Original audited label",
            df["GOLD_LABEL_NORM"],
            df["ORIGINAL_LABEL_NORM"],
        )

    if "FOCAL_COUNTRY" in df.columns:
        df["ORIGINAL_FOCAL_NORM"] = df["FOCAL_COUNTRY"].map(norm)
        print_acc(
            "Original focal country",
            df.loc[focal_mask, "GOLD_FOCAL_NORM"],
            df.loc[focal_mask, "ORIGINAL_FOCAL_NORM"],
        )

    print("\n==================== BY TASK COUNTS ====================")
    print(df["TASK"].value_counts(dropna=False).to_string())

    print_confusion(
        "Overall label confusion: gold x V2 prediction",
        df["GOLD_LABEL_NORM"],
        df["PRED_LABEL_NORM"],
    )

    for task in ["AUT", "DEM", "WEST"]:
        sub = df[df["TASK"] == task]
        print_confusion(
            f"{task} confusion: gold x V2 prediction",
            sub["GOLD_LABEL_NORM"],
            sub["PRED_LABEL_NORM"],
        )

    print_confusion(
        "Focal country confusion: gold x V2 prediction",
        df.loc[focal_mask, "GOLD_FOCAL_NORM"],
        df.loc[focal_mask, "PRED_FOCAL_NORM"],
    )

    print("\n==================== ERROR SUMMARY ====================")

    label_errors = df[
        df["GOLD_LABEL_NORM"].map(is_nonempty_gold)
        & ~df["LABEL_CORRECT"]
    ].copy()

    focal_errors = df[
        focal_mask
        & (df["FOCAL_CORRECT"] == False)
    ].copy()

    print(f"Label errors: {len(label_errors):,}")
    print(f"Focal errors: {len(focal_errors):,}")

    useful_cols = [
        "TASK",
        "sample_from",
        "label_value",
        "AUDIT_GOLD_LABEL_VALUE",
        "PRED_LABEL_NORM",
        "AUT_LABEL_V2",
        "DEM_LABEL_V2",
        "WEST_LABEL_V2",
        "AUDIT_GOLD_FOCAL_COUNTRY",
        "FOCAL_COUNTRY_V2",
        "Google_Trans",
        "text",
        "notes",
        "Discuss",
        "link",
    ]
    useful_cols = [c for c in useful_cols if c in df.columns]

    print("\nFirst 25 label errors:")
    if len(label_errors) == 0:
        print("None")
    else:
        print(label_errors[useful_cols].head(25).to_string(index=False))

    if args.errors_out:
        error_df = pd.concat(
            [
                label_errors.assign(ERROR_TYPE="label"),
                focal_errors.assign(ERROR_TYPE="focal"),
            ],
            ignore_index=True,
        )
        error_df.to_csv(args.errors_out, index=False, encoding="utf-8-sig")
        print(f"\nWrote errors to: {args.errors_out}")


if __name__ == "__main__":
    main()
