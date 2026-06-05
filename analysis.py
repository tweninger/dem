#!/usr/bin/env python3
# pip install pandas numpy scipy scikit-learn

import argparse
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score


LABEL_COLS_DEFAULT = ["AUT_LABEL", "DEM_LABEL", "WEST_LABEL"]


def normalize_label(x: str) -> str:
    """Normalize label values to make counting stable."""
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def update_counter(counter: Counter, series: pd.Series):
    for v in series:
        counter[normalize_label(v)] += 1


def update_pair_counter(counter: Counter, a: pd.Series, b: pd.Series):
    for x, y in zip(a, b):
        counter[(normalize_label(x), normalize_label(y))] += 1


def counter_to_frame(counter: Counter, name: str) -> pd.DataFrame:
    df = pd.DataFrame(counter.items(), columns=[name, "count"])
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    df["pct"] = df["count"] / df["count"].sum()
    return df


def pair_counter_to_frame(counter: Counter, name_a: str, name_b: str) -> pd.DataFrame:
    rows = [{"%s" % name_a: k[0], "%s" % name_b: k[1], "count": v} for k, v in counter.items()]
    df = pd.DataFrame(rows)
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    df["pct"] = df["count"] / df["count"].sum()
    return df


def build_contingency_from_pair_counts(pair_counts: Counter):
    """
    Build contingency table from a Counter[(a,b)] -> count.
    Returns (contingency_df, labels_a, labels_b).
    """
    labels_a = sorted({k[0] for k in pair_counts.keys()})
    labels_b = sorted({k[1] for k in pair_counts.keys()})
    idx_a = {lab: i for i, lab in enumerate(labels_a)}
    idx_b = {lab: j for j, lab in enumerate(labels_b)}

    mat = np.zeros((len(labels_a), len(labels_b)), dtype=np.int64)
    for (a, b), c in pair_counts.items():
        mat[idx_a[a], idx_b[b]] = c

    cont_df = pd.DataFrame(mat, index=labels_a, columns=labels_b)
    return cont_df, labels_a, labels_b


def cramers_v_from_contingency(cont: np.ndarray) -> float:
    """Cramér’s V with bias correction (good for non-square tables)."""
    chi2, _, _, _ = chi2_contingency(cont, correction=False)
    n = cont.sum()
    if n == 0:
        return float("nan")
    r, k = cont.shape
    phi2 = chi2 / n
    # Bias correction (Bergsma 2013-ish correction commonly used)
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0.0
    rcorr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
    kcorr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def association_report(pair_counts: Counter, col_a: str, col_b: str) -> dict:
    cont_df, _, _ = build_contingency_from_pair_counts(pair_counts)
    cont = cont_df.values

    # Chi-square
    chi2, p, dof, _ = chi2_contingency(cont, correction=False)

    # Cramér's V
    v = cramers_v_from_contingency(cont)

    # Mutual information (needs sample-level arrays; reconstruct from counts efficiently)
    # We'll compute MI from contingency directly:
    # MI = sum_{i,j} p(i,j) log( p(i,j) / (p(i)p(j)) )
    n = cont.sum()
    if n == 0:
        mi = float("nan")
    else:
        pij = cont / n
        pi = pij.sum(axis=1, keepdims=True)
        pj = pij.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = pij / (pi * pj)
            term = np.where(pij > 0, pij * np.log(ratio), 0.0)
        mi = float(term.sum())

    return {
        "pair": f"{col_a} x {col_b}",
        "n": int(n),
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": float(v),
        "mutual_info_nats": float(mi),
        "n_rows_in_table": int(cont.shape[0]),
        "n_cols_in_table": int(cont.shape[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Labeled output CSV (e.g., labeled_aut_dem_west.csv)")
    ap.add_argument("--cols", nargs="+", default=LABEL_COLS_DEFAULT, help="Label columns to analyze")
    ap.add_argument("--chunksize", type=int, default=200_000, help="Streaming chunksize")
    ap.add_argument("--out_prefix", default="label_analysis", help="Prefix for output files")
    args = ap.parse_args()

    cols = args.cols
    for c in cols:
        if c is None or not c.strip():
            raise SystemExit("Empty column name in --cols")

    # Streaming counters
    marginals = {c: Counter() for c in cols}
    pair_counts = {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair_counts[(cols[i], cols[j])] = Counter()

    # Simple row-level stats
    total_rows = 0
    all_three_equal = 0
    any_empty = 0

    usecols = cols  # only read label cols
    reader = pd.read_csv(
        args.csv,
        usecols=lambda c: c in set(usecols),
        dtype=str,
        keep_default_na=False,
        chunksize=args.chunksize,
        encoding="utf-8-sig",
        low_memory=False,
    )

    for chunk_idx, df in enumerate(reader, start=1):
        total_rows += len(df)

        # Normalize in-place (vectorized-ish)
        for c in cols:
            df[c] = df[c].astype(str).map(normalize_label)

        # Marginals
        for c in cols:
            update_counter(marginals[c], df[c])

        # Pairwise
        for (a, b), ctr in pair_counts.items():
            update_pair_counter(ctr, df[a], df[b])

        # Row-level quick stats (assumes 3 cols if you want these)
        if len(cols) >= 3:
            a, b, w = cols[0], cols[1], cols[2]
            all_three_equal += int((df[a] == df[b]).to_numpy().astype(int).dot((df[b] == df[w]).to_numpy().astype(int)))
            any_empty += int(((df[a] == "") | (df[b] == "") | (df[w] == "")).sum())

        if chunk_idx % 5 == 0:
            print(f"Processed {total_rows:,} rows...")

    print(f"Done. Total rows analyzed: {total_rows:,}")

    # Write marginals
    for c, ctr in marginals.items():
        out = counter_to_frame(ctr, c)
        out.to_csv(f"{args.out_prefix}__dist__{c}.csv", index=False)
        print(f"Wrote {args.out_prefix}__dist__{c}.csv")

    # Write pairwise top pairs + association metrics
    assoc_rows = []
    for (a, b), ctr in pair_counts.items():
        pair_df = pair_counter_to_frame(ctr, a, b)
        pair_df.head(200).to_csv(f"{args.out_prefix}__pairs__{a}__{b}.csv", index=False)
        print(f"Wrote {args.out_prefix}__pairs__{a}__{b}.csv (top 200 pairs)")

        assoc_rows.append(association_report(ctr, a, b))

    assoc_df = pd.DataFrame(assoc_rows).sort_values("cramers_v", ascending=False)
    assoc_df.to_csv(f"{args.out_prefix}__associations.csv", index=False)
    print(f"Wrote {args.out_prefix}__associations.csv")

    # Write quick row-level summary (if >=3 cols)
    if len(cols) >= 3:
        summary = {
            "total_rows": total_rows,
            "all_three_equal_count": all_three_equal,
            "all_three_equal_pct": (all_three_equal / total_rows) if total_rows else float("nan"),
            "any_empty_count": any_empty,
            "any_empty_pct": (any_empty / total_rows) if total_rows else float("nan"),
        }
        pd.DataFrame([summary]).to_csv(f"{args.out_prefix}__summary.csv", index=False)
        print(f"Wrote {args.out_prefix}__summary.csv")


if __name__ == "__main__":
    main()

