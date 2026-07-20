#!/usr/bin/env python3
"""Draw a stratified post-frame sample and label focal-country valence.

The default is a 12-item pilot (one item per country x account type x frame
domain stratum). Run with ``--sample-size 1000`` only after reviewing the pilot.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import duckdb
import pandas as pd

from DI_framing import (
    LABEL_CUTOFFS,
    LABEL_MAX_TOKENS,
    PARALLEL,
    _call_model_with_truncation,
    _get_client_for_row,
    _open_output_for_append,
    init_clients,
)
from prompts import (
    FOCAL_VALENCE_LABEL_COLUMN,
    get_focal_valence_prompt,
)


VALENCE_ALLOWED_LABELS = ["Pro", "Anti", "Neutral"]
DEFAULT_PARQUET = "derived/analysis_posts_clean.parquet"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw a stratified post-frame sample and produce valence labels."
    )
    parser.add_argument("--parquet", default=DEFAULT_PARQUET, help="Clean analysis Parquet input")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12,
        help="Number of post-frame applications to sample; defaults to a 12-item pilot",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    parser.add_argument("--out", default=None, help="Output CSV path")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Write the deterministic sample without calling the model",
    )
    return parser


def default_output_path(sample_size: int, seed: int) -> str:
    kind = "pilot" if sample_size <= 24 else "sample"
    return f"derived/valence/focal_valence_{kind}_v3_n{sample_size}_seed{seed}.csv"


def build_stratified_sample(parquet_path: str, sample_size: int, seed: int) -> pd.DataFrame:
    """Sample post-frame applications evenly across country, account type, and domain."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Input Parquet not found: {path}")

    query = """
    WITH frame_applications AS (
        SELECT
            merge_key,
            key_occurrence,
            source,
            category,
            post_date,
            post_year,
            post_month,
            text,
            focal_actor,
            frame_domain,
            frame_subframe,
            CASE
                WHEN category LIKE 'China_%' THEN 'China'
                WHEN category LIKE 'RF_%' THEN 'Russia'
                ELSE NULL
            END AS source_country,
            CASE
                WHEN category LIKE '%Media%' THEN 'Media'
                WHEN category LIKE '%Diplomat%' THEN 'Diplomatic'
                ELSE NULL
            END AS account_type
        FROM read_parquet(?)
        CROSS JOIN LATERAL (
            VALUES
                ('AUT', aut_frame),
                ('DEM', dem_frame),
                ('WEST', west_frame)
        ) AS frames(frame_domain, frame_subframe)
        WHERE post_year BETWEEN 2020 AND 2025
          AND text IS NOT NULL
          AND trim(text) <> ''
          AND frame_subframe NOT IN ('No Category', 'Invalid label')
    ),
    eligible AS (
        SELECT *
        FROM frame_applications
        WHERE source_country IS NOT NULL AND account_type IS NOT NULL
    ),
    strata AS (
        SELECT
            source_country,
            account_type,
            frame_domain,
            count(*) AS available,
            row_number() OVER (
                ORDER BY source_country, account_type, frame_domain
            ) AS stratum_order,
            count(*) OVER () AS stratum_count
        FROM eligible
        GROUP BY source_country, account_type, frame_domain
    ),
    quotas AS (
        SELECT
            *,
            floor(? / stratum_count)::INTEGER
                + CASE WHEN stratum_order <= (? % stratum_count) THEN 1 ELSE 0 END AS requested
        FROM strata
    ),
    ranked AS (
        SELECT
            e.*,
            q.available,
            q.requested,
            row_number() OVER (
                PARTITION BY e.source_country, e.account_type, e.frame_domain
                ORDER BY hash(
                    coalesce(e.merge_key, ''),
                    coalesce(cast(e.key_occurrence AS VARCHAR), ''),
                    e.frame_domain,
                    cast(? AS VARCHAR)
                )
            ) AS sample_rank
        FROM eligible AS e
        INNER JOIN quotas AS q
            USING (source_country, account_type, frame_domain)
    )
    SELECT
        merge_key,
        key_occurrence,
        source,
        category,
        source_country,
        account_type,
        post_date,
        post_year,
        post_month,
        frame_domain,
        frame_subframe,
        focal_actor,
        text,
        available AS stratum_available,
        requested AS stratum_requested,
        sample_rank
    FROM ranked
    WHERE sample_rank <= requested
    ORDER BY source_country, account_type, frame_domain, sample_rank
    """

    con = duckdb.connect()
    try:
        sample = con.execute(query, [str(path), sample_size, sample_size, seed]).fetchdf()
    finally:
        con.close()

    if sample.empty:
        raise RuntimeError("No eligible substantive frame applications were found.")
    sample["sample_seed"] = seed
    return sample


def _resume_count(out_csv: str, header: list[str]) -> int:
    if not os.path.exists(out_csv):
        return 0
    with open(out_csv, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != header:
            raise SystemExit(
                "Existing output header does not match this run. Choose a different --out path."
            )
        return sum(1 for _ in reader)


def _format_prompt(prompt_template: str, record: dict[str, object]) -> str:
    return prompt_template.format(
        source_country=record["source_country"],
        account_type=record["account_type"],
        frame_domain=record["frame_domain"],
        frame_subframe=record["frame_subframe"],
        focal_country=record["focal_actor"] or "None",
        text=record["text"],
    )


def _label_prompt(prompt: str, row_idx: int, cache_key_suffix: str) -> str:
    return _call_model_with_truncation(
        client=_get_client_for_row(row_idx),
        prompt_template="{text}",
        text=prompt,
        max_tokens=LABEL_MAX_TOKENS,
        cutoffs=LABEL_CUTOFFS,
        cache_key_suffix=cache_key_suffix,
        task="VALENCE",
        allowed_labels=VALENCE_ALLOWED_LABELS,
    )


def label_record(
    record: dict[str, object],
    row_idx: int,
    focal_prompt_template: str,
) -> dict[str, str]:
    if str(record.get("focal_actor", "")).strip() in {"", "None"}:
        return {FOCAL_VALENCE_LABEL_COLUMN: "Not Applicable"}

    focal_prompt = _format_prompt(focal_prompt_template, record)
    return {
        FOCAL_VALENCE_LABEL_COLUMN: _label_prompt(focal_prompt, row_idx, "valence:focal:v3")
    }


def label_sample(sample: pd.DataFrame, out_csv: str) -> None:
    records = sample.to_dict("records")
    header = list(sample.columns) + [FOCAL_VALENCE_LABEL_COLUMN]
    start = _resume_count(out_csv, header)
    if start >= len(records):
        print(f"Output already contains all {len(records):,} sampled post-frame applications: {out_csv}")
        return

    focal_prompt_template = get_focal_valence_prompt()
    handle, writer = _open_output_for_append(out_csv, header)
    try:
        next_submit = start
        next_write = start
        in_flight = {}
        completed: dict[int, dict[str, str]] = {}

        with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
            while next_submit < len(records) and len(in_flight) < PARALLEL:
                future = executor.submit(
                    label_record,
                    records[next_submit],
                    next_submit,
                    focal_prompt_template,
                )
                in_flight[future] = next_submit
                next_submit += 1

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    index = in_flight.pop(future)
                    completed[index] = future.result()
                    if next_submit < len(records):
                        new_future = executor.submit(
                            label_record,
                            records[next_submit],
                            next_submit,
                            focal_prompt_template,
                        )
                        in_flight[new_future] = next_submit
                        next_submit += 1

                while next_write in completed:
                    row = dict(records[next_write])
                    row.update(completed.pop(next_write))
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
                    print(
                        f"[{next_write + 1:,}/{len(records):,}] "
                        f"{row['source_country']} {row['account_type']} "
                        f"{row['frame_domain']} focal={row[FOCAL_VALENCE_LABEL_COLUMN]}",
                        flush=True,
                    )
                    next_write += 1
    finally:
        handle.close()

    print(f"Wrote {out_csv}")


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be greater than 0")

    out_csv = args.out or default_output_path(args.sample_size, args.seed)
    sample = build_stratified_sample(args.parquet, args.sample_size, args.seed)
    stratum_columns = ["source_country", "account_type", "frame_domain"]
    eligible_records = int(
        sample.drop_duplicates(stratum_columns)["stratum_available"].sum()
    )
    print(f"Drew {len(sample):,} post-frame applications from {eligible_records:,} eligible records.")
    print(sample.groupby(["source_country", "account_type", "frame_domain"], as_index=False).size().to_string(index=False))

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    if args.sample_only:
        sample.to_csv(out_csv, index=False)
        print(f"Wrote sample only: {out_csv}")
        return

    init_clients()
    label_sample(sample, out_csv)


if __name__ == "__main__":
    main()
