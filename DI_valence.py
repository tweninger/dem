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
from tqdm import tqdm

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
    get_focal_valence_general_prompt,
    get_focal_valence_prompt,
)


# GPT occasionally spells out the polarity despite the requested short labels.
# Accept those equivalents at validation time, then always write the canonical form.
FOCAL_VALENCE_ALLOWED_LABELS = ["Pro", "Anti", "Neutral", "Positive", "Negative"]
FOCAL_VALENCE_ALIASES = {"Positive": "Pro", "Negative": "Anti"}
DEFAULT_PARQUET = "derived/analysis_posts_clean.parquet"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce focal valence labels from a stratified sample or every unique post."
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
        "--full",
        action="store_true",
        help="Label each eligible unique post once instead of drawing a stratified sample",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum posts to label in --full mode (useful for smoke tests)",
    )
    parser.add_argument(
        "--chunk-vectors",
        type=int,
        default=10,
        help="DuckDB vectors fetched at once in --full mode (default: 10; about 20K rows)",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Write the deterministic sample without calling the model",
    )
    return parser


def default_output_path(sample_size: int, seed: int) -> str:
    kind = "pilot" if sample_size <= 24 else "sample"
    return f"derived/valence/focal_valence_{kind}_canonical_n{sample_size}_seed{seed}.csv"


def default_full_output_path() -> str:
    return "derived/valence/focal_valence_full_canonical.csv"


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


def print_valence_stats(out_csv: str) -> None:
    """Print quick valence distributions for a completed labeling run."""
    labeled = pd.read_csv(out_csv, low_memory=False)

    focal_stats = pd.crosstab(
        index=labeled["focal_actor"].fillna("None"),
        columns=labeled[FOCAL_VALENCE_LABEL_COLUMN],
    ).reindex(columns=["Pro", "Anti", "Neutral"], fill_value=0)
    focal_stats["Total"] = focal_stats.sum(axis=1)

    print("\nFocal valence by focal actor")
    print(focal_stats.to_string())


def _format_prompt(prompt_template: str, record: dict[str, object]) -> str:
    return prompt_template.format(
        source_country=record["source_country"],
        account_type=record["account_type"],
        frame_domain=record["frame_domain"],
        frame_subframe=record["frame_subframe"],
        focal_country=record["focal_actor"] or "None",
        text=record["text"],
    )


def _label_prompt(
    prompt: str,
    row_idx: int,
    cache_key_suffix: str,
    allowed_labels: list[str],
) -> str:
    label = _call_model_with_truncation(
        client=_get_client_for_row(row_idx),
        prompt_template="{text}",
        text=prompt,
        max_tokens=LABEL_MAX_TOKENS,
        cutoffs=LABEL_CUTOFFS,
        cache_key_suffix=cache_key_suffix,
        task="VALENCE",
        allowed_labels=allowed_labels,
        strict_initial_allowlist=True,
    )
    return FOCAL_VALENCE_ALIASES.get(label, label)


def label_record(
    record: dict[str, object],
    row_idx: int,
    focal_prompt_template: str,
    general_prompt_template: str,
) -> dict[str, str]:
    focal_actor = str(record.get("focal_actor", "")).strip().casefold()
    prompt_template = (
        general_prompt_template if focal_actor in {"", "none", "other"} else focal_prompt_template
    )
    focal_prompt = _format_prompt(prompt_template, record)
    try:
        label = _label_prompt(
            focal_prompt,
            row_idx,
            "valence:focal:canonical",
            FOCAL_VALENCE_ALLOWED_LABELS,
        )
    except Exception as exc:
        print(
            "[warn] valence failed after retries; writing Neutral "
            f"for row={row_idx:,} merge_key={record.get('merge_key', '')}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        label = "Neutral"
    return {FOCAL_VALENCE_LABEL_COLUMN: label}


def label_sample(sample: pd.DataFrame, out_csv: str) -> None:
    records = sample.to_dict("records")
    header = list(sample.columns) + [FOCAL_VALENCE_LABEL_COLUMN]
    start = _resume_count(out_csv, header)
    if start >= len(records):
        print(f"Output already contains all {len(records):,} sampled post-frame applications: {out_csv}")
        print_valence_stats(out_csv)
        return

    focal_prompt_template = get_focal_valence_prompt()
    general_prompt_template = get_focal_valence_general_prompt()
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
                    general_prompt_template,
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
                            general_prompt_template,
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
                        f"{row['frame_domain']}/{row['frame_subframe']} "
                        f"focal_actor={row['focal_actor'] or 'None'} "
                        f"focal_valence={row[FOCAL_VALENCE_LABEL_COLUMN]}",
                        flush=True,
                    )
                    next_write += 1
    finally:
        handle.close()

    print(f"Wrote {out_csv}")
    print_valence_stats(out_csv)


FULL_OUTPUT_COLUMNS = [
    "merge_key", "key_occurrence", "source", "category", "source_country", "account_type",
    "post_date", "post_year", "post_month", "focal_actor", FOCAL_VALENCE_LABEL_COLUMN,
]


def _full_post_query(parquet_path: str) -> str:
    return """
        SELECT
            merge_key,
            key_occurrence,
            source,
            category,
            CASE
                WHEN category LIKE 'China_%' THEN 'China'
                WHEN category LIKE 'RF_%' THEN 'Russia'
                ELSE NULL
            END AS source_country,
            CASE
                WHEN category LIKE '%Media%' THEN 'Media'
                WHEN category LIKE '%Diplomat%' THEN 'Diplomatic'
                ELSE NULL
            END AS account_type,
            post_date,
            post_year,
            post_month,
            focal_actor,
            text,
            '' AS frame_domain,
            '' AS frame_subframe
        FROM read_parquet(?)
        WHERE text IS NOT NULL
          AND trim(text) <> ''
          AND (category LIKE 'China_%' OR category LIKE 'RF_%')
        ORDER BY merge_key, key_occurrence
    """


def _full_post_counts(parquet_path: str) -> tuple[int, int]:
    con = duckdb.connect()
    try:
        posts, none_posts = con.execute(
            """
            SELECT count(*), count(*) FILTER (WHERE focal_actor = 'None')
            FROM read_parquet(?)
            WHERE text IS NOT NULL
              AND trim(text) <> ''
              AND (category LIKE 'China_%' OR category LIKE 'RF_%')
            """,
            [parquet_path],
        ).fetchone()
        return int(posts), int(none_posts)
    finally:
        con.close()


def _label_full_records(
    records: list[dict[str, object]],
    row_offset: int,
    total_rows: int,
    writer: csv.DictWriter,
    handle,
    focal_prompt_template: str,
    general_prompt_template: str,
    progress: tqdm,
) -> None:
    next_submit = 0
    next_write = 0
    in_flight = {}
    completed: dict[int, dict[str, str]] = {}
    flush_every = 100
    since_flush = 0

    with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
        while next_submit < len(records) and len(in_flight) < PARALLEL:
            future = executor.submit(
                label_record,
                records[next_submit],
                row_offset + next_submit,
                focal_prompt_template,
                general_prompt_template,
            )
            in_flight[future] = next_submit
            next_submit += 1

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                local_index = in_flight.pop(future)
                completed[local_index] = future.result()
                if next_submit < len(records):
                    new_future = executor.submit(
                        label_record,
                        records[next_submit],
                        row_offset + next_submit,
                        focal_prompt_template,
                        general_prompt_template,
                    )
                    in_flight[new_future] = next_submit
                    next_submit += 1

            while next_write in completed:
                record = records[next_write]
                row = {column: record.get(column, "") for column in FULL_OUTPUT_COLUMNS}
                row.update(completed.pop(next_write))
                writer.writerow(row)
                since_flush += 1
                if since_flush >= flush_every:
                    handle.flush()
                    os.fsync(handle.fileno())
                    since_flush = 0
                progress.update(1)
                next_write += 1

    handle.flush()
    os.fsync(handle.fileno())


def label_full_corpus(
    parquet_path: str,
    out_csv: str,
    limit: int | None,
    chunk_vectors: int,
) -> None:
    if chunk_vectors <= 0:
        raise ValueError("--chunk-vectors must be greater than zero")
    available, none_posts = _full_post_counts(parquet_path)
    if none_posts == 0:
        raise SystemExit(
            "The input Parquet has zero focal_actor='None' rows. "
            "Rebuild it with scripts/rebuild_canonical_focal_parquet.py before running full valence."
        )
    total_rows = min(available, limit) if limit is not None else available
    start = min(_resume_count(out_csv, FULL_OUTPUT_COLUMNS), total_rows)
    print(f"Eligible unique posts: {available:,}")
    print(f"Posts with focal_actor='None': {none_posts:,}")
    print(f"Resuming at {start:,}/{total_rows:,}")
    if start >= total_rows:
        print(f"Output already contains all {total_rows:,} posts: {out_csv}")
        print_valence_stats(out_csv)
        return

    focal_prompt_template = get_focal_valence_prompt()
    general_prompt_template = get_focal_valence_general_prompt()
    handle, writer = _open_output_for_append(out_csv, FULL_OUTPUT_COLUMNS)
    con = duckdb.connect()
    con.execute("SET threads = 2")
    try:
        cursor = con.execute(_full_post_query(parquet_path), [parquet_path])
        seen = 0
        with tqdm(total=total_rows, initial=start, unit="posts", desc="Full valence") as progress:
            while seen < total_rows:
                chunk = cursor.fetch_df_chunk(vectors_per_chunk=chunk_vectors)
                if chunk.empty:
                    break
                chunk_start = seen
                seen += len(chunk)
                if seen <= start:
                    continue
                records = chunk.iloc[max(0, start - chunk_start): max(0, total_rows - chunk_start)]
                if records.empty:
                    continue
                _label_full_records(
                    records=records.to_dict("records"),
                    row_offset=max(start, chunk_start),
                    total_rows=total_rows,
                    writer=writer,
                    handle=handle,
                    focal_prompt_template=focal_prompt_template,
                    general_prompt_template=general_prompt_template,
                    progress=progress,
                )
    finally:
        con.close()
        handle.close()

    print(f"Wrote {out_csv}")
    print_valence_stats(out_csv)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be greater than 0")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than 0")

    if args.full:
        out_csv = args.out or default_full_output_path()
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        init_clients()
        label_full_corpus(args.parquet, out_csv, args.limit, args.chunk_vectors)
        return

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
