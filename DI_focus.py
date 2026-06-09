#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pandas as pd

from DI_framing import (
    FOCAL_CUTOFFS,
    FOCAL_MAX_TOKENS,
    PARALLEL,
    TEXT_COL,
    _call_model_with_truncation,
    _get_client_for_row,
    _iter_input_chunks,
    _open_output_for_append,
    _progress_label,
    _read_csv_kwargs,
    _read_input_columns,
    _reservoir_sample_rows,
    init_clients,
)
from prompts import FOCAL_LABEL_COLUMN, get_focal_prompt


def _prediction_column_name(pred_prefix: str) -> str:
    return f"{pred_prefix}{FOCAL_LABEL_COLUMN}" if pred_prefix else FOCAL_LABEL_COLUMN


def _build_output_header(input_columns: list[str], focal_col: str) -> list[str]:
    header = list(input_columns)
    if focal_col not in header:
        header.append(focal_col)
    return header


def _default_output_name(pred_prefix: str) -> str:
    prefix = f"{pred_prefix.lower()}" if pred_prefix else ""
    return f"labeled_{prefix}focal_v1.csv"


def _determine_resume_index(out_csv: str, focal_col: str) -> int:
    if not os.path.exists(out_csv):
        return 0
    with open(out_csv, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        return sum(1 for _ in reader)


def _resume_message(start: int, total_rows: int | None) -> str:
    if total_rows is None:
        return f"Completed {start:,} data rows; next input data row is {start + 1:,}"

    if start >= total_rows:
        return f"Completed all {total_rows:,} data rows"

    return (
        f"Completed {start:,}/{total_rows:,} data rows; next input data row is "
        f"{start + 1:,}/{total_rows:,}"
    )


def label_row(
    row: dict[str, str] | pd.Series,
    row_idx: int,
    text_col: str,
    focal_prompt: str,
    pred_prefix: str,
) -> dict[str, str]:
    client = _get_client_for_row(row_idx)
    text = str(row.get(text_col, ""))
    return {
        _prediction_column_name(pred_prefix): _call_model_with_truncation(
            client=client,
            prompt_template=focal_prompt,
            text=text,
            max_tokens=FOCAL_MAX_TOKENS,
            cutoffs=FOCAL_CUTOFFS,
            cache_key_suffix="focal:v1",
        )
    }


def _label_records(
    records: list[dict[str, str]],
    input_columns: list[str],
    row_offset: int,
    total_rows: int | None,
    writer: csv.DictWriter,
    handle,
    text_col: str,
    focal_prompt: str,
    pred_prefix: str,
) -> None:
    if not records:
        return

    focal_col = _prediction_column_name(pred_prefix)
    next_submit = 0
    next_write = 0
    in_flight = {}
    completed = {}
    flush_every = 10
    since_flush = 0

    with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
        while next_submit < len(records) and len(in_flight) < PARALLEL:
            future = executor.submit(
                label_row,
                records[next_submit],
                row_offset + next_submit,
                text_col,
                focal_prompt,
                pred_prefix,
            )
            in_flight[future] = next_submit
            next_submit += 1

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                local_idx = in_flight.pop(future)
                completed[local_idx] = future.result()

                if next_submit < len(records):
                    new_future = executor.submit(
                        label_row,
                        records[next_submit],
                        row_offset + next_submit,
                        text_col,
                        focal_prompt,
                        pred_prefix,
                    )
                    in_flight[new_future] = next_submit
                    next_submit += 1

            while next_write in completed:
                labels = completed.pop(next_write)
                row = {col: records[next_write].get(col, "") for col in input_columns}
                row.update(labels)
                writer.writerow(row)

                since_flush += 1
                if since_flush >= flush_every:
                    handle.flush()
                    os.fsync(handle.fileno())
                    since_flush = 0

                global_idx = row_offset + next_write
                print(
                    _progress_label(global_idx, total_rows) + f" FOCAL={row.get(focal_col, '')}",
                    flush=True,
                )
                next_write += 1

    handle.flush()
    os.fsync(handle.fileno())


def run_labeling(
    df: pd.DataFrame,
    out_csv: str,
    text_col: str,
    focal_prompt: str,
    pred_prefix: str,
) -> None:
    focal_col = _prediction_column_name(pred_prefix)
    header = _build_output_header(list(df.columns), focal_col)

    total_rows = len(df)
    start = max(0, min(_determine_resume_index(out_csv, focal_col), total_rows))
    print(f"Loaded rows: {total_rows:,}")
    print(_resume_message(start, total_rows))

    handle, writer = _open_output_for_append(out_csv, header)
    try:
        records = df.iloc[start:].to_dict("records")
        _label_records(
            records=records,
            input_columns=list(df.columns),
            row_offset=start,
            total_rows=total_rows,
            writer=writer,
            handle=handle,
            text_col=text_col,
            focal_prompt=focal_prompt,
            pred_prefix=pred_prefix,
        )
    finally:
        handle.close()

    print(f"Wrote {out_csv}")


def run_labeling_stream(
    csv_path: str,
    out_csv: str,
    text_col: str,
    focal_prompt: str,
    pred_prefix: str,
    chunk_size: int,
    limit: int | None,
) -> None:
    focal_col = _prediction_column_name(pred_prefix)
    input_columns = _read_input_columns(csv_path)
    header = _build_output_header(input_columns, focal_col)

    total_rows = limit
    start = _determine_resume_index(out_csv, focal_col)
    if total_rows is not None:
        start = min(start, total_rows)

    print(f"Streaming input CSV in chunks of {chunk_size:,} rows")
    print(_resume_message(start, total_rows))

    handle, writer = _open_output_for_append(out_csv, header)
    try:
        seen_rows = 0
        for chunk in _iter_input_chunks(csv_path, chunk_size):
            chunk_rows = len(chunk)
            chunk_start = seen_rows
            chunk_end = seen_rows + chunk_rows
            seen_rows = chunk_end

            if chunk_end <= start:
                continue
            if total_rows is not None and chunk_start >= total_rows:
                break

            local_start = max(0, start - chunk_start)
            local_end = chunk_rows
            if total_rows is not None:
                local_end = min(local_end, total_rows - chunk_start)
            if local_start >= local_end:
                continue

            records = chunk.iloc[local_start:local_end].to_dict("records")
            _label_records(
                records=records,
                input_columns=input_columns,
                row_offset=chunk_start + local_start,
                total_rows=total_rows,
                writer=writer,
                handle=handle,
                text_col=text_col,
                focal_prompt=focal_prompt,
                pred_prefix=pred_prefix,
            )

            if total_rows is not None and chunk_end >= total_rows:
                break
    finally:
        handle.close()

    print(f"Wrote {out_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV with a text column")
    parser.add_argument("--out", default=None, help="Output CSV path")
    parser.add_argument("--col", default=TEXT_COL, help="Column containing text to label")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Rows to read into memory at a time while streaming the input CSV",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows")
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="With --limit, sample N random rows instead of taking the first N",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed used with --random-sample")
    parser.add_argument("--pred-prefix", default="", help="Prefix for prediction column, e.g. TEST_")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    focal_prompt = get_focal_prompt(1)
    out_csv = args.out or _default_output_name(args.pred_prefix)

    if args.chunk_size <= 0:
        print("ERROR: --chunk-size must be greater than 0", file=sys.stderr)
        sys.exit(2)

    try:
        input_columns = _read_input_columns(args.csv)
    except FileNotFoundError:
        print(f"ERROR: Input CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(2)
    if args.col not in input_columns:
        print(f"ERROR: Column '{args.col}' not found in {args.csv}", file=sys.stderr)
        sys.exit(2)

    if args.limit is not None:
        if args.limit <= 0:
            print("ERROR: --limit must be greater than 0", file=sys.stderr)
            sys.exit(2)
    elif args.random_sample:
        print("ERROR: --random-sample requires --limit", file=sys.stderr)
        sys.exit(2)

    init_clients()
    if args.random_sample:
        print(f"Streaming random sample of {args.limit:,} rows with seed {args.random_seed}")
        df = _reservoir_sample_rows(args.csv, args.limit, args.random_seed, args.chunk_size)
        run_labeling(df, out_csv, args.col, focal_prompt, args.pred_prefix)
    else:
        run_labeling_stream(
            csv_path=args.csv,
            out_csv=out_csv,
            text_col=args.col,
            focal_prompt=focal_prompt,
            pred_prefix=args.pred_prefix,
            chunk_size=args.chunk_size,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
