#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import socket
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pandas as pd
from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, OpenAI, RateLimitError
from tqdm import tqdm

from prompts import FOCAL_LABEL_COLUMN, TASK_LABEL_COLUMNS, get_focal_prompt, get_task_prompts
from settings import (
    API_KEY,
    BASE_HOST,
    BASE_PORT,
    BASE_URL,
    FOCAL_MAX_TOKENS,
    LABEL_MAX_TOKENS,
    MAX_PORTS_TO_SCAN,
    MAX_RETRIES,
    MODEL,
    PARALLEL,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_SECONDS,
    chat_completion_kwargs,
    openai_base_url,
    prompt_cache_kwargs,
    token_limit_param_name,
)

TEXT_COL = "text"
NO_CATEGORY = "No Category"
LABEL_CUTOFFS = [2500, 1800, 1200, 800, 400]
FOCAL_CUTOFFS = [2500, 1800, 1200, 800, 400]

_clients: list[OpenAI] = []


def _is_port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _discover_servers(host: str, base_port: int, max_ports: int) -> list[int]:
    return [p for p in range(base_port, base_port + max_ports) if _is_port_open(host, p)]


def init_clients() -> None:
    global _clients
    if BASE_URL:
        print(f"Using configured endpoint: {BASE_URL}")
        _clients = [OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT) for _ in range(max(1, PARALLEL))]
        return

    ports = _discover_servers(BASE_HOST, BASE_PORT, MAX_PORTS_TO_SCAN)
    if not ports:
        print(
            f"ERROR: No llama-server instances found on {BASE_HOST}:{BASE_PORT}-{BASE_PORT + MAX_PORTS_TO_SCAN - 1}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Discovered servers on ports: {ports}")
    _clients = [OpenAI(base_url=openai_base_url(p), api_key=API_KEY, timeout=REQUEST_TIMEOUT) for p in ports]


def _get_client_for_row(idx: int) -> OpenAI:
    if not _clients:
        raise RuntimeError("Clients not initialized; call init_clients() first.")
    return _clients[idx % len(_clients)]


def _parse_model_output(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""
    try:
        arr = json.loads(text)
        if isinstance(arr, list) and arr:
            return str(arr[0]).strip().strip('"').strip("'")
    except Exception:
        pass

    match = re.search(r"\[(.*?)\]", text, flags=re.S)
    if match:
        inside = match.group(1).split(",")
        return inside[0].strip().strip('"').strip("'")

    return text.splitlines()[0].strip().strip('"').strip("'")


def _normalize_label(value: str) -> str:
    s = str(value).strip()
    if not s:
        return ""
    if s.lower().startswith("no categor"):
        return NO_CATEGORY
    return s


def _truncate_text(text: str, max_chars: int) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def _call_model(client: OpenAI, prompt: str, max_tokens: int, cache_key_suffix: str | None = None) -> str:
    token_param = token_limit_param_name()
    last_err = None
    request_kwargs = chat_completion_kwargs()

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                **{token_param: max_tokens},
                **prompt_cache_kwargs(cache_key_suffix),
                **request_kwargs,
            )
            return _normalize_label(_parse_model_output(response.choices[0].message.content))
        except BadRequestError as exc:
            msg = str(exc).lower()
            removed_any = False
            if "unsupported value" in msg and "temperature" in msg and "temperature" in request_kwargs:
                request_kwargs.pop("temperature", None)
                removed_any = True
            if "unsupported value" in msg and "top_p" in msg and "top_p" in request_kwargs:
                request_kwargs.pop("top_p", None)
                removed_any = True
            if "unsupported value" in msg and "reasoning_effort" in msg and "reasoning_effort" in request_kwargs:
                request_kwargs.pop("reasoning_effort", None)
                removed_any = True
            if removed_any:
                print(
                    f"[warn] {MODEL} rejected one or more request controls; retrying without unsupported parameters.",
                    file=sys.stderr,
                )
                continue
            raise
        except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError) as exc:
            last_err = exc
            if attempt >= MAX_RETRIES:
                break
            sleep_seconds = RETRY_BACKOFF_SECONDS * (attempt + 1)
            print(
                f"[retry] attempt={attempt + 1}/{MAX_RETRIES} after {type(exc).__name__}: sleeping {sleep_seconds:.1f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    if last_err is not None:
        raise last_err
    raise RuntimeError("Model call failed without raising a known exception")


def _call_model_with_truncation(
    client: OpenAI,
    prompt_template: str,
    text: str,
    max_tokens: int,
    cutoffs: list[int],
    cache_key_suffix: str | None = None,
) -> str:
    last_err = None
    for max_chars in cutoffs:
        prompt = prompt_template.format(text=_truncate_text(text, max_chars), texts=_truncate_text(text, max_chars))
        try:
            return _call_model(client, prompt, max_tokens=max_tokens, cache_key_suffix=cache_key_suffix)
        except BadRequestError as exc:
            msg = str(exc)
            if "exceeds the available context size" in msg or "exceed_context_size_error" in msg:
                last_err = exc
                continue
            raise
    if last_err:
        raise last_err
    return ""


def _canonical_task(task: str) -> str:
    return task.strip().upper()


def _parse_task_args(raw_tasks: list[str] | None, task_prompts: dict[str, str]) -> list[str]:
    if not raw_tasks:
        return ["AUT", "DEM", "WEST"]

    parsed: list[str] = []
    for raw in raw_tasks:
        for part in raw.split(","):
            task = _canonical_task(part)
            if not task:
                continue
            if task not in task_prompts:
                raise SystemExit(f"Unsupported task '{part}'. Use aut, dem, and/or west.")
            if task not in parsed:
                parsed.append(task)
    if not parsed:
        raise SystemExit("No valid tasks supplied to --type.")
    return parsed


def _prediction_column_name(base_col: str, pred_prefix: str) -> str:
    return f"{pred_prefix}{base_col}" if pred_prefix else base_col


def _output_columns(tasks: list[str], include_focal: bool, pred_prefix: str) -> list[str]:
    cols = [_prediction_column_name(TASK_LABEL_COLUMNS[task], pred_prefix) for task in tasks]
    if include_focal:
        cols.append(_prediction_column_name(FOCAL_LABEL_COLUMN, pred_prefix))
    return cols


def _build_output_header(input_columns: list[str], label_cols: list[str]) -> list[str]:
    header = list(input_columns)
    for col in label_cols:
        if col not in header:
            header.append(col)
    return header


def _row_needs_resume(
    row: dict[str, str],
    tasks: list[str],
    include_focal: bool,
    pred_prefix: str,
) -> bool:
    task_cols = [_prediction_column_name(TASK_LABEL_COLUMNS[task], pred_prefix) for task in tasks]
    for col in task_cols:
        if str(row.get(col, "")).strip() == "":
            return True

    if not include_focal:
        return False

    focal_col = _prediction_column_name(FOCAL_LABEL_COLUMN, pred_prefix)
    has_any_content_label = any(_normalize_label(row.get(col, "")) != NO_CATEGORY for col in task_cols)
    focal_value = str(row.get(focal_col, "")).strip()
    return has_any_content_label and focal_value == ""


def _determine_resume_index(
    out_csv: str,
    label_cols: list[str],
    tasks: list[str],
    include_focal: bool,
    pred_prefix: str,
) -> int:
    if not os.path.exists(out_csv):
        return 0
    with open(out_csv, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        if not all(col in reader.fieldnames for col in label_cols):
            return sum(1 for _ in reader)

        for idx, row in enumerate(reader):
            if _row_needs_resume(row, tasks, include_focal, pred_prefix):
                return idx
        return idx + 1 if "idx" in locals() else 0


def _open_output_for_append(path: str, header: list[str]):
    exists = os.path.exists(path)
    has_header = False

    if exists and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8-sig", newline="") as existing_handle:
            reader = csv.reader(existing_handle)
            existing_header = next(reader, None)
        if existing_header and existing_header != header:
            raise SystemExit(
                "Existing output header does not match the expected schema. "
                "Choose a new --out file or remove the old output before resuming."
            )
        has_header = existing_header is not None

    handle = open(path, "a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=header,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )
    if not has_header:
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
    return handle, writer


def _default_output_name(tasks: list[str], include_focal: bool) -> str:
    suffix = "_".join(task.lower() for task in tasks)
    if include_focal:
        suffix = f"{suffix}_focal"
    return f"labeled_{suffix}.csv"


def _has_any_content_label(labels: dict[str, str], tasks: list[str], pred_prefix: str) -> bool:
    return any(
        _normalize_label(labels.get(_prediction_column_name(TASK_LABEL_COLUMNS[task], pred_prefix), "")) != NO_CATEGORY
        for task in tasks
    )


def label_row(
    row: dict[str, str] | pd.Series,
    row_idx: int,
    tasks: list[str],
    include_focal: bool,
    text_col: str,
    task_prompts: dict[str, str],
    focal_prompt: str,
    pred_prefix: str,
) -> dict[str, str]:
    client = _get_client_for_row(row_idx)
    text = str(row.get(text_col, ""))
    out: dict[str, str] = {}

    for task in tasks:
        out[_prediction_column_name(TASK_LABEL_COLUMNS[task], pred_prefix)] = _call_model_with_truncation(
            client=client,
            prompt_template=task_prompts[task],
            text=text,
            max_tokens=LABEL_MAX_TOKENS,
            cutoffs=LABEL_CUTOFFS,
            cache_key_suffix=f"task:{task}",
        )

    if include_focal:
        if _has_any_content_label(out, tasks, pred_prefix):
            out[_prediction_column_name(FOCAL_LABEL_COLUMN, pred_prefix)] = _call_model_with_truncation(
                client=client,
                prompt_template=focal_prompt,
                text=text,
                max_tokens=FOCAL_MAX_TOKENS,
                cutoffs=FOCAL_CUTOFFS,
                cache_key_suffix="focal",
            )
        else:
            out[_prediction_column_name(FOCAL_LABEL_COLUMN, pred_prefix)] = ""

    return out


def _read_csv_kwargs() -> dict:
    return {
        "dtype": str,
        "keep_default_na": False,
        "encoding": "utf-8-sig",
        "engine": "python",
        "on_bad_lines": "skip",
    }


def _read_input_columns(csv_path: str) -> list[str]:
    return list(pd.read_csv(csv_path, nrows=0, **_read_csv_kwargs()).columns)


def _iter_input_chunks(csv_path: str, chunk_size: int):
    yield from pd.read_csv(csv_path, chunksize=chunk_size, **_read_csv_kwargs())


def _reservoir_sample_rows(csv_path: str, sample_size: int, seed: int, chunk_size: int) -> pd.DataFrame:
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    columns: list[str] = []
    seen = 0

    for chunk in _iter_input_chunks(csv_path, chunk_size):
        if not columns:
            columns = list(chunk.columns)
        for row in chunk.to_dict("records"):
            tagged = {"__row_idx__": seen}
            tagged.update(row)
            if len(reservoir) < sample_size:
                reservoir.append(tagged)
            else:
                chosen = rng.randint(0, seen)
                if chosen < sample_size:
                    reservoir[chosen] = tagged
            seen += 1

    reservoir.sort(key=lambda row: int(row["__row_idx__"]))
    sampled_rows = [{k: v for k, v in row.items() if k != "__row_idx__"} for row in reservoir]
    return pd.DataFrame(sampled_rows, columns=columns)


def _progress_label(row_idx: int, total_rows: int | None) -> str:
    if total_rows is None:
        return f"[{row_idx + 1}]"
    return f"[{row_idx + 1}/{total_rows}]"


def _label_records(
    records: list[dict[str, str]],
    input_columns: list[str],
    row_offset: int,
    total_rows: int | None,
    writer: csv.DictWriter,
    handle,
    tasks: list[str],
    include_focal: bool,
    text_col: str,
    task_prompts: dict[str, str],
    focal_prompt: str,
    pred_prefix: str,
    debug_progress: bool,
) -> None:
    if not records:
        return

    label_cols = _output_columns(tasks, include_focal, pred_prefix)
    next_submit = 0
    next_write = 0
    in_flight = {}
    completed = {}
    flush_every = 10
    since_flush = 0
    pbar = tqdm(total=total_rows and len(records), desc="Labeling", smoothing=0.05) if False else None

    with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
        while next_submit < len(records) and len(in_flight) < PARALLEL:
            future = executor.submit(
                label_row,
                records[next_submit],
                row_offset + next_submit,
                tasks,
                include_focal,
                text_col,
                task_prompts,
                focal_prompt,
                pred_prefix,
            )
            in_flight[future] = next_submit
            next_submit += 1
            if debug_progress:
                tqdm.write(
                    f"[debug] submitted={row_offset + next_submit} in_flight={len(in_flight)} completed_buffered={len(completed)} next_write={row_offset + next_write}"
                )

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                local_idx = in_flight.pop(future)
                global_idx = row_offset + local_idx
                try:
                    completed[local_idx] = future.result()
                except Exception as exc:
                    print(f"[error] row {global_idx} failed: {exc}", file=sys.stderr)
                    traceback.print_exc()
                    completed[local_idx] = {col: "" for col in label_cols}

                if next_submit < len(records):
                    new_future = executor.submit(
                        label_row,
                        records[next_submit],
                        row_offset + next_submit,
                        tasks,
                        include_focal,
                        text_col,
                        task_prompts,
                        focal_prompt,
                        pred_prefix,
                    )
                    in_flight[new_future] = next_submit
                    next_submit += 1
                    if debug_progress:
                        tqdm.write(
                            f"[debug] submitted={row_offset + next_submit} in_flight={len(in_flight)} completed_buffered={len(completed)} next_write={row_offset + next_write}"
                        )

            if debug_progress and completed and next_write not in completed:
                waiting_on = row_offset + next_write
                ready_min = row_offset + min(completed)
                ready_max = row_offset + max(completed)
                tqdm.write(
                    f"[debug] waiting_on_row={waiting_on + 1} in_flight={len(in_flight)} completed_buffered={len(completed)} buffered_range={ready_min + 1}-{ready_max + 1}"
                )

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
                parts = [
                    f"{task}={row.get(_prediction_column_name(TASK_LABEL_COLUMNS[task], pred_prefix), '')}"
                    for task in tasks
                ]
                if include_focal:
                    parts.append(f"FOCAL={row.get(_prediction_column_name(FOCAL_LABEL_COLUMN, pred_prefix), '')}")
                tqdm.write(_progress_label(global_idx, total_rows) + " " + " | ".join(parts))
                next_write += 1

    handle.flush()
    os.fsync(handle.fileno())


def run_labeling(
    df: pd.DataFrame,
    out_csv: str,
    tasks: list[str],
    include_focal: bool,
    text_col: str,
    task_prompts: dict[str, str],
    focal_prompt: str,
    pred_prefix: str,
    debug_progress: bool,
) -> None:
    label_cols = _output_columns(tasks, include_focal, pred_prefix)
    header = _build_output_header(list(df.columns), label_cols)

    total_rows = len(df)
    start = max(0, min(_determine_resume_index(out_csv, label_cols, tasks, include_focal, pred_prefix), total_rows))
    print(f"Loaded rows: {total_rows:,}")
    print(f"Resuming at row {start:,}/{total_rows:,}")

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
            tasks=tasks,
            include_focal=include_focal,
            text_col=text_col,
            task_prompts=task_prompts,
            focal_prompt=focal_prompt,
            pred_prefix=pred_prefix,
            debug_progress=debug_progress,
        )
    finally:
        handle.close()

    print(f"Wrote {out_csv}")


def run_labeling_stream(
    csv_path: str,
    out_csv: str,
    tasks: list[str],
    include_focal: bool,
    text_col: str,
    task_prompts: dict[str, str],
    focal_prompt: str,
    pred_prefix: str,
    debug_progress: bool,
    chunk_size: int,
    limit: int | None,
) -> None:
    label_cols = _output_columns(tasks, include_focal, pred_prefix)
    input_columns = _read_input_columns(csv_path)
    header = _build_output_header(input_columns, label_cols)

    total_rows = limit
    start = _determine_resume_index(out_csv, label_cols, tasks, include_focal, pred_prefix)
    if total_rows is not None:
        start = min(start, total_rows)

    print(f"Streaming input CSV in chunks of {chunk_size:,} rows")
    if total_rows is None:
        print(f"Resuming at row {start:,}")
    else:
        print(f"Resuming at row {start:,}/{total_rows:,}")

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
                tasks=tasks,
                include_focal=include_focal,
                text_col=text_col,
                task_prompts=task_prompts,
                focal_prompt=focal_prompt,
                pred_prefix=pred_prefix,
                debug_progress=debug_progress,
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
    parser.add_argument("--chunk-size", type=int, default=1000, help="Rows to read into memory at a time while streaming the input CSV")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows")
    parser.add_argument("--random-sample", action="store_true", help="With --limit, sample N random rows instead of taking the first N")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed used with --random-sample")
    parser.add_argument("--type", nargs="+", default=None, help="One or more task types: aut dem west")
    parser.add_argument("--focal", action="store_true", help="Also label the focal country/actor when a content frame is found")
    parser.add_argument("--prompt-version", type=int, choices=[1, 2], default=2, help="Prompt set version to use")
    parser.add_argument("--pred-prefix", default="", help="Prefix for prediction columns, e.g. TEST_")
    parser.add_argument("--debug-progress", action="store_true", help="Print extra progress info about queued, in-flight, and buffered rows")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    task_prompts = get_task_prompts(args.prompt_version)
    focal_prompt = get_focal_prompt(args.prompt_version)
    tasks = _parse_task_args(args.type, task_prompts)
    out_csv = args.out or _default_output_name(tasks, args.focal)

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
        run_labeling(df, out_csv, tasks, args.focal, args.col, task_prompts, focal_prompt, args.pred_prefix, args.debug_progress)
    else:
        run_labeling_stream(
            csv_path=args.csv,
            out_csv=out_csv,
            tasks=tasks,
            include_focal=args.focal,
            text_col=args.col,
            task_prompts=task_prompts,
            focal_prompt=focal_prompt,
            pred_prefix=args.pred_prefix,
            debug_progress=args.debug_progress,
            chunk_size=args.chunk_size,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
