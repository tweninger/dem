#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pandas as pd
from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, OpenAI, RateLimitError
from tqdm import tqdm

from prompts import TASK_LABEL_COLUMNS, get_focal_prompt, get_task_prompts
from settings import (
    API_KEY,
    BASE_HOST,
    BASE_PORT,
    BASE_URL,
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

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                **{token_param: max_tokens},
                **prompt_cache_kwargs(cache_key_suffix),
                **chat_completion_kwargs(),
            )
            return _normalize_label(_parse_model_output(response.choices[0].message.content))
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
        cols.append(_prediction_column_name("FOCAL_COUNTRY", pred_prefix))
    return cols


def _determine_resume_index(out_csv: str, label_cols: list[str]) -> int:
    if not os.path.exists(out_csv):
        return 0
    try:
        out_df = pd.read_csv(out_csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if not all(col in out_df.columns for col in label_cols):
            return len(out_df)
        done = pd.Series(True, index=out_df.index)
        for col in label_cols:
            done &= out_df[col].astype(str).str.strip().ne("")
        incomplete = done[~done]
        if len(incomplete) == 0:
            return len(out_df)
        return int(incomplete.index[0])
    except Exception:
        with open(out_csv, "r", encoding="utf-8-sig") as handle:
            return max(0, sum(1 for _ in handle) - 1)


def _open_output_for_append(path: str, header: list[str]):
    exists = os.path.exists(path)
    handle = open(path, "a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=header,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )
    if not exists:
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
    row: pd.Series,
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
            max_tokens=32,
            cutoffs=LABEL_CUTOFFS,
            cache_key_suffix=f"task:{task}",
        )

    if include_focal:
        if _has_any_content_label(out, tasks, pred_prefix):
            out[_prediction_column_name("FOCAL_COUNTRY", pred_prefix)] = _call_model_with_truncation(
                client=client,
                prompt_template=focal_prompt,
                text=text,
                max_tokens=8,
                cutoffs=FOCAL_CUTOFFS,
                cache_key_suffix="focal",
            )
        else:
            out[_prediction_column_name("FOCAL_COUNTRY", pred_prefix)] = ""

    return out


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
    header = list(df.columns)
    for col in label_cols:
        if col not in header:
            header.append(col)

    total_rows = len(df)
    start = max(0, min(_determine_resume_index(out_csv, label_cols), total_rows))
    print(f"Loaded rows: {total_rows:,}")
    print(f"Resuming at row {start:,}/{total_rows:,}")

    handle, writer = _open_output_for_append(out_csv, header)
    try:
        pbar = tqdm(total=total_rows - start, desc="Labeling", smoothing=0.05)
        next_submit = start
        next_write = start
        in_flight = {}
        completed = {}
        flush_every = 10
        since_flush = 0

        with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
            while next_submit < total_rows and len(in_flight) < PARALLEL:
                future = executor.submit(
                    label_row,
                    df.iloc[next_submit],
                    next_submit,
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
                        f"[debug] submitted={next_submit} in_flight={len(in_flight)} completed_buffered={len(completed)} next_write={next_write}"
                    )

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    row_idx = in_flight.pop(future)
                    try:
                        completed[row_idx] = future.result()
                    except Exception as exc:
                        print(f"[error] row {row_idx} failed: {exc}", file=sys.stderr)
                        traceback.print_exc()
                        completed[row_idx] = {col: "" for col in label_cols}

                    if next_submit < total_rows:
                        new_future = executor.submit(
                            label_row,
                            df.iloc[next_submit],
                            next_submit,
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
                                f"[debug] submitted={next_submit} in_flight={len(in_flight)} completed_buffered={len(completed)} next_write={next_write}"
                            )

                if debug_progress and completed and next_write not in completed:
                    waiting_on = next_write
                    ready_min = min(completed)
                    ready_max = max(completed)
                    tqdm.write(
                        f"[debug] waiting_on_row={waiting_on + 1} in_flight={len(in_flight)} completed_buffered={len(completed)} buffered_range={ready_min + 1}-{ready_max + 1}"
                    )

                while next_write in completed:
                    labels = completed.pop(next_write)
                    row = {col: df.iloc[next_write][col] for col in df.columns}
                    row.update(labels)
                    writer.writerow(row)

                    since_flush += 1
                    if since_flush >= flush_every:
                        handle.flush()
                        os.fsync(handle.fileno())
                        since_flush = 0

                    pbar.update(1)
                    parts = [
                        f"{task}={row.get(_prediction_column_name(TASK_LABEL_COLUMNS[task], pred_prefix), '')}"
                        for task in tasks
                    ]
                    if include_focal:
                        parts.append(f"FOCAL={row.get(_prediction_column_name('FOCAL_COUNTRY', pred_prefix), '')}")
                    tqdm.write(f"[{next_write + 1}/{total_rows}] " + " | ".join(parts))
                    next_write += 1

        pbar.close()
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()

    print(f"Wrote {out_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV with a text column")
    parser.add_argument("--out", default=None, help="Output CSV path")
    parser.add_argument("--col", default=TEXT_COL, help="Column containing text to label")
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

    try:
        df = pd.read_csv(
            args.csv,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            engine="python",
            on_bad_lines="skip",
        )
    except FileNotFoundError:
        print(f"ERROR: Input CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(2)
    if args.col not in df.columns:
        print(f"ERROR: Column '{args.col}' not found in {args.csv}", file=sys.stderr)
        sys.exit(2)

    if args.limit is not None:
        if args.limit <= 0:
            print("ERROR: --limit must be greater than 0", file=sys.stderr)
            sys.exit(2)
        if args.random_sample:
            n = min(args.limit, len(df))
            df = df.sample(n=n, random_state=args.random_seed).sort_index().copy()
        else:
            df = df.head(args.limit).copy()
    elif args.random_sample:
        print("ERROR: --random-sample requires --limit", file=sys.stderr)
        sys.exit(2)

    init_clients()
    run_labeling(df, out_csv, tasks, args.focal, args.col, task_prompts, focal_prompt, args.pred_prefix, args.debug_progress)


if __name__ == "__main__":
    main()
