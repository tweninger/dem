#!/usr/bin/env python3
"""Attach canonical focal labels to the analysis Parquet and rebuild it safely.

The focal CSV is keyed with the same content key and duplicate-occurrence logic
used in ``01_prepare_analysis_data.ipynb``.  It therefore does not rely on CSV
row order when matching labels to posts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import pandas as pd

from canonicalize_focal import DEFAULT_OUTPUT as DEFAULT_MAP


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "derived" / "analysis_posts_clean.parquet"
DEFAULT_FOCAL_CSV = REPO_ROOT / "data" / "dem_labeled_focus2.csv"
DEFAULT_OUTPUT = REPO_ROOT / "derived" / "analysis_posts_clean.parquet"
DEFAULT_AUDIT = REPO_ROOT / "derived" / "canonical_focal_audit.csv"
BASE_COLUMNS = [
    "source", "category", "title", "link", "text", "image source",
    "image alt text", "date posted", "username", "hashtags",
    "number of likes", "number of comments", "number of views", "standardized_date",
]


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_INPUT, help="Existing clean analysis Parquet")
    parser.add_argument("--focal-csv", type=Path, default=DEFAULT_FOCAL_CSV, help="Full DI_focus CSV")
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP, help="Canonical focal map")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Rebuilt Parquet")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT, help="Audit CSV output")
    parser.add_argument("--top", type=int, default=1000, help="Number of canonical actors to retain")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing --out after a successful rebuild")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable DuckDB's terminal progress bar",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"label", "canonical"}
    if not required.issubset(mapping.columns):
        raise ValueError(f"{path} must contain {', '.join(sorted(required))}")
    mapping = mapping[["label", "canonical"]].drop_duplicates("label", keep="last")
    return mapping


def create_views(con: duckdb.DuckDBPyConnection, focal_csv: Path) -> None:
    order_columns = ", ".join(quote_identifier(column) for column in BASE_COLUMNS)
    source_key = """
        sha256(concat_ws(chr(31),
            regexp_replace(trim(coalesce(source, '')), '\\s+', ' ', 'g'),
            regexp_replace(trim(coalesce(link, '')), '\\s+', ' ', 'g'),
            regexp_replace(trim(coalesce(standardized_date, '')), '\\s+', ' ', 'g'),
            regexp_replace(trim(coalesce(text, '')), '\\s+', ' ', 'g')
        ))
    """
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW focal_keyed AS
        WITH source_rows AS (
            SELECT *, {source_key} AS merge_key
            FROM read_csv('{sql_path(focal_csv)}', header=true, all_varchar=true,
                          parallel=false, ignore_errors=false)
        )
        SELECT *, row_number() OVER (PARTITION BY merge_key ORDER BY {order_columns}) AS key_occurrence
        FROM source_rows
        """
    )
    # This mirrors canonicalize_focal.normalize_label: parenthetical text and
    # punctuation are removed; lowercase ASCII letters, digits, and spaces remain.
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW mapped_focal_items AS
        SELECT
            f.merge_key,
            f.key_occurrence,
            trim(regexp_replace(regexp_replace(regexp_replace(lower(strip_accents(item)), '\\([^)]*\\)', '', 'g'), '\\s+', ' ', 'g'), '[^a-z0-9 ]+', '', 'g')) AS label,
            m.canonical
        FROM focal_keyed AS f
        CROSS JOIN UNNEST(str_split(coalesce(f.focal, ''), ';')) AS u(item)
        LEFT JOIN focal_map AS m
          ON trim(regexp_replace(regexp_replace(regexp_replace(lower(strip_accents(item)), '\\([^)]*\\)', '', 'g'), '\\s+', ' ', 'g'), '[^a-z0-9 ]+', '', 'g')) = m.label
        """
    )


def main() -> None:
    args = parse_args()
    for path in (args.parquet, args.focal_csv, args.map):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.top < 1:
        raise ValueError("--top must be at least 1")
    if args.out.exists() and args.out.resolve() != args.parquet.resolve() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.out}; use --overwrite")
    if args.out.resolve() == args.parquet.resolve() and not args.overwrite:
        raise FileExistsError("Refusing to replace the input Parquet; rerun with --overwrite")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp_out = args.out.with_name(f"{args.out.stem}.canonical-focal.tmp{args.out.suffix}")
    if temp_out.exists():
        raise FileExistsError(f"Temporary output already exists: {temp_out}")

    con = duckdb.connect()
    con.execute("SET memory_limit = '6GB'")
    con.execute("SET threads = 2")
    con.execute("SET preserve_insertion_order = false")
    if not args.no_progress:
        con.execute("SET enable_progress_bar = true")
        con.execute("SET progress_bar_time = 1000")
    con.register("focal_map", load_mapping(args.map))
    try:
        print("[1/4] Reading and keying the focal CSV...", flush=True)
        create_views(con, args.focal_csv)
        # Canonical labels outside the global top N become Other. Blank, None,
        # unmapped, and intentionally mapped-to-none outputs are handled below.
        print(f"[2/4] Counting canonical actors and selecting the top {args.top}...", flush=True)
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW top_actors AS
            SELECT canonical
            FROM mapped_focal_items
            WHERE label NOT IN ('', 'none')
              AND canonical IS NOT NULL
              AND canonical <> 'none'
            GROUP BY canonical
            ORDER BY count(*) DESC, canonical
            LIMIT {args.top}
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW cleaned_focal_items AS
            SELECT
                merge_key,
                key_occurrence,
                CASE
                    WHEN label IN ('', 'none') OR canonical = 'none' THEN 'None'
                    WHEN canonical IS NULL OR canonical NOT IN (SELECT canonical FROM top_actors) THEN 'Other'
                    ELSE canonical
                END AS focal_component
            FROM mapped_focal_items
            """
        )
        print("[3/4] Applying source-country rules and writing the rebuilt Parquet...", flush=True)
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW canonical_focal AS
            WITH grouped AS (
                SELECT
                    p.merge_key,
                    p.key_occurrence,
                    CASE
                        WHEN p.category LIKE 'China_%' THEN 'china'
                        WHEN p.category LIKE 'RF_%' THEN 'russia'
                        ELSE NULL
                    END AS source_actor,
                    string_agg(DISTINCT i.focal_component, ';' ORDER BY i.focal_component)
                        FILTER (WHERE i.focal_component NOT IN ('Other', 'None')) AS known_actors,
                    string_agg(DISTINCT i.focal_component, ';' ORDER BY i.focal_component)
                        FILTER (
                            WHERE i.focal_component NOT IN ('Other', 'None')
                              AND i.focal_component <> CASE
                                  WHEN p.category LIKE 'China_%' THEN 'china'
                                  WHEN p.category LIKE 'RF_%' THEN 'russia'
                                  ELSE ''
                              END
                        ) AS non_source_actors,
                    bool_or(i.focal_component = CASE
                        WHEN p.category LIKE 'China_%' THEN 'china'
                        WHEN p.category LIKE 'RF_%' THEN 'russia'
                        ELSE ''
                    END) AS has_source_actor,
                    bool_or(i.focal_component = 'Other') AS has_other
                FROM read_parquet('{sql_path(args.parquet)}') AS p
                LEFT JOIN cleaned_focal_items AS i
                  ON p.merge_key = i.merge_key AND p.key_occurrence = i.key_occurrence
                GROUP BY 1, 2, 3
            )
            SELECT
                merge_key,
                key_occurrence,
                CASE
                    WHEN has_source_actor AND non_source_actors IS NOT NULL THEN non_source_actors
                    WHEN has_source_actor AND has_other THEN 'Other'
                    WHEN has_source_actor THEN source_actor
                    WHEN known_actors IS NOT NULL THEN known_actors
                    WHEN has_other THEN 'Other'
                    ELSE 'None'
                END AS focal_actor
            FROM grouped
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT
                    p.* EXCLUDE (focal_actor, has_focal_actor),
                    f.focal_actor,
                    f.focal_actor <> 'None' AS has_focal_actor
                FROM read_parquet('{sql_path(args.parquet)}') AS p
                LEFT JOIN canonical_focal AS f
                  ON p.merge_key = f.merge_key AND p.key_occurrence = f.key_occurrence
            ) TO '{sql_path(temp_out)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        print("[4/4] Reading the rebuilt file and writing the audit...", flush=True)
        audit = con.execute(
            f"""
            SELECT focal_actor, count(*) AS posts
            FROM read_parquet('{sql_path(temp_out)}')
            GROUP BY focal_actor
            ORDER BY posts DESC, focal_actor
            """
        ).fetchdf()
    finally:
        con.close()

    os.replace(temp_out, args.out)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit, index=False)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.audit}")
    print(audit.head(args.top + 2).to_string(index=False))


if __name__ == "__main__":
    main()
