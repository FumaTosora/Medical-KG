import csv
import re
import sqlite3
from pathlib import Path
from typing import Iterable


_IDENTIFIER_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_identifier(name: str) -> str:
    cleaned = _IDENTIFIER_RE.sub("_", (name or "").strip())
    cleaned = cleaned.strip("_").lower()
    if not cleaned:
        raise ValueError("Identifier must not be empty.")
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def _detect_delimiter(csv_path: Path) -> str:
    """Detect the delimiter, ignoring trailing semicolon padding from OMOP files."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        raw_sample = f.read(8192)
    # Strip trailing semicolons from each line before sniffing
    cleaned = "\n".join(line.rstrip(";") for line in raw_sample.splitlines())
    if not cleaned.strip():
        return ","
    try:
        dialect = csv.Sniffer().sniff(cleaned, delimiters=",\t;|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _strip_trailing_semicolons(filepath: Path):
    """Generator that yields lines from a file with trailing semicolons stripped."""
    with filepath.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            yield line.rstrip(";\r\n") + "\n"


def _quoted_ident(name: str) -> str:
    return f'"{name}"'


def _create_table_from_header(conn: sqlite3.Connection, table: str, header: list[str], replace_table: bool) -> None:
    table_name = _sanitize_identifier(table)
    columns = [_sanitize_identifier(col) for col in header]

    if replace_table:
        conn.execute(f"DROP TABLE IF EXISTS {_quoted_ident(table_name)}")

    col_defs = ", ".join(f"{_quoted_ident(col)} TEXT" for col in columns)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {_quoted_ident(table_name)} ({col_defs})")


def _create_lookup_indexes(conn: sqlite3.Connection) -> None:
    # Raw-column indexes — useful when queries don't wrap the column in LOWER().
    conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_name ON concept(concept_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_domain ON concept(domain_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_synonym_name ON concept_synonym(concept_synonym_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_synonym_concept_id ON concept_synonym(concept_id)")

    # Case-insensitive expression indexes — the lookup queries in
    # `src/queries/db_tooling.py:_fetch_omop_candidates` wrap both sides in
    # LOWER() to ignore case, which prevents SQLite from using the raw
    # `idx_concept_name` / `idx_concept_synonym_name` indexes above. Without
    # these LOWER() indexes the planner falls back to scanning by
    # `idx_concept_domain`, which is essentially "scan most of the table" for
    # the big domains (Drug ≈ 4.9M rows). A single lookup went from 150 ms to
    # 2 ms once these were in place on a 6.4M-concept DB.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_name_lower ON concept(LOWER(concept_name))")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_synonym_name_lower ON concept_synonym(LOWER(concept_synonym_name))")

    # The synonym query JOINs `concept` on `concept_id`, but `concept.concept_id`
    # is plain TEXT with no PK / no index — so the JOIN had to use
    # idx_concept_domain (huge) instead. This index makes the JOIN a fast
    # equality probe and was the difference between 2.8 s and 2 ms on a
    # `HAART` synonym lookup against the Drug domain.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_concept_id ON concept(concept_id)")

    # ANALYZE feeds the query planner real selectivity statistics so it picks
    # the right index when there are multiple candidates (e.g. domain vs
    # name). One-time cost (~10–20 s on a 6.4M-row concept table); without
    # it, even with the right indexes available, the planner can still
    # choose poorly.
    conn.execute("ANALYZE")

    # concept_relationship is a 39 M-row table; without a compound index on
    # (concept_id_1, concept_id_2) the lookup_omop_relation tool would do a
    # full-table scan for each edge candidate.  These indexes are created with
    # IF NOT EXISTS so they're skipped on re-import runs.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cr_concept_pair "
        "ON concept_relationship (concept_id_1, concept_id_2)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cr_reverse_pair "
        "ON concept_relationship (concept_id_2, concept_id_1)"
    )
    # The relationship table is small (722 rows) but the JOIN on relationship_id
    # benefits from an explicit index when the planner can't guess selectivity.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_relationship_id "
        "ON relationship (relationship_id)"
    )


def import_csv_to_sqlite(
    csv_path: Path,
    conn: sqlite3.Connection,
    table_name: str | None = None,
    replace_table: bool = False,
    batch_size: int = 5000,
) -> dict:
    target_table = _sanitize_identifier(table_name or csv_path.stem)
    delimiter = _detect_delimiter(csv_path)

    lines = _strip_trailing_semicolons(csv_path)
    # Use QUOTE_NONE so that quotes wrapping entire lines don't swallow delimiters
    reader = csv.reader(lines, delimiter=delimiter, quoting=csv.QUOTE_NONE)
    raw_header = next(reader, None)
    if not raw_header:
        return {"table": target_table, "rows": 0, "status": "empty"}

    # Strip quotes and drop empty trailing columns
    header = [col.strip().strip('"').strip() for col in raw_header]
    valid_indices = [i for i, col in enumerate(header) if col]
    header = [header[i] for i in valid_indices]
    if not header:
        return {"table": target_table, "rows": 0, "status": "empty_header"}

    _create_table_from_header(conn, target_table, header, replace_table)
    columns = [_sanitize_identifier(col) for col in header]
    placeholders = ", ".join(["?"] * len(columns))
    columns_sql = ", ".join(_quoted_ident(col) for col in columns)
    insert_sql = f"INSERT INTO {_quoted_ident(target_table)} ({columns_sql}) VALUES ({placeholders})"

    row_count = 0
    batch: list[list[str]] = []
    for row in reader:
        # Pick only the columns that had non-empty headers; strip quotes from values
        picked = [(row[i].strip().strip('"') if i < len(row) else "") for i in valid_indices]
        if len(picked) < len(columns):
            picked += [""] * (len(columns) - len(picked))
        batch.append(picked)
        if len(batch) >= batch_size:
            conn.executemany(insert_sql, batch)
            row_count += len(batch)
            batch = []

    if batch:
        conn.executemany(insert_sql, batch)
        row_count += len(batch)

    return {"table": target_table, "rows": row_count, "status": "ok"}


def import_omop_folder_to_sqlite(
    omop_folder: str | Path,
    sqlite_db_path: str | Path,
    replace_existing: bool = True,
    include_tables: Iterable[str] | None = None,
) -> list[dict]:
    omop_root = Path(omop_folder)
    if not omop_root.exists():
        raise FileNotFoundError(f"OMOP folder not found: {omop_root}")

    include_set = {t.lower() for t in include_tables} if include_tables else None
    csv_files = sorted(p for p in omop_root.glob("*.csv") if p.is_file())

    db_path = Path(sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)

    results: list[dict] = []
    try:
        for csv_file in csv_files:
            table = csv_file.stem.lower()
            if include_set is not None and table not in include_set:
                continue
            result = import_csv_to_sqlite(
                csv_path=csv_file,
                conn=conn,
                table_name=table,
                replace_table=replace_existing,
            )
            results.append(result)

        _create_lookup_indexes(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return results

