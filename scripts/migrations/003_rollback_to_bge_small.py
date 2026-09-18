"""Roll back to bge-small (384-dim) embeddings.

Inverse of `002_backup_and_switch_sapbert.py`. Drops the 768-dim
`concept_vec` / `concept_embedding_meta` tables (and all sqlite-vec shadow
tables), then renames the `_bge_small` backup tables back to their standard
names, restoring the 384-dim bge-small index.

After running this script, revert the Python constants:
    src/queries/embeddings.py            → EMBEDDING_MODEL_NAME, EMBEDDING_DIM = 384
    scripts/migrations/001_create_...py  → GENERAL_EMBEDDING_DIM = 384

Usage:
    python scripts/migrations/003_rollback_to_bge_small.py
    python scripts/migrations/003_rollback_to_bge_small.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

try:
    import sqlite_vec
    _HAS_SQLITE_VEC = True
except ImportError:
    _HAS_SQLITE_VEC = False

from src.database import get_connection


# Shadow tables that sqlite-vec creates alongside a vec0 virtual table.
# `DROP TABLE concept_vec` may not remove them all on every sqlite-vec version,
# so we drop them explicitly.
_OMOP_DROP = [
    "concept_embedding_meta",
    "concept_vec",
    "concept_vec_chunks",
    "concept_vec_rowids",
    "concept_vec_vector_chunks00",
    "concept_vec_info",
]

_GENERAL_DROP = [
    "general_concept_embedding_meta",
    "general_concept_vec",
    "general_concept_vec_chunks",
    "general_concept_vec_rowids",
    "general_concept_vec_vector_chunks00",
    "general_concept_vec_info",
]

_OMOP_RESTORE = [
    ("concept_vec_bge_small",                   "concept_vec"),
    ("concept_vec_bge_small_chunks",            "concept_vec_chunks"),
    ("concept_vec_bge_small_rowids",            "concept_vec_rowids"),
    ("concept_vec_bge_small_vector_chunks00",   "concept_vec_vector_chunks00"),
    ("concept_vec_bge_small_info",              "concept_vec_info"),
    ("concept_embedding_meta_bge_small",        "concept_embedding_meta"),
]

_GENERAL_RESTORE = [
    ("general_concept_vec_bge_small",                   "general_concept_vec"),
    ("general_concept_vec_bge_small_chunks",            "general_concept_vec_chunks"),
    ("general_concept_vec_bge_small_rowids",            "general_concept_vec_rowids"),
    ("general_concept_vec_bge_small_vector_chunks00",   "general_concept_vec_vector_chunks00"),
    ("general_concept_vec_bge_small_info",              "general_concept_vec_info"),
    ("general_concept_embedding_meta_bge_small",        "general_concept_embedding_meta"),
]


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone() is not None


def _run(dry_run: bool) -> None:
    conn = get_connection()

    if not _HAS_SQLITE_VEC:
        print("ERROR: sqlite-vec is not installed. Install it with `pip install sqlite-vec`.")
        sys.exit(1)

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Idempotency: if backup tables are absent, nothing to restore.
    if not _table_exists(conn, "concept_vec_bge_small") and \
       not _table_exists(conn, "general_concept_vec_bge_small"):
        print("No backup tables found (concept_vec_bge_small / general_concept_vec_bge_small).")
        print("Nothing to roll back.")
        conn.close()
        return

    drop_stmts = [
        f"DROP TABLE IF EXISTS {t}"
        for t in _OMOP_DROP + _GENERAL_DROP
    ]

    restore_pairs = [
        (src, dst)
        for src, dst in _OMOP_RESTORE + _GENERAL_RESTORE
        if _table_exists(conn, src)
    ]
    rename_stmts = [
        f"ALTER TABLE {src} RENAME TO {dst}"
        for src, dst in restore_pairs
    ]

    all_stmts = drop_stmts + rename_stmts

    if dry_run:
        print("-- DRY RUN — no changes will be made\n")
        for s in all_stmts:
            print(s.strip() + ";")
        conn.close()
        return

    try:
        conn.execute("BEGIN")
        for s in all_stmts:
            conn.execute(s)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        conn.close()
        raise

    print("Rollback complete.")
    print(f"Dropped {len(drop_stmts)} 768-dim tables.")
    print(f"Restored {len(rename_stmts)} bge-small backup tables:")
    for src, dst in restore_pairs:
        print(f"  {src} → {dst}")
    print("\nRemember to revert the Python constants:")
    print("  src/queries/embeddings.py: EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5', EMBEDDING_DIM = 384")
    print("  scripts/migrations/001_create_general_concept_tables.py: GENERAL_EMBEDDING_DIM = 384")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the SQL that would run without executing it.",
    )
    args = parser.parse_args()
    _run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
