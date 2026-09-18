"""Back up bge-small embedding tables and create fresh 768-dim shells for SapBERT.

Renames the existing 384-dim vec/meta tables to a `_bge_small` suffix so the
6.4M OMOP embeddings (and 66 general-concept embeddings) are preserved, then
creates empty 768-dim `concept_vec` / `concept_embedding_meta` tables ready
for `scripts/build_concept_embeddings.py` to fill.

Idempotent: exits cleanly if the backup tables already exist (migration
already ran) or if there is nothing to back up (fresh DB).

sqlite-vec `vec0` virtual tables have five shadow tables in addition to the
virtual table entry itself. All five must be renamed individually — renaming
only the virtual table entry leaves the shadows under the original names and
breaks queries against the renamed table.

Usage:
    python scripts/migrations/002_backup_and_switch_sapbert.py
    python scripts/migrations/002_backup_and_switch_sapbert.py --dry-run
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


_OMOP_RENAMES = [
    ("concept_vec",                    "concept_vec_bge_small"),
    ("concept_vec_chunks",             "concept_vec_bge_small_chunks"),
    ("concept_vec_rowids",             "concept_vec_bge_small_rowids"),
    ("concept_vec_vector_chunks00",    "concept_vec_bge_small_vector_chunks00"),
    ("concept_vec_info",               "concept_vec_bge_small_info"),
    ("concept_embedding_meta",         "concept_embedding_meta_bge_small"),
]

_GENERAL_RENAMES = [
    ("general_concept_vec",                    "general_concept_vec_bge_small"),
    ("general_concept_vec_chunks",             "general_concept_vec_bge_small_chunks"),
    ("general_concept_vec_rowids",             "general_concept_vec_bge_small_rowids"),
    ("general_concept_vec_vector_chunks00",    "general_concept_vec_bge_small_vector_chunks00"),
    ("general_concept_vec_info",               "general_concept_vec_bge_small_info"),
    ("general_concept_embedding_meta",         "general_concept_embedding_meta_bge_small"),
]

_NEW_OMOP_DDL = [
    """
    CREATE TABLE concept_embedding_meta (
        rowid       INTEGER PRIMARY KEY,
        concept_id  INTEGER NOT NULL UNIQUE,
        embed_text  TEXT NOT NULL
    )
    """,
    "CREATE VIRTUAL TABLE concept_vec USING vec0(embedding float[768])",
]

_NEW_GENERAL_DDL = [
    """
    CREATE TABLE general_concept_embedding_meta (
        rowid       INTEGER PRIMARY KEY,
        concept_id  INTEGER NOT NULL UNIQUE REFERENCES general_concept(concept_id),
        embed_text  TEXT NOT NULL
    )
    """,
    "CREATE VIRTUAL TABLE general_concept_vec USING vec0(embedding float[768])",
]


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone()
    return row is not None


def _run(dry_run: bool) -> None:
    conn = get_connection()

    if not _HAS_SQLITE_VEC:
        print("ERROR: sqlite-vec is not installed. Install it with `pip install sqlite-vec`.")
        sys.exit(1)

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Idempotency: if backup tables already exist, nothing to do.
    if _table_exists(conn, "concept_vec_bge_small"):
        print("Already migrated — concept_vec_bge_small exists. Nothing to do.")
        conn.close()
        return

    # If concept_vec doesn't exist at all this is a fresh DB; create the
    # 768-dim shells directly without renaming anything.
    omop_vec_present = _table_exists(conn, "concept_vec")
    general_vec_present = _table_exists(conn, "general_concept_vec")

    if not omop_vec_present and not general_vec_present:
        print("No embedding tables found — creating fresh 768-dim shells.")
        stmts = _NEW_OMOP_DDL + _NEW_GENERAL_DDL
        if dry_run:
            for s in stmts:
                print(s.strip())
        else:
            for s in stmts:
                conn.execute(s)
            conn.commit()
            print("Done. Fresh 768-dim concept_vec and general_concept_vec created.")
        conn.close()
        return

    # Normal case: rename old tables then create empty 768-dim ones.
    renames = []
    if omop_vec_present:
        renames.extend(_OMOP_RENAMES)
    if general_vec_present:
        renames.extend(_GENERAL_RENAMES)

    new_ddl = []
    if omop_vec_present:
        new_ddl.extend(_NEW_OMOP_DDL)
    if general_vec_present:
        new_ddl.extend(_NEW_GENERAL_DDL)

    rename_stmts = [
        f"ALTER TABLE {src} RENAME TO {dst}"
        for src, dst in renames
        if _table_exists(conn, src)
    ]
    all_stmts = rename_stmts + new_ddl

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

    renamed = [f"  {src} → {dst}" for src, dst in renames if _table_exists(conn, dst)]
    print("Migration complete.")
    print(f"Renamed {len(rename_stmts)} tables:")
    for src, dst in renames:
        print(f"  {src} → {dst}")
    print("Created fresh 768-dim tables:")
    for s in new_ddl:
        # extract the table name from the DDL
        import re
        m = re.search(r"(?:TABLE|VIEW)\s+(\S+)", s, re.IGNORECASE)
        if m:
            print(f"  {m.group(1)}")
    print("\nNext steps:")
    print("  1. Update EMBEDDING_DIM to 768 in src/queries/embeddings.py")
    print("  2. Update EMBEDDING_MODEL_NAME to pritamdeka/SapBERT-from-PubMedBERT-fulltext")
    print("  3. Run: python scripts/build_concept_embeddings.py")

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
