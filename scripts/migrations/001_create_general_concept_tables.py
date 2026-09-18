"""Create the general-concept registry tables.

This is the parallel of the OMOP `concept` / `concept_synonym` /
`concept_embedding_meta` / `concept_vec` quartet, but for non-clinical
content the LLM extracts from notes (hobbies, foods, devices, activities,
…). It starts empty and is populated by the LLM via `upsert_general_concept`
in `src/queries/general_concepts.py`.

The DDL is idempotent (CREATE IF NOT EXISTS), so this is safe to call on
every connection init from `src/database.py:init_db`.

Note on `general_concept_vec`: this is a sqlite-vec virtual table, so the
extension must be loaded on the connection that runs the CREATE. Callers
should either load sqlite-vec themselves before invoking `apply` or accept
that the virtual-table creation will be skipped (and built lazily on first
`upsert_general_concept`). The non-vec tables always succeed.
"""

from __future__ import annotations

import sqlite3


# Embedding dim must match `src/queries/embeddings.py:EMBEDDING_DIM`. Currently
# 768 for SapBERT. Migration 002_backup_and_switch_sapbert.py renamed the old
# 384-dim tables to _bge_small and created fresh 768-dim shells.
# To revert: run 003_rollback_to_bge_small.py and set this back to 384.
GENERAL_EMBEDDING_DIM = 768


_NONVEC_DDL = [
    # Canonical names live here. `created_by_bin` is best-effort provenance —
    # which bin first coined the canonical — useful when auditing the store
    # to spot canonicals coined from low-quality contexts.
    """
    CREATE TABLE IF NOT EXISTS general_concept (
        concept_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        concept_name    TEXT NOT NULL,
        created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
        created_by_bin  TEXT,
        notes           TEXT
    )
    """,
    # Unique on LOWER() so "Gaming console" / "gaming console" / "GAMING CONSOLE"
    # collapse to one row. The upsert code does the case-insensitive lookup
    # first; the index is the second line of defense.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_general_concept_name_nocase
    ON general_concept (LOWER(concept_name))
    """,
    # Synonyms mirror OMOP's `concept_synonym`. A new source term (e.g.
    # "Xbox" coming in against an existing "Gaming console" canonical) is
    # appended here without creating a new concept row.
    """
    CREATE TABLE IF NOT EXISTS general_concept_synonym (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        concept_id      INTEGER NOT NULL REFERENCES general_concept(concept_id),
        synonym_name    TEXT NOT NULL
    )
    """,
    # SQLite can't enforce UNIQUE(concept_id, LOWER(name)) as a table-level
    # constraint, so we do it via a unique expression index.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_general_synonym_unique
    ON general_concept_synonym (concept_id, LOWER(synonym_name))
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_general_synonym_name_nocase
    ON general_concept_synonym (LOWER(synonym_name))
    """,
    # Mirrors `concept_embedding_meta` from the OMOP store: rowid is the
    # concept_id, embed_text records the exact string we embedded so we can
    # rebuild the vector or audit synonym-enrichment after the fact.
    """
    CREATE TABLE IF NOT EXISTS general_concept_embedding_meta (
        rowid       INTEGER PRIMARY KEY,
        concept_id  INTEGER NOT NULL UNIQUE REFERENCES general_concept(concept_id),
        embed_text  TEXT NOT NULL
    )
    """,
]


def _create_vec_table(conn: sqlite3.Connection) -> bool:
    """Create the `general_concept_vec` virtual table. Returns True on
    success, False if sqlite-vec isn't loaded (the table will be retried
    later from `general_concepts.py` after the extension is loaded)."""
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS general_concept_vec "
            f"USING vec0(embedding float[{GENERAL_EMBEDDING_DIM}])"
        )
        return True
    except sqlite3.OperationalError:
        # `no such module: vec0` — extension not loaded on this connection.
        # Non-fatal: the rest of init_db keeps working; the vec table will
        # be created lazily by `upsert_general_concept`, which opens its
        # own vec-loaded connection.
        return False


def apply(conn: sqlite3.Connection) -> None:
    """Run the migration against an open connection. Idempotent."""
    cursor = conn.cursor()
    try:
        for stmt in _NONVEC_DDL:
            cursor.execute(stmt)
        _create_vec_table(conn)
        conn.commit()
    finally:
        cursor.close()


if __name__ == "__main__":
    # Standalone invocation: open the project DB with sqlite-vec loaded so
    # the virtual table is created on first run too.
    from src.database import get_connection
    try:
        import sqlite_vec
    except ImportError:
        sqlite_vec = None

    conn = get_connection()
    if sqlite_vec is not None:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    try:
        apply(conn)
        print("general-concept tables created (or already present).")
    finally:
        conn.close()
