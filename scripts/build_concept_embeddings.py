"""One-time builder for the OMOP concept embedding index.

For every row in `concept`, composes an enriched embedding text:
    "<concept_name>. Also known as: <syn1>; <syn2>; ..."
where the synonyms come from `concept_synonym`. The whole thing is embedded
with the model configured in src/queries/embeddings.py and stored in
sqlite-vec's `concept_vec` virtual table next to `concept` in the same DB.

Resumable: skips concept_ids already present in `concept_embedding_meta`,
so an interrupted run picks up where it left off.

Usage:
    # Full run (all OMOP concepts):
    python scripts/build_concept_embeddings.py

    # Limit to specific domains for a quick first pass:
    python scripts/build_concept_embeddings.py --domains Drug Condition Measurement

    # Bigger batches if you have a GPU:
    python scripts/build_concept_embeddings.py --batch-size 256

    # Tune how many concepts we pull from SQLite per round-trip:
    python scripts/build_concept_embeddings.py --chunk-size 10000
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import sqlite_vec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.database import get_connection
from src.queries.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    build_embed_text,
    embed_texts,
    get_model,
)


def _connect():
    """Open the project DB with sqlite-vec loaded and pragmas tuned for bulk insert."""
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    # Bulk-insert tuning. WAL is the default we already use; bumping cache
    # and turning off synchronous makes the embedding writes ~3x faster.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -200000")  # ~200 MB
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def _ensure_schema(conn) -> None:
    """Create the meta table and the vector virtual table if they don't exist.

    Raises RuntimeError if concept_vec already exists with a dimension that
    does not match EMBEDDING_DIM — the migration script has not been run yet
    (or the model constant is set incorrectly). Run:
        python scripts/migrations/002_backup_and_switch_sapbert.py
    to rename the old tables and create fresh shells before rebuilding.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'concept_vec'"
    ).fetchone()
    if row is not None:
        ddl = row[0] or ""
        m = re.search(r"float\[(\d+)\]", ddl)
        if m:
            existing_dim = int(m.group(1))
            if existing_dim != EMBEDDING_DIM:
                raise RuntimeError(
                    f"concept_vec exists with dim={existing_dim} but "
                    f"EMBEDDING_DIM={EMBEDDING_DIM}. "
                    f"Run `python scripts/migrations/002_backup_and_switch_sapbert.py` "
                    f"to back up the old tables and create fresh {EMBEDDING_DIM}-dim ones "
                    f"before rebuilding the index."
                )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS concept_embedding_meta (
            rowid       INTEGER PRIMARY KEY,
            concept_id  INTEGER NOT NULL UNIQUE,
            embed_text  TEXT NOT NULL
        )
    """)
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS concept_vec USING vec0(embedding float[{EMBEDDING_DIM}])"
    )
    conn.commit()


def _existing_concept_ids(conn) -> set[int]:
    """All concept_ids already embedded — skipped on resume."""
    cursor = conn.cursor()
    cursor.execute("SELECT concept_id FROM concept_embedding_meta")
    ids = {row[0] for row in cursor.fetchall()}
    cursor.close()
    return ids


def _count_target_concepts(conn, domains: list[str] | None) -> int:
    cursor = conn.cursor()
    if domains:
        placeholders = ",".join(["?"] * len(domains))
        cursor.execute(
            f"SELECT COUNT(*) FROM concept WHERE domain_id IN ({placeholders})",
            domains,
        )
    else:
        cursor.execute("SELECT COUNT(*) FROM concept")
    n = cursor.fetchone()[0]
    cursor.close()
    return int(n)


def _iter_concept_chunks(conn, domains: list[str] | None, chunk_size: int):
    """Yield chunks of (concept_id, concept_name, [synonym, ...]) tuples.

    We stream by ascending concept_id so resume can skip cleanly.
    Synonyms are pulled per-chunk in a second query, batched to avoid the
    cost of a giant LEFT JOIN on the whole `concept` table.
    """
    cursor = conn.cursor()

    base_sql = "SELECT concept_id, concept_name FROM concept"
    params: list = []
    if domains:
        base_sql += " WHERE domain_id IN (" + ",".join(["?"] * len(domains)) + ")"
        params.extend(domains)
    base_sql += " ORDER BY concept_id"

    cursor.execute(base_sql, params)

    chunk: list[tuple[int, str]] = []
    for row in cursor:
        chunk.append((int(row["concept_id"]), row["concept_name"] or ""))
        if len(chunk) >= chunk_size:
            yield _attach_synonyms(conn, chunk)
            chunk = []
    if chunk:
        yield _attach_synonyms(conn, chunk)
    cursor.close()


def _attach_synonyms(conn, chunk: list[tuple[int, str]]) -> list[tuple[int, str, list[str]]]:
    """Given (concept_id, concept_name) pairs, fetch all synonyms in one query."""
    if not chunk:
        return []
    ids = [cid for cid, _name in chunk]
    placeholders = ",".join(["?"] * len(ids))
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT concept_id, concept_synonym_name
        FROM concept_synonym
        WHERE concept_id IN ({placeholders})
        """,
        ids,
    )
    syn_map: dict[int, list[str]] = {cid: [] for cid in ids}
    for row in cursor:
        cid = int(row["concept_id"])
        s = row["concept_synonym_name"]
        if s:
            syn_map[cid].append(s)
    cursor.close()
    return [(cid, name, syn_map.get(cid, [])) for cid, name in chunk]


def _flush_batch(conn, ids: list[int], texts: list[str], vectors) -> None:
    """Insert one batch into both tables atomically."""
    cursor = conn.cursor()
    try:
        # Vectors first; the meta row's rowid must equal the concept_id we
        # write into concept_vec. We use INSERT OR IGNORE so re-running on
        # a partially-built index never errors out.
        cursor.executemany(
            "INSERT OR IGNORE INTO concept_vec(rowid, embedding) VALUES (?, ?)",
            [(cid, vec.tobytes()) for cid, vec in zip(ids, vectors)],
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO concept_embedding_meta(rowid, concept_id, embed_text) VALUES (?, ?, ?)",
            [(cid, cid, txt) for cid, txt in zip(ids, texts)],
        )
        conn.commit()
    finally:
        cursor.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the OMOP concept embedding index.")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Restrict to OMOP domain_id values (e.g. Drug Condition Measurement). "
        "Default: all domains.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size. Bump to 128–256 on GPU, keep around 64 on Apple Silicon.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of concept rows fetched from SQLite per round-trip.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip concept_ids already in concept_embedding_meta. (Default: on.)",
    )
    args = parser.parse_args()

    print(f"Model: {EMBEDDING_MODEL_NAME}  (dim={EMBEDDING_DIM})")
    print(f"Loading model — first run downloads weights, subsequent runs are instant ...")
    get_model()  # warm the cache before opening the DB

    conn = _connect()
    try:
        _ensure_schema(conn)
        existing = _existing_concept_ids(conn) if args.resume else set()
        total = _count_target_concepts(conn, args.domains)
        if args.resume and existing:
            print(f"Resuming: {len(existing):,} concepts already embedded.")
        print(f"Target: {total:,} concepts" + (f" in domains {args.domains}" if args.domains else ""))

        done = 0
        skipped = 0
        empty = 0
        start = time.time()
        last_report = start

        for chunk in _iter_concept_chunks(conn, args.domains, args.chunk_size):
            # Filter out concepts we already embedded.
            todo = [(cid, name, syns) for cid, name, syns in chunk if cid not in existing]
            skipped += len(chunk) - len(todo)
            if not todo:
                continue

            texts = [build_embed_text(name, syns) for _cid, name, syns in todo]
            # Drop concepts whose embed text is empty (concept_name is NULL/blank
            # and no synonyms). They can't be searched against meaningfully.
            keep_idx = [i for i, t in enumerate(texts) if t.strip()]
            empty += len(todo) - len(keep_idx)
            if not keep_idx:
                continue
            ids = [todo[i][0] for i in keep_idx]
            texts = [texts[i] for i in keep_idx]

            # Embed in sub-batches sized to the model, then flush as one DB write.
            vectors = embed_texts(texts, batch_size=args.batch_size, normalize=True)
            _flush_batch(conn, ids, texts, vectors)

            done += len(ids)
            existing.update(ids)

            now = time.time()
            if now - last_report >= 5.0:
                elapsed = now - start
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = max(0, total - skipped - done)
                eta_min = (remaining / rate / 60.0) if rate > 0 else float("inf")
                print(
                    f"  embedded {done:,}  (skip {skipped:,}, empty {empty:,})  "
                    f"{rate:.0f}/s  ETA {eta_min:5.1f} min"
                )
                last_report = now

        print()
        print(f"Done. Embedded {done:,} new concepts; skipped {skipped:,}; empty {empty:,}.")
        print(f"Index now covers {len(existing):,} concepts.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
