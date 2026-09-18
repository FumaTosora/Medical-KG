"""Import SapBERT embeddings from numpy files into the project SQLite DB.

Run this on the main machine after copying sapbert_concept_ids.npy and
sapbert_embeddings.npy back from the embedding machine.

Steps this script performs:
  1. Runs migration 002 to back up the current bge-small tables and create
     fresh 768-dim concept_vec / concept_embedding_meta shells.
  2. Reads the .npy files and the embed-texts file.
  3. Bulk-inserts everything into the main DB.

Usage:
    python sapbert_embedding_builder/import_sapbert_embeddings.py

    # Custom file locations:
    python sapbert_embedding_builder/import_sapbert_embeddings.py \\
        --ids   /tmp/sapbert_concept_ids.npy \\
        --vecs  /tmp/sapbert_embeddings.npy \\
        --texts /tmp/sapbert_embed_texts.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sqlite_vec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.database import get_connection

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_IDS   = SCRIPT_DIR / "sapbert_concept_ids.npy"
DEFAULT_VECS  = SCRIPT_DIR / "sapbert_embeddings.npy"
DEFAULT_TEXTS = SCRIPT_DIR / "sapbert_embed_texts.txt"

EMBEDDING_DIM = 768


def _connect():
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -200000")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def _run_migration():
    """Back up bge-small tables and create 768-dim shells if not done yet."""
    import importlib.util
    migration = PROJECT_ROOT / "scripts" / "migrations" / "002_backup_and_switch_sapbert.py"
    spec = importlib.util.spec_from_file_location("migration_002", migration)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._run(dry_run=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids",   default=str(DEFAULT_IDS))
    parser.add_argument("--vecs",  default=str(DEFAULT_VECS))
    parser.add_argument("--texts", default=str(DEFAULT_TEXTS))
    args = parser.parse_args()

    ids_path   = Path(args.ids)
    vecs_path  = Path(args.vecs)
    texts_path = Path(args.texts)

    for p in (ids_path, vecs_path):
        if not p.exists():
            print(f"ERROR: {p} not found.")
            print("Copy sapbert_concept_ids.npy and sapbert_embeddings.npy into "
                  "sapbert_embedding_builder/ first.")
            sys.exit(1)

    print("Loading numpy files ...")
    concept_ids = np.load(str(ids_path)).astype(np.int32)
    embeddings  = np.load(str(vecs_path)).astype(np.float32)
    assert len(concept_ids) == len(embeddings), "ids/vecs length mismatch"

    embed_texts: list[str] = []
    if texts_path.exists():
        with open(texts_path, encoding="utf-8") as f:
            embed_texts = [line.rstrip("\n") for line in f]
        if len(embed_texts) != len(concept_ids):
            print(f"WARNING: embed_texts length ({len(embed_texts)}) != concept_ids "
                  f"({len(concept_ids)}). Filling missing entries with empty string.")
            embed_texts += [""] * (len(concept_ids) - len(embed_texts))
    else:
        print(f"WARNING: {texts_path} not found — embed_text will be empty.")
        embed_texts = [""] * len(concept_ids)

    print(f"  {len(concept_ids):,} concepts, dim={embeddings.shape[1]}")

    print("\nRunning migration 002 (back up bge-small, create 768-dim shells) ...")
    _run_migration()

    print("\nInserting into DB ...")
    conn = _connect()
    try:
        done = 0
        start = time.time()
        last_report = start
        CHUNK = 5000
        total = len(concept_ids)

        for i in range(0, total, CHUNK):
            chunk_ids   = concept_ids[i: i + CHUNK]
            chunk_vecs  = embeddings[i: i + CHUNK]
            chunk_texts = embed_texts[i: i + CHUNK]

            cursor = conn.cursor()
            cursor.executemany(
                "INSERT OR IGNORE INTO concept_vec(rowid, embedding) VALUES (?, ?)",
                [(int(cid), vec.tobytes()) for cid, vec in zip(chunk_ids, chunk_vecs)],
            )
            cursor.executemany(
                "INSERT OR IGNORE INTO concept_embedding_meta(rowid, concept_id, embed_text) "
                "VALUES (?, ?, ?)",
                [(int(cid), int(cid), txt)
                 for cid, txt in zip(chunk_ids, chunk_texts)],
            )
            conn.commit()
            cursor.close()

            done += len(chunk_ids)
            now = time.time()
            if now - last_report >= 5.0:
                rate = done / (now - start)
                print(f"  inserted {done:,}/{total:,}  {rate:.0f}/s")
                last_report = now

        print(f"\nDone. Inserted {done:,} concepts.")
        print("\nSet these env vars before running the KG pipeline:")
        print("  EMBEDDING_MODEL=cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        print("  EMBEDDING_DIM=768")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
