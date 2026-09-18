"""Build a SapBERT embedding index from OMOP concept CSVs.

Reads CONCEPT.csv and CONCEPT_SYNONYM.csv from the same directory as this
script, embeds every concept with SapBERT
(cambridgeltl/SapBERT-from-PubMedBERT-fulltext, 768-dim), and writes two
numpy files:

    sapbert_concept_ids.npy      — int32 array of concept_ids, shape (N,)
    sapbert_embeddings.npy       — float32 array of vectors,   shape (N, 768)
    sapbert_embed_texts.txt      — one embed-text per line (for debugging)

No SQLite or sqlite-vec needed on this machine. Copy the .npy files back to
the main machine and run import_sapbert_embeddings.py there to load them
into the project DB.

This script is STANDALONE — no imports from the main project's src/.

Usage:
    # Full run (all ~6.4M OMOP concepts):
    python build_sapbert_embeddings.py

    # GPU-optimised batch size:
    python build_sapbert_embeddings.py --batch-size 256

    # Limit to specific domains for a smaller first pass:
    python build_sapbert_embeddings.py --domains Drug Condition Measurement
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as st_models

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
EMBEDDING_DIM = 768
MAX_SYNONYMS_PER_CONCEPT = 8

CONCEPT_CSV = SCRIPT_DIR / "CONCEPT.csv"
SYNONYM_CSV = SCRIPT_DIR / "CONCEPT_SYNONYM.csv"

DEFAULT_IDS_OUT   = SCRIPT_DIR / "sapbert_concept_ids.npy"
DEFAULT_VECS_OUT  = SCRIPT_DIR / "sapbert_embeddings.npy"
DEFAULT_TEXTS_OUT = SCRIPT_DIR / "sapbert_embed_texts.txt"


# ---------------------------------------------------------------------------
# Embed-text composition — identical to build_embed_text in the main project
# ---------------------------------------------------------------------------

def build_embed_text(concept_name: str, synonyms: list[str]) -> str:
    name = (concept_name or "").strip()
    seen = {name.lower()} if name else set()
    cleaned: list[str] = []
    for syn in synonyms or []:
        s = (syn or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
    cleaned.sort(key=lambda s: (len(s), s.lower()))
    cleaned = cleaned[:MAX_SYNONYMS_PER_CONCEPT]
    if not cleaned:
        return name
    return f"{name}. Also known as: " + "; ".join(cleaned) + "."


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print(f"Loading {MODEL_NAME} ...")
        word_model = st_models.Transformer(MODEL_NAME)
        pooling = st_models.Pooling(
            word_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        _MODEL = SentenceTransformer(modules=[word_model, pooling])
        print(f"  Model loaded. Output dim: {_MODEL.get_sentence_embedding_dimension()}")
    return _MODEL


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_concepts(domain_filter: set[str] | None) -> list[tuple[int, str, str]]:
    print(f"Reading {CONCEPT_CSV} ...")
    t0 = time.time()
    rows: list[tuple[int, str, str]] = []
    with open(CONCEPT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                cid = int(row["concept_id"])
            except (KeyError, ValueError):
                continue
            domain = (row.get("domain_id") or "").strip()
            if domain_filter and domain not in domain_filter:
                continue
            name = (row.get("concept_name") or "").strip()
            rows.append((cid, name, domain))
    print(f"  {len(rows):,} concepts in {time.time() - t0:.1f}s"
          + (f" (domains: {sorted(domain_filter)})" if domain_filter else ""))
    return rows


def _load_synonyms(concept_ids: set[int]) -> dict[int, list[str]]:
    if not SYNONYM_CSV.exists():
        print(f"WARNING: {SYNONYM_CSV} not found — proceeding without synonyms.")
        return {}
    print(f"Reading {SYNONYM_CSV} ...")
    t0 = time.time()
    syn_map: dict[int, list[str]] = defaultdict(list)
    with open(SYNONYM_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                cid = int(row["concept_id"])
            except (KeyError, ValueError):
                continue
            if cid not in concept_ids:
                continue
            s = (row.get("concept_synonym_name") or "").strip()
            if s:
                syn_map[cid].append(s)
    print(f"  Synonyms for {len(syn_map):,} concepts in {time.time() - t0:.1f}s")
    return dict(syn_map)


# ---------------------------------------------------------------------------
# Resumable output helpers
# ---------------------------------------------------------------------------

def _load_existing(ids_path: Path) -> set[int]:
    """Return concept_ids already written to disk (for resume)."""
    if ids_path.exists():
        arr = np.load(str(ids_path))
        return set(arr.tolist())
    return set()


def _append_results(
    ids_path: Path,
    vecs_path: Path,
    texts_path: Path,
    ids: list[int],
    vectors: np.ndarray,
    texts: list[str],
) -> None:
    """Append a batch to the three output files."""
    new_ids = np.array(ids, dtype=np.int32)
    if ids_path.exists():
        old_ids = np.load(str(ids_path))
        old_vecs = np.load(str(vecs_path))
        new_ids = np.concatenate([old_ids, new_ids])
        vectors = np.concatenate([old_vecs, vectors])
    np.save(str(ids_path), new_ids)
    np.save(str(vecs_path), vectors)
    with open(texts_path, "a", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding batch size. Use 128–256 on a GPU.")
    parser.add_argument("--domains", nargs="+", default=None, metavar="DOMAIN",
                        help="Restrict to OMOP domain_id values (e.g. Drug Condition).")
    parser.add_argument("--ids-out",   default=str(DEFAULT_IDS_OUT))
    parser.add_argument("--vecs-out",  default=str(DEFAULT_VECS_OUT))
    parser.add_argument("--texts-out", default=str(DEFAULT_TEXTS_OUT))
    args = parser.parse_args()

    ids_path   = Path(args.ids_out)
    vecs_path  = Path(args.vecs_out)
    texts_path = Path(args.texts_out)
    domain_filter = set(args.domains) if args.domains else None

    if not CONCEPT_CSV.exists():
        print(f"ERROR: {CONCEPT_CSV} not found.")
        print("Copy CONCEPT.csv and CONCEPT_SYNONYM.csv into the same folder as this script.")
        sys.exit(1)

    get_model()  # warm before CSV I/O

    concepts  = _load_concepts(domain_filter)
    concept_ids = {cid for cid, _n, _d in concepts}
    syn_map   = _load_synonyms(concept_ids)
    total     = len(concepts)

    existing = _load_existing(ids_path)
    if existing:
        print(f"Resuming: {len(existing):,} concepts already embedded.")
    print(f"Target: {total:,} concepts")

    done = skipped = empty = 0
    start = last_report = time.time()

    CHUNK = 5000
    for chunk_start in range(0, total, CHUNK):
        chunk = concepts[chunk_start: chunk_start + CHUNK]

        todo = [
            (cid, name, syn_map.get(cid, []))
            for cid, name, _domain in chunk
            if cid not in existing
        ]
        skipped += len(chunk) - len(todo)
        if not todo:
            continue

        texts = [build_embed_text(name, syns) for _cid, name, syns in todo]
        keep_idx = [i for i, t in enumerate(texts) if t.strip()]
        empty += len(todo) - len(keep_idx)
        if not keep_idx:
            continue

        ids    = [todo[i][0] for i in keep_idx]
        texts  = [texts[i]   for i in keep_idx]

        vectors = embed_texts(texts, batch_size=args.batch_size)
        _append_results(ids_path, vecs_path, texts_path, ids, vectors, texts)

        done += len(ids)
        existing.update(ids)

        now = time.time()
        if now - last_report >= 5.0:
            elapsed = now - start
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = max(0, total - skipped - done)
            eta_min = (remaining / rate / 60.0) if rate > 0 else float("inf")
            print(f"  embedded {done:,}  (skip {skipped:,}, empty {empty:,})  "
                  f"{rate:.0f}/s  ETA {eta_min:5.1f} min")
            last_report = now

    print()
    print(f"Done. Embedded {done:,} new concepts; skipped {skipped:,}; empty {empty:,}.")
    print(f"Index covers {len(existing):,} concepts total.")
    print()
    print("Output files:")
    for p in (ids_path, vecs_path, texts_path):
        if p.exists():
            print(f"  {p}  ({p.stat().st_size / 1_048_576:.0f} MB)")
    print()
    print("Next steps: copy the .npy files back to the main machine and run:")
    print("  python sapbert_embedding_builder/import_sapbert_embeddings.py")


if __name__ == "__main__":
    main()
