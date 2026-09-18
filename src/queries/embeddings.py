"""Vector-similarity tier for OMOP lookup.

Loads a sentence-transformer model lazily (once per process) and provides a
`vector_search` helper that talks to the `concept_vec` virtual table created
by `scripts/build_concept_embeddings.py`.

Each concept gets ONE embedding, computed from its `concept_name` enriched
with its synonyms as context (e.g. "Atrial fibrillation. Also known as:
AF; AFib; A fib."). The canonical name appears first so it dominates the
vector; synonyms reinforce the meaning without crowding it out.

Schema this module expects:

    CREATE TABLE concept_embedding_meta (
        rowid       INTEGER PRIMARY KEY,        -- = concept_id
        concept_id  INTEGER NOT NULL UNIQUE,
        embed_text  TEXT NOT NULL               -- the string we embedded
    );
    CREATE VIRTUAL TABLE concept_vec USING vec0(embedding float[<DIM>]);

The virtual table's rowid equals concept_embedding_meta.rowid equals
concept_id, so a vector hit gives you the concept_id directly.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer

from src.database import get_connection


# Clinical biomedical entity-linking model (768-dim). Trained on UMLS concept
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


# When building per-concept enriched text, cap synonyms to keep the embedding
# focused on the canonical meaning. Sorted by length ascending — short forms
# (abbreviations) come first as they're the highest-recall payload.
MAX_SYNONYMS_PER_CONCEPT = 8


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once per process. SentenceTransformer auto-picks the
    best device (CUDA > MPS > CPU).

    SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext) is a raw BERT
    model without sentence-transformers config files. We wrap it explicitly
    with mean pooling so SentenceTransformer can load it. For models that
    already ship modules.json (e.g. bge-small), the word_embedding_model /
    pooling approach still works correctly.
    """
    from sentence_transformers import models as st_models

    word_model = st_models.Transformer(EMBEDDING_MODEL_NAME)
    pooling = st_models.Pooling(
        word_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
    )
    return SentenceTransformer(modules=[word_model, pooling])


def build_embed_text(concept_name: str, synonyms: list[str]) -> str:
    """Compose the string we feed the embedder for a single concept.

    Format: "<concept_name>. Also known as: <syn1>; <syn2>; ...".
    The canonical name comes first so it dominates the embedding. Synonyms
    are deduplicated case-insensitively against the name and against each
    other, and capped at MAX_SYNONYMS_PER_CONCEPT shortest entries.
    """
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

    # Shortest-first: abbreviations & common short forms before paraphrases.
    cleaned.sort(key=lambda s: (len(s), s.lower()))
    cleaned = cleaned[:MAX_SYNONYMS_PER_CONCEPT]

    if not cleaned:
        return name
    return f"{name}. Also known as: " + "; ".join(cleaned) + "."


def embed_texts(texts: list[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    """Embed a list of strings. Returns a (N, DIM) float32 array.

    `normalize=True` returns unit vectors, so cosine similarity reduces to a
    dot product — what sqlite-vec uses by default.
    """
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32, copy=False)


def embed_one(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def _connect_with_vec():
    """Open the project DB with the sqlite-vec extension loaded."""
    conn = get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


# Public alias — `src/queries/general_concepts.py` needs a vec-loaded
# connection for its own writes. Keep the `_connect_with_vec` name too so
# the rest of this module continues to read like before.
connect_with_vec = _connect_with_vec


def knn_rowids(
    query_text: str,
    top_k: int = 20,
    vec_table: str = "concept_vec",
    conn=None,
) -> list[tuple[int, float]]:
    """Low-level KNN primitive: return `[(rowid, distance), ...]` for the
    nearest `top_k` rows in any `vec0` virtual table.

    Both the OMOP store (`concept_vec`) and the general-concept store
    (`general_concept_vec`) call this and then JOIN to their own metadata
    tables. Keeping the KNN call separate from the JOIN means the general
    store doesn't have to pretend to have a `concept` table.

    Parameters
    ----------
    query_text : the user/LLM term to search for.
    top_k      : how many rows to fetch from the vec index.
    vec_table  : name of the vec0 virtual table to query. Must be a literal
                 identifier (no user input — we interpolate it into the SQL).
    conn       : optional connection (already vec-loaded). If None, a fresh
                 one is opened and closed inside this call.

    Returns `[]` for empty queries; raises on actual SQL/embedding errors so
    callers can decide whether to swallow or propagate (see
    `_fetch_omop_candidates` for the wrap-with-print pattern).
    """
    if not query_text or not query_text.strip():
        return []

    qvec = embed_one(query_text.strip())
    qblob = qvec.tobytes()

    owns_conn = conn is None
    if conn is None:
        conn = _connect_with_vec()
    cursor = conn.cursor()
    try:
        # vec0's MATCH+k is its own KNN query and must not be combined with
        # an outer ORDER BY/LIMIT — sqlite-vec rejects that. The CTE keeps
        # the KNN call isolated.
        sql = f"""
            SELECT rowid, distance
            FROM {vec_table}
            WHERE embedding MATCH ?
              AND k = ?
            ORDER BY distance ASC
        """
        cursor.execute(sql, (qblob, top_k))
        return [(int(r["rowid"]), float(r["distance"])) for r in cursor.fetchall()]
    finally:
        cursor.close()
        if owns_conn:
            conn.close()


def vector_search(
    query_text: str,
    top_k: int = 20,
    domain_filter: str | None = None,
) -> list[dict]:
    """Top-K OMOP concepts ranked by cosine similarity to `query_text`.

    Returns concept rows with a `score` field in [-1, 1] (1.0 = identical).
    `domain_filter` is used only to over-fetch enough candidates so the
    caller's domain-priority sort has material to work with — it is NOT
    applied as a hard WHERE filter, because a post-KNN domain filter on
    a 6M-row index starves results when the target domain's concepts are
    not in the top-N nearest neighbors globally.
    """
    if not query_text or not query_text.strip():
        return []

    qvec = embed_one(query_text.strip())
    qblob = qvec.tobytes()

    # Over-fetch more when a domain hint is given so the caller's domain-
    # priority ranking has enough candidates from the right domain to pick from.
    raw_k = top_k * 8 if domain_filter else top_k * 2

    conn = _connect_with_vec()
    cursor = conn.cursor()
    try:
        # vec0's MATCH+k is its own KNN query and must not be combined with an
        # outer ORDER BY/LIMIT — sqlite-vec rejects that. Wrap the KNN call in
        # a CTE so vec0 sees only its own constraints, then JOIN for metadata.
        # Note: `concept.concept_id` is TEXT (the OMOP CSVs are imported as
        # all-TEXT, no schema typing), but `concept_vec.rowid` is INTEGER.
        # We CAST to INTEGER for the JOIN; SQLite's type affinity would
        # otherwise treat '44821957' != 44821957 and silently drop every row.
        sql = """
            WITH knn AS (
                SELECT rowid, distance
                FROM concept_vec
                WHERE embedding MATCH ?
                  AND k = ?
            )
            SELECT
                knn.rowid AS concept_id,
                knn.distance AS distance,
                c.concept_name AS concept_name,
                c.domain_id AS domain_id,
                c.vocabulary_id AS vocabulary_id,
                c.standard_concept AS standard_concept,
                c.concept_code AS concept_code,
                cem.embed_text AS embed_text
            FROM knn
            JOIN concept c ON CAST(c.concept_id AS INTEGER) = knn.rowid
            JOIN concept_embedding_meta cem ON cem.rowid = knn.rowid
            ORDER BY knn.distance ASC
        """
        cursor.execute(sql, [qblob, raw_k])
        rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    results = []
    for r in rows[:top_k * 2]:
        # Normalized embeddings → distance = 1 - cosine_similarity.
        score = 1.0 - float(r["distance"])
        results.append({
            "concept_id": int(r["concept_id"]),
            "concept_name": r["concept_name"],
            "domain_id": r["domain_id"],
            "vocabulary_id": r["vocabulary_id"],
            "standard_concept": r["standard_concept"],
            "concept_code": r["concept_code"],
            "embed_text": r["embed_text"],   # for debugging: what was actually embedded
            "score": score,
        })
    return results


@lru_cache(maxsize=1)
def has_vector_index() -> bool:
    """True if the embedding tables exist. `_fetch_omop_candidates` calls
    this so the vector tier degrades gracefully before the build script runs."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE name IN "
            "('concept_embedding_meta', 'concept_vec')"
        )
        names = {row[0] for row in cursor.fetchall()}
        return {"concept_embedding_meta", "concept_vec"} <= names
    finally:
        cursor.close()
        conn.close()
