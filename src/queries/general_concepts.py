"""General-concept registry — parallel of OMOP for non-clinical content.

OMOP covers Drugs/Conditions/Procedures/Measurements/etc. — everything a
clinical vocabulary is designed for. Free-text notes routinely mention
things OMOP has no opinion about: a patient's hobby ("jogging"), the
device they fell off ("e-scooter"), the food they reacted to ("ramen"),
the show they watch ("Netflix"). Without a place to land, the LLM either
coins a different ad-hoc name in every bin (duplicates in Neo4j) or marks
them Unknown.

This module is that place to land. It mirrors the OMOP tables under a
`general_` prefix:

    general_concept             ← canonical names the LLM has coined
    general_concept_synonym     ← source terms / variations that map to one
    general_concept_embedding_meta
    general_concept_vec         ← sqlite-vec virtual table

And it exposes two operations:

    lookup_general_concepts(term, top_k)  — same response shape as
        `lookup_omop_concepts`, three tiers (exact name 1.0, synonym 0.92,
        vector scaled into [0, 0.9]).

    upsert_general_concept(canonical_name, synonyms, source_term, ...)
        — adds a canonical (or appends a new synonym to an existing one)
        and rebuilds that concept's embedding so the vector tier sees it
        on the next query.

The canonical name MUST be a generalization of the source term — "Xbox"
and "PlayStation" both fold into "Gaming console". That is what makes the
store useful: a year of merging clinical notes leaves you with a few
hundred broad categories, not tens of thousands of one-off names. The
LLM's prompt enforces this; the upsert itself doesn't try to police it
(no way to detect a too-narrow canonical at write time without an
expensive LLM call we'd rather avoid).
"""

from __future__ import annotations

import sqlite3

# Lazy imports: this module's lookup must keep working even if the embedding
# stack isn't installed (matches OMOP's graceful-degradation pattern in
# `src/queries/db_tooling.py`). Imported at first use inside functions.


# Vector-tier score ceiling — same scaling as OMOP so a perfect cosine
# match in either store still ranks below an exact-name hit (1.0).
VECTOR_SCORE_CAP = 0.9

# Confidence threshold below which the LLM should consider coining a new
# canonical instead of accepting the best candidate. Matches OMOP's
# `requires_review` threshold so both stores feel uniform.
REVIEW_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def _fetch_general_candidates(
    cursor,
    term: str,
    limit: int,
) -> list[dict]:
    """Three-tier search of the general store: exact name → synonym → vector.

    Mirrors `_fetch_omop_candidates` in db_tooling.py but with the simpler
    general schema (no domain_id, no vocabulary_id, no standard_concept).
    """
    candidates: dict[int, dict] = {}

    def add_candidate(row: dict, score: float, match_type: str):
        cid = int(row["concept_id"])
        payload = {
            "concept_id": cid,
            "concept_name": row.get("concept_name"),
            "score": score,
            "match_type": match_type,
        }
        current = candidates.get(cid)
        if not current or payload["score"] > current["score"]:
            candidates[cid] = payload

    # 1) Exact match on canonical name. Case-insensitive — the LLM may
    #    feed back "playstation" against canonical "PlayStation" / "Gaming
    #    console" and we want both to land.
    cursor.execute(
        """
        SELECT concept_id, concept_name
        FROM general_concept
        WHERE LOWER(concept_name) = LOWER(?)
        LIMIT ?
        """,
        (term, limit),
    )
    for row in cursor.fetchall():
        add_candidate(dict(row), score=1.0, match_type="exact")

    # 2) Exact match on synonym name (this is where most second-occurrence
    #    hits land — "Xbox" was stored as a synonym of "Gaming console" the
    #    first time it showed up).
    cursor.execute(
        """
        SELECT gc.concept_id, gc.concept_name
        FROM general_concept_synonym gcs
        JOIN general_concept gc ON gc.concept_id = gcs.concept_id
        WHERE LOWER(gcs.synonym_name) = LOWER(?)
        LIMIT ?
        """,
        (term, limit),
    )
    for row in cursor.fetchall():
        add_candidate(dict(row), score=0.92, match_type="synonym")

    # 3) Vector tier — paraphrase, morphology, "things in the same neighborhood".
    #    Skipped if no high-confidence exact already exists or the vec index
    #    doesn't exist yet (e.g. on a freshly-migrated empty store, the table
    #    exists but has zero rows — knn_rowids returns []).
    have_high_confidence_exact = any(
        c.get("score", 0) >= 0.92 for c in candidates.values()
    )
    if not have_high_confidence_exact:
        # Pre-check: skip the vector tier entirely if the virtual table
        # hasn't been created yet (init_db couldn't make it without
        # sqlite-vec loaded, and no upsert has run since). Otherwise we'd
        # log a spurious "OperationalError: no such table" on every lookup
        # against an empty store.
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE name = 'general_concept_vec'"
        )
        vec_exists = cursor.fetchone() is not None

        hits = []
        if vec_exists:
            try:
                from src.queries.embeddings import knn_rowids
                hits = knn_rowids(term, top_k=limit * 2, vec_table="general_concept_vec")
            except Exception as exc:
                # Same convention as OMOP: print and degrade rather than crash.
                print(f"[lookup_general_concepts] vector tier failed for {term!r}: {type(exc).__name__}: {exc}")
                hits = []

        if hits:
            # Resolve rowids back to canonical names in one query.
            rowids = [rid for rid, _ in hits]
            placeholders = ",".join("?" * len(rowids))
            cursor.execute(
                f"SELECT concept_id, concept_name FROM general_concept "
                f"WHERE concept_id IN ({placeholders})",
                rowids,
            )
            name_by_id = {int(r["concept_id"]): r["concept_name"] for r in cursor.fetchall()}
            for rid, distance in hits:
                name = name_by_id.get(rid)
                if not name:
                    # Vector index row exists but the concept was deleted —
                    # skip rather than surface a half-row.
                    continue
                # Same scaling as OMOP: cosine [-1, 1] → [0, 0.9] so a perfect
                # vector hit still ranks below an exact-name hit (1.0).
                cos = max(0.0, min(1.0, 1.0 - float(distance)))
                add_candidate(
                    {"concept_id": rid, "concept_name": name},
                    score=VECTOR_SCORE_CAP * cos,
                    match_type="vector",
                )

    # Simple sort: by score descending, name ascending. No domain ranking
    # (general store has no domains), no standard_concept tiebreaker.
    sorted_candidates = sorted(
        candidates.values(),
        key=lambda x: (-(x.get("score") or 0.0), str(x.get("concept_name") or "")),
    )
    return sorted_candidates[:limit]


def lookup_general_concepts(term: str, top_k: int = 5) -> dict:
    """Three-tier search of the general-concept registry.

    Returns the same response shape as `lookup_omop_concepts` so the LLM
    can treat both tools uniformly:

        {
          "ok": True,
          "term": ...,
          "row_count": N,
          "best_candidate": {concept_id, concept_name, score, match_type} | None,
          "confidence": float,
          "requires_review": bool,
          "candidates": [...],
        }
    """
    from src.database import get_connection
    normalized_term = (term or "").strip()
    if not normalized_term:
        return {"ok": False, "error": "Missing required argument: term"}

    k = max(1, min(int(top_k or 5), 25))
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Tolerate the case where the migration hasn't run yet (fresh checkout
        # with no init_db call). Behave like the OMOP lookup when its tables
        # are missing — return ok=True with zero candidates.
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE name = 'general_concept'"
        )
        if not cursor.fetchone():
            return {
                "ok": True,
                "term": normalized_term,
                "row_count": 0,
                "best_candidate": None,
                "confidence": 0.0,
                "requires_review": True,
                "candidates": [],
            }

        candidates = _fetch_general_candidates(cursor, normalized_term, k)
        best = candidates[0] if candidates else None
        return {
            "ok": True,
            "term": normalized_term,
            "row_count": len(candidates),
            "best_candidate": best,
            "confidence": round(float(best["score"]), 3) if best else 0.0,
            "requires_review": not bool(best) or float(best["score"]) < REVIEW_THRESHOLD,
            "candidates": candidates,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "Run init_db (or scripts/migrations/001_create_general_concept_tables.py) to create the general-concept tables.",
        }
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _ensure_vec_table(conn: sqlite3.Connection) -> bool:
    """Create `general_concept_vec` on this connection if it isn't there yet.

    `init_db` tries to create it but silently skips if sqlite-vec wasn't
    loaded on that connection. `upsert_general_concept` always opens its
    own vec-loaded connection, so this retry catches that case.
    """
    try:
        # Import the dim constant from the migration to keep them in sync.
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "_gcm",
            Path(__file__).resolve().parents[2] / "scripts" / "migrations"
            / "001_create_general_concept_tables.py",
        )
        if spec is None or spec.loader is None:
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dim = module.GENERAL_EMBEDDING_DIM
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS general_concept_vec "
            f"USING vec0(embedding float[{dim}])"
        )
        return True
    except sqlite3.OperationalError:
        return False


def upsert_general_concept(
    canonical_name: str,
    synonyms: list[str] | None = None,
    source_term: str | None = None,
    created_by_bin: str | None = None,
    notes: str | None = None,
) -> dict:
    """Register (or extend) a canonical general-concept entry.

    Behavior
    --------
    - If a `general_concept` row with the same canonical_name (case-insensitive)
      already exists: the existing concept_id is returned and any new synonyms
      (including `source_term` if provided) are appended to
      `general_concept_synonym`. The embedding is rebuilt with the expanded
      synonym set so the vector tier picks up the new context next time.
    - If no such row exists: insert it, insert the synonyms, build the
      embedding, write it into `general_concept_vec` and `general_concept_embedding_meta`.

    All writes happen on a single vec-loaded connection inside one
    transaction, so a failed embedding write rolls back the concept row.

    The canonical must be a broader category than `source_term` (that rule
    is enforced by the LLM prompt, not here). The tool's own check is just
    case-insensitive dedup on the canonical name.

    Returns
    -------
    {
      "ok": True,
      "action": "insert" | "update",
      "concept_id": int,
      "canonical_name": str,
      "synonym_count": int,
    }
    """
    name = (canonical_name or "").strip()
    if not name:
        return {"ok": False, "error": "Missing required argument: canonical_name"}

    # Synonym set: deduplicate case-insensitively against itself and the
    # canonical. `source_term` is just another synonym from this tool's POV.
    incoming_synonyms: list[str] = []
    seen = {name.lower()}
    for s in list(synonyms or []) + ([source_term] if source_term else []):
        if not s:
            continue
        s_clean = s.strip()
        if not s_clean:
            continue
        key = s_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        incoming_synonyms.append(s_clean)

    from src.queries.embeddings import connect_with_vec, build_embed_text, embed_one

    conn = connect_with_vec()
    cursor = conn.cursor()
    try:
        _ensure_vec_table(conn)

        # Case-insensitive lookup of the canonical.
        cursor.execute(
            "SELECT concept_id FROM general_concept WHERE LOWER(concept_name) = LOWER(?)",
            (name,),
        )
        row = cursor.fetchone()

        if row is not None:
            concept_id = int(row["concept_id"])
            action = "update"
        else:
            cursor.execute(
                """
                INSERT INTO general_concept (concept_name, created_by_bin, notes)
                VALUES (?, ?, ?)
                """,
                (name, created_by_bin, notes),
            )
            concept_id = int(cursor.lastrowid)
            action = "insert"

        # Append new synonyms idempotently. The unique index on
        # (concept_id, LOWER(synonym_name)) means duplicates are silently
        # dropped by INSERT OR IGNORE — no need to pre-check.
        for syn in incoming_synonyms:
            cursor.execute(
                """
                INSERT OR IGNORE INTO general_concept_synonym (concept_id, synonym_name)
                VALUES (?, ?)
                """,
                (concept_id, syn),
            )

        # Pull the full synonym list back out so the embedding text reflects
        # the post-upsert state (existing synonyms + the new ones we just added).
        cursor.execute(
            "SELECT synonym_name FROM general_concept_synonym WHERE concept_id = ?",
            (concept_id,),
        )
        all_synonyms = [r["synonym_name"] for r in cursor.fetchall()]

        embed_text = build_embed_text(name, all_synonyms)
        vec = embed_one(embed_text)
        vec_bytes = vec.tobytes()

        # Upsert the embedding meta row.
        cursor.execute(
            """
            INSERT INTO general_concept_embedding_meta (rowid, concept_id, embed_text)
            VALUES (?, ?, ?)
            ON CONFLICT(rowid) DO UPDATE SET embed_text = excluded.embed_text
            """,
            (concept_id, concept_id, embed_text),
        )

        # And the vec row. sqlite-vec doesn't support ON CONFLICT on virtual
        # tables, so we delete-then-insert. The rowid equals concept_id by
        # convention (mirrors `concept_vec`).
        cursor.execute("DELETE FROM general_concept_vec WHERE rowid = ?", (concept_id,))
        cursor.execute(
            "INSERT INTO general_concept_vec (rowid, embedding) VALUES (?, ?)",
            (concept_id, vec_bytes),
        )

        conn.commit()

        # Synonym count we report back is the post-upsert total (useful for
        # the LLM to see "this canonical now has 5 synonyms" trend over time).
        cursor.execute(
            "SELECT COUNT(*) AS n FROM general_concept_synonym WHERE concept_id = ?",
            (concept_id,),
        )
        synonym_count = int(cursor.fetchone()["n"])

        return {
            "ok": True,
            "action": action,
            "concept_id": concept_id,
            "canonical_name": name,
            "synonym_count": synonym_count,
        }
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        cursor.close()
        conn.close()
