import json
import os
from typing import Callable
from typing import Any

from src.connection import get_client, get_deployment_name, call_with_retry
from src.database import get_connection
from src.queries.entity_types import canonicalize_type
from src.queries.general_concepts import (
    lookup_general_concepts,
    upsert_general_concept,
)
from src.queries.identity import canonical_etype, derive_natural_key_id

# Embedding tier is optional — the module pulls in numpy / sentence-transformers /
# sqlite-vec, which may not be installed in every environment. Importing lazily
# below means the rest of the lookup pipeline keeps working until the user
# runs `pip install -r requirements.txt` and builds the index.
try:
    from src.queries.embeddings import has_vector_index, vector_search
except ImportError:
    def has_vector_index() -> bool:
        return False
    def vector_search(*_args, **_kwargs):
        return []


EXECUTE_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "Execute SQL against SQLite (SELECT/INSERT/UPDATE/DELETE/DDL).",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL statement to execute.",
                }
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
}

LOOKUP_OMOP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_omop_concepts",
        "description": (
            "Search OMOP concept and concept_synonym for canonical concepts and synonyms. "
            "Combines exact-name, exact-synonym, and vector-similarity matching "
            "(paraphrase / morphology / cross-language) into a single ranked list. "
            "Returns candidates with their OMOP domain_id and a `canonical_type` field — "
            "use the best candidate's `canonical_type` as the entity's KG type. "
            "Each candidate's `match_type` tells you whether it came from "
            "exact / synonym / vector matching."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "Clinical term to map.",
                },
                "domain_hint": {
                    "type": "string",
                    "description": "Optional OMOP domain hint (Condition, Drug, Measurement, Procedure, Observation, ...).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of candidates to return.",
                    "minimum": 1,
                    "maximum": 25,
                },
            },
            "required": ["term"],
            "additionalProperties": False,
        },
    },
}

UPSERT_ENTITY_TOOL = {
    "type": "function",
    "function": {
        "name": "upsert_entity",
        "description": (
            "Insert or update one entity in entities_for_neo4j. Use this for ALL "
            "entity writes — do NOT construct INSERT statements via execute_sql. "
            "The tool owns identity (rewrites the id to a natural key when one "
            "exists from attributes) and attribute merging (non-empty existing "
            "values are preserved when the incoming payload is empty/null; lists "
            "like source_files are unioned). The returned `final_id` may differ "
            "from the input `id` — use it when constructing relations to this entity. "
            "IMPORTANT: Concept entities (Condition, Drug, Measurement, Procedure, "
            "Observation, etc.) must NOT carry timestamps in their attributes — "
            "timestamps belong on the relation edges. This allows a single Condition "
            "node (e.g. HIV) to be shared by all patients who have it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Proposed entity id. May be rewritten to a natural-key id (e.g. patient_<subject_id>).",
                },
                "type": {
                    "type": "string",
                    "description": "Entity type from the whitelist (Patient, Encounter, Drug, Condition, ...).",
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable name. Never JSON. Must be non-empty.",
                },
                "attributes": {
                    "type": "object",
                    "description": (
                        "Attributes object. Will be merged with any existing row's attributes "
                        "(non-empty wins; lists are unioned). Do NOT include 'timestamp' for "
                        "concept entities (Condition, Drug, Measurement, Procedure, Observation) "
                        "— put timestamps in the relation attributes instead."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["id", "type", "name", "attributes"],
            "additionalProperties": False,
        },
    },
}


# --- General-concept registry tools -----------------------------------------
# Parallel of LOOKUP_OMOP_TOOL / a write tool for non-clinical content inside
# notes. The LLM is told (via the merge prompt) to call BOTH lookups for any
# note value and pick the higher-confidence canonical; if neither returns a
# confident hit it calls UPSERT_GENERAL_CONCEPT_TOOL to register a broader
# category (NOT the source term itself — that's what the docstring hammers).

LOOKUP_GENERAL_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_general_concepts",
        "description": (
            "Search the general-concept registry for non-clinical values "
            "extracted from notes (hobbies, foods, devices, activities, "
            "everyday objects — anything OMOP does not cover). Three tiers: "
            "exact name → exact synonym → vector similarity. Returns the "
            "same response shape as lookup_omop_concepts. Call this IN "
            "ADDITION to lookup_omop_concepts for note values; whichever "
            "store returns the higher-confidence candidate wins."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "Term to map.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of candidates to return.",
                    "minimum": 1,
                    "maximum": 25,
                },
            },
            "required": ["term"],
            "additionalProperties": False,
        },
    },
}

UPSERT_GENERAL_CONCEPT_TOOL = {
    "type": "function",
    "function": {
        "name": "upsert_general_concept",
        "description": (
            "Register a new canonical name in the general-concept registry, "
            "OR append a new synonym to an existing canonical. Call this "
            "ONLY when both lookup_omop_concepts and lookup_general_concepts "
            "missed (best score < 0.8 in both) for a note value.\n\n"
            "CRITICAL: `canonical_name` MUST be a BROADER CATEGORY than the "
            "source term — not the source term echoed back. The whole point "
            "of this registry is that later mentions of similar specific "
            "terms collapse onto the same canonical via the vector tier. "
            "Worked examples:\n"
            "  source_term='PlayStation' → canonical_name='Gaming console', synonyms=['PlayStation']\n"
            "  source_term='Xbox'        → canonical_name='Gaming console', synonyms=['Xbox']\n"
            "  source_term='ramen'       → canonical_name='Noodle dish', synonyms=['ramen']\n"
            "  source_term='jogging'     → canonical_name='Physical activity', synonyms=['jogging']\n"
            "  source_term='Netflix'     → canonical_name='Streaming service', synonyms=['Netflix']\n\n"
            "If a suitable canonical (e.g. 'Gaming console') already exists "
            "in the registry from a previous bin, this tool will detect that "
            "case-insensitively and append the new source_term as a synonym "
            "instead of creating a duplicate row. Use the returned "
            "`canonical_name` on the entity going into entities_for_neo4j."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "canonical_name": {
                    "type": "string",
                    "description": "Broader category name (e.g. 'Gaming console'). NOT the source term verbatim.",
                },
                "source_term": {
                    "type": "string",
                    "description": "The original term extracted from the note (e.g. 'PlayStation'). Stored as a synonym.",
                },
                "synonyms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional extra synonyms beyond source_term (paraphrases, abbreviations, alternative spellings).",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional short rationale — why this canonical, why this generalization. Useful for later audit.",
                },
            },
            "required": ["canonical_name", "source_term"],
            "additionalProperties": False,
        },
    },
}


LOOKUP_OMOP_RELATION_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_omop_relation",
        "description": (
            "Look up valid OMOP relationships between two already-mapped concepts. "
            "Call this BEFORE inserting any edge where both the source and target "
            "entity carry an omop_concept_id. If results are returned, use the "
            "relationship_id as the edge type in relations_for_neo4j instead of "
            "the closed relation vocabulary. Returns an empty list when no OMOP "
            "relation exists between the two concepts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "concept_id_1": {
                    "type": "integer",
                    "description": "omop_concept_id of the source entity.",
                },
                "concept_id_2": {
                    "type": "integer",
                    "description": "omop_concept_id of the target entity.",
                },
            },
            "required": ["concept_id_1", "concept_id_2"],
            "additionalProperties": False,
        },
    },
}

DB_TOOLS = [
    EXECUTE_SQL_TOOL,
    LOOKUP_OMOP_TOOL,
    UPSERT_ENTITY_TOOL,
    LOOKUP_GENERAL_TOOL,
    UPSERT_GENERAL_CONCEPT_TOOL,
    LOOKUP_OMOP_RELATION_TOOL,
]


def lookup_omop_relation(concept_id_1: int, concept_id_2: int) -> dict:
    """Return valid OMOP relationships between two concept IDs.

    Queries concept_relationship joined with the relationship registry so each
    result includes both the short relationship_id (used as the KG edge type)
    and the human-readable relationship_name.  Both directions are tried so the
    caller doesn't need to know which concept is the logical source.
    """
    sql = """
    SELECT cr.relationship_id, r.relationship_name, r.is_hierarchical,
           cr.concept_id_1, cr.concept_id_2
    FROM concept_relationship cr
    JOIN relationship r ON r.relationship_id = cr.relationship_id
    WHERE cr.concept_id_1 = ?
      AND cr.concept_id_2 = ?
      AND (cr.invalid_reason IS NULL OR cr.invalid_reason = '')
    ORDER BY r.is_hierarchical DESC, cr.relationship_id
    LIMIT 20
    """
    conn = get_connection()
    try:
        rows = conn.execute(sql, (int(concept_id_1), int(concept_id_2))).fetchall()
        relations = [dict(r) for r in rows]
        return {
            "ok": True,
            "concept_id_1": concept_id_1,
            "concept_id_2": concept_id_2,
            "relation_count": len(relations),
            "relations": relations,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "Ensure concept_relationship and relationship tables are imported into SQLite.",
        }
    finally:
        conn.close()


def _fetch_omop_candidates(cursor, term: str, domain_hint: str | None, limit: int) -> list[dict]:
    candidates: dict[int, dict] = {}

    def add_candidate(row: dict, score: float, match_type: str):
        cid = int(row["concept_id"])
        domain_id = row.get("domain_id")
        payload = {
            "concept_id": cid,
            "concept_name": row.get("concept_name"),
            "domain_id": domain_id,
            "vocabulary_id": row.get("vocabulary_id"),
            "standard_concept": row.get("standard_concept"),
            "concept_code": row.get("concept_code"),
            "canonical_type": canonicalize_type(None, omop_domain=domain_id),
            "score": score,
            "match_type": match_type,
        }
        current = candidates.get(cid)
        if not current or payload["score"] > current["score"]:
            candidates[cid] = payload

    domain_sql = ""
    params_exact = [term]
    if domain_hint:
        domain_sql = " AND c.domain_id = ?"
        params_exact.append(domain_hint)

    # 1) Exact match on concept_name
    cursor.execute(
        f"""
        SELECT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id,
               c.standard_concept, c.concept_code
        FROM concept c
        WHERE LOWER(c.concept_name) = LOWER(?){domain_sql}
        LIMIT ?
        """,
        tuple(params_exact + [limit]),
    )
    for row in cursor.fetchall():
        add_candidate(dict(row), score=1.0, match_type="exact")

    # 2) Exact match on synonym name (catches abbreviations stored as synonyms)
    cursor.execute(
        f"""
        SELECT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id,
               c.standard_concept, c.concept_code
        FROM concept_synonym cs
        JOIN concept c ON c.concept_id = cs.concept_id
        WHERE LOWER(cs.concept_synonym_name) = LOWER(?){domain_sql}
        LIMIT ?
        """,
        tuple(params_exact + [limit]),
    )
    for row in cursor.fetchall():
        add_candidate(dict(row), score=0.92, match_type="synonym")

    # 3) Vector similarity over per-concept enriched embeddings (concept_name +
    #    synonyms used as context). Catches paraphrase, word-order, morphology,
    #    and cross-language misses that exact and synonym-exact can't.
    #    Skipped if the embedding index hasn't been built yet — graceful fallthrough.
    have_high_confidence_exact = any(
        c.get("score", 0) >= 0.92 for c in candidates.values()
    )
    if not have_high_confidence_exact and has_vector_index():
        try:
            vec_hits = vector_search(term, top_k=limit * 2, domain_filter=domain_hint)
        except Exception as exc:
            # Embedding model load / vec query failures must not break lookup,
            # but they ARE bugs — surface them rather than silently degrading.
            print(f"[lookup_omop_concepts] vector tier failed for {term!r}: {type(exc).__name__}: {exc}")
            vec_hits = []
        for hit in vec_hits:
            # Map cosine similarity [-1, 1] into [0, 0.9] so even a perfect
            # cosine (1.0) still ranks below an exact-name hit (1.0). Negative
            # cosines (effectively unrelated) clip to 0.
            cos = max(0.0, min(1.0, float(hit.get("score") or 0.0)))
            scaled = 0.9 * cos
            add_candidate(hit, score=scaled, match_type="vector")

    # Domain priority for tie-breaking when scores match. Lower = preferred.
    # The clinical domains the KG actually uses come first; OMOP metadata-y
    # domains (Meas Value, Type Concept, Metadata) get pushed to the bottom
    # so a Drug match ranks above a Meas Value match for the same name.
    clinical_priority = {
        "Drug": 0, "Condition": 0, "Procedure": 0, "Measurement": 0,
        "Observation": 0, "Device": 0, "Specimen": 0, "Visit": 0,
        "Provider": 1, "Episode": 1, "Route": 1,
        "Geography": 2, "Race": 2, "Ethnicity": 2, "Gender": 2,
        "Spec Anatomic Site": 2, "Unit": 2,
    }
    def _domain_rank(domain: str | None) -> int:
        if not domain:
            return 9
        if domain_hint and domain == domain_hint:
            return -1  # exact hint match always wins
        return clinical_priority.get(domain, 5)

    # Vocabulary priority — ensures the same source term always resolves to the
    # same concept_id across bins even when multiple standard concepts share the
    # same name (e.g. "Heart rate" exists in both LOINC and SNOMED).
    _VOCAB_PRIORITY = {
        "LOINC": 0,     # measurements, labs
        "RxNorm": 0,    # drugs
        "SNOMED": 1,    # conditions, observations, procedures
        "ICD10CM": 2,
        "ICD9CM": 2,
        "NDC": 3,
        "CPT4": 3,
    }
    def _vocab_rank(vocab: str | None) -> int:
        return _VOCAB_PRIORITY.get(vocab or "", 9)

    sorted_candidates = sorted(
        candidates.values(),
        key=lambda x: (
            _domain_rank(x.get("domain_id")),   # hint match (-1) beats everything
            x.get("standard_concept") != "S",
            -(x.get("score") or 0.0),
            _vocab_rank(x.get("vocabulary_id")),
            str(x.get("concept_name") or ""),
        ),
    )
    return sorted_candidates[:limit]


def lookup_omop_concepts(term: str, domain_hint: str | None = None, top_k: int = 5) -> dict:
    normalized_term = (term or "").strip()
    if not normalized_term:
        return {"ok": False, "error": "Missing required argument: term"}

    k = max(1, min(int(top_k or 5), 25))
    conn = get_connection()
    cursor = conn.cursor()
    try:
        candidates = _fetch_omop_candidates(cursor, normalized_term, domain_hint, k)
        best = candidates[0] if candidates else None
        return {
            "ok": True,
            "term": normalized_term,
            "domain_hint": domain_hint,
            "row_count": len(candidates),
            "best_candidate": best,
            "canonical_type": best.get("canonical_type") if best else "Unknown",
            "confidence": round(float(best["score"]), 3) if best else 0.0,
            "requires_review": not bool(best) or float(best["score"]) < 0.8,
            "candidates": candidates,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "Ensure OMOP tables concept and concept_synonym are available in SQLite.",
        }
    finally:
        cursor.close()
        conn.close()


def _merge_attributes(existing: dict, incoming: dict) -> dict:
    """Combine two attribute dicts using a non-empty-wins rule.

    For each key in either dict:
      - if both have non-empty values: incoming wins.
      - if incoming is empty/null/[]/'': keep existing.
      - if existing is empty/null/[]/'': take incoming.
    Lists (e.g. source_files) are unioned by string content rather than
    overwritten, so multiple bins each contributing a different file path
    leave both paths in the final list.
    """
    def _is_empty(v) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and not v.strip():
            return True
        if isinstance(v, (list, dict)) and not v:
            return True
        return False

    merged = dict(existing or {})
    for k, v_in in (incoming or {}).items():
        v_old = merged.get(k)

        # List union (preserves order; dedup by JSON-stringified content for nested items).
        if isinstance(v_old, list) and isinstance(v_in, list):
            seen = {json.dumps(x, sort_keys=True, default=str) for x in v_old}
            for item in v_in:
                key = json.dumps(item, sort_keys=True, default=str)
                if key not in seen:
                    v_old.append(item)
                    seen.add(key)
            merged[k] = v_old
            continue

        if _is_empty(v_in):
            if _is_empty(v_old):
                merged[k] = v_in  # both empty, fine
            # else keep existing non-empty
            continue

        merged[k] = v_in
    return merged


def upsert_entity(id: str, type: str, name: str, attributes: dict | None = None) -> dict:
    """Insert or update one entity, doing identity rewrite + attribute merge.

    Identity: if the (type, attributes) imply a natural-key id (e.g. Patient
    with subject_id -> patient_<subject_id>), the row is written under that
    id regardless of what was passed in. Two bins describing the same
    Patient thus collapse to one row deterministically.

    Attribute merge: when the row already exists, incoming attributes are
    merged via `_merge_attributes` (non-empty wins, lists unioned), so a
    sparse second-bin payload doesn't clobber a rich first-bin one.
    """
    raw_id = (id or "").strip()
    raw_name = (name or "").strip()
    if not raw_id:
        return {"ok": False, "error": "Missing required argument: id"}
    if not raw_name:
        return {"ok": False, "error": "Missing required argument: name"}

    canon_type = canonical_etype(type)
    if not isinstance(attributes, dict):
        attributes = {}

    natural_id = derive_natural_key_id(canon_type, attributes)
    final_id = natural_id or raw_id

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT type, name, attributes FROM entities_for_neo4j WHERE id = ?",
            (final_id,),
        )
        row = cursor.fetchone()
        if row is not None:
            existing_name = row["name"] if "name" in row.keys() else row[1]
            existing_attrs_raw = row["attributes"] if "attributes" in row.keys() else row[2]
            try:
                existing_attrs = json.loads(existing_attrs_raw) if existing_attrs_raw else {}
            except (json.JSONDecodeError, TypeError):
                existing_attrs = {}
            merged_attrs = _merge_attributes(existing_attrs, attributes)
            # Keep the existing non-empty name. Two bins legitimately referring
            # to the same logical entity may use different names ("Subject
            # 10000032" vs "Patient 10000032"); the first one wins. This also
            # avoids tripping the `guard_entity_rename` SQL trigger.
            final_name = existing_name if (existing_name or "").strip() else raw_name
            action = "update"
        else:
            merged_attrs = dict(attributes)
            final_name = raw_name
            action = "insert"

        cursor.execute(
            """
            INSERT INTO entities_for_neo4j (id, type, name, attributes)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              type = excluded.type,
              name = excluded.name,
              attributes = excluded.attributes
            """,
            (final_id, canon_type, final_name, json.dumps(merged_attrs, ensure_ascii=False)),
        )
        conn.commit()
        return {
            "ok": True,
            "action": action,
            "final_id": final_id,
            "id_was_rewritten": final_id != raw_id,
            "type": canon_type,
            "merged_attribute_keys": sorted(merged_attrs.keys()),
        }
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}", "final_id": final_id}
    finally:
        cursor.close()
        conn.close()


def execute_sql(sql: str) -> dict:
    """Execute raw SQL and return either rows (for SELECT) or mutation stats.
    Supports multiple statements separated by ';' for batch inserts."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Check if there are multiple statements (batch mode)
        stripped = sql.strip().rstrip(";")
        statements = [s.strip() for s in stripped.split(";") if s.strip()]

        if len(statements) > 1:
            # Multi-statement batch: use executescript (auto-commits)
            conn.executescript(sql)
            return {
                "ok": True,
                "kind": "mutation",
                "statements_executed": len(statements),
            }

        # Single statement: use cursor.execute for result-set support
        cursor.execute(sql)
        if cursor.description is not None:
            rows = [dict(row) for row in cursor.fetchall()]
            return {
                "ok": True,
                "kind": "result_set",
                "row_count": len(rows),
                "rows": rows,
            }

        conn.commit()
        return {
            "ok": True,
            "kind": "mutation",
            "affected_rows": cursor.rowcount,
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


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                chunks.append(text)
        return "\n".join(chunks)
    return ""


def _summarize_tool_result(name: str, args: dict, result: dict) -> str:
    """One-line description of what a tool call did, for the agent log.

    The agent loop calls many tools per bin; without a summary every line is
    just `[tool] execute_sql` and you can't tell what's actually happening.
    This pulls the most-useful fields out of each handler's return value.
    """
    if not isinstance(result, dict):
        return repr(result)[:80]

    if not result.get("ok", True):
        return f"ERROR: {result.get('error', 'unknown error')}"

    if name == "execute_sql":
        kind = result.get("kind")
        if kind == "result_set":
            return f"SELECT -> {result.get('row_count', 0)} rows"
        if kind == "mutation":
            n = result.get("affected_rows")
            if n is None:
                n = result.get("statements_executed", "?")
                return f"mutation -> {n} statements"
            return f"mutation -> {n} rows affected"
        return str(kind)

    if name == "upsert_entity":
        action = result.get("action", "?")
        final_id = result.get("final_id", "?")
        rewritten = " (id rewritten)" if result.get("id_was_rewritten") else ""
        n_keys = len(result.get("merged_attribute_keys") or [])
        return f"{action} id={final_id} type={result.get('type')!r} attrs={n_keys}{rewritten}"

    if name == "lookup_omop_concepts":
        term = args.get("term", "")
        hint = args.get("domain_hint")
        best = result.get("best_candidate")
        head = f"term={term!r}"
        if hint:
            head += f" hint={hint!r}"
        if not best:
            return f"{head} -> no candidates"
        score = best.get("score")
        score_str = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
        return (
            f"{head} -> {best.get('concept_name')!r} "
            f"({best.get('domain_id')}) via {best.get('match_type')}"
            f"{score_str}"
        )

    if name == "lookup_general_concepts":
        term = args.get("term", "")
        best = result.get("best_candidate")
        if not best:
            return f"term={term!r} -> no candidates"
        score = best.get("score")
        score_str = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
        return (
            f"term={term!r} -> {best.get('concept_name')!r} "
            f"via {best.get('match_type')}{score_str}"
        )

    if name == "upsert_general_concept":
        action = result.get("action", "?")
        cname = result.get("canonical_name", "?")
        cid = result.get("concept_id", "?")
        n_syn = result.get("synonym_count", "?")
        return f"{action} canonical={cname!r} id={cid} synonyms={n_syn}"

    if name == "lookup_omop_relation":
        c1 = args.get("concept_id_1")
        c2 = args.get("concept_id_2")
        count = result.get("relation_count", 0)
        if not result.get("ok", True):
            return f"ERROR: {result.get('error', 'unknown error')}"
        if count == 0:
            return f"concept_id_1={c1} concept_id_2={c2} -> no relations"
        top = result.get("relations", [{}])[0]
        return (
            f"concept_id_1={c1} concept_id_2={c2} -> {count} relations, "
            f"top={top.get('relationship_id')!r} ({top.get('relationship_name')})"
        )

    # Unknown tool: short repr of the result for visibility.
    return repr(result)[:80]


def run_llm_db_session(
    client,
    user_prompt: str,
    system_prompt: str = "Du bist ein SQL-Assistent. Nutze Tools, wenn noetig.",
    model: str | None = None,
    max_steps: int = 10,
    sql_executor: Callable[[str], dict] | None = None,
    extra_tools: list[dict] | None = None,
    extra_tool_handlers: dict[str, Callable[[dict[str, Any]], dict]] | None = None,
) -> str:
    """Run a tool-calling loop where the model can execute unrestricted SQL."""
    executor = sql_executor or execute_sql
    deployment = model or get_deployment_name()
    tools = DB_TOOLS + (extra_tools or [])
    handlers: dict[str, Callable[[dict[str, Any]], dict]] = {
        "execute_sql": lambda args: executor(args.get("sql", "")),
        "lookup_omop_concepts": lambda args: lookup_omop_concepts(
            term=args.get("term", ""),
            domain_hint=args.get("domain_hint"),
            top_k=args.get("top_k", 5),
        ),
        "upsert_entity": lambda args: upsert_entity(
            id=args.get("id", ""),
            type=args.get("type", ""),
            name=args.get("name", ""),
            attributes=args.get("attributes") or {},
        ),
        "lookup_general_concepts": lambda args: lookup_general_concepts(
            term=args.get("term", ""),
            top_k=args.get("top_k", 5),
        ),
        "upsert_general_concept": lambda args: upsert_general_concept(
            canonical_name=args.get("canonical_name", ""),
            source_term=args.get("source_term"),
            synonyms=args.get("synonyms") or [],
            notes=args.get("notes"),
        ),
        "lookup_omop_relation": lambda args: lookup_omop_relation(
            concept_id_1=args.get("concept_id_1", 0),
            concept_id_2=args.get("concept_id_2", 0),
        ),
    }
    if extra_tool_handlers:
        handlers.update(extra_tool_handlers)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    _max_retries = int(os.getenv("LLM_MAX_RETRIES", "100"))
    _base_delay = float(os.getenv("LLM_BASE_DELAY", "1.0"))
    _max_delay = float(os.getenv("LLM_MAX_DELAY", "120.0"))

    for step in range(1, max_steps + 1):
        print(f"[agent step {step}/{max_steps}] requesting next action ...", flush=True)
        response = call_with_retry(
            lambda: client.chat.completions.create(
                model=deployment,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_completion_tokens=16384,
            ),
            max_attempts=_max_retries,
            base_delay=_base_delay,
            max_delay=_max_delay,
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        assistant_text = _content_to_text(getattr(message, "content", ""))

        if tool_calls:
            assistant_tool_calls = []
            for call in tool_calls:
                assistant_tool_calls.append(
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "tool_calls": assistant_tool_calls,
                }
            )

            for call in tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                handler = handlers.get(call.function.name)
                if not handler:
                    result = {"ok": False, "error": f"Unknown tool: {call.function.name}"}
                else:
                    result = handler(args)

                summary = _summarize_tool_result(call.function.name, args, result)
                print(f"  [tool] {call.function.name:22s} {summary}", flush=True)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
            continue

        if assistant_text:
            print(f"[agent done] returning final response ({step} turns used)", flush=True)
            return assistant_text

    print(f"[agent done] step limit reached ({max_steps} turns)", flush=True)
    return "No final result within the tool-step limit."


def ask_database_with_llm(user_prompt: str) -> str:
    client = get_client()
    return run_llm_db_session(client=client, user_prompt=user_prompt)


