import os
import json
import re
import sqlite3
from datetime import datetime as _dt
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(PROJECT_ROOT / "data" / "medical_kg.sqlite"))


def get_connection():
    db_path = Path(SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _apply_general_concept_migration(conn: sqlite3.Connection) -> None:
    """Load the digit-prefixed migration module by path and apply it.

    The filename `001_create_general_concept_tables.py` starts with a digit,
    so a plain `import` won't work — we resolve it via `importlib.util` here.
    The migration is idempotent (CREATE IF NOT EXISTS), so calling this on
    every `init_db` is cheap.
    """
    import importlib.util
    migration_path = PROJECT_ROOT / "scripts" / "migrations" / "001_create_general_concept_tables.py"
    spec = importlib.util.spec_from_file_location(
        "_general_concept_migration", migration_path
    )
    if spec is None or spec.loader is None:
        return  # migration file missing — non-fatal; older checkouts keep working.
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply(conn)


def init_db():
    """Create the entities_for_neo4j and relations_for_neo4j tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities_for_neo4j (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            attributes TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS relations_for_neo4j (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            type TEXT NOT NULL,
            attributes TEXT
        )
    """)

    # Migrate away from the old UNIQUE(source,target,type) table constraint.
    # SQLite can't DROP a table-level UNIQUE constraint in place, so we rebuild
    # the table without it and let idx_rfn_unique (below) take over.
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='sqlite_autoindex_relations_for_neo4j_1'"
    )
    if cursor.fetchone() is not None:
        cursor.executescript("""
            BEGIN;
            ALTER TABLE relations_for_neo4j RENAME TO _relations_old;
            CREATE TABLE relations_for_neo4j (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                source    TEXT NOT NULL,
                target    TEXT NOT NULL,
                type      TEXT NOT NULL,
                attributes TEXT
            );
            INSERT OR IGNORE INTO relations_for_neo4j (id, source, target, type, attributes)
                SELECT id, source, target, type, attributes FROM _relations_old;
            DROP TABLE _relations_old;
            COMMIT;
        """)

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_rfn_unique
        ON relations_for_neo4j (
            source,
            target,
            type,
            COALESCE(json_extract(attributes, '$.timestamp'), '')
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_efn_type ON entities_for_neo4j(type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rfn_type ON relations_for_neo4j(type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rfn_source ON relations_for_neo4j(source)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rfn_target ON relations_for_neo4j(target)")

    # Tripwire: refuse to rename an existing entity to a substantively
    # different name. With per-bin id prefixes in place, this should fire
    # extremely rarely — when it does, it's almost always because two
    # extractions produced the same id for unrelated entities and one is
    # silently about to overwrite the other (the bug that lost patient
    # 10000032 in an earlier run).
    #
    # Legitimate cases that DO fire (and the LLM must handle):
    #   - canonical OMOP rename (e.g. "HIV" -> "Human immunodeficiency virus
    #     infection"). The merge agent should normalize the name in the
    #     same INSERT it creates the row, not in a follow-up upsert.
    # The error message is plain text so the LLM can read it via
    # execute_sql's {"ok": false, "error": ...} and recover.
    cursor.execute("DROP TRIGGER IF EXISTS guard_entity_rename")
    cursor.execute("""
        CREATE TRIGGER guard_entity_rename
        BEFORE UPDATE OF name ON entities_for_neo4j
        WHEN OLD.name IS NOT NULL
         AND NEW.name IS NOT NULL
         AND TRIM(OLD.name) != ''
         AND TRIM(NEW.name) != ''
         AND LOWER(TRIM(OLD.name)) != LOWER(TRIM(NEW.name))
        BEGIN
            SELECT RAISE(ABORT, 'guard_entity_rename: refusing to rename existing entity. Existing id has a different name than the incoming row — usually a bin id-collision. Use a different id, or SELECT first to confirm same entity before renaming.');
        END
    """)

    conn.commit()
    cursor.close()

    # General-concept registry (parallel of OMOP, populated by the LLM at
    # merge time). Non-fatal if sqlite-vec isn't loaded on this connection —
    # the vector virtual table will be retried on first upsert.
    try:
        _apply_general_concept_migration(conn)
    except Exception as exc:
        # Surface the failure but don't break init_db — the structural KG
        # tables above are still in place and the rest of the pipeline can
        # run; the general store just won't be available until this is fixed.
        print(f"[init_db] general-concept migration skipped: {type(exc).__name__}: {exc}")
    conn.close()


def get_schema_summary() -> str:
    """Return a summary of entity types, counts, and relation types for the agent."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT type, COUNT(*) FROM entities_for_neo4j GROUP BY type ORDER BY type")
    entity_types = cursor.fetchall()

    cursor.execute("SELECT type, COUNT(*) FROM relations_for_neo4j GROUP BY type ORDER BY type")
    relation_types = cursor.fetchall()

    cursor.close()
    conn.close()

    lines = ["=== Current KG Schema ==="]
    lines.append("Entity types:")
    for t, c in entity_types:
        lines.append(f"  {t}: {c} nodes")
    lines.append("Relation types:")
    for t, c in relation_types:
        lines.append(f"  {t}: {c} edges")
    lines.append(f"Total: {sum(c for _, c in entity_types)} entities, {sum(c for _, c in relation_types)} relations")
    return "\n".join(lines)


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]


def get_all_entities() -> list[dict]:
    """Return all entities from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, type, name, attributes FROM entities_for_neo4j")
    rows = _rows_to_dicts(cursor.fetchall())
    cursor.close()
    conn.close()
    for r in rows:
        if r["attributes"] and isinstance(r["attributes"], str):
            try:
                r["attributes"] = json.loads(r["attributes"])
            except (json.JSONDecodeError, TypeError):
                r["attributes"] = {}
    return rows


def get_all_relations() -> list[dict]:
    """Return all relations from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT source, target, type, attributes FROM relations_for_neo4j")
    rows = _rows_to_dicts(cursor.fetchall())
    cursor.close()
    conn.close()
    for r in rows:
        if r["attributes"] and isinstance(r["attributes"], str):
            try:
                r["attributes"] = json.loads(r["attributes"])
            except (json.JSONDecodeError, TypeError):
                r["attributes"] = {}
    return rows


NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", ""),
    "database": os.getenv("NEO4J_DATABASE", "medical-kg"),
}


def _sanitize_neo4j_label(name: str) -> str:
    """Sanitize a string to be a valid Neo4j label/relationship type.

    Used for relation types only. Entity labels go through `_canonical_entity_label`
    which additionally clamps to the entity-type whitelist.

    OMOP relationship_ids like "May treat" or "Is a" are uppercased to follow
    Neo4j relationship-type convention (e.g. MAY_TREAT, IS_A).
    """
    import re
    s = name.strip().upper().replace(" ", "_").replace("-", "_").replace("/", "_")
    s = re.sub(r'[^A-Z0-9_]', '', s)
    if s and s[0].isdigit():
        s = "_" + s
    return s or "UNKNOWN"


# Canonical (source_label, target_label) → relationship type.
# Applied at Neo4j write time to ensure graph consistency regardless of what
# the LLM wrote. OMOP-derived edges (identified by omop_relation_id in attrs)
# are exempt — they are already precise and should not be overridden.
# Values are UPPERCASE to match _sanitize_neo4j_label output.
_CANONICAL_REL_TYPE: dict[tuple[str, str], str] = {
    # Patient spine
    ("PATIENT", "HOSPITALADMISSION"): "HAS_ADMISSION",
    ("PATIENT", "EDSTAY"):            "HAS_ED_STAY",       # ED-only fallback (no hadm_id)
    ("PATIENT", "ALLERGY"):           "HAS_ALLERGY",
    ("PATIENT", "CONDITION"):         "HAS_CONDITION_IN_PMH",
    ("PATIENT", "OBSERVATION"):       "HAS_CONDITION_IN_PMH",
    ("PATIENT", "RACE"):              "HAS_DEMOGRAPHICS",
    ("PATIENT", "GENDER"):            "HAS_DEMOGRAPHICS",
    ("PATIENT", "MEASUREMENT"):       "HAS_BASELINE_RESULT",
    ("PATIENT", "PROCEDURE"):         "HAS_BASELINE_RESULT",
    ("PATIENT", "DRUG"):              "HAS_CHRONIC_MEDICATION",
    ("PATIENT", "SOCIALHISTORY"):     "HAS_SOCIAL_HISTORY",
    ("PATIENT", "NOTE"):              "HAS_NOTE",
    # HospitalAdmission
    ("HOSPITALADMISSION", "EDSTAY"):       "HAS_ED_STAY",
    ("HOSPITALADMISSION", "ICUSTAY"):      "HAS_ICU_STAY",
    ("HOSPITALADMISSION", "CONDITION"):    "DOCUMENTS_DIAGNOSIS",
    ("HOSPITALADMISSION", "NOTE"):         "HAS_NOTE",
    ("HOSPITALADMISSION", "TRANSFER"):     "INCLUDES_TRANSFER",
    ("HOSPITALADMISSION", "DEMOGRAPHICS"): "HAS_ENCOUNTER_CONTEXT",
    # EDStay — clinical data hangs off the most specific stay
    ("EDSTAY", "CONDITION"):    "DOCUMENTS_DIAGNOSIS",
    ("EDSTAY", "OBSERVATION"):  "DOCUMENTS_FINDING",
    ("EDSTAY", "DRUG"):         "ADMINISTERED_MED",
    ("EDSTAY", "MEASUREMENT"):  "RECORDED_VITAL",
    ("EDSTAY", "NOTE"):         "HAS_NOTE",
    ("EDSTAY", "SIGNALRECORD"): "HAS_ECG",
    ("EDSTAY", "TRANSFER"):     "INCLUDES_TRANSFER",
    ("EDSTAY", "DEMOGRAPHICS"): "HAS_ENCOUNTER_CONTEXT",
    # ICUStay
    ("ICUSTAY", "CONDITION"):   "DOCUMENTS_DIAGNOSIS",
    ("ICUSTAY", "OBSERVATION"): "DOCUMENTS_FINDING",
    ("ICUSTAY", "DRUG"):        "ADMINISTERED_MED",
    ("ICUSTAY", "MEASUREMENT"): "RECORDED_VITAL",
    ("ICUSTAY", "NOTE"):        "HAS_NOTE",
    # Note
    ("NOTE", "CONDITION"):        "DOCUMENTS_DIAGNOSIS",
    ("NOTE", "DRUG"):             "DOCUMENTS_MEDICATION",
    ("NOTE", "MEASUREMENT"):      "DOCUMENTS_FINDING",
    ("NOTE", "OBSERVATION"):      "DOCUMENTS_FINDING",
    ("NOTE", "PROCEDURE"):        "DOCUMENTS_FINDING",
    ("NOTE", "SOCIALHISTORY"):    "DOCUMENTS_FINDING",
    ("NOTE", "SPECIMEN"):         "DOCUMENTS_FINDING",
    # Other
    ("ALLERGY", "DRUG"):               "ALLERGEN_IS",
    ("SIGNALRECORD", "SIGNALLEAD"):    "HAS_LEAD",
    ("SIGNALRECORD", "FILE"):          "HAS_SOURCE_FILE",
    ("DEMOGRAPHICS", "MEASUREMENT"):   "DOCUMENTS_FINDING",
    ("DEMOGRAPHICS", "OBSERVATION"):   "DOCUMENTS_FINDING",
    ("DRUG", "ROUTE"):                 "ADMINISTERED_VIA",
}


def _canonical_entity_label(raw_type: str, attrs: dict) -> str:
    """Resolve an entity's `type` to a canonical Neo4j label.

    Backstop for the LLM: even if a non-canonical type slips through into
    SQLite, Neo4j only ever sees a label from the whitelist. Prefers the OMOP
    domain stored in attributes (ground truth) over the raw type.
    """
    from src.queries.entity_types import canonicalize_type
    omop_domain = None
    if isinstance(attrs, dict):
        omop_domain = attrs.get("omop_domain_id")
    canonical = canonicalize_type(raw_type, omop_domain=omop_domain)
    return _sanitize_neo4j_label(canonical)


_ISO_DATETIME_RE = re.compile(r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$')
_ISO_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _to_neo4j_datetime(v):
    """Convert ISO date/datetime strings to Python datetime; pass other values through."""
    if not isinstance(v, str):
        return v
    if _ISO_DATETIME_RE.match(v):
        return _dt.fromisoformat(v.replace(' ', 'T'))
    if _ISO_DATE_RE.match(v):
        return _dt.fromisoformat(v + 'T00:00:00')
    return v


def sync_to_neo4j():
    """Sync the entire KG from SQLite (entities_for_neo4j / relations_for_neo4j) to Neo4j.
    Clears Neo4j first, then writes all entities and relations."""
    driver = GraphDatabase.driver(
        NEO4J_CONFIG["uri"],
        auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"]),
    )

    # Verify the connection actually works
    driver.verify_connectivity()
    print("Neo4j connection verified.")

    entities = get_all_entities()
    relations = get_all_relations()

    # Build a lookup: entity_id -> sanitized label (for relation matching)
    entity_label_map = {}

    db_name = NEO4J_CONFIG["database"]

    with driver.session(database=db_name) as session:
        # Clear existing graph
        result = session.run("MATCH (n) DETACH DELETE n")
        result.consume()
        print("  Cleared existing Neo4j graph.")

        # Create entities
        for e in entities:
            attrs = e.get("attributes", {})
            if isinstance(attrs, str):
                try:
                    attrs = json.loads(attrs)
                except (json.JSONDecodeError, TypeError):
                    attrs = {}
            if not isinstance(attrs, dict):
                attrs = {}

            props = {"id": e["id"], "name": e["name"]}
            for k, v in attrs.items():
                if isinstance(v, (str, int, float, bool)):
                    props[k] = _to_neo4j_datetime(v)
                else:
                    props[k] = json.dumps(v, ensure_ascii=False)

            label = _canonical_entity_label(e["type"], attrs)
            entity_label_map[e["id"]] = label
            result = session.run(
                f"MERGE (n:`{label}` {{id: $id}}) SET n += $props",
                id=e["id"],
                props=props,
            )
            result.consume()

        print(f"  Created {len(entities)} entity nodes.")

        # Create relations - match by label for performance and correctness
        for r in relations:
            attrs = r.get("attributes", {})
            if isinstance(attrs, str):
                try:
                    attrs = json.loads(attrs)
                except (json.JSONDecodeError, TypeError):
                    attrs = {}
            if not isinstance(attrs, dict):
                attrs = {}

            flat_attrs = {}
            for k, v in attrs.items():
                if isinstance(v, (str, int, float, bool)):
                    flat_attrs[k] = _to_neo4j_datetime(v)
                else:
                    flat_attrs[k] = json.dumps(v, ensure_ascii=False)

            rel_type = _sanitize_neo4j_label(r["type"])
            src_label = entity_label_map.get(r["source"])
            tgt_label = entity_label_map.get(r["target"])

            # Override with canonical type unless this edge was OMOP-derived.
            # OMOP edges carry an omop_relation_id attribute set by lookup_omop_relation
            # and are already semantically precise — don't remap them.
            if src_label and tgt_label and not attrs.get("omop_relation_id"):
                rel_type = _CANONICAL_REL_TYPE.get(
                    (src_label, tgt_label), rel_type
                )

            if not src_label or not tgt_label:
                print(f"  WARNING: Skipping relation {r['source']} -[{r['type']}]-> {r['target']} (node not found in entity list)")
                continue

            timestamp = _to_neo4j_datetime(flat_attrs.get("timestamp", ""))
            result = session.run(
                f"""
                MATCH (a:`{src_label}` {{id: $source}})
                MATCH (b:`{tgt_label}` {{id: $target}})
                MERGE (a)-[r:`{rel_type}` {{timestamp: $timestamp}}]->(b)
                SET r += $props
                """,
                source=r["source"],
                target=r["target"],
                timestamp=timestamp,
                props=flat_attrs,
            )
            summary = result.consume()
            if summary.counters.relationships_created == 0 and not summary.counters.contains_updates:
                print(f"  WARNING: Relation {r['source']} -[{r['type']}]-> {r['target']} not created (node not found?)")

        print(f"  Created {len(relations)} relationships.")

    # Final verification
    with driver.session(database=db_name) as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"  Neo4j verification: {node_count} nodes, {rel_count} relationships in database '{db_name}'")

    driver.close()
    print(f"Neo4j sync complete: {len(entities)} entities, {len(relations)} relations")
