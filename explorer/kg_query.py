"""Neo4j query logic for the Medical KG Explorer.

Public API used by app.py:
  get_grounded_schema()               — live schema + sampled properties/values (cached)
  classify_message(question, history) -> "query" | "chat"
  normalize_question_terms(question)  — map colloquial terms to OMOP canonical values
  plan_and_answer(question, schema, history, term_hints, ...)
                              — multi-round parallel query pipeline; returns (answer, executed_queries)
  chat_reply(question, history)       — direct LLM reply for non-query messages
  check_neo4j_connection()            — sidebar status

Lower-level building blocks (used internally by plan_and_answer):
  cypher_from_question(...)           — LLM writes a single Cypher query
  run_cypher(cypher)                  — execute a single Cypher query against Neo4j
  answer_from_results(...)            — LLM turns a single query's rows into plain English
  plan_queries(...)                   — LLM plans a batch of Cypher queries
  run_queries_parallel(...)           — execute a batch of queries in parallel threads
  review_and_replan(...)              — LLM reviews round-1 results and optionally plans more
  synthesize_answer(...)              — LLM synthesizes answer from all executed queries
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.connection import get_client, get_deployment_name  # noqa: E402
from src.database import NEO4J_CONFIG  # noqa: E402

SQLITE_PATH = PROJECT_ROOT / "data" / "medical_kg.sqlite"


# ---------------------------------------------------------------------------
# Typed data shapes for the multi-round pipeline
# ---------------------------------------------------------------------------

class PlannedQuery(TypedDict):
    cypher: str
    rationale: str


class ExecutedQuery(TypedDict):
    cypher: str
    rationale: str
    results: list[dict]  # empty on error
    error: str           # empty string on success


# ---------------------------------------------------------------------------
# Neo4j driver
# ---------------------------------------------------------------------------

def _get_driver():
    return GraphDatabase.driver(
        NEO4J_CONFIG["uri"],
        auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"]),
    )


def check_neo4j_connection() -> tuple[bool, str]:
    try:
        driver = _get_driver()
        driver.verify_connectivity()
        driver.close()
        return True, f"Connected to {NEO4J_CONFIG['uri']} / {NEO4J_CONFIG['database']}"
    except ServiceUnavailable as e:
        return False, f"Neo4j unavailable: {e}"
    except Exception as e:
        return False, f"Connection error: {e}"


# ---------------------------------------------------------------------------
# Schema grounding — queries the live graph to discover actual structure
# ---------------------------------------------------------------------------

def _run_discovery(session, cypher: str, default=None):
    """Run a discovery query, return data() or default on any error."""
    try:
        return session.run(cypher).data()
    except Exception:
        return default or []


@lru_cache(maxsize=1)
def get_grounded_schema() -> str:
    """Build a rich schema description from the live graph.

    Queries the actual labels, relationship types, property keys, and
    representative values so the LLM knows exactly what exists — including
    correct casing, real property names, and how ambiguous concepts like
    gender are actually stored.
    """
    try:
        driver = _get_driver()
        with driver.session(database=NEO4J_CONFIG["database"]) as s:
            # Labels and counts
            label_counts = {
                r["label"]: r["cnt"]
                for r in _run_discovery(s,
                    "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt "
                    "ORDER BY cnt DESC"
                )
            }

            # Build a case-insensitive lookup so we can find e.g. "PATIENT" or "Patient"
            _label_ci = {lbl.upper(): lbl for lbl in label_counts}

            def _lbl(name: str) -> str:
                """Return the actual label casing from the live graph, or the input."""
                return _label_ci.get(name.upper(), name)

            # Relationship types (exact casing from DB)
            rel_types = [
                r["relationshipType"]
                for r in _run_discovery(s,
                    "CALL db.relationshipTypes() YIELD relationshipType"
                )
            ]

            # Build a case-insensitive rel type lookup for queries
            _rel_ci = {rt.upper(): rt for rt in rel_types}

            def _rel(name: str) -> str:
                return _rel_ci.get(name.upper(), name)

            # Property keys per label (sample first node)
            label_props: dict[str, list[str]] = {}
            for label in label_counts:
                rows = _run_discovery(s,
                    f"MATCH (n:`{label}`) RETURN keys(n) AS k LIMIT 1"
                )
                if rows:
                    label_props[label] = sorted(rows[0].get("k") or [])

            # Property keys per relationship type (sample first edge)
            rel_props: dict[str, list[str]] = {}
            for rt in rel_types:
                rows = _run_discovery(s,
                    f"MATCH ()-[r:`{rt}`]->() RETURN keys(r) AS k LIMIT 1"
                )
                if rows:
                    keys = sorted(k for k in (rows[0].get("k") or []) if k != "timestamp")
                    if keys:
                        rel_props[rt] = keys

            # Sample values for key properties
            samples: dict[str, list] = {}

            # Generic name samples for every label — shows the LLM what content
            # each label actually holds (e.g. SignalRecord = "ECG 40689238",
            # Note = "History of Present Illness", etc.)
            for label in label_counts:
                name_rows = _run_discovery(s,
                    f"MATCH (n:`{label}`) WHERE n.name IS NOT NULL "
                    f"RETURN n.name AS name LIMIT 5"
                )
                if name_rows:
                    samples[f"{label} names"] = [r["name"] for r in name_rows]

            # Measurement value+unit samples (numeric context the generic loop misses)
            _meas_lbl = _lbl("Measurement")
            measurements = _run_discovery(s,
                f"MATCH (n:`{_meas_lbl}`) WHERE n.value IS NOT NULL "
                f"RETURN n.name AS name, n.value AS value, n.unit AS unit LIMIT 5"
            )
            if measurements:
                samples["Measurement values"] = measurements

            # Gender/Race relationship paths from Patient (needed for ^^^WARNING detection)
            _pat_lbl = _lbl("Patient")
            _gender_lbl = _lbl("Gender")
            _race_lbl = _lbl("Race")

            gender_paths = _run_discovery(s,
                f"MATCH (p:`{_pat_lbl}`)-[r]->(g:`{_gender_lbl}`) "
                f"RETURN type(r) AS rel, g.name AS value LIMIT 4"
            )
            if gender_paths:
                samples[f"{_pat_lbl}→{_gender_lbl} paths"] = gender_paths

            race_paths = _run_discovery(s,
                f"MATCH (p:`{_pat_lbl}`)-[r]->(n:`{_race_lbl}`) "
                f"RETURN type(r) AS rel, n.name AS value LIMIT 3"
            )
            if race_paths:
                samples[f"{_pat_lbl}→{_race_lbl} paths"] = race_paths

            # Outgoing rel types from Patient — count distinct (rel, target) pairs
            # IMPORTANT: do NOT deduplicate — multiple rels to the same target must all appear.
            patient_rels = _run_discovery(s,
                f"MATCH (p:`{_pat_lbl}`)-[r]->(t) "
                f"RETURN type(r) AS rel, labels(t)[0] AS target, count(*) AS cnt "
                f"ORDER BY target, rel"
            )
            patient_rel_pairs = [(row["rel"], row["target"], row["cnt"]) for row in patient_rels]

            # Outgoing rels from HospitalAdmission, EDStay, ICUStay
            _hadm_lbl = _lbl("HospitalAdmission")
            hadm_rels = _run_discovery(s,
                f"MATCH (:`{_pat_lbl}`)-[:`{_rel('has_admission')}`]->(a:`{_hadm_lbl}`)-[r]->(t) "
                f"RETURN type(r) AS rel, labels(t)[0] AS target, count(*) AS cnt "
                f"ORDER BY target, rel"
            )
            hadm_rel_pairs = [(row["rel"], row["target"], row["cnt"]) for row in hadm_rels]

            _ed_lbl = _lbl("EDStay")
            edstay_rels = _run_discovery(s,
                f"MATCH (:`{_hadm_lbl}`)-[:`{_rel('has_ed_stay')}`]->(e:`{_ed_lbl}`)-[r]->(t) "
                f"RETURN type(r) AS rel, labels(t)[0] AS target, count(*) AS cnt "
                f"ORDER BY target, rel"
            )
            edstay_rel_pairs = [(row["rel"], row["target"], row["cnt"]) for row in edstay_rels]

            _icu_lbl = _lbl("ICUStay")
            icustay_rels = _run_discovery(s,
                f"MATCH (:`{_hadm_lbl}`)-[:`{_rel('has_icu_stay')}`]->(i:`{_icu_lbl}`)-[r]->(t) "
                f"RETURN type(r) AS rel, labels(t)[0] AS target, count(*) AS cnt "
                f"ORDER BY target, rel"
            )
            icustay_rel_pairs = [(row["rel"], row["target"], row["cnt"]) for row in icustay_rels]

            # All distinct paths reachable from Patient (up to 4 hops), deduplicated by rel chain
            path_rows = _run_discovery(s,
                f"MATCH path = (p:`{_pat_lbl}`)-[*1..4]->(leaf) "
                f"WITH [r IN relationships(path) | type(r)] AS rels, "
                f"     [n IN nodes(path) | labels(n)[0]] AS node_labels "
                f"RETURN DISTINCT rels, node_labels "
                f"ORDER BY size(rels), node_labels[-1] "
                f"LIMIT 300"
            )

        driver.close()

    except Exception as e:
        return f"(Schema discovery failed: {e})\n\nFallback: graph contains Patient, Encounter, Condition, Drug, Measurement, Observation, Note, Gender, Race, Allergy, Procedure, SignalRecord, SignalLead nodes."

    # ---------------------------------------------------------------------------
    # Assemble the schema text
    # ---------------------------------------------------------------------------
    lines = ["=== GRAPH SCHEMA (live, sampled from actual data) ===\n"]

    lines.append("NODE LABELS AND COUNTS:")
    for label, cnt in label_counts.items():
        props = label_props.get(label, [])
        lines.append(f"  {label} ({cnt} nodes)  properties: {', '.join(props) if props else '(none sampled)'}")

    lines.append("\nRELATIONSHIP TYPES (exact casing — Cypher is case-sensitive):")
    for rt in sorted(rel_types):
        lines.append(f"  {rt}")

    if rel_props:
        lines.append("\nRELATIONSHIP PROPERTIES (use r.<prop> to access — data lives on the edge, not a node):")
        for rt in sorted(rel_props):
            lines.append(f"  [:{rt}]  properties: {', '.join(rel_props[rt])}")

    lines.append("\nPATIENT OUTGOING RELATIONSHIPS (direct, 1 hop) — ALL paths, including multiple rels to same target:")
    # Group by target to detect inconsistency
    from collections import defaultdict
    patient_by_target: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for rel, target, cnt in patient_rel_pairs:
        patient_by_target[target].append((rel, cnt))

    for target in sorted(patient_by_target):
        rels = patient_by_target[target]
        for rel, cnt in sorted(rels):
            lines.append(f"  (Patient)-[:{rel}]->({target})  [{cnt} edges]")
        if len(rels) > 1:
            rel_names = "|".join(r for r, _ in sorted(rels))
            lines.append(
                f"  ^^^ WARNING: {target} is reachable via {len(rels)} different rel types. "
                f"Always query ALL of them: [:{rel_names}]"
            )

    def _emit_outgoing(label: str, pairs: list[tuple[str, str, int]]) -> list[str]:
        out = []
        by_target: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for rel, target, cnt in pairs:
            by_target[target].append((rel, cnt))
        for target in sorted(by_target):
            rels = by_target[target]
            for rel, cnt in sorted(rels):
                out.append(f"  ({label})-[:{rel}]->({target})  [{cnt} edges]")
            if len(rels) > 1:
                rel_names = "|".join(r for r, _ in sorted(rels))
                out.append(
                    f"  ^^^ WARNING: {target} is reachable via {len(rels)} different rel types. "
                    f"Always query ALL of them: [:{rel_names}]"
                )
        return out

    lines.append("\nHOSPITALADMISSION OUTGOING RELATIONSHIPS (via HAS_ADMISSION) — ALL paths:")
    lines.extend(_emit_outgoing("HospitalAdmission", hadm_rel_pairs) or ["  (none found)"])

    lines.append("\nEDSTAY OUTGOING RELATIONSHIPS (via HAS_ED_STAY) — ALL paths:")
    lines.extend(_emit_outgoing("EDStay", edstay_rel_pairs) or ["  (none found)"])

    lines.append("\nICUSTAY OUTGOING RELATIONSHIPS (via HAS_ICU_STAY) — ALL paths:")
    lines.extend(_emit_outgoing("ICUStay", icustay_rel_pairs) or ["  (none found)"])

    if samples:
        lines.append("\nSAMPLED DATA VALUES:")
        for key, vals in samples.items():
            lines.append(f"  {key}:")
            for v in vals:
                lines.append(f"    {v}")

    # Live-discovered reachable paths from Patient
    if path_rows:
        lines.append("\nREACHABLE PATHS FROM PATIENT (live, up to 4 hops, deduplicated):")
        lines.append("  Use these paths to construct MATCH clauses — do not assume a path exists")
        lines.append("  if it is not listed here.")
        seen_chains: set[tuple] = set()
        for row in path_rows:
            rels = row.get("rels") or []
            node_labels = row.get("node_labels") or []
            if not rels or not node_labels:
                continue
            chain_key = tuple(rels)
            if chain_key in seen_chains:
                continue
            seen_chains.add(chain_key)
            # Format as Cypher-style path: (Patient)-[:rel1]->(:Label1)-[:rel2]->(:Label2)
            path_str = "(Patient)"
            for rel, label in zip(rels, node_labels[1:]):
                path_str += f"-[:{rel}]->(:{label})"
            lines.append(f"  {path_str}")
    else:
        lines.append("\nREACHABLE PATHS FROM PATIENT: (not available — graph may be empty)")

    lines.append("""
NOTE: Relationship types are case-sensitive — use them EXACTLY as shown in REACHABLE PATHS.
IMPORTANT: For any node label marked ^^^ WARNING above, always use pipe syntax to query ALL
  relationship types simultaneously, e.g. [:rel_a|rel_b]->(Target).
  Querying only one will silently miss records that used the other type.
""")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Term normalization — map colloquial demographic terms to OMOP canonical values
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_demographic_vocab() -> dict[str, str]:
    """Build a lowercase-synonym → canonical-name map for Race, Gender, Ethnicity.

    Strategy: only include canonical values that actually exist as node names in
    Neo4j (discovered at startup), then map OMOP synonyms of those concepts to
    those canonical values. This prevents mapping "black" → "African race" when
    the actual graph value is "Black or African American".
    """
    if not SQLITE_PATH.exists():
        return {}

    # Step 1: get the actual demographic node values from the live graph
    actual_values: set[str] = set()
    try:
        driver = _get_driver()
        with driver.session(database=NEO4J_CONFIG["database"]) as s:
            rows = _run_discovery(s,
                "MATCH (n) WHERE toUpper(labels(n)[0]) IN ['RACE','GENDER','ETHNICITY'] "
                "RETURN DISTINCT n.name AS name"
            )
            for r in rows:
                if r.get("name"):
                    actual_values.add(r["name"].strip())
        driver.close()
    except Exception:
        return {}

    if not actual_values:
        return {}

    # Step 2: for each actual graph value, find all OMOP synonyms and map them back
    vocab: dict[str, str] = {}
    try:
        conn = sqlite3.connect(str(SQLITE_PATH))
        conn.row_factory = sqlite3.Row

        for canonical in actual_values:
            # The canonical name itself is always a match
            vocab[canonical.lower()] = canonical

            # Add each individual word from the canonical as a match too.
            # "Black or African American" → "black" maps to it, "african" maps to it.
            # This ensures colloquial single-word queries hit the right canonical.
            # Exclude stopwords that would create false matches.
            _stopwords = {"or", "and", "of", "the", "a", "an", "not", "other", "unknown"}
            for word in re.findall(r"[a-z]+", canonical.lower()):
                if word not in _stopwords and len(word) > 2:
                    # Only set if not already set by a more-specific synonym
                    vocab.setdefault(word, canonical)

            # Find synonyms of any OMOP concept with this exact name
            rows = conn.execute(
                "SELECT cs.concept_synonym_name "
                "FROM concept c "
                "JOIN concept_synonym cs ON c.concept_id = cs.concept_id "
                "WHERE c.concept_name = ? "
                "AND c.domain_id IN ('Race', 'Gender', 'Ethnicity')",
                (canonical,)
            ).fetchall()
            for r in rows:
                syn = (r["concept_synonym_name"] or "").strip()
                if syn:
                    vocab[syn.lower()] = canonical

        conn.close()
    except Exception:
        pass

    return vocab


def normalize_question_terms(question: str) -> dict[str, str]:
    """Return a mapping of {original_term: omop_canonical_name} for any word or
    short phrase in the question that matches an OMOP demographic synonym.

    Only returns matches where the canonical form differs from what the user wrote
    (case-normalized), so the caller only injects hints when they actually add info.

    Example: "black patients" → {"black": "Black or African American"}
    """
    if not question:
        return {}

    vocab = _load_demographic_vocab()
    if not vocab:
        return {}

    found: dict[str, str] = {}
    q_lower = question.lower()

    # Try 1-3 word phrases to catch "african american", "not hispanic", etc.
    words = re.findall(r"[a-z]+", q_lower)
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            if phrase in vocab:
                canonical = vocab[phrase]
                # Report if the canonical is meaningfully different from what
                # the user typed — either different casing or a longer form
                if canonical != phrase and canonical.lower() != phrase:
                    found[phrase] = canonical

    return found


# ---------------------------------------------------------------------------
# Message routing — classify before generating Cypher
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """You classify user messages in a medical knowledge graph chat interface.

Reply with exactly one word — either:
  query   — the user is asking for information that requires searching the graph database
             (e.g. "how many patients", "what conditions does patient X have", "list all drugs")
  chat    — the user is asking a follow-up conversational question, asking for an explanation,
             asking why something was done, or making a comment that does not require a new graph query
             (e.g. "why did you write that query", "what does that mean", "thanks", "explain this",
              "can you simplify", "what is Metformin")

Output only the single word, no punctuation."""


def classify_message(question: str, history: list[dict]) -> str:
    """Return 'query' or 'chat'. Defaults to 'query' on any failure."""
    try:
        client = get_client()
        deployment = get_deployment_name()
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                *history[-4:],
                {"role": "user", "content": question},
            ],
            max_completion_tokens=5,
            temperature=0,
        )
        label = (response.choices[0].message.content or "query").strip().lower()
        return "chat" if label.startswith("chat") else "query"
    except Exception:
        return "query"


# ---------------------------------------------------------------------------
# Cypher generation
# ---------------------------------------------------------------------------

_CYPHER_SYSTEM = """You are a Neo4j Cypher expert for a clinical knowledge graph.

{schema}

RULES:
- Return ONLY a valid Cypher query — no explanation, no markdown, no code fences.
- Never use DETACH DELETE, MERGE, SET, CREATE, or DROP — this is a read-only graph.
- Always include a LIMIT clause (default 50 unless the user asks for all results).
- Use case-insensitive string matching: toLower(n.name) CONTAINS toLower('term')
- Node labels must match EXACTLY as listed in NODE LABELS AND COUNTS above — never invent a label that is not listed there (e.g. 'Medication' is not a valid label; use 'Drug').
- Relationship types are case-sensitive — use them EXACTLY as they appear in REACHABLE PATHS.
- Some relationships carry data properties (listed in RELATIONSHIP PROPERTIES in the schema).
  Access them with `r.<prop>` in the MATCH: MATCH (a)-[r:REL_TYPE]->(b) RETURN r.value, r.unit
  Do NOT look for a separate node when the data is on the edge itself.
- For any node label marked ^^^ WARNING in the schema, use pipe syntax to query ALL
  listed rel types simultaneously: (p)-[:rel_a|rel_b]->(Target). Never use just one.
- Before writing any MATCH, check REACHABLE PATHS in the schema above to find the correct
  path to the target node type. Do not assume a path exists if it is not listed.
- Clinical content nodes (Measurement, Condition, Drug, Observation) are reachable via
  MULTIPLE carriers: EDSTAY, HOSPITALADMISSION, ICUSTAY, NOTE, and directly from PATIENT.
  DEFAULT: search ALL carriers with UNION ALL unless the user explicitly narrows the scope.
- Node id format: patient_<subject_id>, admission_<hadm_id>, edstay_<stay_id>,
  icustay_<icustay_id>

EXAMPLES (illustrating how to use REACHABLE PATHS — rel types taken verbatim from schema):
Q: "Find patients who had vomiting"
   # REACHABLE PATHS shows HOSPITALADMISSION and EDSTAY both have DOCUMENTS_DIAGNOSIS->CONDITION
A: MATCH (p:PATIENT)-[:HAS_ADMISSION]->(:HOSPITALADMISSION)-[:DOCUMENTS_DIAGNOSIS]->(c:CONDITION)
   WHERE toLower(c.name) CONTAINS 'vomiting'
   RETURN DISTINCT p.id, p.name LIMIT 50

Q: "Show all drugs given to female patients"
   # HAS_DEMOGRAPHICS for gender; EDSTAY carries ADMINISTERED_MED
A: MATCH (p:PATIENT)-[:HAS_DEMOGRAPHICS]->(g:GENDER)
   WHERE toLower(g.name) CONTAINS 'female'
   MATCH (p)-[:HAS_ADMISSION]->(:HOSPITALADMISSION)-[:HAS_ED_STAY]->(e:EDSTAY)-[:ADMINISTERED_MED]->(d:DRUG)
   RETURN DISTINCT p.id, d.name LIMIT 50

Q: "Which patients have hypertension in their medical history?"
   # REACHABLE PATHS shows (PATIENT)-[:HAS_CONDITION_IN_PMH]->(:CONDITION)
A: MATCH (p:PATIENT)-[:HAS_CONDITION_IN_PMH]->(c:CONDITION)
   WHERE toLower(c.name) CONTAINS 'hypertension'
   RETURN p.id, p.name, c.name LIMIT 50

Q: "Show all blood pressure values"
   # Measurement is reachable via EDSTAY, HOSPITALADMISSION, ICUSTAY, and directly from PATIENT.
   # Always search ALL carriers with UNION ALL — never query just one carrier.
A: MATCH (e:EDSTAY)-[r:RECORDED_VITAL]->(m:MEASUREMENT)
   WHERE toLower(m.name) CONTAINS 'blood pressure'
   RETURN e.id AS source, r.value, r.unit, r.timestamp
   UNION ALL
   MATCH (a:HOSPITALADMISSION)-[r:RECORDED_VITAL]->(m:MEASUREMENT)
   WHERE toLower(m.name) CONTAINS 'blood pressure'
   RETURN a.id AS source, r.value, r.unit, r.timestamp
   UNION ALL
   MATCH (p:PATIENT)-[r:HAS_BASELINE_RESULT]->(m:MEASUREMENT)
   WHERE toLower(m.name) CONTAINS 'blood pressure'
   RETURN p.id AS source, r.value, r.unit, r.timestamp
   LIMIT 50
""".strip()

_ANSWER_SYSTEM = """You are a clinical data assistant. The user asked a question about a medical knowledge graph.
You were given a Cypher query that was run against Neo4j and the raw results.
Write a clear, concise answer in plain English. Use bullet points for lists.
If the results are empty, say clearly that no matching data was found.
Do not mention Cypher or technical implementation details."""

_CHAT_SYSTEM = """You are an assistant for a medical knowledge graph explorer tool.
You have full knowledge of the graph schema, discovered live from the database:

{schema}

Answer the user's conversational question or explanation request.
You can answer schema questions (what nodes exist, what relationships exist,
how to traverse from A to B) directly from the schema above — no query needed.
Be concise. If the user asks about a previous query, explain it clearly.
Do not generate Cypher unless explicitly asked."""

_PLAN_QUERIES_SYSTEM = """You are a Neo4j Cypher planning expert for a clinical knowledge graph.

{schema}

CYPHER RULES (apply to every query you write):
- Never use DETACH DELETE, MERGE, SET, CREATE, or DROP — read-only graph.
- Always include a LIMIT clause (default 50 unless the user asks for all results).
- Use case-insensitive matching: toLower(n.name) CONTAINS toLower('term')
- Node labels must match EXACTLY as listed in NODE LABELS AND COUNTS above — never invent a label that is not listed there (e.g. 'Medication' is not a valid label; use 'Drug').
- Relationship types are case-sensitive — use them EXACTLY as shown in REACHABLE PATHS.
- Relationship properties are on the edge: MATCH (a)-[r:REL]->(b) RETURN r.value, r.unit
- For labels marked ^^^ WARNING in the schema, query ALL listed rel types with pipe syntax.
- Before any MATCH, verify the path exists in REACHABLE PATHS. Do not assume.
- Node id format: patient_<subject_id>, admission_<hadm_id>, edstay_<stay_id>, icustay_<icustay_id>
- Timestamps are native Neo4j datetime values — use date() for date filtering:
    date(r.storetime) = date('2125-09-29')
  For ranges: date(r.storetime) >= date('2125-09-29') AND date(r.storetime) <= date('2125-09-30')
  Never compare raw datetime properties with datetime() — use date() to strip timezone.

PLANNING STRATEGY:
- CONCEPT-FIRST: For any clinical content question, always plan at least one query that
  starts directly from the concept node type most relevant to the question and returns it
  without any additional MATCH or JOIN. Reason: "what node type best represents what I am
  looking for?" — match it by name, return it, done.
  Example for an imaging question:
    MATCH (proc:PROCEDURE)
    WHERE toLower(proc.name) CONTAINS toLower('angiography')
    RETURN proc.name, proc.id, proc.source_term LIMIT 50
  Example for a lab question:
    MATCH (m:MEASUREMENT)
    WHERE toLower(m.name) CONTAINS toLower('urea nitrogen')
    RETURN m.name, m.id, m.source_term LIMIT 50
  CRITICAL: Do NOT add a second MATCH or a spine join (PERFORMED_PROCEDURE, HAS_ADMISSION,
  etc.) after the concept node match — that defeats the purpose and misses nodes only
  reachable via NOTE subtrees. The graph has a single patient; no scoping is needed.
- SPINE COMPLEMENT: Also plan separate queries starting from the encounter spine
  (EDSTAY, HOSPITALADMISSION) via structured relationship types, as these may hold
  instances the concept-first search misses (e.g. a drug administered but not documented
  in any note).
- For questions about demographics, spine, or structural nodes (Patient, Gender, Race,
  Allergy), a single spine query is usually sufficient.
- Each query must be independently runnable and retrieve a meaningfully different slice of data.
  Do not combine all carriers into one UNION ALL.
- Include route/administration-mode filtering only when the question explicitly asks for it;
  otherwise return all instances and let the synthesis step group by route.

OUTPUT FORMAT — return a JSON object with exactly this structure:
{{
  "queries": [
    {{
      "cypher": "<valid Cypher — no markdown, no code fences>",
      "rationale": "<one sentence: what this query retrieves and why>"
    }}
  ]
}}
Return {{"queries": []}} if no queries are needed.
Return at most {max_queries} queries. Each query must be independently runnable.
Do NOT wrap the JSON in markdown code fences.""".strip()

_SYNTHESIZE_SYSTEM = """You are a clinical data assistant. The user asked a question about a \
medical knowledge graph. Multiple Cypher queries searched different carrier paths in the graph.

BEFORE writing a single word of your answer, execute this reasoning protocol in order:

STEP 1 — SET ASIDE EMPTY RESULTS
List all queries that returned no rows or an error. Do not use these to draw any conclusion.
Empty = "this path had nothing." It does NOT mean the concept is absent overall.

STEP 2 — SCORE EACH NON-EMPTY RESULT FOR RELEVANCE
For every query that returned rows, judge each row against the user's question:
  STRONG match  — the row names or describes the exact concept/exam/drug/condition asked about,
                  including clinical synonyms, abbreviations, and paraphrases
                  (e.g. "CT angiography of chest with contrast" is a STRONG match for
                  "chest CT with contrast"; "CTA thorax" is also a STRONG match)
  PARTIAL match — the row is related but not the exact concept
  UNRELATED     — the row clearly belongs to a different concept
                  (e.g. "Lung segmentectomy" for a chest CT question is UNRELATED)

STEP 3 — CONCLUDE FROM THE STRONGEST MATCH FOUND
  • If any STRONG or PARTIAL match was found on any carrier → answer affirmatively.
    The carrier type (radiology report section, structured procedure, note, etc.) does NOT
    matter — a match is a match regardless of which path it came from.
  • Only answer "not found" if STEP 2 produced zero STRONG or PARTIAL matches across
    ALL non-empty results.

STEP 4 — CITE PROVENANCE FOR EVERY FINDING
For each confirmed finding state: concept name, carrier path, and timestamp if available.
Example: "CT angiography of chest with contrast — documented in Radiology Report →
EXAMINATION section at 2125-09-28 20:31, tied to ED stay 31293660."

STEP 5 — MENTION GAPS AS CONTEXT ONLY
After reporting findings, briefly note which carrier paths returned nothing — framed as
"not found on the structured procedure path", not "not found."

OUTPUT RULES:
- Write in clear, concise plain English.
- Use bullet points for lists.
- Do not mention Cypher, Neo4j, query rounds, or implementation details.
- Synthesize into a single coherent answer — do not give a per-query breakdown."""

_MAX_TOTAL_QUERIES = 10
_DEFAULT_ROUND1_BUDGET = 6


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:cypher)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Multi-round parallel query pipeline
# ---------------------------------------------------------------------------

def _format_results_for_review(executed: list[ExecutedQuery]) -> str:
    parts = []
    for i, q in enumerate(executed, 1):
        parts.append(f"Query {i}: {q['rationale']}")
        cypher_preview = q["cypher"][:200] + ("..." if len(q["cypher"]) > 200 else "")
        parts.append(f"  Cypher: {cypher_preview}")
        if q["error"]:
            parts.append(f"  Result: ERROR — {q['error']}")
        elif not q["results"]:
            parts.append("  Result: (no rows returned)")
        else:
            row_preview = "\n".join(f"    {row}" for row in q["results"][:5])
            parts.append(f"  Result: {len(q['results'])} rows\n{row_preview}")
    return "\n".join(parts)


def plan_queries(
    question: str,
    schema: str,
    history: list[dict],
    max_queries: int,
    term_hints: dict[str, str] | None = None,
) -> list[PlannedQuery]:
    """Ask the LLM to plan a batch of Cypher queries for the question.

    Returns a list of PlannedQuery dicts. Returns an empty list if planning
    fails (LLM error, JSON parse error) or the LLM decides no queries are needed.
    max_queries is clamped to [1, 10].
    """
    max_queries = max(1, min(_MAX_TOTAL_QUERIES, max_queries))
    client = get_client()
    deployment = get_deployment_name()

    user_content = (
        f"Question: {question}\n\n"
        f"Plan up to {max_queries} Cypher queries to answer this question completely.\n"
        f"Each query should retrieve different, complementary information.\n"
        f"Avoid redundant queries that would return the same data."
    )
    if term_hints:
        hints_text = "\n".join(
            f'  "{term}" → use "{canonical}" in the Cypher (OMOP canonical value)'
            for term, canonical in term_hints.items()
        )
        user_content += f"\n\nTERM HINTS — use these exact OMOP values:\n{hints_text}"

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": _PLAN_QUERIES_SYSTEM.format(schema=schema, max_queries=max_queries),
                },
                *history[-6:],
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=2048,
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        queries = parsed.get("queries", [])
        result: list[PlannedQuery] = []
        for item in queries[:max_queries]:
            cypher = (item.get("cypher") or "").strip()
            rationale = (item.get("rationale") or "").strip()
            if cypher:
                result.append({"cypher": cypher, "rationale": rationale or "(no rationale)"})
        return result
    except Exception:
        return []


def run_queries_parallel(
    planned_queries: list[PlannedQuery],
    *,
    max_workers: int = 5,
) -> list[ExecutedQuery]:
    """Execute each planned query in a thread pool.

    Returns results in the same order as planned_queries. Each result carries
    either populated results (on success) or a populated error string (on failure).
    Never raises.
    """
    if not planned_queries:
        return []

    n = len(planned_queries)
    output: list[ExecutedQuery] = [
        {"cypher": q["cypher"], "rationale": q["rationale"], "results": [], "error": ""}
        for q in planned_queries
    ]

    def _run_one(index: int, cypher: str) -> tuple[int, list[dict], str]:
        try:
            rows = run_cypher(cypher)
            return index, rows, ""
        except Exception as exc:
            return index, [], str(exc)

    workers = min(max_workers, n)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one, i, planned_queries[i]["cypher"]): i
            for i in range(n)
        }
        for future in as_completed(futures):
            idx, rows, err = future.result()
            output[idx]["results"] = rows
            output[idx]["error"] = err

    return output


def review_and_replan(
    question: str,
    schema: str,
    history: list[dict],
    round1_results: list[ExecutedQuery],
    budget_remaining: int,
) -> list[PlannedQuery]:
    """LLM reviews round-1 results and optionally plans more queries.

    Returns empty list if: budget_remaining <= 0, LLM decides answer is complete,
    or any error occurs.
    """
    if budget_remaining <= 0:
        return []

    client = get_client()
    deployment = get_deployment_name()
    results_summary = _format_results_for_review(round1_results)

    user_content = (
        f"Original question: {question}\n\n"
        f"Round 1 queries and results:\n{results_summary}\n\n"
        f"Review the results above carefully.\n"
        f'If the question is fully answered, return {{"queries": []}}.\n'
        f"If key information is missing, empty, or errored, plan up to {budget_remaining} "
        f"additional Cypher queries to fill the gaps.\n"
        f"Do not repeat queries that already returned data."
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": _PLAN_QUERIES_SYSTEM.format(
                        schema=schema, max_queries=budget_remaining
                    ),
                },
                *history[-6:],
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=2048,
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        queries = parsed.get("queries", [])
        result: list[PlannedQuery] = []
        for item in queries[:budget_remaining]:
            cypher = (item.get("cypher") or "").strip()
            rationale = (item.get("rationale") or "").strip()
            if cypher:
                result.append({"cypher": cypher, "rationale": rationale or "(no rationale)"})
        return result
    except Exception:
        return []


def synthesize_answer(
    question: str,
    all_results: list[ExecutedQuery],
    history: list[dict],
) -> str:
    """Generate a final plain-English answer from all executed queries across all rounds."""
    client = get_client()
    deployment = get_deployment_name()

    parts = [f"Question: {question}\n"]
    for i, q in enumerate(all_results, 1):
        parts.append(f"Query {i} ({q['rationale']}):")
        if q["error"]:
            parts.append(f"  [FAILED: {q['error']}]")
        elif not q["results"]:
            parts.append("  (no results)")
        else:
            rows_text = "\n".join(f"  {row}" for row in q["results"][:100])
            parts.append(f"  {len(q['results'])} rows:\n{rows_text}")

    user_content = "\n".join(parts)

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": _SYNTHESIZE_SYSTEM},
                *history[-4:],
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=2048,
        )
        return response.choices[0].message.content or "(no answer generated)"
    except Exception as exc:
        return f"(Error generating answer: {exc})"


def plan_and_answer(
    question: str,
    schema: str,
    history: list[dict],
    term_hints: dict[str, str] | None = None,
    *,
    round1_budget: int = _DEFAULT_ROUND1_BUDGET,
    progress_callback: "Any | None" = None,
) -> tuple[str, list[ExecutedQuery]]:
    """Multi-round parallel Cypher query pipeline.

    Round 1: LLM plans up to round1_budget queries, all execute in parallel.
    Round 2 (optional): LLM reviews results, plans follow-up queries if gaps found,
    those execute in parallel.
    Total queries across both rounds is capped at _MAX_TOTAL_QUERIES (10).

    Returns (answer_text, all_executed_queries). Never raises — errors are captured
    into ExecutedQuery.error or returned as the answer string.
    progress_callback(msg) is called only from the main thread, never from workers.
    """
    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    n1 = max(1, min(_MAX_TOTAL_QUERIES, round1_budget))

    _progress("Planning queries...")
    round1_planned = plan_queries(question, schema, history, n1, term_hints)

    if not round1_planned:
        _progress("Generating answer...")
        answer = synthesize_answer(question, [], history)
        return answer, []

    q_word = "query" if len(round1_planned) == 1 else "queries"
    _progress(f"Running {len(round1_planned)} {q_word}...")
    round1_executed = run_queries_parallel(round1_planned)
    all_executed = round1_executed[:]

    budget_remaining = _MAX_TOTAL_QUERIES - len(round1_planned)
    if budget_remaining > 0:
        _progress("Reviewing results, checking for gaps...")
        round2_planned = review_and_replan(
            question, schema, history, round1_executed, budget_remaining
        )
        if round2_planned:
            q_word2 = "query" if len(round2_planned) == 1 else "queries"
            _progress(f"Running {len(round2_planned)} follow-up {q_word2}...")
            round2_executed = run_queries_parallel(round2_planned)
            all_executed.extend(round2_executed)

    _progress("Generating answer...")
    answer = synthesize_answer(question, all_executed, history)
    return answer, all_executed


# ---------------------------------------------------------------------------
# Single-query building blocks (used internally or directly)
# ---------------------------------------------------------------------------


def cypher_from_question(
    question: str,
    schema: str,
    history: list[dict],
    term_hints: dict[str, str] | None = None,
) -> str:
    """Generate Cypher from the question.

    term_hints maps colloquial terms to OMOP canonical values, e.g.
    {"black": "Black or African American"}. When present, they are appended
    to the user message so the LLM uses the exact graph values.
    """
    client = get_client()
    deployment = get_deployment_name()

    user_content = question
    if term_hints:
        hints_text = "\n".join(
            f'  "{term}" → use "{canonical}" in the Cypher (OMOP canonical value)'
            for term, canonical in term_hints.items()
        )
        user_content = (
            f"{question}\n\n"
            f"TERM HINTS — use these exact OMOP values in the query:\n{hints_text}"
        )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _CYPHER_SYSTEM.format(schema=schema)},
            *history[-6:],
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=1024,
        temperature=0,
    )
    return _strip_code_fence(response.choices[0].message.content or "")


def answer_from_results(question: str, cypher: str, results: list[dict[str, Any]]) -> str:
    client = get_client()
    deployment = get_deployment_name()
    results_text = (
        "\n".join(str(row) for row in results[:100]) if results else "(no results returned)"
    )
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Cypher query used:\n{cypher}\n\n"
                f"Query results ({len(results)} rows):\n{results_text}"
            )},
        ],
        max_completion_tokens=2048,
    )
    return response.choices[0].message.content or "(no answer generated)"


def chat_reply(question: str, history: list[dict], schema: str = "") -> str:
    """Direct conversational reply — no Cypher, no Neo4j."""
    client = get_client()
    deployment = get_deployment_name()
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _CHAT_SYSTEM.format(schema=schema or "(not available)")},
            *history[-6:],
            {"role": "user", "content": question},
        ],
        max_completion_tokens=1024,
    )
    return response.choices[0].message.content or "(no reply generated)"


# ---------------------------------------------------------------------------
# Cypher execution
# ---------------------------------------------------------------------------

def run_cypher(cypher: str) -> list[dict[str, Any]]:
    driver = _get_driver()
    try:
        with driver.session(database=NEO4J_CONFIG["database"]) as session:
            result = session.run(cypher)
            rows = [dict(record) for record in result]
        return rows
    finally:
        driver.close()
