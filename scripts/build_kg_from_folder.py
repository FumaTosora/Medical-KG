import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.connection import get_client
from src.queries.upload_folder import send_folder_to_model
from src.queries.kg_queries import _parse_csv_response, _format_as_csv
from src.queries.db_tooling import run_llm_db_session, execute_sql
from src.queries.entity_types import allowed_types_enum
from src.queries.identity import derive_natural_key_id
from src.database import (
    init_db,
    get_schema_summary,
    sync_to_neo4j,
)

OUTPUT_DIR = str(PROJECT_ROOT / "logs" / "build_kg_output")


_ID_SAFE = re.compile(r"[^a-z0-9]+")


def _bin_id_prefix(data_path: str) -> str:
    """Derive a stable, globally-unique ID prefix from a bin folder path.

    A path like
        data/.../patient-time-bins/10000032/time_bins_stay_35968195/bin_0
    becomes
        p10000032_s35968195_b0

    Each bin gets its own prefix so entity IDs from different bins never
    collide in the staging tables. The merge agent treats id collisions as
    'this is the same entity, merge it', which silently overwrites cross-bin
    data when the extraction step happens to reuse short ids like e1, e2...
    Prefixing eliminates that whole class of bug at the source.
    """
    parts = Path(data_path).parts
    bin_seg = parts[-1] if len(parts) >= 1 else "bin_0"
    stay_seg = parts[-2] if len(parts) >= 2 else "stay"
    patient_seg = parts[-3] if len(parts) >= 3 else "patient"

    bin_id = bin_seg.removeprefix("bin_")
    stay_id = stay_seg.removeprefix("time_bins_stay_")

    def _clean(s: str) -> str:
        return _ID_SAFE.sub("", s.lower()) or "x"

    return f"p{_clean(patient_seg)}_s{_clean(stay_id)}_b{_clean(bin_id)}"


def _enforce_id_prefix(parsed: dict, prefix: str) -> dict:
    """Belt-and-suspenders: even if the LLM ignored the prefix instruction,
    re-stamp every entity id and every relation source/target to start with
    the bin's prefix. Idempotent — ids that already start with the prefix
    are left alone.

    Globally-stable ids (patient_*, encounter_*, file_*, gender_*, race_*)
    are exempt — they must NOT be bin-prefixed so they can be shared.
    """
    _STABLE_PREFIXES = (
        "patient_", "admission_", "edstay_", "icustay_", "file_", "gender_", "race_",
        "measurement_", "observation_", "condition_", "drug_",
        "procedure_", "allergy_", "specimen_", "unit_", "route_",
        "socialhistory_", "familyhistory_", "immunization_",
        "note_", "signalrecord_", "demographics_", "transfer_",
    )

    def stamp(raw_id: str) -> str:
        rid = (raw_id or "").strip()
        if not rid:
            return rid
        if rid.startswith(prefix + "_"):
            return rid
        if any(rid.startswith(p) for p in _STABLE_PREFIXES):
            return rid
        # Strip any leading underscores from the original id so we don't get '__'
        return f"{prefix}_{rid.lstrip('_')}"

    for e in parsed.get("entities", []) or []:
        e["id"] = stamp(e.get("id", ""))
    for r in parsed.get("relations", []) or []:
        r["source"] = stamp(r.get("source", ""))
        r["target"] = stamp(r.get("target", ""))
    return parsed


_VALUE_FIELDS_TO_STRIP = {"value", "valuenum", "valueuom", "unit"}
_CONCEPT_TYPES_NO_VALUES = {"Measurement", "Observation"}


def _strip_values_from_concept_entities(parsed: dict) -> dict:
    """Remove value/unit fields from Measurement and Observation entity attributes.

    Values belong on the relation edge (with timestamp), not on the concept node.
    This is a backstop in case the LLM still includes them despite the prompt rule.
    """
    for e in parsed.get("entities", []) or []:
        if e.get("type") in _CONCEPT_TYPES_NO_VALUES and isinstance(e.get("attributes"), dict):
            for f in _VALUE_FIELDS_TO_STRIP:
                e["attributes"].pop(f, None)
    return parsed


def _assign_natural_key_ids(parsed: dict) -> dict:
    """Rewrite entity ids to globally-stable natural keys where one exists.

    Two bins describing the same logical Patient (subject_id 10000032) used
    to produce different bin-prefixed ids — the merge agent then either
    duplicated the row or "explored first" inconsistently. Natural-key ids
    sidestep that entirely: the same logical entity always gets the same id,
    so SQLite's INSERT ... ON CONFLICT(id) DO UPDATE collapses cross-bin
    duplicates without any LLM cooperation.

    Identity rules (Patient.subject_id, HospitalAdmission.hadm_id,
    EDStay.stay_id, ICUStay.icustay_id, File.source_path)
    live in `src/queries/identity.derive_natural_key_id` so the same logic
    runs both here (driver pre-pass) and inside the `upsert_entity` LLM tool.
    """
    id_map: dict[str, str] = {}
    for e in parsed.get("entities", []) or []:
        old_id = e.get("id", "")
        if not old_id:
            continue
        attrs = e.get("attributes") or {}
        if not isinstance(attrs, dict):
            attrs = {}
        new_id = derive_natural_key_id(e.get("type"), attrs)
        if new_id and new_id != old_id:
            e["id"] = new_id
            id_map[old_id] = new_id

    if id_map:
        for r in parsed.get("relations", []) or []:
            src = r.get("source")
            tgt = r.get("target")
            if src in id_map:
                r["source"] = id_map[src]
            if tgt in id_map:
                r["target"] = id_map[tgt]

    return parsed


def _guarded_incremental_sql(sql: str) -> dict:
    """Allow incremental SQL, but block destructive operations on core KG tables."""
    normalized = " ".join(sql.strip().lower().split())
    core_table_destructive_prefixes = (
        "delete from entities_for_neo4j",
        "delete from relations_for_neo4j",
        "update entities_for_neo4j set id",
    )
    blocked = {
        "truncate table entities_for_neo4j",
        "truncate table relations_for_neo4j",
        "drop table entities_for_neo4j",
        "drop table relations_for_neo4j",
    }
    if normalized in blocked or any(normalized.startswith(prefix) for prefix in core_table_destructive_prefixes):
        return {
            "ok": False,
            "error": "Blocked in iterative mode: destructive operations on core KG tables are not allowed.",
        }
    return execute_sql(sql)


def _run_incremental_llm_iteration(client, new_data: dict, bin_name: str, data_path: str) -> str:
    """Run OMOP-guided normalization + incremental merge for one bin.

    `data_path` is the absolute bin folder path. Its segments encode the
    natural-key spine — patient subject_id and stay_id — which we surface
    to the merge agent so it can stitch orphan top-level entities even
    when this bin's CSV didn't carry an explicit Patient/Encounter row.
    """
    try:
        schema = get_schema_summary()
    except Exception:
        schema = "Schema summary unavailable (likely missing tables)."
    new_entities_csv = _format_as_csv(new_data.get("entities", []), ["id", "type", "name", "attributes"])
    new_relations_csv = _format_as_csv(new_data.get("relations", []), ["source", "target", "type", "attributes"])

    type_enum = allowed_types_enum()

    # Parse the bin path for the natural-key spine. Mirrors the segment
    # logic in `_bin_id_prefix` so the same values land in the merge prompt
    # as concrete strings the agent can use verbatim.
    _path_parts = Path(data_path).parts
    _patient_seg = _path_parts[-3] if len(_path_parts) >= 3 else ""
    _stay_seg = _path_parts[-2] if len(_path_parts) >= 2 else ""
    path_subject_id = _ID_SAFE.sub("", _patient_seg.lower()) or "unknown"
    path_stay_id = _ID_SAFE.sub("", _stay_seg.removeprefix("time_bins_stay_").lower()) or "unknown"
    path_patient_id = f"patient_{path_subject_id}"
    path_edstay_id = f"edstay_{path_stay_id}"

    prompt = f"""Incrementally merge this bin into the medical KG with OMOP normalization.

Rules:
- Entity ids for Patient, HospitalAdmission, EDStay, ICUStay, and File are globally stable across bins (e.g. `patient_10000032`, `admission_25742920`, `edstay_35968195`, `file_<hash>`). Before upserting any of these, follow the "Cross-bin dedup protocol" below — if the entity already exists in the DB, REUSE its id; do NOT insert a new row with a bin-prefixed id. Other entity ids are bin-local and inherently unique.
- A SQL trigger `guard_entity_rename` will REFUSE any UPDATE that changes an existing entity's `name` to a substantively different value. If you see this error from `execute_sql`, it means the id you tried to upsert already belongs to a different entity. Pick a different id, do not retry the same statement.
- Explore first with SELECT, then apply minimal SQL updates.
- Create `entities_for_neo4j`/`relations_for_neo4j` tables if missing.
- **Use the `upsert_entity` tool for ALL entity writes — do NOT construct `INSERT INTO entities_for_neo4j ...` statements via `execute_sql`.** The tool owns identity (rewrites bin-prefixed ids to natural-key ids when possible) and attribute merging (non-empty existing values are preserved when the new payload is empty/null; lists like `source_files` are unioned across bins). Its returned `final_id` may differ from the id you passed in — use that `final_id` when constructing relations to that entity.
- Use `execute_sql` for: SELECT exploration, relation INSERTs (`INSERT OR IGNORE INTO relations_for_neo4j (source, target, type, attributes) VALUES (...)`), and DDL.
- Reuse existing entities/relations; avoid duplicates and synonym-type variants.
- **Timestamps AND recorded values belong on EDGES, not on entity nodes.** Concept nodes (Condition, Drug, Measurement, Procedure, Observation, etc.) MUST be free of timestamps, value, valuenum, and valueuom so they can be shared across patients. Structural container nodes (Note, SignalRecord, Demographics, Allergy, EDStay, ICUStay) may keep a `timestamp` in their attributes as a label.
- `name` must be human-readable text (never JSON), note-like entities must have non-empty names.
- Keep file-source metadata in attributes, remove iteration-creation metadata.
- Keep entity ids EXACTLY as provided in the input CSV when calling `upsert_entity` — the tool will rewrite them as needed.
- **Use ONLY the relation types from the vocabulary below (CLOSED SET).** If no type fits exactly, pick the nearest and add `"relation_note":"closest match: ..."` to the edge attributes. Do NOT invent new names or caps/punctuation variants.
- Batch multiple relation INSERT statements into a single `execute_sql` call using "INSERT ... ; INSERT ...;" to minimize tool calls.

OMOP-guided edge selection (MANDATORY when both nodes are OMOP-mapped):
- Before inserting ANY edge where BOTH the source and target entity already have
  an `omop_concept_id` stored in their attributes, call `lookup_omop_relation`
  with those two concept IDs.
- If `lookup_omop_relation` returns ≥1 result:
    • Use the returned `relationship_id` as the edge `type` in
      `relations_for_neo4j`. This takes precedence over the closed relation
      vocabulary below.
    • Store `{{"omop_relation": true, "relationship_name": "<name>"}}` in the
      edge's attributes alongside any timestamp.
    • Prefer `is_hierarchical=1` relations ("Is a", "Subsumes") for taxonomy
      edges; prefer clinical pharmacology relations ("May treat", "May prevent",
      "Has MoA") for drug↔condition edges.
- If `lookup_omop_relation` returns 0 results AND both nodes are mapped:
    • Reconsider whether this edge is needed at all. Do NOT add the edge just
      to satisfy structural rules — only insert it if it genuinely carries
      meaning not already captured by another edge in the graph.
    • If you still judge the edge is needed, fall back to the nearest
      closed-vocabulary type below and add
      `{{"omop_relation": false, "relation_note": "no OMOP relation found"}}`.
    • Do NOT invent extra edges or intermediate nodes to force OMOP coverage.
- If only one of the two endpoints is OMOP-mapped (or neither is), skip the
  lookup and use the closed relation vocabulary directly.

Relation vocabulary (CLOSED SET — used when no OMOP relation applies):
  Spine:         has_admission (Patient → HospitalAdmission),
                 has_ed_stay (HospitalAdmission → EDStay, or Patient → EDStay for ED-only),
                 has_icu_stay (HospitalAdmission → ICUStay),
                 includes_transfer, has_encounter_context
  Baseline:      has_allergy, allergen_is, has_condition_in_pmh,
                 has_demographics (Patient → Race/Gender),
                 has_baseline_result (Patient → Measurement/Procedure from history),
                 has_chronic_medication (Patient → Drug, e.g. HAART),
                 has_social_history (Patient → SocialHistory only)
  Notes:         has_note, has_note_section, documents_diagnosis,
                 documents_finding, documents_medication (Note → Drug),
                 documents_allergy, documents_pmh
  Vitals:        recorded_vital (EDStay/ICUStay → Measurement)
  Diagnoses:     documents_diagnosis (EDStay/ICUStay/HospitalAdmission → Condition OR Note → Condition)
  Medications:   reconciled_med, prescribed_med, dispensed_med, administered_med
  Lab:           generated_specimen, has_lab_result, ordered_lab
  Microbiology:  grew_organism, antibiotic_sensitivity
  Procedures:    performed_procedure
  Signals:       has_ecg, has_lead, has_source_file, interpretation_finding
  Services:      assigned_service, has_drg
  Route:         administered_via (Drug → Route)

**`mentions`, `part_of`, `has_chief_complaint`, `has_encounter`, `documents_procedure`, and `documents_home_med` are NOT in the vocabulary — never use them.** Common substitutes:
- SignalRecord → File: use `has_source_file`
- Patient → Race/Gender: use `has_demographics` (NOT has_social_history or documents_finding)
- Patient → baseline Measurement/Procedure: use `has_baseline_result`
- Patient → chronic Drug (e.g. home medications): use `has_chronic_medication`
- EDStay/ICUStay → Observation (clinical findings, arrival mode, etc.): use `documents_finding`
- EDStay/ICUStay → Condition (chief complaint OR confirmed diagnosis): use `documents_diagnosis`
- EDStay/ICUStay → Demographics container: use `has_encounter_context`
- Note → Drug (any drug mentioned): use `documents_medication`
- Any other case: pick the nearest vocabulary type and add `relation_note`.

Cross-bin dedup protocol (Patient / HospitalAdmission / EDStay / ICUStay):
Patients and stays live across bins — the same real-world patient appears in every one of their bins, and a hospital admission appears in every bin within the same stay. Before you upsert a Patient, HospitalAdmission, EDStay, or ICUStay from this bin's CSV, SELECT against `entities_for_neo4j` to find an existing row that describes the same real-world entity. If one exists, REUSE its `id` — do NOT call `upsert_entity` to create a duplicate.

How to look up:
- Patient — the subject_id is in this bin's file paths (e.g. `.../patient-time-bins/10000032/...`) and usually in the source CSVs. Find it, then:
  ```
  SELECT id FROM entities_for_neo4j
  WHERE type='Patient'
    AND (json_extract(attributes,'$.subject_id')='<subject_id>'
         OR name LIKE '%<subject_id>%');
  ```
- HospitalAdmission — look for hadm_id:
  ```
  SELECT id FROM entities_for_neo4j
  WHERE type='HospitalAdmission'
    AND json_extract(attributes,'$.hadm_id')='<hadm_id>';
  ```
- EDStay — look for stay_id:
  ```
  SELECT id FROM entities_for_neo4j
  WHERE type='EDStay'
    AND json_extract(attributes,'$.stay_id')='<stay_id>';
  ```
- ICUStay — look for icustay_id:
  ```
  SELECT id FROM entities_for_neo4j
  WHERE type='ICUStay'
    AND json_extract(attributes,'$.icustay_id')='<icustay_id>';
  ```

If the SELECT returns nothing, THEN upsert. When you do, MAKE SURE attributes carry:
- Patient: `subject_id`
- HospitalAdmission: `hadm_id`
- EDStay: `stay_id` AND `subject_id`
- ICUStay: `icustay_id` AND `subject_id`
The upsert tool uses these to give the row a stable id (`admission_<hadm_id>`, `edstay_<stay_id>`, etc.) — which is what the NEXT bin's SELECT will find.

Path-derived spine for THIS bin:
This bin's content belongs to **{path_patient_id}** and ED stay **{path_edstay_id}**.
If any CSV in this bin contains a `hadm_id` column, create a `HospitalAdmission` node with id `admission_<hadm_id>` and link:
  `(admission_<hadm_id>)-[has_ed_stay]->({path_edstay_id})`
  `({path_patient_id})-[has_admission]->(admission_<hadm_id>)`
If NO `hadm_id` is found anywhere in this bin (ED-only visit), link the EDStay directly to the patient:
  `({path_patient_id})-[has_ed_stay]->({path_edstay_id})`
Clinical data (vitals, notes, ECG, diagnoses) MUST attach to the most specific stay node — `EDStay` for ED data, `ICUStay` for ICU data, `HospitalAdmission` for admission-level data (discharge notes, inpatient diagnoses).
After all your upserts, every top-level container (Note, SignalRecord, triage Observation, lab panel, Demographics, Allergy event) in this bin's parsed CSV MUST have at least one incoming or outgoing edge that puts it on a path to **{path_patient_id}** — either directly, or via **{path_edstay_id}** / `admission_<hadm_id>`. If you see a top-level container with no upward edge after extraction, add the missing edge via `execute_sql` against `relations_for_neo4j` (use the `INSERT ... ON CONFLICT(source,target,type) DO UPDATE` pattern). Typical missing edges look like:
  ```
  INSERT OR IGNORE INTO relations_for_neo4j (source, target, type, attributes)
  VALUES ('{path_edstay_id}', '<bin_prefix>_note_hpi', 'has_note', '{{"timestamp": "<note_timestamp_if_known>"}}');
  ```
Do NOT add stitching edges for leaf concepts (Condition, Drug, Measurement) — those already have specific edges from their container. Only fix orphan containers.

Entity type whitelist (closed set — every entity's `type` MUST be one of these, exactly):
  {type_enum}

Picking the right type:
- For structural entities (Patient, HospitalAdmission, EDStay, ICUStay, Transfer, File, Note, SignalRecord, SignalLead, Demographics, Allergy, Immunization, FamilyHistory, SocialHistory): assign the type directly from the whitelist; do NOT call any lookup tool for these.
- **For structural container nodes (Allergy, SocialHistory, FamilyHistory, Immunization, Note, SignalRecord, Demographics, Transfer), always include `subject_id` (the patient's subject id — available from the bin path as `{path_subject_id}`) and `source_term` (exact wording from the source CSV) in the entity's attributes.** The `upsert_entity` tool uses these to derive a stable patient-scoped id (e.g. `note_{path_subject_id}_historyofpresentillness`) so the same content from different bins collapses to one node automatically. Without `subject_id` + `source_term` in attributes, the node gets a bin-prefixed id and will duplicate.
- For every other entity where `source_term` is non-empty: lookup is MANDATORY — see "OMOP mapping requirements" below. Use the returned `canonical_type` as the entity's type.
- If an entity has no `source_term` and is not structural (e.g. a pure container like "ED triage" with an empty name), assign the most specific whitelist type you can from context; no lookup needed.
- If a clinical entity cannot be confidently mapped after lookup, set type = "Unknown" and mapping_status = "needs_review".
- **All OMOP-mapped entities are globally stable (shared across ALL bins and patients).** After OMOP lookup, derive the entity id as `<type_lowercase>_<omop_concept_id>` (e.g. `measurement_3027018` for Heart rate, `gender_8532` for Female, `race_8516` for Black or African American). Do NOT bin-prefix these ids. The `upsert_entity` tool does this automatically when `omop_concept_id` is present in attributes — you only need to ensure it is set. Before upserting, SELECT to check if the row already exists; if so, reuse its id rather than creating a duplicate.

OMOP mapping requirements:
- **For every entity where `source_term` is non-empty, call `lookup_omop_concepts` unconditionally** — including routine measurements (Heart rate, Blood pressure, Temperature, O2 saturation, Respiratory rate, Pain score), common conditions, drugs, and procedures. Do NOT skip this step because the term seems obvious. A score-1.0 exact hit costs one tool call and locks in the omop_concept_id, which is required for cross-bin deduplication.
- Pass:
  - term = `source_term` from attributes (preferred) or entity name
  - domain_hint = best guess from current type/context (e.g. "Drug", "Condition", "Measurement"). The hint is used as a soft tiebreaker — passing it improves precision.
- If the term is an abbreviation or short form (≤ 6 characters, or well-known clinical acronym such as COPD, PTSD, HIV, HAART, HCV, PFT, FVC, FEV1, AF, JVD, RRR, CTAB, INR, o2sat, sbp, dbp), FIRST expand it to the full medical term and call `lookup_omop_concepts` with that expanded form (e.g. "COPD" → "Chronic obstructive pulmonary disease", "HIV" → "Human immunodeficiency virus infection", "HAART" → "Highly active antiretroviral therapy", "sbp" → "Systolic blood pressure", "dbp" → "Diastolic blood pressure", "o2sat" → "Oxygen saturation"). If the expanded form scores ≥ 0.8, use it. Only fall back to the abbreviated form if the expanded form scores < 0.8.
- If multiple candidates are returned, use the clinical context of the patient (other diagnoses, medications, specialty) to choose the best match. Do not blindly pick the first result.
- Do not invent OMOP IDs. Use only returned candidates.
- If mapped (best score ≥ 0.8), store in attributes: omop_concept_id, omop_concept_name, omop_domain_id, omop_vocabulary_id, mapping_confidence, mapping_status="mapped", and source_term. Canonicalize entity name to omop_concept_name. Set the entity's `type` to the chosen candidate's `canonical_type`. Do NOT add a timestamp, value, valuenum, or valueuom to the entity attributes — these go on the relation edge only.
- If OMOP misses (best score < 0.8): do NOT immediately mark unmapped. Instead follow the fallback chain in "General-concept registry" below.
- Merge/dedup by omop_concept_id where applicable to avoid synonym duplicates.

General-concept registry (fallback for any entity OMOP misses):
- This registry covers two categories: (a) non-clinical note values (hobbies, foods, devices, brands) that OMOP never covers, and (b) clinical entities where OMOP returns no confident match — negated physical exam findings (no JVD, no rales, CTAB, RRR), triage-specific values (acuity), or unusual observations.
- **Fallback chain for any entity with source_term where OMOP scored < 0.8:**
  1. Call `lookup_general_concepts` (term = source_term).
  2. If best score ≥ 0.8: use that canonical name and mark mapping_status="mapped_general".
  3. If both OMOP and general registry missed (both < 0.8): call `upsert_general_concept` to coin a new canonical. The `canonical_name` MUST be a broader category than the source term — not the source term echoed back. Examples:
     - source_term="no JVD" → canonical_name="Absent jugular venous distension"
     - source_term="CTAB" → canonical_name="Clear to auscultation bilaterally"
     - source_term="RRR" → canonical_name="Regular rate and rhythm"
     - source_term="AAOx3" → canonical_name="Alert and oriented to person place and time"
     - source_term="NABS" → canonical_name="Normoactive bowel sounds"
     - source_term="acuity" → canonical_name="ED triage acuity score"
     - source_term="PlayStation" → canonical_name="Gaming console"
  4. After `upsert_general_concept` returns, use the returned `canonical_name` as the entity's name and set mapping_status="mapped_general".
- After any general-concept match or coin, set the entity `type` from the whitelist based on context (Observation for physical exam findings and triage values, Measurement for numeric scores). Do NOT set type="Unknown" just because OMOP missed — use the general registry fallback first.
- Do NOT call `upsert_general_concept` for structural entities (Patient, Encounter, File, Note, ...).

Schema:
{schema}

New entities CSV:
{new_entities_csv}

New relations CSV:
{new_relations_csv}

Return a short summary of changes including mapped/unmapped counts.

Reference DDL (SQLite, if needed):
CREATE TABLE IF NOT EXISTS entities_for_neo4j (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  name TEXT NOT NULL,
  attributes TEXT
);

CREATE TABLE IF NOT EXISTS relations_for_neo4j (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  target TEXT NOT NULL,
  type TEXT NOT NULL,   -- either a closed-vocab type OR an OMOP relationship_id (e.g. "May treat", "Is a")
  attributes TEXT
);

-- Unique index: same (source, target, type, timestamp) is one edge.
-- Different timestamps produce separate edges (e.g. two diagnoses on different dates).
CREATE UNIQUE INDEX IF NOT EXISTS idx_rfn_unique
ON relations_for_neo4j (
    source, target, type,
    COALESCE(json_extract(attributes, '$.timestamp'), '')
);

CREATE INDEX IF NOT EXISTS idx_efn_type ON entities_for_neo4j(type);
CREATE INDEX IF NOT EXISTS idx_rfn_type ON relations_for_neo4j(type);
CREATE INDEX IF NOT EXISTS idx_rfn_source ON relations_for_neo4j(source);
CREATE INDEX IF NOT EXISTS idx_rfn_target ON relations_for_neo4j(target);
"""

    return run_llm_db_session(
        client=client,
        user_prompt=prompt,
        system_prompt="You are a SQL agent for iterative KG maintenance with strict OMOP mapping. Explore first, then update. Minimize tool calls by batching SQL statements.",
        sql_executor=_guarded_incremental_sql,
        max_steps=50,
    )


def process_bin_llm_driven(data_path: str):
    """Process a single bin folder with the LLM doing all the heavy lifting:
    entity/relation extraction and incremental merge without timeline graph construction."""
    bin_name = os.path.basename(data_path)

    # Empty bins (0 input files) used to invoke the LLM anyway, which then
    # hallucinated placeholder entities like "id"/"type"/"source"/"target"
    # from CSV header artifacts. Skip them up front.
    has_files = any(
        not name.startswith(".")
        for _root, _dirs, files in os.walk(data_path)
        for name in files
    )
    if not has_files:
        print(f"=== Skipping '{bin_name}' (no input files) ===", flush=True)
        return

    client = get_client()
    init_db()

    id_prefix = _bin_id_prefix(data_path)
    type_enum = allowed_types_enum()
    prompt = f"""Extract KG updates from this bin.

Your entire response MUST follow this exact layout — nothing before the first
marker, nothing after the last:

===ENTITIES===
id,type,name,attributes
<one row per entity>
===RELATIONS===
source,target,type,attributes
<one row per relation>
===END===

Rules for the output format:
- The three marker lines (===ENTITIES===, ===RELATIONS===, ===END===) appear
  verbatim on their own lines, with no surrounding text.
- The CSV header line immediately follows each marker — no blank line between.
- No markdown, no prose, no explanation outside the markers.
- attributes must be valid JSON with double-quoted keys, or {{}} if empty.

## The one rule that matters

Every concept that a medical or general vocabulary could plausibly look up
gets its own entity row. Do NOT pack multiple concepts into one row's
attributes — concepts buried inside a `text` blob or a nested attribute are
invisible to the normalization pass that runs next, and the whole pipeline
silently degrades.

Containers (notes, triage events, lab panels, signal recordings) become
thin SHELLS whose attributes hold only provenance (timestamp, source_file,
short label). Every concept they contain — a diagnosis, a drug, a vital
value, a chief complaint, a hobby — is its own row, linked back to the
shell using the CLOSED SET relation types above (e.g. `documents_finding`,
`recorded_vital`, `has_note_section`, `has_lead`). Do NOT use `mentions`
or `part_of` — those are not in the vocabulary. Use your judgement on what
counts as a separate concept; when in doubt, split.

## The spine: every bin needs a Patient and an Encounter

Every bin you produce output for MUST include a Patient entity AND an
Encounter entity. They are the trunk every other entity in this bin hangs
off — without them, this bin's content floats unreachable from the rest
of the KG. The `subject_id` and `stay_id` are ALWAYS in the bin's file
path (e.g. `.../patient-time-bins/<subject_id>/time_bins_stay_<stay_id>/bin_N`)
and are usually in the source CSVs too. Put `subject_id` in the Patient's
attributes, and `stay_id` (plus `hadm_id` if any CSV gives you one) in
the Encounter's. These keys are what makes the merge pass collapse
cross-bin duplicates onto a single canonical Patient/Encounter row.

**Gender and Race entities are globally stable across ALL patients.** Use
the id pattern `gender_<value>` / `race_<value>` (e.g. `gender_female`,
`race_white`, `race_black_african_american`) — NOT a bin-prefixed id.
This allows the same Female node to be shared by every female patient.

Every top-level container (Note, SignalRecord, Transfer, Demographics, …)
MUST have at least one edge pointing UP toward the spine. Containers
without an upward edge become orphan clouds.

## Entity-type whitelist (use ONE of these EXACT strings for every `type`)

  {type_enum}

Do not invent new types. If nothing fits, use Observation.

## Concept attributes

For every concept entity put the exact source wording in
attributes.source_term — that is what the merge pass looks up.
For Measurement and Observation entities store ONLY concept-identity fields:
source_term, ref_range_lower, ref_range_upper, and unit_of_measure (the
canonical unit for this concept, e.g. "mEq/L" — NOT a specific recorded value).
Do NOT put value, valuenum, or valueuom on the entity — those belong
exclusively on the relation edge that connects the encounter/note to this
Measurement. Example: {{"source_term":"K","ref_range_lower":3.5,"ref_range_upper":5.0,"unit_of_measure":"mEq/L"}}.

## Relation vocabulary (CLOSED SET — use ONLY these types)

Use the most specific type that fits the data source. If no type fits
exactly, pick the nearest and add `"relation_note":"closest match: ..."` to
the edge attributes.

Spine:
  has_encounter         Patient → Encounter          {{timestamp: admittime}}
  includes_transfer     Encounter → Transfer         {{timestamp: intime, sequence: 0,1,2,...}}
  has_encounter_context Encounter → Demographics     {{timestamp: intime}}

Baseline (known before arrival; from bin_0, allergies.csv, past_medical_history.csv):
  has_allergy           Patient → Allergy            {{}}
  allergen_is           Allergy → Drug               {{}}
  has_condition_in_pmh  Patient → Condition          {{source: "past_medical_history"}}
  has_condition_in_pmh  Patient → Observation        {{source: "past_medical_history"}}
  has_social_history    Patient → SocialHistory      {{source: "social_history"}}
  has_demographics      Patient → Race               {{source: "patient_demographics"}}
  has_demographics      Patient → Gender             {{source: "patient_demographics"}}
  has_baseline_result   Patient → Measurement        {{source: "social_history"|"outpatient"}}
  has_baseline_result   Patient → Procedure          {{source: "social_history"|"outpatient"}}
  has_chronic_medication Patient → Drug              {{source: "social_history", note: "home/chronic medication"}}

Notes and document structure:
  has_note              Encounter → Note             {{timestamp: charttime}}
  has_note              Patient → Note               {{timestamp: charttime}}
  has_note_section      Note → Note                  {{}}
  documents_diagnosis   Note → Condition             {{timestamp, seq_num, icd_code, icd_version, context: "ed_diagnosis"|"discharge_summary"|"radiology"|"HPI"|"pmh"}}
  documents_diagnosis   Encounter → Condition        {{timestamp, seq_num, icd_code, icd_version, context: "chief_complaint"|"ed_diagnosis"|"hosp_diagnoses"}}
  documents_finding     Note → Observation           {{timestamp, value, source_section}}
  documents_finding     Encounter → Observation      {{timestamp, source_section}}
  documents_medication  Note → Drug                  {{timestamp, gsn, ndc, source_section}}
  documents_allergy     Note → Allergy               {{timestamp}}

Vital signs (from ed_edstays_triage AND ed_vitalsign — use the SAME type for both):
  recorded_vital        Encounter → Measurement      {{timestamp: charttime, value: float, valueuom: str, source: "ed_edstays_triage"|"ed_vitalsign"}}

Medication lifecycle — use the CORRECT type based on the source file:
  reconciled_med        Encounter → Drug             {{timestamp: charttime, gsn, ndc, etc_description}}               ← ed_medrecon only (home meds at arrival)
  prescribed_med        Encounter → Drug             {{timestamp: starttime, stop_time, dose_val_rx, dose_unit_rx, route, frequency, pharmacy_id, poe_id}}  ← hosp_prescriptions only
  dispensed_med         Encounter → Drug             {{timestamp: charttime, status, route, frequency, source: "hosp_pharmacy"|"ed_pyxis"}}                  ← hosp_pharmacy or ed_pyxis
  administered_med      Encounter → Drug             {{timestamp: charttime, event_txt, dose_given, dose_given_unit, route, emar_id}}                        ← hosp_emar only

Laboratory:
  generated_specimen    Encounter → Specimen         {{timestamp: charttime, specimen_id}}
  has_lab_result        Specimen → Measurement       {{timestamp: charttime, storetime, value, valuenum, flag, labevent_id}}
  ordered_lab           Encounter → Measurement      {{timestamp: charttime, priority, specimen_id}}

Microbiology:
  grew_organism         Specimen → Observation       {{timestamp: storedate, test_name, isolate_num}}
  antibiotic_sensitivity Observation → Drug          {{ab_name, interpretation, dilution_text}}

Procedures:
  performed_procedure   Encounter → Procedure        {{timestamp: chartdate, seq_num, icd_code, icd_version}}

Signals (ECG):
  has_ecg               Encounter → SignalRecord     {{timestamp: ecg_time}}
  has_lead              SignalRecord → SignalLead    {{}}
  has_source_file       SignalRecord → File          {{}}
  administered_via      Drug → Route                 {{}}
  interpretation_finding SignalRecord → Observation  {{timestamp, report_slot: "report_0"|..., report_text}}

Hospital services and billing:
  assigned_service      Encounter → Observation      {{timestamp: transfertime, prev_service, curr_service}}
  has_drg               Encounter → Observation      {{drg_type: "APR"|"HCFA", drg_severity, drg_mortality}}

## Worked example

If a bin contains an ed_edstays_triage row for 2180-08-05 20:58 (HR=96,
Temp=98.5°F, O2sat=100%, chief complaint "n/v/d, Abd pain", acuity=3)
and an HPI note saying "Patient with HIV on HAART, given Morphine Sulfate 5mg IV. K 5.3.":

===ENTITIES===
id,type,name,attributes
  {id_prefix}_patient,Patient,Patient,"{{""subject_id"":""<from path>""}}"
  {id_prefix}_encounter,Encounter,Encounter,"{{""stay_id"":""<from path>"",""hadm_id"":""<from CSV if known>"",""admission_type"":""EW EMER.""}}"
  {id_prefix}_note_hpi,Note,HPI,"{{""charttime"":""2180-08-05 20:58:00"",""source_file"":""...""}}"
  {id_prefix}_cond_hiv,Condition,HIV,"{{""source_term"":""HIV""}}"
  {id_prefix}_drug_haart,Drug,HAART,"{{""source_term"":""HAART"",""etc_description"":""Antiretroviral""}}"
  {id_prefix}_drug_morphine,Drug,Morphine Sulfate,"{{""source_term"":""Morphine Sulfate""}}"
  {id_prefix}_meas_k,Measurement,Potassium,"{{""source_term"":""K"",""ref_range_lower"":3.5,""ref_range_upper"":5.0}}"
  {id_prefix}_meas_hr,Measurement,Heart Rate,"{{""source_term"":""heartrate""}}"
  {id_prefix}_meas_temp,Measurement,Body Temperature,"{{""source_term"":""temperature""}}"
  {id_prefix}_obs_cc,Observation,Abdominal pain and nausea,"{{""source_term"":""n/v/d, Abd pain""}}"
===RELATIONS===
source,target,type,attributes
  {id_prefix}_patient,{id_prefix}_encounter,has_encounter,"{{"timestamp"":""2180-08-05 20:58:00""}}"
  {id_prefix}_encounter,{id_prefix}_obs_cc,documents_finding,"{{"timestamp"":""2180-08-05 20:58:00"",""acuity"":3,""arrival_transport"":""AMBULANCE""}}"
  {id_prefix}_encounter,{id_prefix}_meas_hr,recorded_vital,"{{"timestamp"":""2180-08-05 20:58:00"",""value"":96,""valueuom"":""bpm"",""source"":""ed_edstays_triage""}}"
  {id_prefix}_encounter,{id_prefix}_meas_temp,recorded_vital,"{{"timestamp"":""2180-08-05 20:58:00"",""value"":98.5,""valueuom"":""°F"",""source"":""ed_edstays_triage""}}"
  {id_prefix}_encounter,{id_prefix}_note_hpi,has_note,"{{"timestamp"":""2180-08-05 20:58:00""}}"
  {id_prefix}_note_hpi,{id_prefix}_cond_hiv,documents_diagnosis,"{{"timestamp"":""2180-08-05 20:58:00"",""context"":""HPI""}}"
  {id_prefix}_note_hpi,{id_prefix}_drug_haart,documents_medication,"{{"timestamp"":""2180-08-05 20:58:00"",""source_section"":""HPI""}}"
  {id_prefix}_note_hpi,{id_prefix}_drug_morphine,administered_med,"{{"timestamp"":""2180-08-05 20:58:00"",""dose_given"":""5"",""dose_given_unit"":""mg"",""route"":""IV""}}"
  {id_prefix}_note_hpi,{id_prefix}_meas_k,has_lab_result,"{{"timestamp"":""2180-08-05 20:58:00"",""value"":""5.3"",""valuenum"":5.3,""valueuom"":""mEq/L""}}"
===END===

Key rules demonstrated:
- `{id_prefix}_cond_hiv` has NO timestamp — all patients with HIV share one node.
- Triage vitals are individual `recorded_vital` edges (same type as serial ed_vitalsign rows).
- HAART from HPI → `documents_medication` (drug mentioned in clinical note).
- Morphine given IV → `administered_med` (event in clinical note).
- Chief complaint observation → `documents_finding` edge from Encounter (not has_chief_complaint).

## Mechanical rules

- `name` is human-readable text; never JSON. Put metadata in `attributes`
  (valid JSON, double-quoted).
- **Timestamps AND recorded values belong on EDGES, not on concept nodes.**
  Concept entities (Condition, Drug, Measurement, Procedure, Observation, etc.)
  MUST have NO timestamp, value, valuenum, or valueuom in their attributes.
  The edge that connects the encounter/note to a Measurement carries the
  value+timestamp pair — a single Creatinine node accumulates multiple edges,
  one per timepoint, each with its own value and timestamp.
  Structural container nodes (Note, SignalRecord, Demographics) may keep a
  `timestamp` in their own attributes as a label.
- **Use ONLY the relation types from the vocabulary above.** Do not invent
  new relation names. If none fits, pick the nearest and add
  `"relation_note":"closest match: ..."` to the edge attributes.
- Every `id` MUST start with `{id_prefix}_`, and every relation's `source`
  and `target` must too — the prefix guarantees cross-bin uniqueness.
  Keep the prefix's underscores intact.
- Do not create timeline structures (TimeNote, TimeBin, NEXT, OCCURS_IN).
- For an unreadable file, emit a File entity with whatever metadata you
  can extract: {{"filename":"...","reason_unreadable":"..."}}.
"""

    print(f"=== Processing '{bin_name}' (LLM-driven, id_prefix={id_prefix}) ===", flush=True)
    response_text = send_folder_to_model(client, data_path, prompt)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    debug_path = os.path.join(OUTPUT_DIR, f"debug_raw_response_{id_prefix}.txt")
    with open(debug_path, "w") as f:
        f.write(response_text)
    print(f"Raw response written to {debug_path}")

    result = _parse_csv_response(response_text)
    # Strip value/unit from Measurement and Observation nodes — values belong on edges.
    result = _strip_values_from_concept_entities(result)
    # Re-stamp ids to enforce the prefix even if the LLM ignored the rule.
    result = _enforce_id_prefix(result, id_prefix)
    # Rewrite Patient/Encounter/File ids to natural keys so cross-bin
    # duplicates collapse on the SQLite upsert path.
    result = _assign_natural_key_ids(result)
    print(f"LLM returned: {len(result['entities'])} entities, {len(result['relations'])} relations")

    # Let the LLM explore DB state and apply SQL updates autonomously.
    llm_report = _run_incremental_llm_iteration(client, result, bin_name, data_path)
    print(f"KG iteration input: {len(result.get('entities', []))} entities, {len(result.get('relations', []))} relations")
    print("LLM DB iteration report:")
    print(llm_report)

    print(f"KG saved in SQLite.")
    print(get_schema_summary())


def _bin_sort_key(name: str):
    suffix = name.split("_", 1)[1] if "_" in name else ""
    return (0, int(suffix)) if suffix.isdigit() else (1, suffix)


def sync_kg_to_neo4j():
    """Sync the current SQLite KG to Neo4j. Call once after all bins are processed."""
    print(f"\n=== Syncing KG to Neo4j ===")
    sync_to_neo4j()
    print(f"\n=== Done! KG is stored in SQLite and Neo4j. ===")
    print(get_schema_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a KG from a time-bins stay folder.",
    )
    parser.add_argument(
        "stay_dir",
        help="Path to a time_bins_stay_* folder (e.g. data/testData/Patients/time_bins_stay_31293660)",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N bins (sorted). Omit to process all bins.",
    )
    args = parser.parse_args()

    stay_dir = os.path.abspath(args.stay_dir)
    if not os.path.isdir(stay_dir):
        raise SystemExit(f"Stay directory not found: {stay_dir}")

    bins = [
        d for d in os.listdir(stay_dir)
        if os.path.isdir(os.path.join(stay_dir, d)) and d.startswith("bin_")
    ]
    bins.sort(key=_bin_sort_key)

    if args.max_bins is not None:
        bins = bins[:args.max_bins]

    bin_list = [os.path.join(stay_dir, b) for b in bins]

    print(f"Processing {len(bin_list)} bin(s) from {stay_dir}:")
    for p in bin_list:
        print(f"  {p}")

    init_db()

    try:
        from src.queries.embeddings import get_model
        get_model()
        print("[warmup] Embedding model loaded.")
    except Exception as _warmup_err:
        print(f"[warmup] Embedding model load failed (vector tier will be slow on first use): {_warmup_err}")

    failed_bins = []
    t_start = time.perf_counter()

    for data_path in bin_list:
        try:
            process_bin_llm_driven(data_path)
        except Exception as exc:
            print(f"=== FAILED '{os.path.basename(data_path)}': {type(exc).__name__}: {exc} ===", flush=True)
            traceback.print_exc()
            failed_bins.append(data_path)

    elapsed = time.perf_counter() - t_start
    minutes, seconds = divmod(elapsed, 60)

    if failed_bins:
        print(f"\n=== {len(failed_bins)} bin(s) failed ===")
        for p in failed_bins:
            print(f"  FAILED: {p}")
    else:
        print("\n=== All bins processed successfully ===")

    print(f"=== Total time: {elapsed:.1f}s ({int(minutes)}m {seconds:.1f}s) ===")

    sync_kg_to_neo4j()
