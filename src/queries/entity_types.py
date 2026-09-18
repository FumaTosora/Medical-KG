"""Canonical entity-type whitelist for the Neo4j KG.

The KG only ever exposes labels from a fixed, curated set:

  1. **OMOP_DOMAIN_TYPES** — the OMOP `domain_id` values that make sense as
     Neo4j labels. We keep OMOP's names verbatim (Condition, Drug, ...) so
     the KG self-documents against the vocabulary it normalizes against.
     Compound OMOP domains (Condition/Drug, Drug/Measurement, ...) and
     metadata-only domains (Type Concept, Metadata, Relationship, ...) are
     intentionally excluded — they're internal OMOP plumbing, not entity
     types in our graph.

  2. **NON_OMOP_TYPES** — things OMOP does not model: the patient record
     itself, source files, signal recordings, hospital transfers between
     stations.

`canonicalize_type` is the single resolver used in three places:
  - the merge prompt (as an enum the LLM must pick from),
  - inside `lookup_omop_concepts` (to surface the right type to the LLM),
  - at Neo4j sync (as the final backstop).
"""

from __future__ import annotations


# Curated OMOP domains we expose as Neo4j labels (out of the 50 in `domain`).
# Compound domains (Condition/Drug etc.) and metadata domains (Type Concept,
# Metadata, Relationship, Plan Stop Reason, ...) are excluded — they're not
# meaningful entity types in our graph.
OMOP_DOMAIN_TYPES: frozenset[str] = frozenset({
    "Condition",
    "Drug",
    "Procedure",
    "Measurement",
    "Observation",
    "Device",
    "Specimen",
    # OMOP "Visit" is intentionally excluded — it semantically overlaps with
    # our structural HospitalAdmission / EDStay / ICUStay nodes. The alias map
    # below maps "visit" → "HospitalAdmission".
    # OMOP "Person" is intentionally excluded — it's a demographic-registry
    # type. The patient identity in our KG is the non-OMOP "Patient" below;
    # any incoming "Person" gets aliased to "Patient".
    "Provider",
    "Geography",
    "Episode",
    "Route",         # drug administration route
    "Unit",          # measurement units
    "Spec Anatomic Site",
    "Race",
    "Ethnicity",
    "Gender",
})


# Structural / non-clinical types OMOP does not cover. Filled by hand —
# anything outside the OMOP domain set belongs here or it should not exist.
NON_OMOP_TYPES: frozenset[str] = frozenset({
    # The patient record and its structure.
    "Patient",              # the person whose record this is
    "Demographics",         # demographic bundle (race, gender, dob, ...)
    "HospitalAdmission",    # inpatient hospital admission (hadm_id)
    "EDStay",               # emergency department stay (ed stay_id)
    "ICUStay",              # intensive care unit stay (icu stay_id)
    "Transfer",             # movement between stations / wards within a stay
    # Source artefacts.
    "File",             # any source document (PDF, image, scan, attachment)
    "Note",             # clinical free-text note distinct from OMOP Observation
    # Physiological signals (waveforms are not OMOP concepts).
    "SignalRecord",     # an ECG / EEG / waveform recording as a whole
    "SignalLead",       # an individual lead/channel within a SignalRecord
    # History sections that aren't single OMOP concepts.
    "Allergy",          # allergy entry (substance + reaction + severity)
    "Immunization",     # vaccination event
    "FamilyHistory",    # family-history entry
    "SocialHistory",     # social/medical-history entry (smoking, alcohol, occupation)
    # Catch-all for things the pipeline could not classify.
    "Unknown",
})


# The full canonical set. Neo4j labels must come from here.
ALLOWED_TYPES: frozenset[str] = OMOP_DOMAIN_TYPES | NON_OMOP_TYPES


# Aliases the LLM commonly emits, normalized to canonical types.
# Keys are matched case- and punctuation-insensitively (see `_normalize_key`).
# Only entries that map to a *different* canonical name belong here — a type
# already canonical (e.g. "Drug" -> "Drug") needs no alias.
_ALIASES: dict[str, str] = {
    # Conditions -> OMOP "Condition"
    "diagnosis": "Condition",
    "disease": "Condition",
    "disorder": "Condition",
    "finding": "Condition",
    "clinicalfinding": "Condition",
    "symptom": "Condition",
    "problem": "Condition",
    # Drugs -> OMOP "Drug"
    "medication": "Drug",
    "med": "Drug",
    "pharmaceutical": "Drug",
    "prescription": "Drug",
    "rx": "Drug",
    "substance": "Drug",
    "substanceuse": "Drug",
    # Procedures -> OMOP "Procedure"
    "operation": "Procedure",
    "surgery": "Procedure",
    "intervention": "Procedure",
    # Measurements -> OMOP "Measurement"
    "labmeasurement": "Measurement",
    "lab": "Measurement",
    "labresult": "Measurement",
    "labvalue": "Measurement",
    "vital": "Measurement",
    "vitalsign": "Measurement",
    "assessment": "Measurement",
    # Observations -> OMOP "Observation"
    "observationnote": "Observation",
    # People — OMOP's "Person" is a demographic-registry type, not the patient
    # identity our KG cares about; we always fold it into "Patient".
    "person": "Patient",
    "subject": "Patient",
    "doctor": "Provider",
    "physician": "Provider",
    "clinician": "Provider",
    "staff": "Provider",
    # Demographics
    "demographic": "Demographics",
    "sex": "Demographics",
    # Files
    "document": "File",
    "attachment": "File",
    "image": "File",
    "scan": "File",
    "pdf": "File",
    # Notes
    "clinicalnote": "Note",
    "report": "Note",
    # Signals
    "signal": "SignalRecord",
    "ecg": "SignalRecord",
    "eeg": "SignalRecord",
    "waveform": "SignalRecord",
    "lead": "SignalLead",
    "channel": "SignalLead",
    # Encounters / transfers
    "admission": "HospitalAdmission",
    "hospitaladmission": "HospitalAdmission",
    "hospitalstay": "HospitalAdmission",
    "inpatientadmission": "HospitalAdmission",
    # OMOP's "Visit" domain maps to HospitalAdmission (the closest structural equivalent).
    "visit": "HospitalAdmission",
    "encounter": "HospitalAdmission",
    "stay": "EDStay",
    "edstay": "EDStay",
    "edvisit": "EDStay",
    "emergencystay": "EDStay",
    "icustay": "ICUStay",
    "intensivecarestay": "ICUStay",
    "icu": "ICUStay",
    "transport": "Transfer",
    # Histories
    "vaccine": "Immunization",
    "vaccination": "Immunization",
    "socialhistory": "SocialHistory",
    "medicalhistory": "SocialHistory",
    "pastmedicalhistory": "SocialHistory",
    "familyhistory": "FamilyHistory",
    "allergy": "Allergy",
    "allergyentry": "Allergy",
    # Devices / specimens
    "implant": "Device",
    "sample": "Specimen",
    # Geography
    "location": "Geography",
    "place": "Geography",
    "address": "Geography",
}


def _normalize_key(value: str) -> str:
    """Lowercase + strip non-alphanumerics so 'Lab Result', 'lab_result', and
    'LabResult' all collapse to the same alias key."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def canonicalize_type(raw_type: str | None, omop_domain: str | None = None) -> str:
    """Resolve an LLM-emitted type string to a canonical KG type.

    Resolution order:
      1. OMOP domain (ground truth when the entity is OMOP-mapped) — used as-is
         if it's in OMOP_DOMAIN_TYPES.
      2. Exact match against ALLOWED_TYPES.
      3. Alias map (case- and punctuation-insensitive).
      4. Case-insensitive direct hit against ALLOWED_TYPES.
      5. Fallback: 'Unknown'.
    """
    if omop_domain:
        domain = omop_domain.strip()
        if domain in OMOP_DOMAIN_TYPES:
            return domain

    if not raw_type:
        return "Unknown"

    stripped = raw_type.strip()
    if stripped in ALLOWED_TYPES:
        return stripped

    key = _normalize_key(stripped)
    if not key:
        return "Unknown"

    alias = _ALIASES.get(key)
    if alias:
        return alias

    for candidate in ALLOWED_TYPES:
        if _normalize_key(candidate) == key:
            return candidate

    return "Unknown"


def allowed_types_enum() -> str:
    """Stable, sorted, comma-separated string of canonical types — for prompts."""
    return ", ".join(sorted(ALLOWED_TYPES))
