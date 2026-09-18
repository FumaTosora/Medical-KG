"""Natural-key entity-id derivation.

The KG agent gets the same logical Patient or Encounter described in multiple
bins (different `id_prefix`) but with the same identifying attributes
(`subject_id`, `hadm_id`, `stay_id`). Without a stable id, the staging tables
end up with one row per bin instead of one row per logical entity. This
module derives a deterministic id from those attributes so cross-bin
duplicates collapse on the SQLite upsert path.

The derivation is shared between two callers:
  - the driver's pre-pass `_assign_natural_key_ids` (rewrites ids before the
    merge agent ever sees them), and
  - the `upsert_entity` LLM tool (re-derives at write time as a safety net).

`_canonical_etype` strips an LLM-emitted `type_` prefix and resolves the
remaining string through the project-wide type whitelist. Without this,
LLM variants like "type_Patient" silently bypass the natural-key rewrite.
"""

from __future__ import annotations

import hashlib
import re

from src.queries.entity_types import canonicalize_type


_ID_SAFE = re.compile(r"[^a-z0-9]+")


def _safe_segment(value: str) -> str:
    """Lowercase + strip non-alphanumerics so '10000032' and ' 10000032 '
    collapse to the same id segment."""
    return _ID_SAFE.sub("", (value or "").lower()) or "x"


def _coerce(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def canonical_etype(raw: str | None) -> str:
    """Resolve an LLM-emitted type string to its canonical whitelist form.

    Strips a leading "type_" if present (the LLM occasionally adds this
    despite the prompt asking for the bare name), then runs through
    `canonicalize_type` so aliases like "Diagnosis" or "Medication" also
    land on the same canonical name.
    """
    stripped = (raw or "").strip()
    if stripped.lower().startswith("type_"):
        stripped = stripped[len("type_"):]
    return canonicalize_type(stripped)


def derive_natural_key_id(raw_type: str | None, attrs: dict | None) -> str | None:
    """Return a deterministic id for entities that have a natural key, else None.

      Patient          + attributes.subject_id  -> patient_<subject_id>
      HospitalAdmission + attributes.hadm_id    -> admission_<hadm_id>
      EDStay           + attributes.stay_id     -> edstay_<stay_id>
      ICUStay          + attributes.icustay_id  -> icustay_<icustay_id>
      File             + attributes.source_path -> file_<sha1(path)[:12]>
      Gender           + attributes.omop_concept_id -> gender_<concept_id>
      Race             + attributes.omop_concept_id -> race_<concept_id>

    OMOP-mapped clinical concept nodes (Measurement, Condition, Drug, …) get
    a stable id from their omop_concept_id so the same concept collapses across
    bins regardless of which bin processed it.

    Structural container nodes (Allergy, Note, SignalRecord, …) are patient-scoped:
    same patient + same source_term/name collapses cross-bin duplicates.

    Returns None if the type is not one of these or the relevant key is missing.
    """
    if not isinstance(attrs, dict):
        return None
    etype = canonical_etype(raw_type)

    if etype == "Patient":
        sid = _coerce(attrs.get("subject_id"))
        if sid:
            return f"patient_{_safe_segment(sid)}"
        return None

    if etype == "HospitalAdmission":
        hadm = _coerce(attrs.get("hadm_id"))
        if hadm:
            return f"admission_{_safe_segment(hadm)}"
        return None

    if etype == "EDStay":
        stay = _coerce(attrs.get("stay_id"))
        if stay:
            return f"edstay_{_safe_segment(stay)}"
        return None

    if etype == "ICUStay":
        icustay = _coerce(attrs.get("icustay_id"))
        if icustay:
            return f"icustay_{_safe_segment(icustay)}"
        return None

    if etype == "File":
        path = _coerce(attrs.get("source_path"))
        if path:
            digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
            return f"file_{digest}"
        return None

    if etype == "Gender":
        concept_id = _coerce(attrs.get("omop_concept_id"))
        if concept_id:
            return f"gender_{_safe_segment(concept_id)}"
        # Fallback: slugify the name so at least the same value within a bin
        # gets a consistent id — the merge pass will still create duplicates
        # across bins, but only until the OMOP lookup fills in concept_id.
        name = _coerce(attrs.get("name") or attrs.get("source_term"))
        if name:
            return f"gender_{_safe_segment(name)}"
        return None

    if etype == "Race":
        concept_id = _coerce(attrs.get("omop_concept_id"))
        if concept_id:
            return f"race_{_safe_segment(concept_id)}"
        name = _coerce(attrs.get("name") or attrs.get("source_term"))
        if name:
            return f"race_{_safe_segment(name)}"
        return None

    # OMOP-mapped clinical concept nodes are globally stable — same concept_id
    # means the same real-world concept regardless of which bin processed it.
    _OMOP_CONCEPT_TYPES = {
        "Measurement", "Observation", "Condition", "Drug", "Procedure",
        "Allergy", "Specimen", "Unit", "Route",
    }
    if etype in _OMOP_CONCEPT_TYPES:
        concept_id = _coerce(attrs.get("omop_concept_id"))
        if concept_id:
            type_slug = _safe_segment(etype)
            return f"{type_slug}_{_safe_segment(concept_id)}"

    # Structural container nodes are patient-scoped: same patient + same content
    # slug collapses cross-bin duplicates. Uses source_term (exact CSV wording,
    # most stable) with name as fallback.
    _STRUCTURAL_CONTAINER_TYPES = {
        "Allergy", "SocialHistory", "FamilyHistory", "Immunization",
        "Note", "SignalRecord", "Demographics", "Transfer",
    }
    if etype in _STRUCTURAL_CONTAINER_TYPES:
        subject_id = _coerce(attrs.get("subject_id"))
        term = _coerce(attrs.get("source_term") or attrs.get("name"))
        if subject_id and term:
            type_slug = _safe_segment(etype)
            return f"{type_slug}_{_safe_segment(subject_id)}_{_safe_segment(term)}"

    return None
