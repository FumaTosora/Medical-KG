import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.queries.entity_types import (
    ALLOWED_TYPES,
    NON_OMOP_TYPES,
    OMOP_DOMAIN_TYPES,
    allowed_types_enum,
    canonicalize_type,
)


class WhitelistShapeTests(unittest.TestCase):
    """The whitelist itself must stay coherent — OMOP and non-OMOP sets
    disjoint, the union exposed as ALLOWED_TYPES, no empties."""

    def test_omop_and_non_omop_are_disjoint(self):
        self.assertEqual(OMOP_DOMAIN_TYPES & NON_OMOP_TYPES, frozenset())

    def test_allowed_is_the_union(self):
        self.assertEqual(ALLOWED_TYPES, OMOP_DOMAIN_TYPES | NON_OMOP_TYPES)

    def test_no_empty_or_whitespace_entries(self):
        for t in ALLOWED_TYPES:
            self.assertTrue(t and t.strip(), f"empty/blank type in whitelist: {t!r}")

    def test_unknown_is_in_the_whitelist(self):
        # Fallback type must be a legal canonical type itself.
        self.assertIn("Unknown", ALLOWED_TYPES)


class CanonicalizeTypeTests(unittest.TestCase):
    """Resolution order: OMOP domain (if recognized) > exact match >
    alias > case-insensitive match > 'Unknown'."""

    def test_omop_domain_wins_over_raw_type(self):
        # OMOP domain is ground truth — overrides whatever the LLM wrote.
        self.assertEqual(canonicalize_type("Drug", omop_domain="Condition"), "Condition")
        self.assertEqual(canonicalize_type("anything", omop_domain="Measurement"), "Measurement")

    def test_unrecognized_omop_domain_falls_through(self):
        # Compound OMOP domains (Drug/Measurement) and metadata domains aren't
        # in OMOP_DOMAIN_TYPES; the raw type should be used instead.
        self.assertEqual(canonicalize_type("Drug", omop_domain="Drug/Measurement"), "Drug")
        self.assertEqual(canonicalize_type("Condition", omop_domain="Type Concept"), "Condition")

    def test_canonical_passthrough(self):
        for canonical in ["Drug", "Condition", "Measurement", "Patient", "File", "Transfer"]:
            self.assertEqual(canonicalize_type(canonical), canonical)

    def test_alias_resolution_renames_to_omop_native(self):
        # The whole point of the alias map: collapse renamed/synonym variants.
        self.assertEqual(canonicalize_type("Medication"), "Drug")
        self.assertEqual(canonicalize_type("ClinicalFinding"), "Condition")
        self.assertEqual(canonicalize_type("LabMeasurement"), "Measurement")
        self.assertEqual(canonicalize_type("ObservationNote"), "Observation")
        self.assertEqual(canonicalize_type("Diagnosis"), "Condition")
        self.assertEqual(canonicalize_type("Assessment"), "Measurement")

    def test_alias_is_case_and_punctuation_insensitive(self):
        self.assertEqual(canonicalize_type("medication"), "Drug")
        self.assertEqual(canonicalize_type("MEDICATION"), "Drug")
        self.assertEqual(canonicalize_type("lab_result"), "Measurement")
        self.assertEqual(canonicalize_type("Lab Result"), "Measurement")
        self.assertEqual(canonicalize_type("clinical-finding"), "Condition")

    def test_person_always_aliases_to_patient(self):
        # OMOP "Person" is demographic registry; our patient identity is "Patient".
        # This must hold whether the OMOP domain is given or not.
        self.assertEqual(canonicalize_type("Person"), "Patient")
        self.assertEqual(canonicalize_type("person"), "Patient")
        self.assertEqual(canonicalize_type("Person", omop_domain="Person"), "Patient")

    def test_transport_aliases_to_transfer(self):
        # Hospital inter-station movement — Transfer is the canonical type.
        self.assertEqual(canonicalize_type("Transport"), "Transfer")
        self.assertEqual(canonicalize_type("transport"), "Transfer")

    def test_file_subtypes_collapse(self):
        for raw in ["PDF", "pdf", "image", "Scan", "Document", "attachment"]:
            self.assertEqual(canonicalize_type(raw), "File")

    def test_unknown_for_garbage_input(self):
        self.assertEqual(canonicalize_type(""), "Unknown")
        self.assertEqual(canonicalize_type(None), "Unknown")
        self.assertEqual(canonicalize_type("   "), "Unknown")
        self.assertEqual(canonicalize_type("xyznosuchthing"), "Unknown")

    def test_every_alias_target_is_a_canonical_type(self):
        # Guards against typos in the alias map. Every value the alias map
        # can produce must itself be a valid canonical type.
        from src.queries.entity_types import _ALIASES
        for alias, canonical in _ALIASES.items():
            self.assertIn(
                canonical, ALLOWED_TYPES,
                f"alias {alias!r} maps to {canonical!r} which is not in ALLOWED_TYPES",
            )

    def test_idempotent(self):
        # Running canonicalize twice must equal once — backstop against
        # cycles or pingponging in the alias map.
        for raw in ["Medication", "Drug", "person", "Transport", "PDF", "garbage"]:
            once = canonicalize_type(raw)
            twice = canonicalize_type(once)
            self.assertEqual(once, twice, f"non-idempotent for {raw!r}: {once!r} -> {twice!r}")


class AllowedTypesEnumTests(unittest.TestCase):
    def test_enum_lists_every_canonical_type(self):
        enum_str = allowed_types_enum()
        listed = {t.strip() for t in enum_str.split(",")}
        self.assertEqual(listed, set(ALLOWED_TYPES))

    def test_enum_is_sorted_for_stable_prompts(self):
        enum_str = allowed_types_enum()
        listed = [t.strip() for t in enum_str.split(",")]
        self.assertEqual(listed, sorted(listed))


if __name__ == "__main__":
    unittest.main()
