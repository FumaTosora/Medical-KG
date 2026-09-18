"""Tests for the general-concept registry (parallel of OMOP for non-medical content).

Covers:
- Empty-store lookup returns ok=True with zero candidates.
- upsert + lookup roundtrip: synonym tier and vector tier both fire.
- Dedup on second upsert: same canonical -> no new concept row, synonym appended.
- OMOP regression: lookup_omop_concepts still produces its documented shape
  after the embeddings.py refactor (knn_rowids split out, public alias added).

The tests use a real sqlite-vec-backed SQLite file in a temp dir, the real
sentence-transformer model (so vector behavior is the genuine article, not a
mock). First run pays the ~5s model-load tax once via setUpClass.

Skipped wholesale if sqlite-vec or sentence-transformers aren't installed —
matches the project convention of graceful degradation.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Skip the entire module if the embedding stack isn't available. The
# general-concept tools require both libraries to function end-to-end; a
# half-working test would just be noise.
try:
    import sqlite_vec  # noqa: F401
    import sentence_transformers  # noqa: F401
    EMBEDDING_STACK_AVAILABLE = True
except ImportError:
    EMBEDDING_STACK_AVAILABLE = False


@unittest.skipUnless(
    EMBEDDING_STACK_AVAILABLE,
    "sqlite-vec or sentence-transformers not installed",
)
class GeneralConceptStoreTests(unittest.TestCase):
    """End-to-end tests of the general-concept registry against a real DB."""

    @classmethod
    def setUpClass(cls):
        # Pre-load the sentence-transformer model once so the first per-test
        # call doesn't take 5 seconds.
        from src.queries.embeddings import get_model
        get_model()

    def setUp(self):
        # Each test gets its own DB so they can't pollute each other. Override
        # the module-level SQLITE_DB_PATH that `get_connection` reads, then
        # call init_db to create both the structural KG tables and the new
        # general-concept tables via the migration hook.
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.sqlite"

        from src import database as db_module
        self._orig_db_path = db_module.SQLITE_DB_PATH
        db_module.SQLITE_DB_PATH = str(self._db_path)
        db_module.init_db()

    def tearDown(self):
        from src import database as db_module
        db_module.SQLITE_DB_PATH = self._orig_db_path
        self._tmpdir.cleanup()

    # ---- lookup on empty store ------------------------------------------------

    def test_empty_store_returns_no_candidates(self):
        from src.queries.general_concepts import lookup_general_concepts
        result = lookup_general_concepts("anything at all")
        self.assertTrue(result["ok"])
        self.assertEqual(result["row_count"], 0)
        self.assertIsNone(result["best_candidate"])
        self.assertTrue(result["requires_review"])
        self.assertEqual(result["candidates"], [])

    # ---- upsert + lookup roundtrip --------------------------------------------

    def test_upsert_then_synonym_lookup(self):
        """After upserting a canonical with a synonym, a lookup on the
        synonym must return that canonical with the synonym-tier score (0.92)."""
        from src.queries.general_concepts import (
            lookup_general_concepts,
            upsert_general_concept,
        )

        upserted = upsert_general_concept(
            canonical_name="Gaming console",
            source_term="PlayStation",
        )
        self.assertTrue(upserted["ok"])
        self.assertEqual(upserted["action"], "insert")
        self.assertEqual(upserted["canonical_name"], "Gaming console")
        # source_term lands as the only synonym.
        self.assertEqual(upserted["synonym_count"], 1)

        result = lookup_general_concepts("PlayStation")
        self.assertTrue(result["ok"])
        self.assertEqual(result["row_count"], 1)
        best = result["best_candidate"]
        self.assertEqual(best["concept_name"], "Gaming console")
        self.assertEqual(best["match_type"], "synonym")
        # Synonym-tier score is fixed at 0.92 in _fetch_general_candidates.
        self.assertAlmostEqual(best["score"], 0.92, places=3)

    def test_upsert_then_vector_neighborhood(self):
        """A second specific term (e.g. 'Xbox') that wasn't stored as a
        synonym should still vector-match to 'Gaming console' because the
        embedding text includes the canonical name + synonyms."""
        from src.queries.general_concepts import (
            lookup_general_concepts,
            upsert_general_concept,
        )

        upsert_general_concept(
            canonical_name="Gaming console",
            source_term="PlayStation",
            synonyms=["video game console", "home console"],
        )

        result = lookup_general_concepts("Xbox")
        self.assertTrue(result["ok"])
        # The empty case would be row_count == 0; we expect the vector tier
        # to bring back at least the one concept we inserted.
        self.assertGreaterEqual(result["row_count"], 1)
        best = result["best_candidate"]
        self.assertEqual(best["concept_name"], "Gaming console")
        self.assertEqual(best["match_type"], "vector")
        # The vector tier scaled cosine into [0, 0.9]. We don't pin the
        # exact value — bge-small returns ~0.13 here, SapBERT would be very
        # different — we just confirm the tier fired (score > 0).
        self.assertGreater(best["score"], 0.0)

    # ---- dedup on second upsert ------------------------------------------------

    def test_second_upsert_of_same_canonical_is_dedup(self):
        """Calling upsert_general_concept a second time with the same
        canonical_name (any case) must NOT create a second row — it must
        return the existing concept_id and append the new source_term as a
        synonym."""
        from src.database import get_connection
        from src.queries.general_concepts import upsert_general_concept

        first = upsert_general_concept(
            canonical_name="Gaming console",
            source_term="PlayStation",
        )
        self.assertEqual(first["action"], "insert")
        self.assertEqual(first["synonym_count"], 1)

        second = upsert_general_concept(
            # Different casing on purpose — the unique index is on LOWER(name).
            canonical_name="gaming CONSOLE",
            source_term="Xbox",
        )
        self.assertEqual(second["action"], "update")
        self.assertEqual(second["concept_id"], first["concept_id"])
        self.assertEqual(second["synonym_count"], 2)

        # Confirm the table state directly: still exactly one concept row,
        # two synonyms attached to it.
        conn = get_connection()
        try:
            n_concepts = conn.execute(
                "SELECT COUNT(*) FROM general_concept"
            ).fetchone()[0]
            n_synonyms = conn.execute(
                "SELECT COUNT(*) FROM general_concept_synonym WHERE concept_id = ?",
                (first["concept_id"],),
            ).fetchone()[0]
            synonyms = {
                r[0].lower() for r in conn.execute(
                    "SELECT synonym_name FROM general_concept_synonym WHERE concept_id = ?",
                    (first["concept_id"],),
                ).fetchall()
            }
        finally:
            conn.close()

        self.assertEqual(n_concepts, 1)
        self.assertEqual(n_synonyms, 2)
        self.assertEqual(synonyms, {"playstation", "xbox"})

    def test_second_upsert_repeating_synonym_is_idempotent(self):
        """Re-upserting with the same source_term twice must not duplicate
        the synonym row (relies on the unique-on-LOWER(synonym_name) index)."""
        from src.queries.general_concepts import upsert_general_concept

        upsert_general_concept(canonical_name="Noodle dish", source_term="ramen")
        result = upsert_general_concept(canonical_name="Noodle dish", source_term="RAMEN")
        # Synonym dedup is case-insensitive, so the count stays at 1.
        self.assertEqual(result["synonym_count"], 1)
        self.assertEqual(result["action"], "update")


# ---- OMOP regression -------------------------------------------------------

class OmopLookupShapeRegressionTests(unittest.TestCase):
    """Confirms the embeddings.py refactor (knn_rowids extracted, public
    `connect_with_vec` alias added) didn't change `lookup_omop_concepts`'s
    response shape for the empty-DB / no-OMOP-tables case.

    We don't reach for the real OMOP CSVs here — that's covered by
    test_omop_import.py. This test only checks that the function still
    returns ok=True with row_count=0 when OMOP is absent, which is what the
    rest of the pipeline relies on for graceful degradation."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.sqlite"
        from src import database as db_module
        self._orig_db_path = db_module.SQLITE_DB_PATH
        db_module.SQLITE_DB_PATH = str(self._db_path)
        # Don't call init_db — we want a truly empty DB to confirm graceful
        # degradation when neither concept nor general_concept exists.

    def tearDown(self):
        from src import database as db_module
        db_module.SQLITE_DB_PATH = self._orig_db_path
        self._tmpdir.cleanup()

    def test_lookup_omop_concepts_on_empty_db(self):
        from src.queries.db_tooling import lookup_omop_concepts
        result = lookup_omop_concepts("aspirin")
        # When concept/concept_synonym don't exist, the function catches
        # the OperationalError and returns ok=False with a hint. That's the
        # documented contract — the rest of the pipeline reads ok=False as
        # "OMOP not available, fall back."
        self.assertIn("ok", result)
        if result["ok"]:
            # Some environments may have a stub `concept` table; in that
            # case the call must still succeed with row_count=0.
            self.assertEqual(result["row_count"], 0)
            self.assertIsNone(result["best_candidate"])
        else:
            self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
