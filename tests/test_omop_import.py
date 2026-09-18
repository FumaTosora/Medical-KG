import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.omop_import import import_omop_folder_to_sqlite


class OmopImportTests(unittest.TestCase):
    def test_imports_concept_and_synonym_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            omop_dir = tmp_path / "omop"
            omop_dir.mkdir(parents=True, exist_ok=True)
            sqlite_path = tmp_path / "kg.sqlite"

            (omop_dir / "CONCEPT.csv").write_text(
                "concept_id,concept_name,domain_id,vocabulary_id,standard_concept,concept_code\n"
                "1,Headache,Condition,SNOMED,S,25064002\n",
                encoding="utf-8",
            )
            (omop_dir / "CONCEPT_SYNONYM.csv").write_text(
                "concept_id,concept_synonym_name,language_concept_id\n"
                "1,Cephalalgia,4180186\n",
                encoding="utf-8",
            )

            results = import_omop_folder_to_sqlite(omop_dir, sqlite_path)
            self.assertEqual(len(results), 2)

            conn = sqlite3.connect(sqlite_path)
            try:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM concept")
                self.assertEqual(c.fetchone()[0], 1)
                c.execute("SELECT COUNT(*) FROM concept_synonym")
                self.assertEqual(c.fetchone()[0], 1)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()

