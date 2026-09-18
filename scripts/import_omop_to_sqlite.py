import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import SQLITE_DB_PATH, init_db
from src.omop_import import import_omop_folder_to_sqlite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import OMOP CSV files into the Docker-hosted SQLite database. "
                    "Run inside the container: docker compose exec kg-app python scripts/import_omop_to_sqlite.py"
    )
    parser.add_argument(
        "--omop-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "OMOP"),
        help="Path to folder containing OMOP CSV files.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Append to existing OMOP tables instead of replacing them.",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=None,
        help="Optional list of table names to import (example: concept concept_synonym).",
    )
    args = parser.parse_args()

    init_db()
    results = import_omop_folder_to_sqlite(
        omop_folder=args.omop_dir,
        sqlite_db_path=SQLITE_DB_PATH,
        replace_existing=not args.keep_existing,
        include_tables=args.tables,
    )

    print(f"Imported {len(results)} OMOP table(s) into SQLite: {SQLITE_DB_PATH}")
    for item in results:
        print(f"  - {item['table']}: {item['rows']} rows")


if __name__ == "__main__":
    main()
