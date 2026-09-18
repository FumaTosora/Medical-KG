import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.queries.db_tooling import ask_database_with_llm


def main():
    parser = argparse.ArgumentParser(description="LLM + SQLite Tool-Loop Runner")
    parser.add_argument("prompt", help="Frage oder Auftrag fuer das LLM")
    args = parser.parse_args()

    result = ask_database_with_llm(args.prompt)
    print(result)


if __name__ == "__main__":
    main()


