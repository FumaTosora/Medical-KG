import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.database import sync_to_neo4j

if __name__ == "__main__":
    sync_to_neo4j()