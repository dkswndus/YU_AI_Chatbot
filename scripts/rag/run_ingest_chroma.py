"""ChromaDB 적재 실행."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.chroma_ingest import ingest_all

if __name__ == "__main__":
    ingest_all()
