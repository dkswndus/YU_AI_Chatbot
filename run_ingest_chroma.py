"""청크를 ChromaDB에 적재. (run_chunk.py 실행 후 사용)"""
from app.retrieval.chroma_ingest import ingest

if __name__ == "__main__":
    ingest()
