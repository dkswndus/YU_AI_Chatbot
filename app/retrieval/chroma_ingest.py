"""청크 JSON을 ChromaDB에 적재."""
import json
from pathlib import Path

from app.retrieval.chroma_client import CHROMA_PATH, COLLECTION_NAME, get_client, get_collection

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"


def load_all_chunks():
    """data/processed/chunks/{category}/*.json 에서 모든 청크 로드."""
    chunks = []
    if not CHUNKS_DIR.exists():
        return chunks
    for json_path in sorted(CHUNKS_DIR.rglob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            chunks.extend(data)
        else:
            chunks.append(data)
    return chunks


def ingest(chroma_path=None, reset=True):
    """청크를 ChromaDB에 적재. reset=True면 기존 컬렉션 삭제 후 처음부터 적재."""
    chroma_path = chroma_path or CHROMA_PATH
    chroma_path.mkdir(parents=True, exist_ok=True)

    all_chunks = load_all_chunks()
    if not all_chunks:
        print("로드된 청크 없음. run_chunk.py 먼저 실행하세요.")
        return 0

    client = get_client(chroma_path)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    coll = get_collection(client)

    ids = []
    documents = []
    metadatas = []

    for c in all_chunks:
        ids.append(c["chunk_id"])
        documents.append(c["text"])
        metadatas.append({
            "doc_id": c["doc_id"],
            "title": c["title"],
            "category": c["category"],
            "section": c.get("section", ""),
        })

    # ChromaDB add는 한 번에 많이 넣어도 됨 (내부에서 배치)
    coll.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"ChromaDB 적재 완료: {len(ids)}개 청크 -> {chroma_path}")
    return len(ids)


if __name__ == "__main__":
    ingest()
