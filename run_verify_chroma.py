"""ChromaDB 적재 여부 확인: 청크 개수 + 샘플 조회."""
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "yu_docs"


def main():
    if not CHROMA_PATH.exists():
        print(f"ChromaDB 폴더 없음: {CHROMA_PATH}")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    coll = client.get_collection(COLLECTION_NAME)

    count = coll.count()
    print(f"[ChromaDB] 컬렉션: {COLLECTION_NAME}")
    print(f"[ChromaDB] 저장 경로: {CHROMA_PATH}")
    print(f"[ChromaDB] 청크 개수: {count}개")
    print()

    # 샘플 3개 조회 (id, 문서 일부, 메타데이터)
    sample = coll.peek(limit=3)
    print("--- 샘플 3건 ---")
    for i, id_ in enumerate(sample["ids"], 1):
        doc = sample["documents"][sample["ids"].index(id_)] if sample["documents"] else ""
        meta = sample["metadatas"][sample["ids"].index(id_)] if sample["metadatas"] else {}
        text_preview = (doc[:120] + "…") if len(doc) > 120 else doc
        print(f"{i}. id: {id_}")
        print(f"   category: {meta.get('category')} | title: {meta.get('title')}")
        print(f"   text: {text_preview}")
        print()


if __name__ == "__main__":
    main()
