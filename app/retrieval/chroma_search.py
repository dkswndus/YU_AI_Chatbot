"""ChromaDB 유사도 검색."""
from app.retrieval.chroma_client import get_collection, get_client


def search(query, n_results=5, category=None):
    """
    질문과 유사한 청크 검색.
    category 를 지정하면 해당 카테고리만 검색.
    """
    coll = get_collection()
    where = {"category": category} if category else None
    result = coll.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    if result["ids"] and result["ids"][0]:
        for i, chunk_id in enumerate(result["ids"][0]):
            out.append({
                "chunk_id": chunk_id,
                "text": result["documents"][0][i] if result["documents"] else "",
                "metadata": result["metadatas"][0][i] if result["metadatas"] else {},
                "distance": result["distances"][0][i] if result.get("distances") else None,
            })
    return out
