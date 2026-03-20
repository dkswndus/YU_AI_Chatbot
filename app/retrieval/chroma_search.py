"""ChromaDB 유사도 검색."""
from typing import Any, Dict, List, Optional

from app.retrieval.chroma_client import get_collection


def search(
    query: str,
    n_results: int = 10,
    category: Optional[str] = None,
    type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    collection = get_collection()
    where = {}
    if category and type_filter:
        where = {"$and": [{"category": category}, {"type": type_filter}]}
    elif category:
        where = {"category": category}
    elif type_filter:
        where = {"type": type_filter}

    kwargs = {"query_texts": [query], "n_results": n_results}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return output
