"""ChromaDB 클라이언트: 청크 적재 및 유사도 검색."""
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# 한글 포함 검색용 멀티링구얼 임베딩
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "yu_docs"


def get_client(path: Path | None = None) -> chromadb.PersistentClient:
    """영구 저장 클라이언트 반환."""
    return chromadb.PersistentClient(path=str(path or CHROMA_PATH))


def get_embedding_function():
    """SentenceTransformer 기반 임베딩 함수."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )


def get_collection(client=None, embedding_function=None):
    """컬렉션 반환 (없으면 생성)."""
    client = client or get_client()
    ef = embedding_function or get_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "YU_AI_Chatbot 문서 청크"},
    )
