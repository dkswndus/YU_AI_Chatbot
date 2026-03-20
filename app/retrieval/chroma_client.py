"""ChromaDB 클라이언트 및 임베딩 함수."""
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "yu_docs"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(CHROMA_DB_PATH))


def get_collection(client: chromadb.PersistentClient = None):
    if client is None:
        client = get_client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
