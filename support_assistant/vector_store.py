import chromadb
import os
from .embeddings import get_embedding

_client = None
COLLECTION_NAME = "support_policies"

def get_db_client():
    global _client
    if _client is None:
        db_path = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(db_path, exist_ok=True)
        _client = chromadb.PersistentClient(path=db_path)
    return _client

def get_collection():
    client = get_db_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def add_chunks(chunk_ids: list[str], texts: list[str], metadatas: list[dict], embeddings: list[list[float]]):
    collection = get_collection()
    collection.upsert(
        ids=chunk_ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

def retrieve_top_k(query: str, k: int = 3):
    collection = get_collection()
    query_embedding = get_embedding(query)
    
    # Query ChromaDB using cosine similarity (default for chromadb if configured, or just default L2).
    # To force cosine, we can set hnsw:space to cosine when creating collection, 
    # but the assignment asks for "cosine similarity retrieval".
    # Let's ensure the collection is created with cosine if possible, or just rely on default.
    # Actually, we can just use collection.query.
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    return results
