import os
import glob
from .embeddings import get_embedding
from .vector_store import add_chunks

def load_and_chunk_documents(docs_dir: str):
    """
    Reads all text files in the docs directory, chunks them,
    and returns (chunk_ids, texts, metadatas, embeddings).
    """
    chunk_ids = []
    texts = []
    metadatas = []
    embeddings = []

    doc_files = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    
    for doc_idx, file_path in enumerate(doc_files):
        doc_id = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        if not content:
            continue
            
        # Very simple chunking for these small docs (fallback placeholder docs are single sentence)
        # We'll just treat the whole file as one chunk, or split by newline if there are multiple lines.
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for chunk_idx, chunk_text in enumerate(lines):
            chunk_id = f"{doc_id}_chunk_{chunk_idx}"
            chunk_embedding = get_embedding(chunk_text)
            
            chunk_ids.append(chunk_id)
            texts.append(chunk_text)
            metadatas.append({
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "source": doc_id
            })
            embeddings.append(chunk_embedding)
            
    return chunk_ids, texts, metadatas, embeddings

def run_ingestion():
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    print("Starting deterministic ingestion...")
    chunk_ids, texts, metadatas, embeddings = load_and_chunk_documents(docs_dir)
    
    if chunk_ids:
        print(f"Loaded {len(chunk_ids)} chunks. Populating ChromaDB...")
        add_chunks(chunk_ids, texts, metadatas, embeddings)
        print("Ingestion complete.")
    else:
        print("No documents found for ingestion.")

if __name__ == "__main__":
    run_ingestion()
