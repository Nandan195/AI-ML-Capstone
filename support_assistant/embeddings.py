from sentence_transformers import SentenceTransformer
import os

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        # Load the specified model
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_embedding(text: str) -> list[float]:
    model = get_embedding_model()
    # Returns a list of floats natively compatible with ChromaDB
    return model.encode(text).tolist()
