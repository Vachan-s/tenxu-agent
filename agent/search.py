from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=1)
def load_embedding_model():
    """Load embedding model once and cache it."""
    return SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def embed_query(text: str) -> list:
    """Convert a text query into a 384-dimension embedding vector."""
    model = load_embedding_model()
    return model.encode(text).tolist()