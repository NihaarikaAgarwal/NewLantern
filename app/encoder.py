import numpy as np
import os


_embedder = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """
    Lazily load the embedding model unless disabled via DISABLE_EMBEDDINGS=1.
    sentence-transformers is not installed in the deployment image when embeddings are disabled.
    """
    global _embedder
    if os.environ.get("DISABLE_EMBEDDINGS", "0") == "1":
        return None
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(model_name)
    return _embedder


def embed_texts(texts):
    """
    Return embeddings as numpy array. If embeddings are disabled, return zero vectors
    of appropriate shape to keep downstream code working while avoiding large memory usage.
    """
    model = get_embedder()
    if model is None:
        # return small zero vectors (384 dims matches MiniLM default)
        dim = 384
        return np.zeros((len(texts), dim), dtype=float)
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    if isinstance(emb, list):
        emb = np.array(emb)
    return emb
