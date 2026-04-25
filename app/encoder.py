from sentence_transformers import SentenceTransformer
import numpy as np


_embedder = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name)
    return _embedder


def embed_texts(texts):
    model = get_embedder()
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    if isinstance(emb, list):
        emb = np.array(emb)
    return emb
