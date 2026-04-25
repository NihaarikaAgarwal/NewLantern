import numpy as np
import joblib
from pathlib import Path

TFIDF_PATH = Path(__file__).parent / "tfidf.joblib"
_tfidf = None


def _get_tfidf():
    global _tfidf
    if _tfidf is None and TFIDF_PATH.exists():
        _tfidf = joblib.load(TFIDF_PATH)
    return _tfidf


def embed_texts(texts):
    tfidf = _get_tfidf()
    if tfidf is None:
        return np.zeros((len(texts), 1), dtype=float)
    return tfidf.transform(texts).toarray().astype(float)
