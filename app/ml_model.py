from functools import lru_cache
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from .encoder import embed_texts
from .model import _tokens, days_between
import os

MODEL_PATH = Path(__file__).parent / "classifier.joblib"
SCALER_PATH = Path(__file__).parent / "scaler.joblib"


def _same_modality_flag(cur_desc: str, pri_desc: str) -> int:
    modalities = ["ct", "mri", "xray", "xr", "ultrasound", "us"]
    cur = set(_t for _t in cur_desc.lower().split())
    pri = set(_t for _t in pri_desc.lower().split())
    for m in modalities:
        if m in cur and m in pri:
            return 1
    return 0


def build_features(current_desc: str, current_date: str, prior_desc: str, prior_date: str) -> np.ndarray:
    # features: cosine_sim, token_overlap_frac, recency_days, same_modality, rule_pred
    cur_tokens = _tokens(current_desc)
    pri_tokens = _tokens(prior_desc)
    overlap = len(cur_tokens & pri_tokens)
    smaller = min(max(1, len(cur_tokens)), max(1, len(pri_tokens)))
    token_frac = overlap / smaller

    # embeddings
    emb = embed_texts([current_desc, prior_desc])
    a, b = emb[0], emb[1]
    # cosine similarity. If embeddings are zero-vectors (disabled), fall back to token_frac.
    norm_product = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm_product < 1e-6:
        cos_sim = token_frac
    else:
        cos_sim = float(np.dot(a, b)) / norm_product

    recency = days_between(current_date, prior_date)
    same_mod = _same_modality_flag(current_desc, prior_desc)

    # include existing rule baseline as a feature
    # simple rule: token_frac >= 0.5 or recency < 365 or same_modality
    rule_pred = 1 if (token_frac >= 0.5 or recency < 365 or same_mod == 1) else 0

    return np.array([cos_sim, token_frac, recency, same_mod, rule_pred], dtype=float)


def load_model() -> Tuple[LogisticRegression, StandardScaler]:
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run train_and_eval.py to train a model.")
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, scaler


@lru_cache(maxsize=32768)
def predict_proba(case_id: str, current_desc: str, current_date: str, prior_id: str, prior_desc: str, prior_date: str) -> float:
    clf, scaler = load_model()
    feats = build_features(current_desc, current_date, prior_desc, prior_date)
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    prob = float(clf.predict_proba(feats_scaled)[0, 1])
    return prob


@lru_cache(maxsize=32768)
def predict_ml(case_id: str, current_desc: str, current_date: str, prior_id: str, prior_desc: str, prior_date: str) -> bool:
    prob = predict_proba(case_id, current_desc, current_date, prior_id, prior_desc, prior_date)
    return bool(prob >= 0.5)
