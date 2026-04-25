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


_MODALITIES = ["ct", "mri", "xray", "xr", "ultrasound", "us"]

_REGIONS = {
    "brain", "head", "chest", "thorax", "abdomen", "abdominal",
    "pelvis", "pelvic", "spine", "spinal", "lumbar", "cervical",
    "thoracic", "neck", "knee", "shoulder", "hip", "ankle",
    "wrist", "elbow", "orbit", "orbits", "facial", "sinus",
    "sinuses", "cardiac", "heart", "liver", "renal", "kidney",
    "breast", "extremity", "extremities", "hand", "foot",
}


def _same_modality_flag(cur_desc: str, pri_desc: str) -> int:
    cur = set(cur_desc.lower().split())
    pri = set(pri_desc.lower().split())
    for m in _MODALITIES:
        if m in cur and m in pri:
            return 1
    return 0


def _same_region_flag(cur_desc: str, pri_desc: str) -> int:
    cur = set(cur_desc.lower().split()) & _REGIONS
    pri = set(pri_desc.lower().split()) & _REGIONS
    if not cur or not pri:
        return -1  # unknown — no region keyword found in one or both
    return 1 if cur & pri else 0


def build_features(current_desc: str, current_date: str, prior_desc: str, prior_date: str) -> np.ndarray:
    # features: cosine_sim, token_overlap_frac, recency_days, same_modality, same_region, rule_pred
    cur_tokens = _tokens(current_desc)
    pri_tokens = _tokens(prior_desc)
    overlap = len(cur_tokens & pri_tokens)
    smaller = min(max(1, len(cur_tokens)), max(1, len(pri_tokens)))
    token_frac = overlap / smaller

    emb = embed_texts([current_desc, prior_desc])
    a, b = emb[0], emb[1]
    norm_product = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm_product < 1e-6:
        cos_sim = token_frac
    else:
        cos_sim = float(np.dot(a, b)) / norm_product

    recency = days_between(current_date, prior_date)
    same_mod = _same_modality_flag(current_desc, pri_desc=prior_desc)
    same_region = _same_region_flag(current_desc, prior_desc)
    rule_pred = 1 if (token_frac >= 0.5 or recency < 365 or same_mod == 1) else 0

    return np.array([cos_sim, token_frac, recency, same_mod, same_region, rule_pred], dtype=float)


_clf: LogisticRegression | None = None
_scaler: StandardScaler | None = None


def load_model() -> Tuple[LogisticRegression, StandardScaler]:
    global _clf, _scaler
    if _clf is None or _scaler is None:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            raise FileNotFoundError("Trained model not found. Run train_and_eval.py to train a model.")
        _clf = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _clf, _scaler


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
