import pytest
import numpy as np
from app.model import _tokens, days_between, rule_based_decision
from app.ml_model import build_features, _same_modality_flag


# --- _tokens ---

def test_tokens_normalises_case():
    assert _tokens("MRI BRAIN") == _tokens("mri brain")

def test_tokens_strips_punctuation():
    # slash is replaced by space, so "W/O" splits into tokens "w" and "o"
    assert "/" not in " ".join(_tokens("MRI W/O CONTRAST"))

def test_tokens_empty():
    assert _tokens("") == set()


# --- days_between ---

def test_days_between_same_date():
    assert days_between("2024-01-01", "2024-01-01") == 0

def test_days_between_one_year():
    assert days_between("2024-01-01", "2023-01-01") == 365

def test_days_between_bad_date():
    assert days_between("not-a-date", "2024-01-01") == 99999


# --- _same_modality_flag ---

def test_same_modality_both_mri():
    assert _same_modality_flag("MRI BRAIN WITHOUT CONTRAST", "MRI SPINE") == 1

def test_same_modality_ct_vs_mri():
    assert _same_modality_flag("CT CHEST", "MRI BRAIN") == 0

def test_same_modality_no_keyword():
    assert _same_modality_flag("ANGIOGRAM CEREBRAL", "ANGIOGRAM RENAL") == 0


# --- build_features ---

def test_build_features_shape():
    feats = build_features("MRI BRAIN", "2024-01-01", "MRI BRAIN", "2023-01-01")
    assert feats.shape == (5,)

def test_build_features_identical_descriptions():
    feats = build_features("CT CHEST WITHOUT CONTRAST", "2024-01-01", "CT CHEST WITHOUT CONTRAST", "2024-01-01")
    cos_sim, token_frac, recency, same_mod, rule_pred = feats
    assert token_frac == 1.0
    assert recency == 0
    assert same_mod == 1
    assert rule_pred == 1

def test_build_features_different_modality():
    feats = build_features("CT CHEST", "2024-01-01", "MRI BRAIN", "2020-01-01")
    _, _, _, same_mod, _ = feats
    assert same_mod == 0

def test_build_features_recent_triggers_rule():
    feats = build_features("XRAY CHEST", "2024-06-01", "CT ABDOMEN", "2024-05-01")
    _, _, recency, _, rule_pred = feats
    assert recency < 365
    assert rule_pred == 1


# --- rule_based_decision ---

def test_rule_same_description_is_relevant():
    assert rule_based_decision("c1", "MRI BRAIN", "2024-01-01", "s1", "MRI BRAIN", "2020-01-01") is True

def test_rule_recent_prior_is_relevant():
    assert rule_based_decision("c1", "CT CHEST", "2024-06-01", "s1", "MRI ABDOMEN", "2024-05-15") is True

def test_rule_old_unrelated_is_not_relevant():
    assert rule_based_decision("c1", "CT CHEST", "2024-01-01", "s1", "BONE SCAN", "2010-01-01") is False


# --- schema validation ---

def test_schema_valid_request():
    from app.schemas import RequestSchema
    payload = {
        "challenge_id": "relevant-priors-v1",
        "schema_version": 1,
        "generated_at": "2026-01-01T00:00:00Z",
        "cases": [{
            "case_id": "001",
            "patient_id": "p1",
            "patient_name": "Doe, Jane",
            "current_study": {"study_id": "s1", "study_description": "MRI BRAIN", "study_date": "2024-01-01"},
            "prior_studies": [{"study_id": "s2", "study_description": "CT HEAD", "study_date": "2023-01-01"}]
        }]
    }
    req = RequestSchema(**payload)
    assert len(req.cases) == 1
    assert len(req.cases[0].prior_studies) == 1

def test_schema_missing_cases_raises():
    from app.schemas import RequestSchema
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        RequestSchema(challenge_id="x", schema_version=1, generated_at="2026-01-01")


# --- end-to-end prediction ---

def test_predict_proba_returns_float_in_range():
    from app.ml_model import predict_proba
    prob = predict_proba("c1", "MRI BRAIN WITHOUT CONTRAST", "2024-01-01", "s1", "MRI BRAIN WITHOUT CONTRAST", "2023-01-01")
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

def test_predict_proba_same_description_high_prob():
    from app.ml_model import predict_proba
    prob = predict_proba("c1", "CT CHEST WITHOUT CONTRAST", "2024-01-01", "s1", "CT CHEST WITHOUT CONTRAST", "2022-01-01")
    assert prob > 0.5

def test_predict_proba_unrelated_low_prob():
    from app.ml_model import predict_proba
    prob = predict_proba("c1", "MRI BRAIN", "2024-01-01", "s1", "BONE SCAN WHOLE BODY", "2010-01-01")
    assert prob < 0.5
