import json
from pathlib import Path
import numpy as np
import joblib
from app.ml_model import build_features
from app.ml_model import load_model


def load_public(path: str):
    data = json.loads(Path(path).read_text())
    truth = {}
    for item in data.get("truth", []):
        truth[(item["case_id"], item["study_id"])] = bool(item["is_relevant_to_current"]) 
    cases = data.get("cases", [])
    return cases, truth


def build_dataset(cases, truth):
    X = []
    y = []
    keys = []
    for case in cases:
        cid = case["case_id"]
        cur = case["current_study"]
        for prior in case.get("prior_studies", []):
            key = (cid, prior.get("study_id"))
            if key not in truth:
                continue
            feats = build_features(cur.get("study_description",""), cur.get("study_date",""), prior.get("study_description",""), prior.get("study_date",""))
            X.append(feats)
            y.append(1 if truth[key] else 0)
            keys.append(key)
    return np.vstack(X) if X else np.zeros((0,5)), np.array(y), keys


def evaluate(public_json_path: str = "relevant_priors_public.json"):
    cases, truth = load_public(public_json_path)
    X, y, keys = build_dataset(cases, truth)
    if X.shape[0] == 0:
        print("No labeled pairs found to evaluate.")
        return
    clf, scaler = load_model()
    Xs = scaler.transform(X)
    probs = clf.predict_proba(Xs)[:, 1]
    preds = (probs >= 0.5).astype(int)
    correct = (preds == y).sum()
    incorrect = (preds != y).sum()
    acc = correct / (correct + incorrect)
    print(f"ML evaluation on public split: pairs={len(y)} accuracy={acc:.4f} (correct={correct} incorrect={incorrect})")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "relevant_priors_public.json"
    evaluate(path)
