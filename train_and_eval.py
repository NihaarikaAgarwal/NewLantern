import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
from app.model import _tokens, days_between
from app.ml_model import build_features


def load_public(path: str):
    data = json.loads(Path(path).read_text())
    truth = {}
    for item in data.get("truth", []):
        truth[(item["case_id"], item["study_id"])] = bool(item["is_relevant_to_current"])
    cases = data.get("cases", [])
    return cases, truth


def fit_tfidf(cases):
    descriptions = []
    for case in cases:
        descriptions.append(case["current_study"].get("study_description", ""))
        for prior in case.get("prior_studies", []):
            descriptions.append(prior.get("study_description", ""))
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    tfidf.fit(descriptions)
    joblib.dump(tfidf, Path("app") / "tfidf.joblib")
    print(f"TF-IDF fitted on {len(descriptions)} descriptions, vocab size: {len(tfidf.vocabulary_)}")
    return tfidf


def build_dataset(cases, truth):
    X = []
    y = []
    for case in cases:
        cid = case["case_id"]
        cur = case["current_study"]
        for prior in case.get("prior_studies", []):
            key = (cid, prior.get("study_id"))
            if key not in truth:
                continue
            feats = build_features(cur.get("study_description", ""), cur.get("study_date", ""), prior.get("study_description", ""), prior.get("study_date", ""))
            X.append(feats)
            y.append(1 if truth[key] else 0)
    return np.vstack(X) if X else np.zeros((0, 5)), np.array(y)


def train(public_json_path: str = "relevant_priors_public.json", test_size: float = 0.2, random_state: int = 42):
    cases, truth = load_public(public_json_path)
    X, y = build_dataset(cases, truth)
    if X.shape[0] == 0:
        print("No labeled pairs found to train on.")
        return

    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None)

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_hold = scaler.transform(X_hold)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xs_train, y_train)

    joblib.dump(clf, Path("app") / "classifier.joblib")
    joblib.dump(scaler, Path("app") / "scaler.joblib")

    preds_train = (clf.predict_proba(Xs_train)[:, 1] >= 0.5).astype(int)
    correct_train = (preds_train == y_train).sum()
    acc_train = correct_train / len(y_train)

    preds_hold = (clf.predict_proba(Xs_hold)[:, 1] >= 0.5).astype(int)
    correct_hold = (preds_hold == y_hold).sum()
    incorrect_hold = (preds_hold != y_hold).sum()
    acc_hold = correct_hold / len(y_hold)

    print(f"Trained on {len(y_train)} pairs — train accuracy: {acc_train:.4f} (correct={correct_train} incorrect={len(y_train)-correct_train})")
    print(f"Holdout on {len(y_hold)} pairs — holdout accuracy: {acc_hold:.4f} (correct={correct_hold} incorrect={incorrect_hold})")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "relevant_priors_public.json"
    cases, _ = load_public(path)
    fit_tfidf(cases)
    train(path)
