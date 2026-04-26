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
    X, y = [], []
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
    return np.vstack(X) if X else np.zeros((0, 6)), np.array(y)


def train(public_json_path: str = "relevant_priors_public.json", test_size: float = 0.2, random_state: int = 42):
    cases, truth = load_public(public_json_path)

    # Split by patient_id (falling back to case_id if absent) so the same patient
    # never appears in both train and holdout — prevents study-pair leakage.
    split_key = "patient_id" if all(c.get("patient_id") for c in cases) else "case_id"
    unique_ids = list({c[split_key] for c in cases})
    train_ids, hold_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    train_ids_set, hold_ids_set = set(train_ids), set(hold_ids)
    train_cases = [c for c in cases if c[split_key] in train_ids_set]
    hold_cases  = [c for c in cases if c[split_key] in hold_ids_set]
    print(f"Split key: {split_key} — {len(train_ids_set)} train / {len(hold_ids_set)} holdout")

    # Fit TF-IDF only on training descriptions to prevent leakage into holdout
    fit_tfidf(train_cases)

    X_train, y_train = build_dataset(train_cases, truth)
    X_hold, y_hold = build_dataset(hold_cases, truth)

    if X_train.shape[0] == 0:
        print("No labeled pairs found to train on.")
        return

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_hold = scaler.transform(X_hold)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xs_train, y_train)

    joblib.dump(clf, Path("app") / "classifier.joblib")
    joblib.dump(scaler, Path("app") / "scaler.joblib")

    preds_train = (clf.predict_proba(Xs_train)[:, 1] >= 0.5).astype(int)
    acc_train = (preds_train == y_train).mean()

    preds_hold = (clf.predict_proba(Xs_hold)[:, 1] >= 0.5).astype(int)
    correct_hold = (preds_hold == y_hold).sum()
    incorrect_hold = (preds_hold != y_hold).sum()
    acc_hold = correct_hold / len(y_hold)

    print(f"Split: {len(train_cases)} train cases ({len(y_train)} pairs) / {len(hold_cases)} holdout cases ({len(y_hold)} pairs)")
    print(f"Train accuracy:   {acc_train:.4f}")
    print(f"Holdout accuracy: {acc_hold:.4f} (correct={correct_hold} incorrect={incorrect_hold})")

    _error_analysis(clf, scaler, hold_cases, truth, y_hold, preds_hold)


def _error_analysis(clf, scaler, hold_cases, truth, y_hold, preds_hold):
    from app.model import _tokens as tokenize
    from app.ml_model import _MODALITIES

    def get_modality(desc):
        tokens = tokenize(desc)
        for m in _MODALITIES:
            if m in tokens:
                return m
        return "other"

    def recency_bucket(days):
        if days < 30:
            return "<30d"
        if days < 365:
            return "30-365d"
        if days < 1095:
            return "1-3y"
        return ">3y"

    rows = []
    for case in hold_cases:
        cid = case["case_id"]
        cur = case["current_study"]
        for prior in case.get("prior_studies", []):
            key = (cid, prior.get("study_id"))
            if key not in truth:
                continue
            rows.append({
                "modality": get_modality(cur.get("study_description", "")),
                "recency": recency_bucket(abs((
                    __import__("datetime").date.fromisoformat(cur.get("study_date", "2000-01-01")) -
                    __import__("datetime").date.fromisoformat(prior.get("study_date", "2000-01-01"))
                ).days) if cur.get("study_date") and prior.get("study_date") else 9999),
                "truth": truth[key],
            })

    if not rows:
        return
    corrects = [p == r["truth"] for p, r in zip(preds_hold, rows)]

    def summarise(key):
        buckets = {}
        for row, correct in zip(rows, corrects):
            k = row[key]
            buckets.setdefault(k, []).append(correct)
        return buckets

    print("\n--- Error analysis by modality ---")
    for mod, vals in sorted(summarise("modality").items()):
        print(f"  {mod:12s}: {sum(vals)/len(vals):.3f}  (n={len(vals)})")

    print("\n--- Error analysis by recency ---")
    for bucket in ["<30d", "30-365d", "1-3y", ">3y"]:
        vals = summarise("recency").get(bucket, [])
        if vals:
            print(f"  {bucket:10s}: {sum(vals)/len(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "relevant_priors_public.json"
    train(path)
