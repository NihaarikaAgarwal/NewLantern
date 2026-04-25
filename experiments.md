# Experiments — Relevant Priors Challenge

## Methodology

The task is binary classification: given a current radiology study and a prior study for the same patient, predict whether the prior is relevant for the radiologist reading the current study.

### Feature Engineering (5 features per study pair)

All features are derived from study descriptions and dates — no external data sources.

**1. TF-IDF Cosine Similarity**
Fit a `TfidfVectorizer` (unigrams + bigrams, `min_df=2`, `sublinear_tf=True`) on all 28,610 study descriptions in the public split. At inference time, transform both descriptions into TF-IDF vectors and compute cosine similarity. This captures terminology overlap (e.g. "MRI BRAIN" vs "MRI BRAIN WITHOUT CONTRAST" scores high; "CT CHEST" vs "MRI BRAIN" scores low). Medical imaging descriptions are short and structured, making TF-IDF highly effective.

**2. Token Overlap Fraction**
Normalize descriptions (lowercase, strip punctuation), compute the intersection of token sets, divided by the smaller set size. A fast, lightweight signal for exact or near-exact description matches.

**3. Recency (days)**
Absolute day delta between current and prior study dates. More recent priors are generally more relevant.

**4. Same Modality Flag**
Binary: 1 if both descriptions share a modality keyword (`ct`, `mri`, `xray`, `xr`, `ultrasound`, `us`). Captures cross-description modality matching independent of word order.

**5. Rule Baseline Prediction**
Deterministic signal: 1 if `token_frac ≥ 0.5` OR `recency < 365 days` OR `same modality`. Encodes domain knowledge directly as a feature so the classifier can learn when to trust or override it.

### Model

**Logistic Regression** (`sklearn`, `max_iter=1000`) with `StandardScaler` normalization on all 5 features. Trained on an 80/20 stratified split of the 27,614 labeled public pairs.

Artifacts saved to `app/`:
- `tfidf.joblib` — fitted TF-IDF vectorizer
- `classifier.joblib` — trained logistic regression
- `scaler.joblib` — fitted StandardScaler

### Inference Pipeline

```
Request → FastAPI /predict
    → per prior: build_features(current_desc, current_date, prior_desc, prior_date)
        → TF-IDF cosine sim + token overlap + recency + modality flag + rule pred
    → scaler.transform → clf.predict_proba → threshold 0.5 → bool
    → return predictions[]
```

Results are cached with `@lru_cache(maxsize=32768)` keyed on (case_id, current_desc, current_date, prior_id, prior_desc, prior_date) to skip recomputation on retries.

---

## Results

| Approach | Public Holdout Accuracy | Smoke Test (10 cases, 173 priors) |
|---|---|---|
| Rule baseline (token overlap + recency + modality) | 0.6780 | — |
| Logistic regression, no TF-IDF (token features only) | 0.8378 | 0.7803 |
| **Logistic regression + TF-IDF cosine similarity** | **0.8376** | **0.9249** |

The TF-IDF model matches the holdout accuracy of the no-embedding version (as expected — same training data), but the smoke test jumped from 78% to 92.49%, suggesting the TF-IDF cosine similarity feature is the dominant signal for the harder cases in the evaluation set.

Training details (final model):
- TF-IDF vocab: 1,642 terms (unigrams + bigrams, min_df=2)
- Train pairs: 22,091 — accuracy: 0.8383
- Holdout pairs: 5,523 — accuracy: **0.8376**
- Public smoke test: 173 priors — accuracy: **0.9249**

---

## API Design & Hints Coverage

- **Logging**: every request logs a UUID `request_id`, case count, prior count, and total processing time.
- **Timeout**: 330s server-side `asyncio.wait_for` timeout (under the 360s evaluator limit). On timeout, falls back synchronously to the rule-based baseline.
- **Batched inference**: all priors in a request are processed in a single loop — no per-exam external calls.
- **Caching**: `@lru_cache` on both `predict_proba` and `rule_based_decision` — retries and duplicate pairs are free.
- **Response**: `predicted_is_relevant` is a strict Python `bool`, serialized as JSON `true`/`false`.
- **Catch-all route**: `/{full_path:path}` forwards any POST to `/predict` in case the evaluator hits a different path.

---

## Deployment

- Docker container on Render (Starter plan, 512MB RAM).
- Stack: FastAPI + uvicorn + scikit-learn only. No PyTorch — sentence-transformers was evaluated but replaced with TF-IDF to stay within the 512MB memory budget.
- All model artifacts (`tfidf.joblib`, `classifier.joblib`, `scaler.joblib`) committed to the repo and loaded at startup in ~50ms.

---

## Next Steps / Improvements

1. **Gradient boosting** — XGBoost or LightGBM would capture non-linear interactions (e.g. same modality AND very recent) that logistic regression cannot.
2. **Body region extraction** — parse anatomy keywords (brain, chest, abdomen, spine) from descriptions as an explicit feature; two studies of the same body region are almost always relevant to each other.
3. **Date-aware modality weighting** — a CT from 10 years ago is less relevant than an MRI from last month, even if same modality; interaction features between recency and modality/description similarity.
4. **Threshold tuning** — optimize the 0.5 decision threshold on the public split for F1 rather than raw accuracy.
5. **Persistent cache** — Redis to deduplicate predictions across separate evaluator requests (survives restarts).
6. **Sentence transformers on larger host** — a 1GB+ instance could load `all-MiniLM-L6-v2`; may add ~2% accuracy for edge cases where semantic meaning differs from surface form.
