# Experiments — Relevant Priors Challenge

## Overview
This service exposes a POST `/predict` endpoint that accepts the challenge request schema and returns one boolean prediction per prior study. The final model is a logistic regression classifier trained on engineered features derived from study descriptions and dates.

## Approach

### Features (5 per study pair)
1. **Cosine similarity** — sentence-transformer embeddings (`all-MiniLM-L6-v2`) of current vs. prior description. When `DISABLE_EMBEDDINGS=1` (deployed config), this falls back to token-overlap fraction to avoid OOM on constrained hosts.
2. **Token overlap fraction** — normalized token overlap between descriptions, divided by the smaller set size.
3. **Recency (days)** — absolute day delta between current and prior study dates.
4. **Same modality flag** — 1 if both descriptions share a modality keyword (ct, mri, xray, xr, ultrasound, us).
5. **Rule baseline prediction** — 1 if token_frac ≥ 0.5 OR recency < 365 OR same modality (deterministic signal).

### Model
- **Logistic Regression** (`sklearn`, `max_iter=1000`) with `StandardScaler` normalization.
- Trained on 80% of the 27,614 labeled public pairs; evaluated on the 20% holdout.
- Model artifacts (`classifier.joblib`, `scaler.joblib`) committed to the repo and loaded at startup.

## Results

| Approach | Public Split Accuracy |
|---|---|
| Rule baseline (token overlap + recency + modality) | 0.6780 |
| Logistic regression (features above, no embeddings) | **0.8378** |

Training run details:
- Train (22,091 pairs): accuracy 0.8386 (correct=18,525, incorrect=3,566)
- Holdout (5,523 pairs): accuracy **0.8378** (correct=4,627, incorrect=896)

## API behavior & hints coverage
- **Logging**: each request logs a UUID `request_id`, `case_count`, `prior_count`, processing time, and errors.
- **Timeout**: 330s server-side timeout (under the evaluator's 360s limit); on timeout, falls back to the fast rule-based baseline.
- **Batched inference**: all priors processed in a single loop per request — no per-exam external calls.
- **Caching**: `predict_proba` and `rule_based_decision` both use `@lru_cache(maxsize=32768)` to skip repeated pairs.
- **Memory**: deployed with `DISABLE_EMBEDDINGS=1` to avoid loading PyTorch/sentence-transformers (~500MB) on free-tier hosts, preventing OOM/502 errors. Features degrade gracefully (cosine sim → token_frac fallback).

## Deployment
- Hosted on Render (free plan) via Docker.
- `DISABLE_EMBEDDINGS=1` set as environment variable in `render.yaml`.
- `render.yaml` + `Dockerfile` + `Procfile` included for one-click deploy.

## Next steps / Improvements
1. **Train with embeddings on a larger host** — run training with `DISABLE_EMBEDDINGS=0` on a machine with ≥2GB RAM and deploy on a paid plan. Embeddings add meaningful signal, especially for cross-modality pairs with similar body regions.
2. **Gradient boosting** — replace logistic regression with XGBoost or LightGBM; non-linear feature interactions (e.g., same modality AND recent) may improve accuracy.
3. **Body-region extraction** — parse anatomy keywords (brain, chest, abdomen) from descriptions as an additional feature.
4. **Persistent cache** — use Redis to deduplicate predictions across separate evaluator requests (survives process restarts).
5. **Threshold tuning** — optimize the 0.5 decision threshold on the public split to improve F1 rather than raw accuracy.
