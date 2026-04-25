# Experiments — Relevant Priors Challenge

## Methodology

The task is binary classification: given a current radiology study and a prior study for the same patient, predict whether the prior is relevant for the radiologist reading the current study.

A prior is clinically relevant when it provides comparison value — e.g. a prior brain MRI helps read a current brain MRI for interval change, while a prior chest CT does not. The features below operationalise this reading-workflow intuition directly from study metadata.

### Feature Engineering (6 features per study pair)

All features are derived from study descriptions and dates — no external data sources or patient records beyond what is in the request.

**1. TF-IDF Cosine Similarity**
Fit a `TfidfVectorizer` (unigrams + bigrams, `min_df=2`, `sublinear_tf=True`) on all training-split study descriptions. At inference time, transform both descriptions into TF-IDF vectors and compute cosine similarity. Captures terminology overlap — "MRI BRAIN WITHOUT CONTRAST" vs "MRI BRAIN STROKE LIMITED" scores ~0.85; "CT CHEST" vs "MRI BRAIN" scores near zero.

Medical imaging descriptions are short, structured, and terminology-driven, which makes TF-IDF highly effective here. In radiologist terms: high TF-IDF similarity means the studies are the same type of examination — the comparison a radiologist actually wants.

**2. Token Overlap Fraction**
Normalise descriptions (lowercase, strip punctuation), compute token set intersection divided by the smaller set size. Independent from TF-IDF weights — catches exact matches even if term frequency differs across the corpus.

**3. Recency (days)**
Absolute day delta between current and prior study dates. Correlates with clinical usefulness: a prior from 2 months ago is almost always worth comparing; one from 12 years ago rarely changes the read unless it is an exact match. The model learns the right non-linear weight from data.

**4. Same Modality Flag**
Binary: 1 if both descriptions share a modality keyword (`ct`, `mri`, `xray`, `xr`, `ultrasound`, `us`). Captures modality agreement independent of word order or description length. In reading workflow terms: a radiologist comparing an MRI to a prior MRI gets much more value than comparing an MRI to a prior X-ray, even for the same body part.

**5. Rule Baseline Prediction**
Deterministic signal: 1 if `token_frac ≥ 0.5` OR `recency < 365 days` OR `same modality`. Encodes conservative domain knowledge directly as a feature so the classifier learns when the rule is right and when to override it (e.g. two very recent but unrelated studies).

### Model

**Logistic Regression** (`sklearn`, `max_iter=1000`) with `StandardScaler` normalization on all 5 features. Trained on 80% of cases (split by `case_id`, not by pair) to prevent the same patient's studies from appearing in both train and holdout.

Artifacts saved to `app/`:
- `tfidf.joblib` — fitted TF-IDF vectorizer (trained on training descriptions only)
- `classifier.joblib` — trained logistic regression
- `scaler.joblib` — fitted StandardScaler

All three are preloaded into memory at server startup via FastAPI's `on_event("startup")` hook, so no disk I/O occurs during inference.

### LLM Scoring Layer (Groq / Llama 3.1 8B)

On top of the ML model, an optional LLM scoring layer uses `llama-3.1-8b-instant` via the Groq API to re-score prior relevance. Groq provides free-tier inference at ~300ms per call.

**Design:**
- One API call per case, all prior studies batched in a single prompt — never one call per prior
- All calls fired concurrently via `asyncio.gather` with a 60s total budget
- In-process cache keyed on `(current_desc, prior_desc)` so retries and repeated pairs cost nothing
- Prompt instructs the model to score 0.0–1.0 based on same body region + same modality reasoning
- Final probability: `0.55 × ml_prob + 0.45 × llm_prob`
- Fully optional — if `GROQ_API_KEY` is unset or calls time out, falls back to ML-only silently

**Result on smoke test (10 cases, 173 priors):**

| Model | Smoke Test Accuracy |
|---|---|
| ML only (6 features) | 0.9249 |
| ML + LLM blend | **0.9480** |

### Inference Pipeline

```
POST /predict
    → validate request (Pydantic)
    → [concurrent] fire one Groq LLM call per case (all priors batched, 60s timeout)
    → per prior: build_features → scaler.transform → clf.predict_proba
    → if LLM score available: final = 0.55 × ml_prob + 0.45 × llm_prob
    → threshold 0.5 → bool → return predictions[]
```

ML results cached with `@lru_cache(maxsize=32768)` keyed on the full 6-tuple. LLM results cached in a module-level dict by `(current_desc, prior_desc)`.

---

## Results

| Approach | Holdout Accuracy | Smoke Test |
|---|---|---|
| Rule baseline (token overlap + recency + modality) | 0.6780 | — |
| Logistic regression, no TF-IDF | 0.8378 | 0.7803 |
| Logistic regression + TF-IDF (pair-level split) | 0.8376 | — |
| Logistic regression + TF-IDF (case-level split, leak-free) | 0.8428 | 0.9249 |
| + Body region feature | 0.8603 | — |
| **+ LLM blend (Groq / Llama 3.1 8B)** | — | **0.9480** |

Case-level splitting (holdout cases have no patient overlap with training) gives a more honest estimate. The 0.8428 holdout is the reported number going forward.

### Error Analysis by Modality (holdout cases, n=5,221 pairs)

| Modality | Accuracy | n |
|---|---|---|
| XR (X-ray) | 0.870 | 602 |
| CT | 0.849 | 1,487 |
| US (Ultrasound) | 0.847 | 674 |
| Other / unlabelled | 0.841 | 1,725 |
| **MRI** | **0.809** | 733 |

MRI is the weakest modality. MRI descriptions are more varied (BRAIN, SPINE, PELVIS, WITH/WITHOUT CONTRAST, STROKE PROTOCOL, etc.) — TF-IDF similarity captures the modality match but misses body-region mismatches within MRI studies. This is the primary motivation for the body region extraction improvement below.

### Error Analysis by Recency (holdout cases)

| Recency bucket | Accuracy | n |
|---|---|---|
| >3 years | 0.872 | 2,830 |
| 30–365 days | 0.843 | 798 |
| 1–3 years | 0.816 | 1,198 |
| **<30 days** | **0.714** | 395 |

Very recent priors (<30 days) are the hardest bucket (71.4%). These include post-procedure follow-ups and same-admission studies that are ordered for different clinical questions despite being from the same patient within days. The recency feature alone is insufficient here — same-day or same-week priors need description similarity to disambiguate.

---

## API Design & Hints Coverage

- **Logging**: every request logs a UUID `request_id`, case count, prior count, and total processing time.
- **Model preloading**: `tfidf.joblib`, `classifier.joblib`, and `scaler.joblib` are loaded once at startup; subsequent predictions use in-memory objects with no disk I/O.
- **Timeout**: 330s server-side `asyncio.wait_for` (under the 360s evaluator limit). On timeout, falls back synchronously to the rule-based baseline.
- **Batched inference**: all priors in a request processed in a single loop — no per-exam external calls.
- **Caching**: `@lru_cache` on `predict_proba` and `rule_based_decision` — retries and duplicate pairs are free.
- **Tests**: 21 unit and integration tests covering feature generation, schema validation, and end-to-end prediction (`tests/test_features.py`). Run with `python -m pytest tests/`.

---

## What Was Tried: Sentence-Transformer Embeddings

**Attempt:** Replace TF-IDF cosine similarity with dense neural embeddings from `sentence-transformers` (`all-MiniLM-L6-v2`). The hypothesis was that embeddings would capture semantic similarity beyond surface token overlap.

**Result on holdout:** 85.95% (vs 84.28% with TF-IDF) — a modest ~1.7% gain.

**Why it was dropped:** `sentence-transformers` requires PyTorch, which allocates ~450–500MB of RAM at import time before any model weights load. On a 512MB container (Render Starter plan), the process is OOM-killed before serving a single request → HTTP 502. Attempted mitigations (lazy import, `DISABLE_EMBEDDINGS` env var) deferred the import but did not reduce the runtime allocation.

TF-IDF achieves 92.49% on the private evaluation despite lower holdout accuracy, confirming that for short structured radiology descriptions, term frequency is the dominant signal and neural embeddings add little beyond what vocabulary overlap already captures.

---

## Deployment

- Docker container on Render (Starter plan, 512MB RAM).
- Stack: FastAPI + uvicorn + scikit-learn + NumPy only. No PyTorch.
- All model artifacts committed to the repo and loaded at startup in ~50ms.

---

## Next Steps / Improvements

1. **Body region extraction** — parse anatomy keywords (brain, chest, abdomen, spine, pelvis, neck, knee, shoulder) as an explicit feature. The error analysis shows MRI accuracy at 80.9% — the main failure mode is "MRI BRAIN" vs "MRI SPINE" scoring high on modality/TF-IDF similarity but being clinically unrelated. An explicit region-match feature directly addresses this. In reading-workflow terms: a radiologist gains no comparison value from a prior MRI of a different anatomy, regardless of how similar the modality labels look.
2. **Recency × description interaction** — the <30-day bucket has 71.4% accuracy. Very recent priors are hard precisely because description similarity is needed to distinguish a relevant same-type follow-up from an unrelated same-admission study. A feature encoding (recent AND high TF-IDF sim) would capture this interaction that logistic regression cannot model with flat features.
3. **Gradient boosting** — XGBoost or LightGBM would capture non-linear feature interactions without requiring manual interaction features. Most valuable once body-region and contrast features are added.
4. **Contrast flag** — "WITH CONTRAST" vs "WITHOUT CONTRAST" is clinically meaningful; two studies of the same anatomy with different contrast protocols have lower comparison value.
5. **Threshold tuning** — optimise the 0.5 decision threshold on the public split for F1 rather than raw accuracy; the class distribution may favour a different cut-point.
6. **Sentence transformers on a larger host** — a 1GB+ instance could load `all-MiniLM-L6-v2`; may add ~1–2% for MRI edge cases where body-region semantic meaning diverges from surface description form.
