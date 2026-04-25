# Experiments — Relevant Priors Challenge

## Overview
This repository implements a deterministic rule-based baseline API for the Relevant Priors challenge. The service exposes a POST `/predict` endpoint that accepts the challenge request schema and returns one boolean prediction per prior study.

## Baseline rule
- Token overlap: normalize descriptions and compute token overlap. If overlap fraction >= 0.5 -> relevant.
- Modality heuristic: if both descriptions contain modality keywords (e.g., `ct`, `mri`, `xray`) -> relevant.
- Recency: if prior is within 365 days of current study -> relevant.
- Decisions are cached via an in-memory LRU cache (`functools.lru_cache`).

## Local evaluation
- Command: `python local_eval.py relevant_priors_public.json`
- Public split result (ran locally): Evaluated 27,614 priors — accuracy: 0.6780 (correct=18,721 incorrect=8,893)

## API behavior & hints coverage
- Logs: each request logs a UUID `request_id`, `case_count`, `prior_count`, processing time, and errors to stdout.
- Timeout: server enforces a 330s processing timeout; if exceeded, it falls back to the rule baseline and returns predictions.
- Batched inference: the service processes priors per case in-batch (no per-prior external calls).
- Caching: rule decisions cached by (case_id, study_id) pair to avoid recomputation on retries.
- Outputs: `predicted_is_relevant` is a strict boolean `true`/`false`.

## Next steps / Improvements
1. Add TF-IDF + lightweight classifier trained on public split (fast, deterministic).  
2. Add sentence-transformer embeddings + cosine similarity thresholding.  
3. Optional: LLM per-case batched probe with caching and token limits; only use as fallback for difficult cases.  
4. Add persistent cache (Redis) for evaluator runs to deduplicate across requests.  
5. Add unit tests and CI for the API contract and scoring.

## How to run
1. Create and activate venv, install deps: see `README.md`.  
2. Start server: `uvicorn app.main:app --host 0.0.0.0 --port 8000`  
3. Example request: send the challenge-formatted JSON to `/predict`.
