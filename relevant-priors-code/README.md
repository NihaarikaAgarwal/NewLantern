FastAPI baseline for Relevant Priors challenge

Run the server locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The `/predict` endpoint accepts the request schema from the challenge and returns one prediction per prior.

Use `python local_eval.py` to run the rule-baseline on the provided `relevant_priors_public.json` for a local accuracy check.
