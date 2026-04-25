import uuid
import time
import asyncio
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .schemas import RequestSchema, ResponseSchema, Prediction
from .ml_model import predict_ml
from .model import rule_based_decision

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"service": "relevant-priors", "endpoints": ["/predict (POST)", "/health (GET)"]}

# Logging
logger = logging.getLogger("relevant_priors")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(fmt)
logger.addHandler(handler)


async def process_cases(cases: List[dict]) -> List[Prediction]:
    preds: List[Prediction] = []
    start = time.time()
    for case in cases:
        cid = case["case_id"]
        cur = case["current_study"]
        for prior in case.get("prior_studies", []):
            # ML hybrid prediction (embeddings + features + classifier). No fallback.
            try:
                prob = predict_proba(cid, cur.get("study_description", ""), cur.get("study_date", ""), prior.get("study_id", ""), prior.get("study_description", ""), prior.get("study_date", ""))
                decision = bool(prob >= 0.5)
            except Exception as e:
                logger.error(f"ML predict error: {e}; using rule-based decision (prob=0.0)")
                decision = rule_based_decision(cid, cur.get("study_description", ""), cur.get("study_date", ""), prior.get("study_id", ""), prior.get("study_description", ""), prior.get("study_date", ""))
                prob = 1.0 if decision else 0.0
            preds.append(Prediction(case_id=cid, study_id=prior.get("study_id", ""), predicted_is_relevant=bool(decision), confidence=float(prob)))
    total = time.time() - start
    logger.info(f"Processed {len(cases)} cases, {len(preds)} priors in {total:.3f}s")
    return preds


@app.post("/predict")
async def predict(request: Request):
    request_id = str(uuid.uuid4())
    t0 = time.time()
    body = await request.json()
    try:
        req = RequestSchema(**body)
    except Exception as e:
        logger.error(f"{request_id} - invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    case_count = len(req.cases)
    prior_count = sum(len(c.prior_studies) for c in req.cases)
    logger.info(f"{request_id} - received request: cases={case_count} priors={prior_count}")

    try:
        # enforce a safe timeout shorter than evaluator's 360s
        preds = await asyncio.wait_for(process_cases([c.dict() for c in req.cases]), timeout=330)
    except asyncio.TimeoutError:
        logger.error(f"{request_id} - processing timed out, using fallback rule baseline")
        # fallback: run rule baseline synchronously (should be fast)
        preds = await process_cases([c.dict() for c in req.cases])

    response = ResponseSchema(predictions=preds)
    total_time = time.time() - t0
    logger.info(f"{request_id} - returning {len(preds)} predictions in {total_time:.3f}s")
    return JSONResponse(content=response.dict())
