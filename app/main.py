import uuid
import time
import asyncio
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import RequestSchema, ResponseSchema, Prediction
from .ml_model import predict_ml, predict_proba
from .model import rule_based_decision
from .llm_scorer import score_all_cases

logger = logging.getLogger("relevant_priors")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(fmt)
logger.addHandler(handler)

app = FastAPI()


@app.on_event("startup")
async def preload_models():
    from .ml_model import load_model
    from .encoder import _get_tfidf
    _get_tfidf()
    load_model()
    logger.info("Models preloaded at startup")


# Allow CORS and common methods so evaluator preflight/OPTIONS or POST to root won't be blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: method={request.method} path={request.url.path}")
    try:
        resp = await call_next(request)
    except Exception as e:
        logger.exception(f"Error handling request: {e}")
        raise
    return resp


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.api_route("/", methods=["GET", "POST", "OPTIONS", "HEAD"])
async def root_any(request: Request):
    # Support GET for human check, POST for evaluator (forward to /predict logic)
    if request.method == "GET":
        return {"service": "relevant-priors", "endpoints": ["/predict (POST)", "/health (GET)"]}
    # For POST/OPTIONS/HEAD delegate to the same handler used by /predict
    return await predict(request)

LLM_WEIGHT = 0.45
ML_WEIGHT = 0.55


async def process_cases(cases: List[dict]) -> List[Prediction]:
    start = time.time()

    # Step 1: get LLM scores for all cases concurrently (60s budget)
    llm_scores = await score_all_cases(cases, timeout=60.0)
    llm_time = time.time() - start
    if llm_scores:
        logger.info(f"LLM scoring done in {llm_time:.2f}s — {len(llm_scores)} pairs scored")

    # Step 2: ML predictions, blended with LLM where available
    preds: List[Prediction] = []
    for case in cases:
        cid = case["case_id"]
        cur = case["current_study"]
        for prior in case.get("prior_studies", []):
            sid = prior.get("study_id", "")
            try:
                ml_prob = predict_proba(
                    cid,
                    cur.get("study_description", ""), cur.get("study_date", ""),
                    sid,
                    prior.get("study_description", ""), prior.get("study_date", ""),
                )
            except Exception as e:
                logger.error(f"ML predict error: {e}; falling back to rule baseline")
                decision = rule_based_decision(
                    cid,
                    cur.get("study_description", ""), cur.get("study_date", ""),
                    sid,
                    prior.get("study_description", ""), prior.get("study_date", ""),
                )
                ml_prob = 1.0 if decision else 0.0

            llm_prob = llm_scores.get((cid, sid))
            if llm_prob is not None:
                prob = ML_WEIGHT * ml_prob + LLM_WEIGHT * llm_prob
            else:
                prob = ml_prob

            preds.append(Prediction(
                case_id=cid,
                study_id=sid,
                predicted_is_relevant=bool(prob >= 0.5),
                confidence=float(prob),
            ))

    total = time.time() - start
    logger.info(f"Processed {len(cases)} cases, {len(preds)} priors in {total:.3f}s")
    return preds


@app.api_route("/predict", methods=["POST", "OPTIONS", "GET", "HEAD"])
async def predict(request: Request):
    request_id = str(uuid.uuid4())
    t0 = time.time()
    # Handle preflight or non-POST quickly
    if request.method in ("OPTIONS", "HEAD"):
        return JSONResponse(content={})
    if request.method == "GET":
        return {"service": "relevant-priors", "endpoints": ["/predict (POST)", "/health (GET)"]}

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
        logger.error(f"{request_id} - processing timed out, using rule-based fallback")
        preds = []
        for case in req.cases:
            cid = case.case_id
            cur = case.current_study
            for prior in case.prior_studies:
                decision = rule_based_decision(cid, cur.study_description, cur.study_date, prior.study_id, prior.study_description, prior.study_date)
                preds.append(Prediction(case_id=cid, study_id=prior.study_id, predicted_is_relevant=bool(decision), confidence=1.0 if decision else 0.0))

    response = ResponseSchema(predictions=preds)
    total_time = time.time() - t0
    logger.info(f"{request_id} - returning {len(preds)} predictions in {total_time:.3f}s")
    return JSONResponse(content=response.dict())


# Catch-all for any POST/OPTIONS/HEAD paths not matched by earlier routes.
# This ensures external evaluators that POST to a different path on the base URL
# still get forwarded to our prediction handler instead of a 405 from the proxy.
@app.api_route("/{full_path:path}", methods=["POST", "OPTIONS", "HEAD"])
async def catch_all_forward(request: Request):
    return await predict(request)
