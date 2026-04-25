import os
import json
import asyncio
import logging

logger = logging.getLogger("relevant_priors")

# In-process cache: (current_desc, prior_desc) -> llm_score
_pair_cache: dict[tuple[str, str], float] = {}
_async_client = None


def _get_client():
    global _async_client
    if _async_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None
        from groq import AsyncGroq
        _async_client = AsyncGroq(api_key=api_key)
    return _async_client


async def score_case(current_desc: str, current_date: str, priors: list) -> dict:
    """
    One LLM call per case — all priors batched in a single prompt.
    Returns {study_id: score} where score is 0.0–1.0.
    Cached by (current_desc, prior_desc) so retries are free.
    """
    client = _get_client()
    if client is None:
        return {}

    result = {}
    uncached = []
    for p in priors:
        key = (current_desc, p.get("study_description", ""))
        if key in _pair_cache:
            result[p["study_id"]] = _pair_cache[key]
        else:
            uncached.append(p)

    if not uncached:
        return result

    prior_lines = "\n".join(
        f"{i+1}. {p.get('study_description', '')} ({p.get('study_date', '')})"
        for i, p in enumerate(uncached)
    )

    prompt = (
        f"A radiologist is reading: {current_desc} ({current_date})\n\n"
        f"Rate each prior study's relevance for comparison.\n"
        f"Relevant = same body region AND similar modality (e.g. prior brain MRI helps "
        f"read current brain MRI; prior chest CT does NOT).\n\n"
        f"Priors:\n{prior_lines}\n\n"
        f"Reply with ONLY valid JSON: {{\"scores\": [0.9, 0.1, ...]}} — "
        f"one float 0.0–1.0 per prior in listed order."
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=8.0,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        scores = json.loads(text)["scores"]
        for p, s in zip(uncached, scores):
            key = (current_desc, p.get("study_description", ""))
            _pair_cache[key] = float(s)
            result[p["study_id"]] = float(s)
    except Exception as e:
        logger.warning(f"LLM scorer failed for case: {e}")

    return result


async def score_all_cases(cases: list, timeout: float = 60.0) -> dict:
    """
    Fire one LLM call per case concurrently, wait up to `timeout` seconds total.
    Returns {(case_id, study_id): llm_score}.
    """
    if _get_client() is None:
        return {}

    async def _score_one(case):
        cid = case["case_id"]
        cur = case["current_study"]
        scores = await score_case(
            cur.get("study_description", ""),
            cur.get("study_date", ""),
            case.get("prior_studies", []),
        )
        return {(cid, sid): s for sid, s in scores.items()}

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[_score_one(c) for c in cases], return_exceptions=True),
            timeout=timeout,
        )
        merged = {}
        for r in results:
            if isinstance(r, dict):
                merged.update(r)
        logger.info(f"LLM scored {len(merged)} prior pairs")
        return merged
    except asyncio.TimeoutError:
        logger.warning("LLM scoring timed out; using ML-only predictions")
        return {}
