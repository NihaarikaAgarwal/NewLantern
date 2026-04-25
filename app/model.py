from functools import lru_cache
from datetime import date
import re
from typing import Tuple


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> set:
    return set(_normalize(text).split())


def days_between(d1: str, d2: str) -> int:
    try:
        a = date.fromisoformat(d1)
        b = date.fromisoformat(d2)
        return abs((a - b).days)
    except Exception:
        return 99999


@lru_cache(maxsize=32768)
def rule_based_decision(case_id: str, current_desc: str, current_date: str, prior_study_id: str, prior_desc: str, prior_date: str) -> bool:
    """
    Simple deterministic rule baseline:
    - If same normalized description (or high token overlap) -> relevant
    - Or if delta days < 365 (within 1 year) -> relevant
    Otherwise not relevant.
    Cached by study pair to avoid recomputation.
    """
    cur_tokens = _tokens(current_desc)
    pri_tokens = _tokens(prior_desc)
    if not cur_tokens or not pri_tokens:
        return False

    overlap = len(cur_tokens & pri_tokens)
    smaller = min(len(cur_tokens), len(pri_tokens))
    frac = overlap / max(1, smaller)
    if frac >= 0.5:
        return True

    # modality exact-match heuristic (CT / MRI / XR keywords)
    modalities = ["ct", "mri", "xray", "xr", "ultrasound", "us"]
    for m in modalities:
        if m in cur_tokens and m in pri_tokens:
            return True

    if days_between(current_date, prior_date) < 365:
        return True

    return False
