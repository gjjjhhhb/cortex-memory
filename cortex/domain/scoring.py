import re
from datetime import datetime
from typing import Any
from cortex.domain.models import RecallHit


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def bm25_token_score(query_tokens: set[str], text: str) -> float:
    text_tokens = tokenize(text)
    if not text_tokens or not query_tokens:
        return 0.0
    score = float(len(query_tokens.intersection(text_tokens)))
    for q in query_tokens:
        if len(q) >= 4:
            q_stem = q[:4]
            if any(t.startswith(q_stem) for t in text_tokens if t != q):
                score += 0.5
    return score


def parse_iso(val: Any) -> datetime | None:
    if not val:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except Exception:
        return None


def match_lens(scope: str, lens: Any) -> bool:
    if not lens:
        return True
    if isinstance(lens, list):
        if len(lens) > 0 and isinstance(lens[0], list):
            lens_str = "/".join(str(x) for x in lens[0])
        else:
            lens_str = "/".join(str(x) for x in lens)
    else:
        lens_str = str(lens)
    return scope == "global" or scope.startswith(lens_str) or lens_str.startswith(scope)


def is_temporally_valid(
    created_at: Any,
    valid_from: Any,
    valid_until: Any,
    superseded_by: Any,
    as_of: datetime | None,
    now: datetime,
) -> bool:
    c_time = parse_iso(created_at)
    v_from = parse_iso(valid_from)
    v_until = parse_iso(valid_until)

    if as_of:
        if c_time and c_time > as_of:
            return False
        if v_from and v_from > as_of:
            return False
        if v_until and v_until <= as_of:
            return False
        return True

    if v_until and v_until <= now:
        return False
    if superseded_by:
        return False
    return True


def rank_candidates(
    candidates: list[dict[str, Any]],
    query_emb: list[float],
    query_tokens: set[str],
    mode: str,
    k: int,
) -> list[RecallHit]:
    for cand in candidates:
        cand_emb = cand.get("embedding") or []
        cand["_vec_score"] = cosine_similarity(query_emb, cand_emb)
        cand["_bm25_score"] = bm25_token_score(query_tokens, cand.get("_text", ""))

    vec_sorted = sorted(candidates, key=lambda x: x["_vec_score"], reverse=True)
    for rank, cand in enumerate(vec_sorted):
        cand["_vec_rank"] = rank + 1

    bm25_sorted = sorted(candidates, key=lambda x: x["_bm25_score"], reverse=True)
    for rank, cand in enumerate(bm25_sorted):
        cand["_bm25_rank"] = rank + 1

    for cand in candidates:
        if mode == "vector":
            cand["_final_score"] = cand["_vec_score"]
        elif mode == "bm25":
            cand["_final_score"] = cand["_bm25_score"]
        else:
            cand["_final_score"] = (1.0 / (60.0 + cand["_vec_rank"])) + (
                1.0 / (60.0 + cand["_bm25_rank"])
            )

    ranked = sorted(candidates, key=lambda x: x["_final_score"], reverse=True)
    top_k = ranked[:k]

    hits: list[RecallHit] = []
    for cand in top_k:
        hits.append(
            RecallHit(
                id=str(cand.get("id", "")),
                text=cand.get("_text", ""),
                statement=cand.get("_text", ""),
                score=round(float(cand["_final_score"]), 4),
                source=cand.get("_source_type", "attribute"),
                memory_category=cand.get("memory_category"),
                entity_name=str(cand.get("entity", "")),
                key=cand.get("key"),
                value=cand.get("value"),
                valid_from=parse_iso(cand.get("valid_from")),
                valid_until=parse_iso(cand.get("valid_until")),
                occurred_at=parse_iso(cand.get("created_at")),
                source_span=cand.get("source_span"),
                vec_score=round(float(cand.get("_vec_score", 0.0)), 4),
                bm25_score=round(float(cand.get("_bm25_score", 0.0)), 4),
            )
        )
    return hits
