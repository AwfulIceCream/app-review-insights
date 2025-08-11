from fastapi import APIRouter, HTTPException
from app.api.schemas import MetricsRequest, MetricsResponse
from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields, average_rating, rating_distribution
from app.core.cache import get_cache, set_cache

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.post("", response_model=MetricsResponse)
async def compute_metrics(payload: MetricsRequest):
    cache_key = f"metrics:v2:{payload.app_id}:{payload.lang}:{payload.country}:{payload.count}"
    cached = get_cache(cache_key)
    if cached:
        return cached

    try:
        raw = fetch_reviews(
            app_id=payload.app_id,
            count=payload.count,
            lang=payload.lang,
            country=payload.country,
        )
    except CollectError as ce:
        raise HTTPException(status_code=400, detail=str(ce))

    rows = extract_fields(raw)
    # rows should have normalized keys like: text, rating, reply, thumbs_up
    # but we’ll gracefully fall back to raw keys if needed.
    texts = [(r.get("text") or r.get("content") or "").strip() for r in rows]
    ratings = [r.get("rating") for r in rows if r.get("rating") in (1, 2, 3, 4, 5)]

    word_counts = [len(t.split()) for t in texts if t != ""]
    char_counts = [len(t) for t in texts]
    has_reply_flags = [
        bool(r.get("reply") or r.get("replyContent"))
        for r in rows
    ]
    thumbs = [
        int(r.get("thumbs_up", r.get("thumbsUpCount", 0)) or 0)
        for r in rows
    ]

    resp = MetricsResponse(
        app_id=payload.app_id,
        count=len(ratings),
        avg_rating=average_rating(ratings),
        rating_distribution=rating_distribution(ratings),
        # NEW aggregates
        avg_word_count=round(sum(word_counts) / len(word_counts), 2) if word_counts else 0.0,
        avg_char_count=round(sum(char_counts) / len(char_counts), 2) if char_counts else 0.0,
        pct_with_reply=round(100.0 * sum(1 for x in has_reply_flags if x) / len(has_reply_flags),
                             2) if has_reply_flags else 0.0,
        total_thumbs_up=sum(thumbs),
        avg_thumbs_up=round(sum(thumbs) / len(thumbs), 3) if thumbs else 0.0,
    )

    set_cache(cache_key, resp, ttl_seconds=600)
    return resp
