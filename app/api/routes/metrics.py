import logging
from fastapi import APIRouter, HTTPException
from app.api.schemas import MetricsRequest, MetricsResponse
from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields, average_rating, rating_distribution
from app.core.cache import get_cache, set_cache

router = APIRouter(prefix="/metrics", tags=["metrics"])
log = logging.getLogger(__name__)


@router.post("", response_model=MetricsResponse, response_model_exclude_none=True)
async def compute_metrics(payload: MetricsRequest):
    cache_key = f"metrics:v2:{payload.app_id}:{payload.lang}:{payload.country}:{payload.count}"
    log.info("metrics_request_start", extra=payload.dict())

    # Cache check
    cached = get_cache(cache_key)
    if cached:
        log.info("metrics_cache_hit", extra={"cache_key": cache_key})
        return cached

    # Fetch reviews
    try:
        raw = fetch_reviews(
            app_id=payload.app_id,
            count=payload.count,
            lang=payload.lang,
            country=payload.country,
        )
        log.info("reviews_fetched", extra={"count": len(raw)})
    except CollectError as ce:
        log.warning("collect_error", extra={"error": str(ce)})
        raise HTTPException(status_code=400, detail=str(ce))
    except Exception:
        log.exception("unexpected_collect_error")
        raise HTTPException(status_code=502, detail="Failed to collect reviews")

    # Normalize review fields
    try:
        rows = extract_fields(raw)
    except Exception:
        log.exception("extract_fields_failed")
        raise HTTPException(status_code=500, detail="Failed to normalize reviews")

    if not rows:
        log.info("no_reviews_found")
        return MetricsResponse(app_id=payload.app_id, count=0)

    # Extract metrics
    texts = [(r.get("text") or r.get("content") or "").strip() for r in rows]
    ratings = [r.get("rating") for r in rows if r.get("rating") in (1, 2, 3, 4, 5)]
    word_counts = [len(t.split()) for t in texts if t]
    char_counts = [len(t) for t in texts]
    has_reply_flags = [bool(r.get("reply") or r.get("replyContent")) for r in rows]
    thumbs = [int(r.get("thumbs_up", r.get("thumbsUpCount", 0)) or 0) for r in rows]

    resp = MetricsResponse(
        app_id=payload.app_id,
        count=len(ratings),
        avg_rating=average_rating(ratings),
        rating_distribution=rating_distribution(ratings),
        avg_word_count=round(sum(word_counts) / len(word_counts), 2) if word_counts else 0.0,
        avg_char_count=round(sum(char_counts) / len(char_counts), 2) if char_counts else 0.0,
        pct_with_reply=round(
            100.0 * sum(1 for x in has_reply_flags if x) / len(has_reply_flags), 2
        ) if has_reply_flags else 0.0,
        total_thumbs_up=sum(thumbs),
        avg_thumbs_up=round(sum(thumbs) / len(thumbs), 3) if thumbs else 0.0,
    )

    # Cache result
    set_cache(cache_key, resp, ttl_seconds=600)
    log.info("metrics_computed", extra={
        "avg_rating": resp.avg_rating,
        "review_count": resp.count,
        "cache_key": cache_key
    })

    return resp
