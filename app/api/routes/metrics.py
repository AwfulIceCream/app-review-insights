from fastapi import APIRouter, HTTPException
from app.api.schemas import MetricsRequest, MetricsResponse
from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields, average_rating, rating_distribution
from app.core.cache import get_cache, set_cache

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.post("", response_model=MetricsResponse)
async def compute_metrics(payload: MetricsRequest):
    cache_key = f"metrics:{payload.app_id}:{payload.lang}:{payload.country}:{payload.count}"
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
    scores = [r["rating"] for r in rows if r["rating"] in (1, 2, 3, 4, 5)]
    resp = MetricsResponse(
        app_id=payload.app_id,
        count=len(scores),
        avg_rating=average_rating(scores),
        rating_distribution=rating_distribution(scores),
    )
    set_cache(cache_key, resp, ttl_seconds=600)
    return resp
