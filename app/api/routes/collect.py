from fastapi import APIRouter, HTTPException
import logging

from app.api.schemas import CollectRequest, CollectResponse
from app.collectors.google_play import fetch_reviews, CollectError

router = APIRouter(prefix="/collect", tags=["collect"])
log = logging.getLogger(__name__)


@router.post("", response_model=CollectResponse, response_model_exclude_none=True)
async def collect_reviews(payload: CollectRequest):
    """
    POST /collect
    - Fetch reviews from Google Play
    - Logs key steps and failures
    - Maps known errors to HTTP 400
    - Excludes null fields from response
    """
    log.info("collect_reviews_start", extra={
        "app_id": payload.app_id,
        "count": payload.count,
        "lang": payload.lang,
        "country": payload.country
    })

    try:
        data = fetch_reviews(
            app_id=payload.app_id,
            count=payload.count,
            lang=payload.lang,
            country=payload.country,
        )
        log.info("collect_reviews_success", extra={
            "app_id": payload.app_id,
            "fetched_count": len(data)
        })
        return CollectResponse(app_id=payload.app_id, count=len(data), reviews=data)

    except CollectError as ce:
        log.warning("collect_reviews_failed", extra={
            "app_id": payload.app_id,
            "error": str(ce)
        })
        raise HTTPException(status_code=400, detail=str(ce))

    except Exception as e:
        log.exception("collect_reviews_unexpected_error", extra={
            "app_id": payload.app_id
        })
        raise HTTPException(status_code=500, detail="Internal server error")
