from fastapi import APIRouter, HTTPException
from app.api.schemas import CollectRequest, CollectResponse
from app.collectors.google_play import fetch_reviews, CollectError

router = APIRouter(prefix="/collect", tags=["collect"])

@router.post("", response_model=CollectResponse)
async def collect_reviews(payload: CollectRequest):
    try:
        data = fetch_reviews(
            app_id=payload.app_id,
            count=payload.count,
            lang=payload.lang,
            country=payload.country,
        )
        # Optional: sample/shuffle here if you want "random"
        return CollectResponse(app_id=payload.app_id, count=len(data), reviews=data)
    except CollectError as ce:
        raise HTTPException(status_code=400, detail=str(ce))
