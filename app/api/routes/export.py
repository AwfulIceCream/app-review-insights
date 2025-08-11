from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from io import StringIO
import csv
import json

from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("/export")
async def export_reviews(
        app_id: str = Query(..., description="Google Play app ID"),
        count: int = Query(100, ge=1, le=500),
        lang: str = Query("en"),
        country: str = Query("us"),
        format: str = Query("csv", regex="^(csv|jsonl)$")
):
    """
    Export raw normalized reviews in CSV or JSONL format.
    """
    try:
        raw = fetch_reviews(app_id=app_id, count=count, lang=lang, country=country)
    except CollectError as ce:
        raise HTTPException(status_code=400, detail=str(ce))

    rows = extract_fields(raw)
    if not rows:
        raise HTTPException(status_code=404, detail="No reviews found")

    if format == "csv":
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={app_id}_reviews.csv"}
        )

    elif format == "jsonl":
        output = StringIO()
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={app_id}_reviews.jsonl"}
        )
