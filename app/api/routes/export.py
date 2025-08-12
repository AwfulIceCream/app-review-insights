from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from io import StringIO
import csv
import json
import logging

from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields

router = APIRouter(prefix="/reviews", tags=["reviews"])
log = logging.getLogger(__name__)


@router.get("/export")
async def export_reviews(
        app_id: str = Query(..., description="Google Play app ID"),
        count: int = Query(100, ge=1, le=500),
        lang: str = Query("en"),
        country: str = Query("us"),
        format: str = Query("csv", regex="^(csv|jsonl)$"),
):
    # Start log
    log.info("export_reviews_start", extra={
        "app_id": app_id, "count": count, "lang": lang, "country": country, "format": format
    })

    # Collect
    try:
        raw = fetch_reviews(app_id=app_id, count=count, lang=lang, country=country)
    except CollectError as ce:
        log.warning("export_reviews_collect_error", extra={"app_id": app_id, "error": str(ce)})
        raise HTTPException(status_code=400, detail=str(ce))
    except Exception as e:
        log.exception("export_reviews_unexpected_collect_error", extra={"app_id": app_id})
        raise HTTPException(status_code=502, detail="Upstream collection failed")

    # Normalize
    try:
        rows = extract_fields(raw)
    except Exception:
        log.exception("export_reviews_extract_fields_failed", extra={"app_id": app_id})
        raise HTTPException(status_code=500, detail="Failed to normalize reviews")

    if not rows:
        log.info("export_reviews_no_content", extra={"app_id": app_id})
        raise HTTPException(status_code=404, detail="No reviews found")

    # Stream CSV
    if format == "csv":
        try:
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            output.seek(0)
            log.info("export_reviews_success", extra={
                "app_id": app_id, "exported": len(rows), "format": "csv"
            })
            return StreamingResponse(
                output,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={app_id}_reviews.csv"},
            )
        except Exception:
            log.exception("export_reviews_csv_stream_error", extra={"app_id": app_id})
            raise HTTPException(status_code=500, detail="Failed to build CSV response")

    # Stream JSONL
    try:
        output = StringIO()
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.seek(0)
        log.info("export_reviews_success", extra={
            "app_id": app_id, "exported": len(rows), "format": "jsonl"
        })
        return StreamingResponse(
            output,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={app_id}_reviews.jsonl"},
        )
    except Exception:
        log.exception("export_reviews_jsonl_stream_error", extra={"app_id": app_id})
        raise HTTPException(status_code=500, detail="Failed to build JSONL response")
