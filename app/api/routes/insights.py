import torch
import logging
from fastapi import APIRouter, HTTPException
from typing import List
from transformers import pipeline

from app.api.schemas import InsightsRequest, InsightsResponse
from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields
from app.nlp.keywords import extract_keywords_contrastive, merge_near_duplicate_terms

router = APIRouter(prefix="/insights", tags=["insights"])
log = logging.getLogger(__name__)

# Load ML sentiment model once
device = 0 if torch.backends.mps.is_available() else -1
_sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device,
)


def get_sentiment(text: str) -> str:
    """Classify sentiment as positive / neutral / negative using the ML model."""
    result = _sentiment_model(text[:512])[0]
    return result["label"].lower()


def _brand_stopwords(app_id: str) -> set[str]:
    """Extract brand-specific stopwords from app_id."""
    parts = app_id.replace("-", ".").split(".")
    return {p.lower() for p in parts if p}


@router.post("", response_model=InsightsResponse, response_model_exclude_none=True)
async def generate_insights(payload: InsightsRequest):
    log.info("insights_generation_start", extra=payload.dict())

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
    except Exception as e:
        log.exception("unexpected_collect_error")
        raise HTTPException(status_code=502, detail="Failed to collect reviews")

    # Extract and clean
    try:
        rows = extract_fields(raw)
    except Exception:
        log.exception("extract_fields_failed")
        raise HTTPException(status_code=500, detail="Failed to normalize reviews")

    rows = [r for r in rows if r.get("text") and r["text"].strip()]
    if not rows:
        log.info("no_reviews_after_cleaning")
        return InsightsResponse(sentiment_distribution={"positive": 0, "neutral": 0, "negative": 0})

    # Classify reviews
    sentiment_distribution = {"positive": 0, "neutral": 0, "negative": 0}
    negative_texts: List[str] = []
    all_texts: List[str] = []

    for r in rows:
        text = r["text"].strip()
        label = get_sentiment(text)
        sentiment_distribution[label] += 1
        all_texts.append(text)
        if label == "negative":
            negative_texts.append(text)

    log.info("sentiment_classification_done", extra={
        "distribution": sentiment_distribution,
        "negatives": len(negative_texts)
    })

    # No negatives → no insights
    if not negative_texts:
        log.info("no_negative_reviews")
        return InsightsResponse(sentiment_distribution=sentiment_distribution)

    # Extract keywords from negatives
    try:
        picked = extract_keywords_contrastive(
            negatives=negative_texts,
            all_texts=all_texts,
            top_n=7,
            app_id=payload.app_id,
            ngram_range=(1, 3),
            max_doc_ratio=0.6,
            return_evidence=True
        )
        log.info("keywords_extracted", extra={"keywords_count": len(picked)})
    except Exception:
        log.exception("keyword_extraction_failed")
        raise HTTPException(status_code=500, detail="Failed to extract keywords")

    # Merge near duplicates
    picked = merge_near_duplicate_terms(picked, threshold=0.5)
    log.info("keywords_deduplicated", extra={"final_keywords": len(picked)})

    # Build actionable insights
    actionable_insights = [
        {
            "issue": term,
            "evidence_examples": [negative_texts[i] for i in idxs[:2]],
            "suggestion": f"Investigate and address '{term}' issues reported by users."
        }
        for term, idxs in picked
    ]
    log.info("actionable_insights_ready", extra={"count": len(actionable_insights)})

    return InsightsResponse(
        sentiment_distribution=sentiment_distribution,
        top_negative_keywords=[term for term, _ in picked],
        actionable_insights=actionable_insights
    )
