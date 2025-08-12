from fastapi import APIRouter, HTTPException
from typing import List
from transformers import pipeline

from app.api.schemas import InsightsRequest, InsightsResponse
from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields
from app.nlp.keywords import extract_keywords_contrastive, merge_near_duplicate_terms

router = APIRouter(prefix="/insights", tags=["insights"])

# Load ML sentiment model once
_sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


def is_negative_review(text: str, rating: int | None = None) -> bool:
    """
    Decide if a review is negative using ML model.
    """
    # ML fallback
    result = _sentiment_model(text[:512])[0]
    if result["label"] == "negative" and rating > 2:
        print(rating, result["label"], text)
    return "negative" in result["label"].lower()


def _brand_stopwords(app_id: str) -> set[str]:
    """Extract brand-specific stopwords from app_id."""
    parts = app_id.replace("-", ".").split(".")
    return {p.lower() for p in parts if p}


@router.post("", response_model=InsightsResponse)
async def generate_insights(payload: InsightsRequest):
    """
    Minimal version:
      1. Use ML model (or rating) to keep only negative reviews.
      2. Extract most common keywords/phrases in those reviews.
    """
    try:
        raw = fetch_reviews(
            app_id=payload.app_id,
            count=payload.count,
            lang=payload.lang,
            country=payload.country,
        )
    except CollectError as ce:
        raise HTTPException(status_code=400, detail=str(ce))

    # Clean and normalize
    rows = extract_fields(raw)
    rows = [r for r in rows if r.get("text") and r["text"].strip()]

    if not rows:
        return InsightsResponse(
            sentiment_distribution={"positive": 0, "neutral": 0, "negative": 0},
            top_negative_keywords=[],
            actionable_insights=[]
        )

    # Keep only negative reviews
    negative_texts: List[str] = [
        r["text"].strip()
        for r in rows
        if is_negative_review(r["text"], r.get("rating"))
    ]

    if not negative_texts:
        return InsightsResponse(
            sentiment_distribution={"positive": 0, "neutral": 0, "negative": 0},
            top_negative_keywords=[],
            actionable_insights=[]
        )

    # Extract top keywords from negatives
    picked = extract_keywords_contrastive(
        negatives=negative_texts,
        all_texts=[r["text"] for r in rows],
        top_n=7,
        app_id=payload.app_id,
        ngram_range=(1, 3),
        max_doc_ratio=0.6,
        return_evidence=True
    )
    picked = merge_near_duplicate_terms(picked, threshold=0.5)

    actionable_insights = [
        {
            "issue": term,
            "evidence_examples": [negative_texts[i] for i in idxs[:2]],
            "suggestion": f"Investigate and address '{term}' issues reported by users."
        }
        for term, idxs in picked
    ]

    return InsightsResponse(
        sentiment_distribution={
            "positive": 0,
            "neutral": 0,
            "negative": len(negative_texts)
        },
        top_negative_keywords=[term for term, _ in picked],
        actionable_insights=actionable_insights
    )
