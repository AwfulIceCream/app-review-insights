import torch

from fastapi import APIRouter, HTTPException
from typing import List
from transformers import pipeline

from app.api.schemas import InsightsRequest, InsightsResponse
from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields
from app.nlp.keywords import extract_keywords_contrastive, merge_near_duplicate_terms

router = APIRouter(prefix="/insights", tags=["insights"])

# Load ML sentiment model once
device = 0 if torch.backends.mps.is_available() else -1
_sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device,
)


def get_sentiment(text: str) -> str:
    """
    Classify sentiment as positive / neutral / negative using the ML model.
    """
    result = _sentiment_model(text[:512])[0]
    return result["label"].lower()


def _brand_stopwords(app_id: str) -> set[str]:
    """Extract brand-specific stopwords from app_id."""
    parts = app_id.replace("-", ".").split(".")
    return {p.lower() for p in parts if p}


@router.post("", response_model=InsightsResponse)
async def generate_insights(payload: InsightsRequest):
    """
    Generate insights from app reviews:
      1. Classify each review as positive, neutral, or negative (ML model).
      2. Extract most common keywords/phrases from negative reviews only.
      3. Return sentiment distribution + actionable insights.
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

    # Extract and clean
    rows = extract_fields(raw)
    rows = [r for r in rows if r.get("text") and r["text"].strip()]
    if not rows:
        return InsightsResponse(
            sentiment_distribution={"positive": 0, "neutral": 0, "negative": 0},
            top_negative_keywords=[],
            actionable_insights=[]
        )

    # Classify all reviews
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

    # No negative reviews → no insights
    if not negative_texts:
        return InsightsResponse(
            sentiment_distribution=sentiment_distribution,
            top_negative_keywords=[],
            actionable_insights=[]
        )

    # Extract keywords from negatives
    picked = extract_keywords_contrastive(
        negatives=negative_texts,
        all_texts=all_texts,
        top_n=7,
        app_id=payload.app_id,
        ngram_range=(1, 3),
        max_doc_ratio=0.6,
        return_evidence=True
    )

    # Merge near-duplicate terms
    picked = merge_near_duplicate_terms(picked, threshold=0.5)

    # Build actionable insights
    actionable_insights = [
        {
            "issue": term,
            "evidence_examples": [negative_texts[i] for i in idxs[:2]],
            "suggestion": f"Investigate and address '{term}' issues reported by users."
        }
        for term, idxs in picked
    ]

    return InsightsResponse(
        sentiment_distribution=sentiment_distribution,
        top_negative_keywords=[term for term, _ in picked],
        actionable_insights=actionable_insights
    )
