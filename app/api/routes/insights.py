from fastapi import APIRouter, HTTPException
from typing import List, Dict

from app.api.schemas import InsightsRequest, InsightsResponse, InsightItem
from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields
from app.nlp.sentiment import analyze_sentiment
from app.nlp.keywords import extract_keywords_contrastive
from app.nlp.topics import make_actionable_insights

router = APIRouter(prefix="/insights", tags=["insights"])


def _brand_stopwords(app_id: str) -> set[str]:
    """Extract brand-specific stopwords from app_id."""
    parts = app_id.replace("-", ".").split(".")
    return {p.lower() for p in parts if p}


@router.post("", response_model=InsightsResponse)
async def generate_insights(payload: InsightsRequest):
    """
    Generate insights for an app:
      - Sentiment distribution (rating first, fallback to text sentiment)
      - Top negative keywords (contrastive)
      - Actionable insights from topic clusters
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

    # Normalize and clean
    rows = extract_fields(raw)
    rows = [r for r in rows if r.get("text") and r["text"].strip()]

    if not rows:
        return InsightsResponse(
            sentiment_distribution={"positive": 0, "neutral": 0, "negative": 0},
            top_negative_keywords=[],
            actionable_insights=[],
        )

    # Sentiment classification (hybrid: rating → text)
    texts: List[str] = []
    labels: List[str] = []
    for r in rows:
        text = r["text"].strip()
        rating = r.get("rating")
        label = analyze_sentiment(text, rating)
        texts.append(text)
        labels.append(label)

    # Sentiment distribution
    dist: Dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
    for label in labels:
        if label in dist:
            dist[label] += 1

    # Focus only on negative reviews
    negative_texts = [t for t, label in zip(texts, labels) if label == "negative"]

    # Extract top negative keywords
    top_neg_keywords: List[str] = []
    if len(negative_texts) >= 2:
        top_neg_keywords = extract_keywords_contrastive(
            negatives=negative_texts,
            all_texts=texts,
            top_n=5,
            min_doc_neg=2,
            extra_stopwords=_brand_stopwords(payload.app_id),
        )

    negative_texts = [
        r["text"] for r in rows
        if analyze_sentiment(r["text"], r.get("rating")) == "negative"
    ]

    # Pass only negatives to actionable insights
    topic_items = make_actionable_insights(
        negative_texts=negative_texts,
        all_texts=texts,  # still used for contrastive analysis
        min_cluster_size=2,
        validate_sentiment=True  # optional param to skip non-negative clusters
    )
    actionable = [
        InsightItem(
            issue=item["issue"],
            evidence_examples=item["evidence_examples"],
            suggestion=item["suggestion"],
        )
        for item in topic_items
    ]

    return InsightsResponse(
        sentiment_distribution=dist,
        top_negative_keywords=top_neg_keywords,
        actionable_insights=actionable,
    )
