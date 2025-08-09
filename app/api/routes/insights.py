from fastapi import APIRouter, HTTPException
from app.api.schemas import InsightsRequest, InsightsResponse, InsightItem
from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields
from app.nlp.sentiment import sentiment_distribution, analyze_sentiment
from app.nlp.keywords import extract_keywords

router = APIRouter(prefix="/insights", tags=["insights"])


@router.post("", response_model=InsightsResponse)
async def generate_insights(payload: InsightsRequest):
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
    texts = [r["text"] for r in rows if r["text"]]
    dist = sentiment_distribution(texts)

    # Find top negative keywords
    negative_texts = [t for t in texts if analyze_sentiment(t) == "negative"]
    top_neg_keywords = extract_keywords(negative_texts, top_n=5)

    # Very simple actionable insights from keywords
    actionable = []
    for kw in top_neg_keywords:
        examples = [t for t in negative_texts if kw in t.lower()][:2]
        actionable.append(
            InsightItem(
                issue=f"Issue related to '{kw}'",
                evidence_examples=examples,
                suggestion=f"Investigate and address '{kw}' issues reported by users."
            )
        )

    return InsightsResponse(
        sentiment_distribution=dist,
        top_negative_keywords=top_neg_keywords,
        actionable_insights=actionable
    )
