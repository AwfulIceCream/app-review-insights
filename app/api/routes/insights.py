from fastapi import APIRouter, HTTPException
from app.api.schemas import InsightsRequest, InsightsResponse, InsightItem
from app.collectors.google_play import fetch_reviews, CollectError
from app.utils.metrics import extract_fields
from app.nlp.sentiment import sentiment_distribution, analyze_sentiment
from app.nlp.keywords import extract_keywords

router = APIRouter(prefix="/insights", tags=["insights"])


def _suggestion_for(keyword: str) -> str:
    k = keyword.lower()
    if "call" in k:
        return "Investigate call connectivity and VOIP stability (Wi‑Fi/BT headset edge cases)."
    if "account" in k or "block" in k or "ban" in k:
        return "Review account access/ban triggers, improve in‑app guidance and appeal flow."
    if "open" in k or "launch" in k or "crash" in k:
        return "Fix app start/launch reliability, add safe‑mode fallback and crash telemetry."
    if "login" in k or "otp" in k or "verify" in k:
        return "Harden login/OTP/verification flows; reduce friction and transient failures."
    return f"Investigate and address '{keyword}' issues reported by users."


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
    negative_texts = [t for t in texts if analyze_sentiment(t) == "negative"]

    top_neg_keywords = extract_keywords(negative_texts, top_n=5)

    actionable = []
    for kw in top_neg_keywords:
        kw_l = kw.lower()
        examples = [t for t in negative_texts if kw_l in t.lower()][:2]
        actionable.append(
            InsightItem(
                issue=kw.capitalize(),
                evidence_examples=examples,
                suggestion=_suggestion_for(kw),
            )
        )

    return InsightsResponse(
        sentiment_distribution=dist,
        top_negative_keywords=top_neg_keywords,
        actionable_insights=actionable
    )
