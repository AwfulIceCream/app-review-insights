from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CollectRequest(BaseModel):
    app_id: str = Field(..., examples=["com.whatsapp"])
    count: int = Field(100, ge=1, le=1000)
    lang: str = Field("en", min_length=2, max_length=5, description="ISO language")
    country: str = Field("us", min_length=2, max_length=2, description="ISO country")


class CollectResponse(BaseModel):
    app_id: str
    count: int
    cached: bool = False  # placeholder for later caching
    reviews: list


class MetricsRequest(BaseModel):
    app_id: str
    count: int = 100
    lang: str = "en"
    country: str = "us"


class MetricsResponse(BaseModel):
    app_id: str
    count: int
    avg_rating: Optional[float]
    rating_distribution: Dict[int, int]
    avg_word_count: float
    avg_char_count: float
    pct_with_reply: float  # 0..100
    total_thumbs_up: int
    avg_thumbs_up: float


class InsightsRequest(BaseModel):
    app_id: str = Field(..., examples=["com.whatsapp"])
    count: int = Field(100, ge=1, le=1000)
    lang: str = Field("en", min_length=2, max_length=5)
    country: str = Field("us", min_length=2, max_length=2)


class InsightItem(BaseModel):
    issue: str
    evidence_examples: List[str]
    suggestion: str


class InsightsResponse(BaseModel):
    sentiment_distribution: Dict[str, int]
    top_negative_keywords: List[str]
    actionable_insights: List[InsightItem]
