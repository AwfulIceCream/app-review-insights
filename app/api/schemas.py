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