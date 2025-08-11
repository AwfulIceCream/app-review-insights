from fastapi import FastAPI

from app.api.routes.export import router as export_router
from app.api.routes.collect import router as collect_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.insights import router as insights_router

app = FastAPI(title="App Review Insights API")


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}


app.include_router(collect_router)
app.include_router(metrics_router)
app.include_router(insights_router)
app.include_router(export_router)
