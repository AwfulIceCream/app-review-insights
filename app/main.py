from fastapi import FastAPI
from app.api.routes.collect import router as collect_router
from app.api.routes.metrics import router as metrics_router

app = FastAPI(title="App Review Insights API")


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}


app.include_router(collect_router)
app.include_router(metrics_router)
