# podreel_asd/main.py
import modal
from fastapi import FastAPI
from podreel_asd.asd_core.columbia_pipeline import run_pipeline
from podreel_asd.routes import detect

modal_app = modal.App("PodReel-ASD")

app = FastAPI(title="ASD Service")
app.include_router(detect.router)


@app.get("/health")
def health():
    return {"status": "ok"}
