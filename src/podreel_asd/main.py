# podreel_asd/main.py
import modal
import numpy as np
from fastapi import FastAPI
import torch
from podreel_asd.asd_core.columbia_pipeline import run_pipeline
from podreel_asd.routes import detect
import pprint

modal_app = modal.App("PodReel-ASD")

app = FastAPI(title="ASD Service")
app.include_router(detect.router)


@app.get("/health")
def health():
    return {"status": "ok"}
