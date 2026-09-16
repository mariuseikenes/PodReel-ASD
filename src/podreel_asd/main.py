# podreel_asd/main.py
from fastapi import FastAPI
from podreel_asd.routes import detect

app = FastAPI(title="ASD Service")
app.include_router(detect.router)


@app.get("/health")
def health():
    return {"status": "ok"}
