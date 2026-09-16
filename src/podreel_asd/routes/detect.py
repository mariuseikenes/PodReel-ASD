# app/routes/detect.py

from podreel_asd.models import DetectRequest, DetectResponse
from fastapi import APIRouter

router = APIRouter()


@router.post("/detect-active-faces", response_model=DetectResponse)
def detect_active_faces(req: DetectRequest):
    # TODO: Download clip and run tracking
    return DetectResponse(clip_id="stub", segments=[])
