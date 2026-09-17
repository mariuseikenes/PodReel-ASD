# app/routes/detect.py

import subprocess

from podreel_asd.asd_core.columbia_pipeline import run_pipeline
from podreel_asd.models import DetectRequest, DetectResponse
from fastapi import APIRouter

router = APIRouter()


@router.post("/detect-active-faces", response_model=DetectResponse)
def detect_active_faces(req: DetectRequest):
    # TODO: Download clip and run tracking

    command = f'ffmpeg -i "{req.clip_url}" -c:v libx264 -preset slow -crf 22 "/tmp/{req.clip_id}.{req.clip_ext}"'
    subprocess.call(command, shell=True, stdout=None)
    run_pipeline(req.clip_id, "/tmp/", f"/tmp/{req.clip_id}")

    return DetectResponse(clip_id="stub", segments=[])
