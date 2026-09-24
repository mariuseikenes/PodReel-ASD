from pydantic.main import BaseModel


class DetectRequest(BaseModel):
    clip_url: str  # Signed S3 Url
    clip_id: str
    clip_ext: str
    start: int
    end: int


class Proc(BaseModel):
    x: float
    y: float
    s: float


class Detection(BaseModel):
    conf: float
    proc: Proc


class DetectedFrame(BaseModel):
    frame: int
    detections: list[Detection]


class DetectResponse(BaseModel):
    frames: list[DetectedFrame]
