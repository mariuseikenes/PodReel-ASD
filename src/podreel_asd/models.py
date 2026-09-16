from pydantic.main import BaseModel


class DetectRequest(BaseModel):
    clip_url: str  # Signed S3 Url


class SpeakerBox(BaseModel):
    tracklet_id: str
    x: float
    y: float
    w: float
    h: float
    confidence: float
    role: str  # speaking | reacting


class Segment(BaseModel):
    start: int
    end: int
    layout: str
    speakers: list[SpeakerBox]


class DetectResponse(BaseModel):
    clip_id: str
    segments: list[Segment]
