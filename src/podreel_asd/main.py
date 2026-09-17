# podreel_asd/main.py
import numpy as np
from fastapi import FastAPI
from podreel_asd.asd_core.columbia_pipeline import run_pipeline
from podreel_asd.routes import detect
import pprint

app = FastAPI(title="ASD Service")
app.include_router(detect.router)


@app.get("/health")
def health():
    return {"status": "ok"}


# run_pipeline("test-clip", ".", "./output")
import pickle

with open("output/pywork/tracks.pckl", "rb") as f:
    tracks = pickle.load(f)

with open("output/pywork/scores.pckl", "rb") as f:
    scores = pickle.load(f)

"""
Ønska format:
{
    frame: int;
    detections: {
        conf: float;
        proc: {
        x: float;
        y: float;
        s: float;
        }
    }[]
}
"""

first_frame = 0
end_frame = tracks[-1]["track"]["frame"][-1]
CROP_SCALE = 0.4
print(f"{first_frame = }, {end_frame = }")
formatted_frames = []

for frame in range(first_frame, end_frame + 1):
    detections = []
    for track, score in zip(tracks, scores):
        track_frames = track["track"]["frame"]
        if frame not in track_frames:
            continue

        idx = np.where(track_frames == frame)[0][0]
        if idx >= len(score):
            continue

        x = track["proc_track"]["x"][idx]
        y = track["proc_track"]["y"][idx]
        s = track["proc_track"]["s"][idx]
        pprint.pp(score)
        pprint.pp(track["proc_track"]["x"])
        conf = float(score[idx])

        detections.append({"conf": conf, "proc": {"x": x, "y": y, "s": s}})

    formatted_frames.append({"frame": frame, "detections": detections})

pprint.pp(formatted_frames)
