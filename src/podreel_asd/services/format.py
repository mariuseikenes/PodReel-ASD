import numpy as np
import pickle


def format_detection(folder: str):
    with open(folder + "/pywork/tracks.pckl", "rb") as f:
        tracks = pickle.load(f)

    with open(folder + "/pywork/scores.pckl", "rb") as f:
        scores = pickle.load(f)

    first_frame = 0
    end_frame = tracks[-1]["track"]["frame"][-1]
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

            conf = float(score[idx])
            detections.append({"conf": conf, "proc": {"x": x, "y": y, "s": s}})

        formatted_frames.append({"frame": frame, "detections": detections})

    return formatted_frames
