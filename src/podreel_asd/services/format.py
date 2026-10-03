from pathlib import Path
import pickle

import numpy as np


def format_detection(folder: str, total_frames: int) -> list[dict]:
    work = Path(folder) / "pywork"

    with (work / "tracks.pckl").open("rb") as f:
        tracks = pickle.load(f)

    with (work / "scores.pckl").open("rb") as f:
        scores = pickle.load(f)

    if len(tracks) != len(scores):
        raise ValueError(
            f"Expected one score array per track: "
            f"{len(tracks)} tracks, {len(scores)} score arrays"
        )

    formatted_frames = [
        {"frame": frame, "detections": []} for frame in range(total_frames)
    ]
    for track_id, (track, track_scores) in enumerate(zip(tracks, scores, strict=True)):
        frames = np.asarray(track["track"]["frame"])
        if len(frames) != len(track_scores):
            print(
                f"Track {track_id}: frames={len(frames)}, "
                f"scores={len(track_scores)}, "
                f"first_frame={frames[0] if len(frames) else None}, "
                f"last_frame={frames[-1] if len(frames) else None}"
            )
    for track_id, (track, scores_for_track) in enumerate(
        zip(tracks, scores, strict=True)
    ):
        frames = np.asarray(track["track"]["frame"])
        x = np.asarray(track["proc_track"]["x"])
        y = np.asarray(track["proc_track"]["y"])
        s = np.asarray(track["proc_track"]["s"])
        values = np.asarray(scores_for_track)

        track_length = len(frames)
        lengths = (track_length, len(x), len(y), len(s))
        missing_scores = track_length - len(values)

        if len(set(lengths)) != 1 or not 0 <= missing_scores <= 2:
            raise ValueError(
                f"Track {track_id} has misaligned arrays: " f"{(*lengths, len(values))}"
            )

        for index, source_frame in enumerate(frames[: len(values)]):
            frame = int(source_frame)
            if not 0 <= frame < total_frames:
                raise ValueError(
                    f"Track {track_id} contains out-of-range frame {frame}"
                )

            formatted_frames[frame]["detections"].append(
                {
                    "trackId": track_id,
                    "score": float(values[index]),
                    "proc": {
                        "x": float(x[index]),
                        "y": float(y[index]),
                        "s": float(s[index]),
                    },
                }
            )

    return formatted_frames
