import subprocess
from pathlib import Path


def probe_video(path: Path) -> None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "format=duration,size:stream=codec_type,codec_name,width,height,"
            "duration,nb_frames,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )

    print(f"Probe path: {path}", flush=True)
    print(f"File size: {path.stat().st_size} bytes", flush=True)
    print(f"ffprobe exit code: {result.returncode}", flush=True)
    print(result.stdout or result.stderr, flush=True)
