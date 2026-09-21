# modal_app.py
import modal

app = modal.App("lr-asd-podreel")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsm6", "libxext6")  # cv2/av deps
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "tqdm",
        "opencv-python-headless",
        "scipy",
        "scikit-learn",
        "gdown",
        "httpx",
        "mediapipe",
        "pandas",
        "python_speech_features",  # whatever LR-ASD needs
    )
    .add_local_dir("src/podreel_asd", remote_path="/root/podreel_asd")
)

# Bake/download model weights into the image once, at build time,
# instead of on every cold start.
weights_vol = modal.Volume.from_name("lr-asd-weights", create_if_missing=True)


@app.function(image=image, gpu="L4", timeout=600, scaledown_window=60)
def run_pipeline_gpu(clip_id: str, video_bytes: bytes) -> list:
    print("Running Pipeling on Modal GPU")
    import sys, pathlib, tempfile

    sys.path.insert(0, "/root")
    from podreel_asd.asd_core.columbia_pipeline import run_pipeline
    from podreel_asd.services.format import format_detection

    tmp_dir = tempfile.mkdtemp()
    work_dir = pathlib.Path(tmp_dir) / clip_id
    work_dir.mkdir(parents=True)

    input_path = pathlib.Path(tmp_dir) / f"{clip_id}.mp4"
    input_path.write_bytes(video_bytes)

    run_pipeline(clip_id, tmp_dir, str(work_dir))
    print("Returning formatted detection")
    return format_detection(str(work_dir))
