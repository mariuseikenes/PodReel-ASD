# app/routes/detect.py

from enum import Enum
import shutil
import subprocess
import uuid
from typing import Optional
import httpx
import asyncio
import modal
from pydantic import BaseModel

from podreel_asd.asd_core.columbia_pipeline import run_pipeline
from podreel_asd.models import DetectRequest, DetectResponse
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pathlib import Path
from podreel_asd.services.format import format_detection
from podreel_asd.services.probe import probe_video

router = APIRouter()
MAX_CONCURRENT_JOBS = 1  # tune to your GPU capacity
pipeline_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class Job(BaseModel):
    status: JobStatus
    result: Optional[DetectResponse] = None
    error: Optional[str] = None


jobs: dict[str, Job] = {}

_run_pipeline_gpu = modal.Function.from_name("lr-asd-podreel", "run_pipeline_gpu")


async def process_clip(job_id: str, req: DetectRequest):
    print("Process clip start")
    async with pipeline_semaphore:
        jobs[job_id].status = JobStatus.running
        input_path = Path(f"/tmp/{req.clip_id}.mp4")
        print("Past the input_path declaration")
        # await execa`ffmpeg -ss ${clipRecord.startMs / 1000} -i "${presignedS3Url}" -t ${duration / 1000} -c copy ${outputPath} `;
        try:
            command = [
                "ffmpeg",
                "-y",
                "-i",
                req.clip_url,
                "-ss",
                str(req.start / 1000),
                "-t",
                str((req.end - req.start) / 1000),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(input_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

            print(f"Requested interval: {req.start}–{req.end} ms", flush=True)
            print(f"ffmpeg exit code: {result.returncode}", flush=True)
            print(f"ffmpeg stderr:\n{result.stderr}", flush=True)
            probe_video(input_path)
            print("After ffmpeg extract")
            video_bytes = input_path.read_bytes()
            print("Running function on Modal")
            segments = await _run_pipeline_gpu.remote.aio(req.clip_id, video_bytes)

            jobs[job_id].result = DetectResponse(frames=segments)
            jobs[job_id].status = JobStatus.done

        except Exception as e:
            print(Exception, e)
            jobs[job_id].status = JobStatus.failed
            jobs[job_id].error = str(e)

        finally:
            input_path.unlink(missing_ok=True)


@router.post("/detect-active-faces")
def detect_active_faces(req: DetectRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    print("Received request: " + req.clip_id)
    jobs[job_id] = Job(status=JobStatus.pending)
    background_tasks.add_task(process_clip, job_id, req)
    print("Returning response, background task added.")
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job
