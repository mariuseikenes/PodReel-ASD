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
    async with pipeline_semaphore:
        jobs[job_id].status = JobStatus.running
        input_path = Path(f"/tmp/{req.clip_id}.{req.clip_ext}")

        try:
            command = [
                "ffmpeg",
                "-i",
                req.clip_url,
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "22",
                str(input_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

            video_bytes = input_path.read_bytes()
            print("Running function on Modal")
            segments = await _run_pipeline_gpu.remote.aio(req.clip_id, video_bytes)

            jobs[job_id].result = DetectResponse(frames=segments)
            jobs[job_id].status = JobStatus.done

        except Exception as e:
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
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job
