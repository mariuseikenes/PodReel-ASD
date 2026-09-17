# app/routes/detect.py

from enum import Enum
import shutil
import subprocess
import uuid
from typing import Optional
import httpx
import asyncio
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


async def process_clip(job_id: str, req: DetectRequest):
    async with pipeline_semaphore:
        jobs[job_id].status = JobStatus.running
        input_path = Path(f"/tmp/{req.clip_id}.{req.clip_ext}")
        work_dir = Path(f"/tmp/{req.clip_id}")

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

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, run_pipeline, req.clip_id, "/tmp/", str(work_dir)
            )
            segments = format_detection(str(work_dir))

            jobs[job_id].result = DetectResponse(frames=segments)
            jobs[job_id].status = JobStatus.done

        except Exception as e:
            jobs[job_id].status = JobStatus.failed
            jobs[job_id].error = str(e)

        finally:
            input_path.unlink(missing_ok=True)
            shutil.rmtree(work_dir, ignore_errors=True)


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
