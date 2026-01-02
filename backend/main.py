"""
Face Blur API - FastAPI Backend
Automatic face detection and blurring for videos
Optimized for Oracle Cloud ARM deployment
"""

import os
import uuid
import asyncio
import shutil
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import aiofiles

from processor import VideoProcessor

# Configuration
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MAX_DURATION_SECONDS = 300  # 5 minutes
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi'}
TEMP_DIR = Path('/tmp/face-blur')
CLEANUP_AFTER_HOURS = 1
MAX_REQUESTS_PER_HOUR = 10

# Rate limiting storage (in production, use Redis)
rate_limit_store: Dict[str, list] = {}

# Job storage
jobs: Dict[str, dict] = {}

# Create temp directory
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Face Blur API",
    description="Automatic face detection and blurring for videos",
    version="1.0.0"
)

# CORS configuration - Update with your Cloudflare Pages domain
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    # Add your Cloudflare Pages domain here:
    # "https://your-app.pages.dev",
    # "https://your-custom-domain.com",
]

# Allow all origins in development (update for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ALLOWED_ORIGINS in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize video processor
processor = VideoProcessor()


def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit."""
    now = datetime.now()
    hour_ago = now - timedelta(hours=1)
    
    if client_ip not in rate_limit_store:
        rate_limit_store[client_ip] = []
    
    # Remove old entries
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip] if t > hour_ago
    ]
    
    if len(rate_limit_store[client_ip]) >= MAX_REQUESTS_PER_HOUR:
        return False
    
    rate_limit_store[client_ip].append(now)
    return True


def validate_file_extension(filename: str) -> bool:
    """Validate file extension."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


async def cleanup_old_jobs():
    """Clean up jobs and files older than CLEANUP_AFTER_HOURS."""
    now = datetime.now()
    cutoff = now - timedelta(hours=CLEANUP_AFTER_HOURS)
    
    jobs_to_remove = []
    for job_id, job_data in jobs.items():
        if job_data.get('created_at', now) < cutoff:
            jobs_to_remove.append(job_id)
    
    for job_id in jobs_to_remove:
        job_dir = TEMP_DIR / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        jobs.pop(job_id, None)
    
    # Also clean orphaned directories
    if TEMP_DIR.exists():
        for item in TEMP_DIR.iterdir():
            if item.is_dir():
                try:
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    if mtime < cutoff:
                        shutil.rmtree(item, ignore_errors=True)
                except Exception:
                    pass


async def process_video_task(job_id: str, input_path: Path, output_path: Path):
    """Background task to process video."""
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 0
        
        # Process video with progress callback
        def progress_callback(progress: int):
            jobs[job_id]['progress'] = progress
        
        success = await asyncio.to_thread(
            processor.process_video,
            str(input_path),
            str(output_path),
            progress_callback
        )
        
        if success and output_path.exists():
            jobs[job_id]['status'] = 'complete'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['output_path'] = str(output_path)
        else:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = 'Processing failed'
            
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)


@app.on_event("startup")
async def startup_event():
    """Startup event - clean old jobs."""
    await cleanup_old_jobs()


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - cleanup."""
    pass


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Face Blur API",
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health_check():
    """API health check."""
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a video file for face blurring.
    
    Returns a job_id for tracking progress.
    """
    # Get client IP for rate limiting
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 10 uploads per hour."
        )
    
    # Validate file extension
    if not file.filename or not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine file extension
    ext = Path(file.filename).suffix.lower()
    input_path = job_dir / f"input{ext}"
    output_path = job_dir / "output.mp4"
    
    # Save uploaded file with size checking
    try:
        total_size = 0
        async with aiofiles.open(input_path, 'wb') as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    await f.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
                    )
                await f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Validate video duration
    try:
        duration = processor.get_video_duration(str(input_path))
        if duration > MAX_DURATION_SECONDS:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Video too long. Maximum duration: {MAX_DURATION_SECONDS // 60} minutes"
            )
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Invalid video file: {str(e)}")
    
    # Create job entry
    jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'created_at': datetime.now(),
        'input_path': str(input_path),
        'output_path': None,
        'error': None
    }
    
    # Start background processing
    background_tasks.add_task(process_video_task, job_id, input_path, output_path)
    
    # Cleanup old jobs periodically
    background_tasks.add_task(cleanup_old_jobs)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video uploaded successfully. Processing started."
    }


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """
    Get processing status for a job.
    
    Returns status (queued/processing/complete/error) and progress (0-100).
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job['status'],
        "progress": job['progress']
    }
    
    if job['status'] == 'error':
        response['error'] = job.get('error', 'Unknown error')
    
    return response


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    """
    Download the processed video.
    
    Only available when status is 'complete'.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job['status'] != 'complete':
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready. Current status: {job['status']}"
        )
    
    output_path = job.get('output_path')
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"blurred_video_{job_id[:8]}.mp4",
        headers={
            "Content-Disposition": f'attachment; filename="blurred_video_{job_id[:8]}.mp4"'
        }
    )


@app.post("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """
    Clean up temporary files for a job.
    
    Call this after successful download.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dir = TEMP_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    
    jobs.pop(job_id, None)
    
    return {"message": "Cleanup successful", "job_id": job_id}


@app.get("/api/jobs")
async def list_jobs():
    """
    List all active jobs (for debugging).
    Returns summary without sensitive paths.
    """
    return {
        "total_jobs": len(jobs),
        "jobs": [
            {
                "job_id": jid,
                "status": jdata['status'],
                "progress": jdata['progress'],
                "created_at": jdata['created_at'].isoformat()
            }
            for jid, jdata in jobs.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
