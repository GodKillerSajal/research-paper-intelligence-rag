from fastapi import APIRouter
import time

router = APIRouter(tags=["health"])

_START_TIME = time.time()


@router.get("/")
def root():
    return {"status": "running", "uptime_seconds": round(time.time() - _START_TIME, 1)}


@router.get("/health")
def health():
    return {"status": "healthy", "uptime_seconds": round(time.time() - _START_TIME, 1)}
