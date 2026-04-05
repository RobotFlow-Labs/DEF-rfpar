"""RFPAR FastAPI service — adversarial pixel perturbation inference."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from ..serve import RFPARNode

node = RFPARNode()
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = os.environ.get("ANIMA_WEIGHT_DIR", "/data/weights") + "/best.pth"
    node.setup_inference(ckpt)
    yield


app = FastAPI(title="anima-rfpar", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "module": "DEF-rfpar",
        "uptime_s": round(time.time() - _start_time, 1),
    }


@app.get("/ready")
def ready() -> dict:
    if not node.weights_loaded:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "module": "DEF-rfpar", "weights_loaded": False},
        )
    return {
        "ready": True,
        "module": "DEF-rfpar",
        "version": "0.1.0",
        "weights_loaded": True,
    }


@app.get("/info")
def info() -> dict:
    return {
        "module": "DEF-rfpar",
        "version": "0.1.0",
        "description": "RFPAR: Remember and Forget Pixel Attack using RL",
        "paper": "ArXiv 2502.07821",
        "capabilities": ["classification_attack", "detection_attack"],
        **node.get_status(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:  # noqa: B008
    image_bytes = await file.read()
    return node.predict(image_bytes)
