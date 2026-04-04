from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="anima-rfpar", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready", "note": "scaffold service"}
