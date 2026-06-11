from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from face_tracker import (
    DEFAULT_RECOGNITION_THRESHOLD,
    Detection,
    KNOWN_FACES_DIR,
    CentroidTracker,
    detect_faces,
    ensure_dirs,
    identify,
    load_cascade,
    load_recognizer,
    normalize_face,
    slug_name,
    train as train_identifier,
)

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(title="Face Tracker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = load_cascade()
recognizer, labels = load_recognizer()
trackers: dict[str, tuple[CentroidTracker, float]] = {}
TRACKER_TTL = 120.0


def reload_model() -> dict:
    global recognizer, labels
    recognizer, labels = load_recognizer()
    if recognizer is None:
        return {"loaded": False, "people": 0}
    return {"loaded": True, "people": len(labels)}


def decode_image(data: bytes, grayscale: bool = False) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(arr, flag)
    return image


def get_tracker(session_id: str) -> CentroidTracker:
    now = time.time()
    stale = [sid for sid, (_, seen) in trackers.items() if now - seen > TRACKER_TTL]
    for sid in stale:
        del trackers[sid]

    if session_id not in trackers:
        trackers[session_id] = (CentroidTracker(), now)
    else:
        tracker, _ = trackers[session_id]
        trackers[session_id] = (tracker, now)
    return trackers[session_id][0]


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content=None, status_code=204)


@app.get("/api/model")
async def model_info():
    info = reload_model()
    return {
        **info,
        "threshold_default": DEFAULT_RECOGNITION_THRESHOLD,
    }


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
    min_face: int = 80,
    session_id: str = Form(""),
):
    data = await file.read()
    frame = decode_image(data)
    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_size = (max(1, min_face), max(1, min_face))
    faces = detect_faces(gray, detector, min_size=min_size)

    detections: list[Detection] = []
    for box in faces:
        name, confidence = identify(gray, box, recognizer, labels, threshold)
        detections.append(Detection(box=box, name=name, confidence=confidence))

    sid = session_id.strip() or str(uuid.uuid4())
    tracker = get_tracker(sid)
    tracks = tracker.update(detections)

    results = []
    for track in tracks:
        if track.missed != 0:
            continue
        x, y, w, h = track.box
        results.append(
            {
                "track_id": track.track_id,
                "box": [x, y, w, h],
                "name": track.name,
                "confidence": track.confidence,
            }
        )

    return JSONResponse({
        "session_id": sid,
        "faces": len(faces),
        "tracks": results,
    })


@app.post("/api/enroll/{name}")
async def enroll_frame(name: str, file: UploadFile = File(...), min_face: int = 80):
    ensure_dirs()
    try:
        person = slug_name(name)
    except ValueError as exc:
        return JSONResponse({"saved": False, "reason": str(exc)}, status_code=400)

    person_dir = KNOWN_FACES_DIR / person
    person_dir.mkdir(parents=True, exist_ok=True)

    data = await file.read()
    gray = decode_image(data, grayscale=True)
    if gray is None:
        return JSONResponse({"saved": False, "reason": "Invalid image"}, status_code=400)

    min_size = (max(1, min_face), max(1, min_face))
    faces = detect_faces(gray, detector, min_size=min_size)
    if len(faces) != 1:
        return JSONResponse({"saved": False, "reason": "need exactly one face"})

    face = normalize_face(gray, faces[0])
    path = person_dir / f"{int(time.time() * 1000)}.png"
    cv2.imwrite(str(path), face)
    count = sum(1 for f in person_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    return JSONResponse({"saved": True, "person": person, "count": count})


@app.post("/api/train")
async def train():
    try:
        train_identifier(argparse.Namespace())
    except SystemExit as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    info = reload_model()
    return JSONResponse({"ok": True, **info})


@app.get("/api/people")
async def people():
    if not KNOWN_FACES_DIR.exists():
        return JSONResponse({"people": []})
    result = []
    for person_dir in sorted(p for p in KNOWN_FACES_DIR.iterdir() if p.is_dir()):
        count = sum(
            1
            for f in person_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        result.append({"name": person_dir.name, "count": count})
    return JSONResponse({"people": result})


@app.post("/api/reset-tracker")
async def reset_tracker(session_id: str = Form("")):
    if session_id and session_id in trackers:
        del trackers[session_id]
    return JSONResponse({"ok": True})


@app.get("/style.css")
async def style_css():
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@app.get("/script.js")
async def script_js():
    return FileResponse(FRONTEND_DIR / "script.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
