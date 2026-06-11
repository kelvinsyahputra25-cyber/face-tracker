from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
KNOWN_FACES_DIR = ROOT / "known_faces"
DATA_DIR = ROOT / "data"
MODEL_PATH = DATA_DIR / "lbph_model.yml"
LABELS_PATH = DATA_DIR / "labels.json"
FACE_SIZE = (200, 200)
DEFAULT_RECOGNITION_THRESHOLD = 52.0
SINGLE_PERSON_THRESHOLD_CAP = 45.0
TRACK_HISTORY_SIZE = 7
TRACK_MIN_KNOWN_VOTES = 3


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    name: str = "Unknown"
    confidence: float | None = None

    @property
    def centroid(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return x + w // 2, y + h // 2


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    name: str = "Unknown"
    confidence: float | None = None
    missed: int = 0
    recent_names: list[str] = field(default_factory=list)

    @property
    def centroid(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return x + w // 2, y + h // 2

    def update(self, detection: Detection) -> None:
        self.box = detection.box
        self.confidence = detection.confidence
        self.missed = 0

        self.recent_names.append(detection.name)
        if len(self.recent_names) > TRACK_HISTORY_SIZE:
            self.recent_names.pop(0)

        known_names = [name for name in self.recent_names if name != "Unknown"]
        if not known_names:
            self.name = "Unknown"
            return

        best_name = max(set(known_names), key=known_names.count)
        best_votes = known_names.count(best_name)
        if best_votes >= TRACK_MIN_KNOWN_VOTES and best_votes / len(self.recent_names) >= 0.5:
            self.name = best_name
        else:
            self.name = "Unknown"


class CentroidTracker:
    def __init__(self, max_distance: int = 90, max_missed: int = 8) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        if not detections:
            self._age_tracks()
            return list(self.tracks.values())

        unmatched_tracks = set(self.tracks)
        unmatched_detections = set(range(len(detections)))
        matches: list[tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            tx, ty = track.centroid
            for index, detection in enumerate(detections):
                dx, dy = detection.centroid
                distance = math.hypot(tx - dx, ty - dy)
                if distance <= self.max_distance:
                    matches.append((distance, track_id, index))

        for _, track_id, detection_index in sorted(matches):
            if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            self.tracks[track_id].update(detections[detection_index])
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_index)

        for track_id in list(unmatched_tracks):
            self.tracks[track_id].missed += 1
            if self.tracks[track_id].missed > self.max_missed:
                del self.tracks[track_id]

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            track = Track(
                track_id=self.next_id,
                box=detection.box,
                name="Unknown",
                confidence=detection.confidence,
            )
            track.update(detection)
            self.tracks[self.next_id] = track
            self.next_id += 1

        return list(self.tracks.values())

    def _age_tracks(self) -> None:
        for track_id in list(self.tracks):
            self.tracks[track_id].missed += 1
            if self.tracks[track_id].missed > self.max_missed:
                del self.tracks[track_id]


def ensure_dirs() -> None:
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


def require_face_module() -> None:
    if not hasattr(cv2, "face"):
        print(
            "OpenCV was installed without the face recognizer module.\n"
            "Install this project's requirements with:\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)


def load_cascade() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade at {cascade_path}")
    return detector


def detect_faces(
    gray: np.ndarray,
    detector: cv2.CascadeClassifier,
    min_size: tuple[int, int] = (80, 80),
) -> list[tuple[int, int, int, int]]:
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return sorted((tuple(map(int, face)) for face in faces), key=lambda box: box[2] * box[3], reverse=True)


def normalize_face(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    face = gray[y : y + h, x : x + w]
    face = cv2.resize(face, FACE_SIZE)
    return cv2.equalizeHist(face)


def slug_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_. -]+", "", name).strip().replace(" ", "_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        raise ValueError("Name must contain at least one letter or number.")
    return slug


def open_camera(index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    return camera


def enroll(args: argparse.Namespace) -> None:
    if args.samples <= 0:
        raise SystemExit("--samples must be greater than 0")
    if args.sample_every <= 0:
        raise SystemExit("--sample-every must be greater than 0")
    if args.min_seconds < 0:
        raise SystemExit("--min-seconds cannot be negative")
    if args.min_face_size <= 0:
        raise SystemExit("--min-face-size must be greater than 0")

    ensure_dirs()
    person = slug_name(args.name)
    person_dir = KNOWN_FACES_DIR / person
    person_dir.mkdir(parents=True, exist_ok=True)

    detector = load_cascade()
    camera = open_camera(args.camera)
    saved = 0
    frame_count = 0
    last_saved_at = 0.0

    print(f"Enrolling {person}. Keep one face visible. Press q to stop early.")
    try:
        while saved < args.samples:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame failed.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, detector, min_size=(args.min_face_size, args.min_face_size))

            if len(faces) == 1:
                x, y, w, h = faces[0]
                now = time.time()
                if frame_count % args.sample_every == 0 and now - last_saved_at >= args.min_seconds:
                    face = normalize_face(gray, faces[0])
                    image_path = person_dir / f"{int(now)}_{saved + 1:03d}.png"
                    cv2.imwrite(str(image_path), face)
                    saved += 1
                    last_saved_at = now
                cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 180, 70), 2)
            elif len(faces) > 1:
                cv2.putText(frame, "Keep only one face in frame", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)

            cv2.putText(frame, f"{person}: {saved}/{args.samples}", (20, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Enroll face", frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} sample(s) to {person_dir}")
    if saved:
        print("Next: python face_tracker.py train")


def iter_training_images() -> Iterable[tuple[np.ndarray, str, Path]]:
    if not KNOWN_FACES_DIR.exists():
        return
    for person_dir in sorted(path for path in KNOWN_FACES_DIR.iterdir() if path.is_dir()):
        for image_path in sorted(person_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, FACE_SIZE)
            yield cv2.equalizeHist(image), person_dir.name, image_path


def train(_: argparse.Namespace) -> None:
    ensure_dirs()
    require_face_module()

    faces: list[np.ndarray] = []
    labels: list[int] = []
    label_by_name: dict[str, int] = {}
    count_by_name: dict[str, int] = {}

    for image, name, _ in iter_training_images():
        label = label_by_name.setdefault(name, len(label_by_name))
        faces.append(image)
        labels.append(label)
        count_by_name[name] = count_by_name.get(name, 0) + 1

    if not faces:
        raise SystemExit("No training images found. Run: python face_tracker.py enroll --name YourName")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))

    labels_payload = {
        "labels": {str(label): name for name, label in label_by_name.items()},
        "counts": count_by_name,
        "trained_at": int(time.time()),
    }
    LABELS_PATH.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

    print(f"Trained {len(faces)} image(s) for {len(label_by_name)} person(s).")
    if len(label_by_name) == 1:
        print(f"Only one person is trained, so recognition will use a stricter {SINGLE_PERSON_THRESHOLD_CAP:.0f} threshold cap.")
    print(f"Model: {MODEL_PATH}")


def load_recognizer() -> tuple[object | None, dict[int, str]]:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, {}
    require_face_module()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    labels_payload = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labels = {int(label): name for label, name in labels_payload["labels"].items()}
    return recognizer, labels


def effective_threshold(labels: dict[int, str], threshold: float) -> float:
    if len(labels) <= 1:
        return min(threshold, SINGLE_PERSON_THRESHOLD_CAP)
    return threshold


def identify(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    recognizer: object | None,
    labels: dict[int, str],
    threshold: float,
) -> tuple[str, float | None]:
    if recognizer is None:
        return "Unknown", None
    face = normalize_face(gray, box)
    label, confidence = recognizer.predict(face)
    if confidence <= effective_threshold(labels, threshold):
        return labels.get(int(label), "Unknown"), float(confidence)
    return "Unknown", float(confidence)


def draw_track(frame: np.ndarray, track: Track) -> None:
    x, y, w, h = track.box
    identified = track.name != "Unknown"
    color = (40, 190, 80) if identified else (0, 170, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    confidence = "" if track.confidence is None else f" {track.confidence:.0f}"
    label = f"ID {track.track_id}: {track.name}{confidence}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y = max(22, y - 8)
    cv2.rectangle(frame, (x, label_y - label_size[1] - 8), (x + label_size[0] + 8, label_y + 4), color, -1)
    cv2.putText(frame, label, (x + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 15, 15), 2)


def run(args: argparse.Namespace) -> None:
    if args.threshold < 0:
        raise SystemExit("--threshold cannot be negative")
    if args.track_distance <= 0:
        raise SystemExit("--track-distance must be greater than 0")
    if args.max_missed < 0:
        raise SystemExit("--max-missed cannot be negative")
    if args.min_face_size <= 0:
        raise SystemExit("--min-face-size must be greater than 0")

    ensure_dirs()
    detector = load_cascade()
    recognizer, labels = load_recognizer()
    if recognizer is None:
        print("No trained model found. Tracking will run, but faces will be Unknown.")
        print("Use: python face_tracker.py enroll --name YourName")
        print("Then: python face_tracker.py train")

    camera = open_camera(args.camera)
    tracker = CentroidTracker(max_distance=args.track_distance, max_missed=args.max_missed)

    print("Running. Press q to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame failed.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, detector, min_size=(args.min_face_size, args.min_face_size))
            detections = []
            for box in faces:
                name, confidence = identify(gray, box, recognizer, labels, args.threshold)
                detections.append(Detection(box=box, name=name, confidence=confidence))

            tracks = tracker.update(detections)
            for track in tracks:
                if track.missed == 0:
                    draw_track(frame, track)

            cv2.putText(frame, "q: quit", (20, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Face tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Webcam face tracker and identifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll_parser = subparsers.add_parser("enroll", help="Capture face samples for one person.")
    enroll_parser.add_argument("--name", required=True, help="Person name to identify.")
    enroll_parser.add_argument("--samples", type=int, default=60, help="Number of face samples to save.")
    enroll_parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    enroll_parser.add_argument("--sample-every", type=int, default=4, help="Save at most one sample every N frames.")
    enroll_parser.add_argument("--min-seconds", type=float, default=0.08, help="Minimum seconds between saved samples.")
    enroll_parser.add_argument("--min-face-size", type=int, default=80, help="Smallest face size to detect.")
    enroll_parser.set_defaults(func=enroll)

    train_parser = subparsers.add_parser("train", help="Train the face identifier from known_faces.")
    train_parser.set_defaults(func=train)

    run_parser = subparsers.add_parser("run", help="Run live tracking and identification.")
    run_parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    run_parser.add_argument("--threshold", type=float, default=DEFAULT_RECOGNITION_THRESHOLD, help="Lower is stricter for LBPH recognition.")
    run_parser.add_argument("--track-distance", type=int, default=90, help="Max centroid distance for same tracked face.")
    run_parser.add_argument("--max-missed", type=int, default=8, help="Frames before a missing track is removed.")
    run_parser.add_argument("--min-face-size", type=int, default=80, help="Smallest face size to detect.")
    run_parser.set_defaults(func=run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
