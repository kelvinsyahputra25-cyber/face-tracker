from __future__ import annotations

import argparse
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2

from face_tracker import (
    CentroidTracker,
    DATA_DIR,
    DEFAULT_RECOGNITION_THRESHOLD,
    Detection,
    KNOWN_FACES_DIR,
    SINGLE_PERSON_THRESHOLD_CAP,
    draw_track,
    detect_faces,
    ensure_dirs,
    identify,
    load_cascade,
    load_recognizer,
    normalize_face,
    open_camera,
    slug_name,
    train as train_identifier,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

ACCENT        = "#185fa5"
ACCENT_LIGHT  = "#e6f1fb"
SUCCESS       = "#1d9e75"
SUCCESS_LIGHT = "#e1f5ee"
WARN          = "#ef9f27"
BG_PAGE       = "#f0f2f5"
BG_PANEL      = "#ffffff"
BG_SECTION    = "#f6f8fa"
TEXT_PRIMARY  = "#17202a"
TEXT_MUTED    = "#59636e"
BORDER        = "#d8dde3"


class FaceTrackerApp:
    def __init__(self, root: tk.Tk) -> None:
        ensure_dirs()
        self.root = root
        self.root.title("Face Tracker")
        self.root.geometry("1200x720")
        self.root.minsize(1000, 640)
        self.root.configure(bg=BG_PAGE)

        self.camera = None
        self.detector = None
        self.recognizer = None
        self.labels: dict[int, str] = {}
        self.tracker = CentroidTracker()
        self.running = False
        self.photo = None
        self.frame_count = 0

        self.enrolling = False
        self.enroll_name = ""
        self.enroll_dir: Path | None = None
        self.enroll_target = 60
        self.enroll_saved = 0
        self.last_saved_at = 0.0

        self.train_thread: threading.Thread | None = None
        self.train_results: queue.Queue[tuple[str, str]] = queue.Queue()

        self._frame_times: list[float] = []
        self._fps: float = 0.0

        self.camera_var   = tk.IntVar(value=0)
        self.name_var     = tk.StringVar()
        self.samples_var  = tk.IntVar(value=60)
        self.threshold_var = tk.DoubleVar(value=DEFAULT_RECOGNITION_THRESHOLD)
        self.min_face_var = tk.IntVar(value=80)
        self.status_var   = tk.StringVar(value="Ready")
        self.model_var    = tk.StringVar(value="▸ not loaded")
        self.faces_var    = tk.StringVar(value="0 faces  ·  0 tracks")
        self.fps_var      = tk.StringVar(value="")
        self.enroll_var   = tk.StringVar(value="Idle")

        self._build_style()
        self._build_ui()
        self.reload_model(show_errors=False)
        self.refresh_people()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------ styles

    def _build_style(self) -> None:
        style = ttk.Style()
        style.configure("TFrame",        background=BG_PANEL)
        style.configure("Page.TFrame",   background=BG_PAGE)
        style.configure("Section.TFrame",background=BG_SECTION)
        style.configure("TLabel",        background=BG_PANEL, foreground=TEXT_PRIMARY, font=("Segoe UI", 10))
        style.configure("Muted.TLabel",  background=BG_PANEL, foreground=TEXT_MUTED,   font=("Segoe UI", 10))
        style.configure("Micro.TLabel",  background=BG_PANEL, foreground=TEXT_MUTED,   font=("Segoe UI", 9))
        style.configure("Cap.TLabel",    background=BG_PANEL, foreground=TEXT_MUTED,   font=("Segoe UI", 9, "bold"))
        style.configure("Title.TLabel",  background=BG_PANEL, foreground=TEXT_PRIMARY, font=("Segoe UI", 14, "bold"))
        style.configure("Bold.TLabel",   background=BG_PANEL, foreground=TEXT_PRIMARY, font=("Segoe UI", 10, "bold"))
        style.configure("Model.TLabel",  background=BG_PANEL, foreground=ACCENT,       font=("Segoe UI", 9))
        style.configure("TButton",        padding=(8, 5), font=("Segoe UI", 10))
        style.configure("Accent.TButton", padding=(8, 5), font=("Segoe UI", 10))
        style.configure("TEntry",         padding=(6, 4))

    # ------------------------------------------------------------------ UI build

    def _build_ui(self) -> None:
        shell = tk.Frame(self.root, bg=BG_PAGE, padx=14, pady=14)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(0, weight=1)
        shell.columnconfigure(1, weight=0)
        shell.rowconfigure(0, weight=1)

        # ---- left: video + status bar
        left = tk.Frame(shell, bg=BG_PAGE)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.video_label = tk.Label(
            left, bg="#101820", fg="#6a8a9a",
            text="Camera off",
            font=("Segoe UI", 15), anchor="center",
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        status_bar = tk.Frame(left, bg=BG_PANEL, padx=10, pady=7,
                              highlightbackground=BORDER, highlightthickness=1)
        status_bar.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        status_bar.columnconfigure(1, weight=1)

        self._status_dot = tk.Canvas(status_bar, width=10, height=10,
                                     bg=BG_PANEL, highlightthickness=0)
        self._status_dot.grid(row=0, column=0, padx=(0, 6))
        self._dot_id = self._status_dot.create_oval(1, 1, 9, 9, fill="#aab0b8", outline="")

        tk.Label(status_bar, textvariable=self.status_var,
                 bg=BG_PANEL, fg=TEXT_MUTED, font=("Segoe UI", 10)
                 ).grid(row=0, column=1, sticky="w")

        tk.Label(status_bar, textvariable=self.faces_var,
                 bg=BG_PANEL, fg=TEXT_MUTED, font=("Segoe UI", 10)
                 ).grid(row=0, column=2, sticky="e", padx=(16, 0))

        tk.Label(status_bar, textvariable=self.fps_var,
                 bg=BG_PANEL, fg=ACCENT, font=("Segoe UI", 10, "bold")
                 ).grid(row=0, column=3, sticky="e", padx=(14, 0))

        # ---- right: control panel
        panel_outer = tk.Frame(shell, bg=BG_PANEL,
                               highlightbackground=BORDER, highlightthickness=1)
        panel_outer.grid(row=0, column=1, sticky="ns")

        canvas = tk.Canvas(panel_outer, bg=BG_PANEL, highlightthickness=0, width=240)
        scrollbar = ttk.Scrollbar(panel_outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = tk.Frame(canvas, bg=BG_PANEL, padx=14, pady=14)
        panel_window = canvas.create_window((0, 0), window=panel, anchor="nw")

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(panel_window, width=canvas.winfo_width())

        panel.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(panel_window, width=e.width))

        panel.columnconfigure(0, weight=1)
        row = 0

        # Header
        tk.Label(panel, text="Face Tracker", bg=BG_PANEL, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 14, "bold")).grid(row=row, column=0, sticky="w")
        row += 1
        tk.Label(panel, textvariable=self.model_var, bg=BG_PANEL,
                 fg=ACCENT, font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w", pady=(2, 14))
        row += 1

        # Camera section
        row = self._section(panel, "Camera", row)
        cam_row = tk.Frame(panel, bg=BG_PANEL)
        cam_row.grid(row=row, column=0, sticky="ew"); row += 1
        cam_row.columnconfigure(1, weight=1)
        tk.Label(cam_row, text="Index", bg=BG_PANEL, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w")
        tk.Spinbox(cam_row, from_=0, to=8, textvariable=self.camera_var,
                   width=5, font=("Segoe UI", 10)).grid(row=0, column=1, sticky="e")

        btn_row = tk.Frame(panel, bg=BG_PANEL)
        btn_row.grid(row=row, column=0, sticky="ew", pady=(8, 16)); row += 1
        btn_row.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(btn_row, text="▶  Start", command=self.start_camera)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.stop_button = ttk.Button(btn_row, text="■  Stop", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # Detection section
        row = self._section(panel, "Detection", row)
        thr_row = tk.Frame(panel, bg=BG_PANEL)
        thr_row.grid(row=row, column=0, sticky="ew"); row += 1
        thr_row.columnconfigure(0, weight=1)
        tk.Label(thr_row, text="Recognition threshold", bg=BG_PANEL,
                 fg=TEXT_MUTED, font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w")
        self.threshold_value = tk.Label(thr_row, text=f"{DEFAULT_RECOGNITION_THRESHOLD:.0f}",
                                        bg=BG_PANEL, fg=TEXT_PRIMARY, font=("Segoe UI", 10, "bold"))
        self.threshold_value.grid(row=0, column=1, sticky="e")
        ttk.Scale(panel, from_=25, to=100, orient=tk.HORIZONTAL,
                  variable=self.threshold_var,
                  command=self._update_threshold_label
                  ).grid(row=row, column=0, sticky="ew", pady=(4, 10)); row += 1

        face_row = tk.Frame(panel, bg=BG_PANEL)
        face_row.grid(row=row, column=0, sticky="ew", pady=(0, 16)); row += 1
        face_row.columnconfigure(1, weight=1)
        tk.Label(face_row, text="Min face size (px)", bg=BG_PANEL,
                 fg=TEXT_MUTED, font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w")
        tk.Spinbox(face_row, from_=40, to=240, increment=10,
                   textvariable=self.min_face_var, width=6,
                   font=("Segoe UI", 10)).grid(row=0, column=1, sticky="e")

        # Enrollment section
        row = self._section(panel, "Enrollment", row)
        ttk.Entry(panel, textvariable=self.name_var,
                  font=("Segoe UI", 10)).grid(row=row, column=0, sticky="ew"); row += 1

        samp_row = tk.Frame(panel, bg=BG_PANEL)
        samp_row.grid(row=row, column=0, sticky="ew", pady=(6, 8)); row += 1
        samp_row.columnconfigure(1, weight=1)
        tk.Label(samp_row, text="Samples", bg=BG_PANEL, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w")
        tk.Spinbox(samp_row, from_=10, to=200, increment=10,
                   textvariable=self.samples_var, width=6,
                   font=("Segoe UI", 10)).grid(row=0, column=1, sticky="e")

        enr_btn_row = tk.Frame(panel, bg=BG_PANEL)
        enr_btn_row.grid(row=row, column=0, sticky="ew"); row += 1
        enr_btn_row.columnconfigure((0, 1), weight=1)
        self.enroll_button = ttk.Button(enr_btn_row, text="Start Enroll", command=self.start_enrollment)
        self.enroll_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.cancel_enroll_button = ttk.Button(enr_btn_row, text="Cancel",
                                               command=self.cancel_enrollment, state=tk.DISABLED)
        self.cancel_enroll_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # Progress bar + label
        enroll_info = tk.Frame(panel, bg=BG_PANEL)
        enroll_info.grid(row=row, column=0, sticky="ew", pady=(8, 0)); row += 1
        enroll_info.columnconfigure(0, weight=1)
        self.enroll_label = tk.Label(enroll_info, textvariable=self.enroll_var,
                                     bg=BG_PANEL, fg=TEXT_MUTED, font=("Segoe UI", 9))
        self.enroll_label.grid(row=0, column=0, sticky="w")
        self.enroll_count_label = tk.Label(enroll_info, text="",
                                           bg=BG_PANEL, fg=TEXT_MUTED, font=("Segoe UI", 9))
        self.enroll_count_label.grid(row=0, column=1, sticky="e")
        self.enroll_progress = ttk.Progressbar(panel, orient=tk.HORIZONTAL,
                                               mode="determinate", maximum=100)
        self.enroll_progress.grid(row=row, column=0, sticky="ew", pady=(4, 16)); row += 1
        self.enroll_progress["value"] = 0

        # Model section
        row = self._section(panel, "Model", row)
        mdl_row = tk.Frame(panel, bg=BG_PANEL)
        mdl_row.grid(row=row, column=0, sticky="ew", pady=(0, 16)); row += 1
        mdl_row.columnconfigure((0, 1), weight=1)
        self.train_button = ttk.Button(mdl_row, text="Train", command=self.train_model)
        self.train_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(mdl_row, text="Reload", command=self.reload_model
                   ).grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # People section
        row = self._section(panel, "People", row)
        self.people_list = tk.Listbox(
            panel, height=7, activestyle="none",
            relief=tk.FLAT, highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=ACCENT,
            selectbackground=ACCENT_LIGHT, selectforeground=TEXT_PRIMARY,
            font=("Segoe UI", 10), bg=BG_SECTION,
        )
        self.people_list.grid(row=row, column=0, sticky="ew"); row += 1
        ttk.Button(panel, text="Refresh", command=self.refresh_people
                   ).grid(row=row, column=0, sticky="ew", pady=(8, 0)); row += 1

    def _section(self, parent: tk.Frame, title: str, row: int) -> int:
        """Render a small caps section divider; returns the next available row."""
        sep_frame = tk.Frame(parent, bg=BORDER, height=1)
        sep_frame.grid(row=row, column=0, sticky="ew", pady=(4, 6))
        row += 1
        tk.Label(parent, text=title.upper(), bg=BG_PANEL,
                 fg=TEXT_MUTED, font=("Segoe UI", 8, "bold")
                 ).grid(row=row, column=0, sticky="w", pady=(0, 6))
        return row + 1

    # ------------------------------------------------------------------ camera

    def start_camera(self) -> bool:
        if self.running:
            return True
        try:
            self.detector = load_cascade()
            self.camera = open_camera(int(self.camera_var.get()))
        except Exception as exc:
            self._set_status("Camera failed", error=True)
            messagebox.showerror("Camera", str(exc))
            return False

        self.tracker = CentroidTracker()
        self.frame_count = 0
        self.running = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self._set_status("Camera running", active=True)
        self._tick()
        return True

    def stop_camera(self) -> None:
        self.running = False
        self.cancel_enrollment()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.video_label.configure(image="", text="Camera off")
        self.photo = None
        self.faces_var.set("0 faces  ·  0 tracks")
        self.fps_var.set("")
        self._frame_times.clear()
        self._fps = 0.0
        self._set_status("Camera stopped")

    # ------------------------------------------------------------------ enrollment

    def start_enrollment(self) -> None:
        try:
            name = slug_name(self.name_var.get())
            samples = int(self.samples_var.get())
        except ValueError as exc:
            messagebox.showerror("Enrollment", str(exc))
            return

        if samples <= 0:
            messagebox.showerror("Enrollment", "Samples must be greater than 0.")
            return
        if not self.running and not self.start_camera():
            return

        self.enroll_name = name
        self.enroll_dir = KNOWN_FACES_DIR / name
        self.enroll_dir.mkdir(parents=True, exist_ok=True)
        self.enroll_target = samples
        self.enroll_saved = 0
        self.last_saved_at = 0.0
        self.enrolling = True
        self.enroll_button.configure(state=tk.DISABLED)
        self.cancel_enroll_button.configure(state=tk.NORMAL)
        self._update_enroll_ui()
        self._set_status(f"Enrolling {name}", active=True)

    def cancel_enrollment(self) -> None:
        if not self.enrolling:
            return
        self.enrolling = False
        self.enroll_button.configure(state=tk.NORMAL)
        self.cancel_enroll_button.configure(state=tk.DISABLED)
        self.enroll_var.set("Idle")
        self.enroll_count_label.configure(text="")
        self.enroll_progress["value"] = 0
        self._set_status("Enrollment cancelled")

    def finish_enrollment(self) -> None:
        name, saved = self.enroll_name, self.enroll_saved
        self.enrolling = False
        self.enroll_button.configure(state=tk.NORMAL)
        self.cancel_enroll_button.configure(state=tk.DISABLED)
        self.enroll_var.set("Complete")
        self.enroll_count_label.configure(text=f"{saved} saved")
        self.enroll_progress["value"] = 100
        self._set_status(f"Saved {saved} samples for {name}")
        self.refresh_people()

    def _update_enroll_ui(self) -> None:
        pct = (self.enroll_saved / self.enroll_target * 100) if self.enroll_target else 0
        self.enroll_var.set(f"Enrolling {self.enroll_name}")
        self.enroll_count_label.configure(text=f"{self.enroll_saved} / {self.enroll_target}")
        self.enroll_progress["value"] = pct

    # ------------------------------------------------------------------ training

    def train_model(self) -> None:
        if self.train_thread is not None and self.train_thread.is_alive():
            return
        self.train_button.configure(state=tk.DISABLED)
        self._set_status("Training model…")

        def worker() -> None:
            try:
                train_identifier(argparse.Namespace())
            except SystemExit as exc:
                self.train_results.put(("error", str(exc.code)))
            except Exception as exc:
                self.train_results.put(("error", str(exc)))
            else:
                self.train_results.put(("ok", "Training complete"))

        self.train_thread = threading.Thread(target=worker, daemon=True)
        self.train_thread.start()
        self.root.after(100, self._poll_training)

    def _poll_training(self) -> None:
        try:
            result, message = self.train_results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_training)
            return

        self.train_button.configure(state=tk.NORMAL)
        if result == "ok":
            self._set_status(message)
            self.reload_model(show_errors=False)
            self.refresh_people()
        else:
            self._set_status("Training failed", error=True)
            messagebox.showerror("Training", message)

    # ------------------------------------------------------------------ model

    def reload_model(self, show_errors: bool = True) -> None:
        try:
            self.recognizer, self.labels = load_recognizer()
        except Exception as exc:
            self.recognizer = None
            self.labels = {}
            self.model_var.set("▸ load failed")
            if show_errors:
                messagebox.showerror("Model", str(exc))
            return

        if self.recognizer is None:
            self.model_var.set("▸ no model")
        elif len(self.labels) == 1:
            self.model_var.set(f"▸ 1 person · strict {SINGLE_PERSON_THRESHOLD_CAP:.0f}")
        else:
            self.model_var.set(f"▸ {len(self.labels)} people loaded")

    def refresh_people(self) -> None:
        self.people_list.delete(0, tk.END)
        people = []
        if KNOWN_FACES_DIR.exists():
            for d in sorted(p for p in KNOWN_FACES_DIR.iterdir() if p.is_dir()):
                count = sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
                people.append((d.name, count))

        if not people:
            self.people_list.insert(tk.END, "  No enrolled people")
            return
        for name, count in people:
            self.people_list.insert(tk.END, f"  {name}  ({count} imgs)")

    # ------------------------------------------------------------------ frame loop

    def _tick(self) -> None:
        if not self.running or self.camera is None or self.detector is None:
            return

        ok, frame = self.camera.read()
        if not ok:
            self._set_status("Frame read failed", error=True)
            self.root.after(100, self._tick)
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_face = max(1, int(self.min_face_var.get()))
        faces = detect_faces(gray, self.detector, min_size=(min_face, min_face))

        detections = []
        for box in faces:
            name, confidence = identify(gray, box, self.recognizer, self.labels, self.threshold_var.get())
            detections.append(Detection(box=box, name=name, confidence=confidence))

        tracks = self.tracker.update(detections)
        for track in tracks:
            if track.missed == 0:
                draw_track(frame, track)

        if self.enrolling:
            self._handle_enrollment(frame, gray, faces)

        self._draw_live_badge(frame)
        self.faces_var.set(f"{len(faces)} faces  ·  {len(tracks)} tracks")
        self._tick_fps()
        self._render_frame(frame)
        self.frame_count += 1
        self.root.after(15, self._tick)

    def _tick_fps(self) -> None:
        now = time.time()
        self._frame_times.append(now)
        # keep only the last 30 timestamps (~1 sec window at 30 fps)
        if len(self._frame_times) > 30:
            self._frame_times = self._frame_times[-30:]
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            self._fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0
        self.fps_var.set(f"{self._fps:.0f} fps")

    def _draw_live_badge(self, frame) -> None:
        h, w = frame.shape[:2]
        label = "● LIVE"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x, y = w - tw - 20, 18
        cv2.rectangle(frame, (x - 6, y - th - 4), (x + tw + 4, y + 4), (15, 110, 56), -1)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (93, 218, 165), 1)

        fps_label = f"{self._fps:.0f} fps"
        (fw, fh), _ = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        fx, fy = 12, 18
        cv2.rectangle(frame, (fx - 4, fy - fh - 4), (fx + fw + 4, fy + 4), (20, 30, 45), -1)
        cv2.putText(frame, fps_label, (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1)

    def _handle_enrollment(self, frame, gray, faces: list[tuple[int, int, int, int]]) -> None:
        if self.enroll_dir is None:
            return

        color = (40, 190, 80) if len(faces) == 1 else (0, 170, 255)
        cv2.putText(
            frame,
            f"Enroll {self.enroll_name}: {self.enroll_saved}/{self.enroll_target}",
            (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
        )

        if len(faces) != 1:
            self._set_status("Enrollment needs exactly one face", active=True)
            return

        now = time.time()
        if self.frame_count % 4 != 0 or now - self.last_saved_at < 0.08:
            return

        face = normalize_face(gray, faces[0])
        image_path = self.enroll_dir / f"{int(now)}_{self.enroll_saved + 1:03d}.png"
        if cv2.imwrite(str(image_path), face):
            self.enroll_saved += 1
            self.last_saved_at = now
            self._update_enroll_ui()

        if self.enroll_saved >= self.enroll_target:
            self.finish_enrollment()

    def _render_frame(self, frame) -> None:
        lw = max(self.video_label.winfo_width(), 320)
        lh = max(self.video_label.winfo_height(), 240)
        h, w = frame.shape[:2]
        scale = min(lw / w, lh / h)
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
        if (dw, dh) != (w, h):
            frame = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h2, w2 = rgb.shape[:2]
        data = f"P6\n{w2} {h2}\n255\n".encode() + rgb.tobytes()
        self.photo = tk.PhotoImage(data=data, format="PPM")
        self.video_label.configure(image=self.photo, text="")

    # ------------------------------------------------------------------ helpers

    def _set_status(self, text: str, active: bool = False, error: bool = False) -> None:
        self.status_var.set(text)
        if error:
            color = "#c0392b"
        elif active:
            color = SUCCESS
        else:
            color = "#aab0b8"
        self._status_dot.itemconfig(self._dot_id, fill=color)

    def _update_threshold_label(self, _: str | None = None) -> None:
        self.threshold_value.configure(text=f"{self.threshold_var.get():.0f}")

    def close(self) -> None:
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    FaceTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()