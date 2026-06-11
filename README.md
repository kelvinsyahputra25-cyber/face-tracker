# Face Tracker

Python webcam face tracker and identifier using OpenCV.

## Setup

```powershell
cd "C:\g.s\Python\Face tracker"
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Use

### Web app (recommended)

Start the server, then open the page in your browser:

```powershell
python server.py
```

Or double-click `run_web.bat`, then visit [http://127.0.0.1:8000](http://127.0.0.1:8000).

The web UI lets you start your webcam, enroll people, train the model, and run live face tracking with labels.

### Desktop UI

Start the desktop UI:

```powershell
python app.py
```

You can also double-click `run_ui.bat`.

Or use the command line:

Enroll a person:

```powershell
python face_tracker.py enroll --name "Your Name"
```

Train the identifier:

```powershell
python face_tracker.py train
```

Run live tracking and identification:

```powershell
python face_tracker.py run
```

Press `q` in the camera window to quit.

## Notes

- Enrolled face images are stored in `known_faces/<name>/`.
- The trained model is stored in `data/lbph_model.yml`.
- If recognition is too strict or too loose, adjust the run threshold:

```powershell
python face_tracker.py run --threshold 85
python face_tracker.py run --threshold 40
```

Higher threshold values accept more matches. Lower values are stricter.

If other people are being identified as you, lower the threshold. When only one person is trained, the app automatically caps recognition at a stricter threshold so weak matches show as `Unknown`. Enroll and train another person if you want that person to be named instead of shown as `Unknown`.
