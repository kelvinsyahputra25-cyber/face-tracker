@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

echo Starting Face Tracker web app at http://127.0.0.1:8000
".venv\Scripts\python.exe" server.py
pause
