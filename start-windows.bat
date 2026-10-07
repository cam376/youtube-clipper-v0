@echo off
REM One-click start for Windows. Creates the venv on first run, installs the
REM pinned dependencies, then launches the server on http://localhost:8000
setlocal
cd /d "%~dp0"

where ffmpeg >nul 2>nul || (
  echo [error] ffmpeg not found on PATH. See WINDOWS_SETUP.md, step 2.
  pause & exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [setup] creating virtual environment...
  py -3 -m venv .venv || python -m venv .venv || (echo [error] Python 3 not found. See WINDOWS_SETUP.md, step 1. & pause & exit /b 1)
  echo [setup] installing dependencies...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt || (echo [error] pip install failed. & pause & exit /b 1)
)

curl -s -o nul http://localhost:11434/api/tags 2>nul || echo [warn] Ollama does not answer on localhost:11434. Clips will use heuristic ranking until you start Ollama.

echo [run] open http://localhost:8000
".venv\Scripts\python.exe" app\main.py
pause
