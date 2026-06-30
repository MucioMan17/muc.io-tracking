@echo off
REM Launch the tracker. Any arguments are forwarded to tracker.py.
REM   windows\run.bat                 (default camera)
REM   windows\run.bat --list-cameras
REM   windows\run.bat --gopro
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo No virtualenv yet - running setup first...
  call "%~dp0setup.bat" || exit /b 1
)
.venv\Scripts\python.exe tracker.py %*
