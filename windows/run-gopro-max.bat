@echo off
REM Best-quality LIVE GoPro preset, tuned for the camera's 640x480 WiFi preview:
REM   - yolov8m-seg (bigger, more accurate model)
REM   - native imgsz 640 + motion layer (catches tiny movers)
REM NOTE: --imgsz 1280 and --tiles only help on RECORDED full-res footage,
REM       not the low-res live preview - on the GoPro they just cost FPS.
REM Extra args you pass are appended (e.g. --gopro-ip 10.5.5.9).
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo No virtualenv yet - running setup first...
  call "%~dp0setup.bat" || exit /b 1
)
.venv\Scripts\python.exe tracker.py --gopro --model yolov8m.pt %*
