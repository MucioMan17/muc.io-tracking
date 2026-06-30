@echo off
REM High-quality GoPro preset for a powerful Windows GPU:
REM  - yolov8m-seg (bigger, more accurate model)
REM  - imgsz 1280  (sees small / far objects)
REM  - 2x2 tiled inference + motion layer (catch tiny movers)
REM Extra args you pass are appended (e.g. --gopro-ip 10.5.5.9).
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo No virtualenv yet - running setup first...
  call "%~dp0setup.bat" || exit /b 1
)
.venv\Scripts\python.exe tracker.py --gopro --model yolov8m-seg.pt --imgsz 1280 --tiles 2 --motion %*
