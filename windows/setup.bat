@echo off
REM ============================================================
REM  muc.io tracker - one-time Windows setup
REM  Creates a Python 3.12 venv, installs deps (CUDA PyTorch if
REM  an NVIDIA GPU is present), and caches models for offline use.
REM ============================================================
setlocal
cd /d "%~dp0\.."
echo === muc.io tracker : Windows setup ===
echo.

where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv ^(Python toolchain manager^)...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>nul
if errorlevel 1 (
  echo.
  echo uv was installed but isn't on PATH yet. Close this window,
  echo open a NEW terminal, and run windows\setup.bat again.
  exit /b 1
)

echo Creating Python 3.12 virtualenv in .venv ...
uv venv --python 3.12 .venv || goto :err

echo Installing dependencies ...
uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :err

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo.
  echo NVIDIA GPU detected - installing the CUDA build of PyTorch for max FPS...
  uv pip install --python .venv\Scripts\python.exe --reinstall torch torchvision ^
     --index-url https://download.pytorch.org/whl/cu124
) else (
  echo No NVIDIA GPU detected - keeping the CPU build of PyTorch.
)

echo.
echo Caching detection models ^(so the GoPro works offline^)...
.venv\Scripts\python.exe -c "from ultralytics import YOLO; [YOLO(m) for m in ['yolov8n-seg.pt','yolov8s-seg.pt','yolov8m-seg.pt']]"
.venv\Scripts\python.exe -c "import imageio_ffmpeg; print('ffmpeg:', imageio_ffmpeg.get_ffmpeg_exe())"

echo.
echo ============================================================
echo  Setup complete. Try:
echo     windows\run.bat --list-cameras
echo     windows\run-gopro-max.bat        ^(GoPro, high quality^)
echo ============================================================
goto :eof

:err
echo.
echo Setup FAILED - see the messages above.
exit /b 1
