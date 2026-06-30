#!/usr/bin/env bash
# Launch the tracker. Any arguments are forwarded to tracker.py.
#   ./run.sh                     # default camera
#   ./run.sh --list-cameras      # see available cameras
#   ./run.sh --source 1          # use camera #1
#   ./run.sh --source clip.mp4   # use a video file
#   ./run.sh --model yolov8s-seg.pt   # more accurate (a bit slower)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "No virtualenv yet — running setup first…"
  ./setup.sh
fi

exec .venv/bin/python tracker.py "$@"
