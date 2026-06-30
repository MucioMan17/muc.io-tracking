# Running muc.io tracker on Windows (with the GoPro)

The exact same `tracker.py` runs on Windows — it auto-selects the DirectShow
camera backend, and the GoPro path (FFmpeg + keep-alive) is cross-platform. A
strong Windows GPU is the best place to run the heavy stuff (bigger model,
higher resolution, tiled + motion detection) at high FPS.

## One-time setup

1. Copy the whole project folder to your Windows machine.
2. Double-click **`windows\setup.bat`** (or run it in a terminal). It will:
   - install `uv` and a private **Python 3.12** environment,
   - install all dependencies,
   - if you have an **NVIDIA GPU**, install the **CUDA build of PyTorch** (this
     is what gives you the big FPS jump — don't skip it),
   - cache the YOLO models + FFmpeg so the GoPro works with no internet.

   > If it says "uv isn't on PATH yet", just open a new terminal and run
   > `windows\setup.bat` again.

## Run it

```bat
windows\run.bat --list-cameras        ::  see your cameras
windows\run.bat                       ::  default webcam
windows\run.bat --source 1            ::  another camera
windows\run-gopro-max.bat             ::  GoPro, high-quality preset ⭐
```

`run-gopro-max.bat` uses the heavy settings your GPU can handle:
`--model yolov8m-seg.pt --imgsz 1280 --tiles 2 --motion`. Dial it up or down:

- **`--imgsz 1536`** — even better on small/distant objects (slower).
- **`--tiles 3`** — 9 tiles, finds the tiniest objects (much slower).
- **`--model yolov8l-seg.pt`** — largest model (most accurate).
- Drop `--tiles`/lower `--imgsz` if FPS is too low.

In the window, press **N** to toggle the motion layer and **?** for all keys.

## GoPro on Windows — two gotchas

1. **Camera WiFi:** turn the GoPro on, set it to connect to the GoPro app
   (Preferences → Connections), then on Windows join its `GPxxxxxxxx` WiFi.
   While on it you'll have no internet — that's fine (models are cached).
2. **Windows Firewall** can block the camera's incoming UDP video. If you get a
   black/"Connecting..." screen on `--gopro`, allow it once (run an
   **Administrator** terminal):
   ```bat
   netsh advfirewall firewall add rule name="muc.io gopro" dir=in action=allow protocol=UDP localport=8554
   ```
   (Or just click **Allow** if Windows pops a firewall prompt the first time.)

## Camera permission

Windows Settings → **Privacy & security → Camera** → enable
**"Let desktop apps access your camera."**

## Performance notes

- The HUD shows the active model + inference size, e.g. `NET 8M@1280x4`
  (model · imgsz · tile count) and live FPS, so you can see the tradeoff.
- Tiled inference runs the detector once per tile, so `--tiles 2` is ~4–5× the
  cost of no tiling. That's exactly what a beefy GPU is for — but watch the FPS
  readout and back off if it dips too low for your use.
