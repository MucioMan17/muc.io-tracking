# muc.io tracker

A live, "thinking" object tracker for macOS. Point it at any camera (built-in,
USB webcam, or even an iPhone via Continuity Camera) and it watches the whole
frame, figures out what each object is, and paints an analytical overlay on top
of the video.

It detects and classifies **80 object types** out of the box — including
`person`, `car`, `dog`, `cat`, `bird`, `airplane`, `bicycle`, `bus`, `truck`,
and many more.

The whole thing is rendered as a **tactical SIGINT-terminal HUD**: phosphor-green
monospace telemetry, cyan corner-bracket targets with crosshair reticles,
"MAG-TRK" magnifier insets with coordinate readouts and amber tether lines, an
optional radar sweep, and a tactical toolbar.

## What you see on screen

| Overlay | What it is |
| --- | --- |
| **Outline dots** | The silhouette of each object, sampled from its segmentation mask — literally what the model is "looking at". |
| **Smart labels** | The **stabilised** class + confidence, e.g. `dog 87%`. Predictions are smoothed over time (majority vote) so labels don't flicker. Includes a track id, a `moving` tag, and pixel speed. |
| **Tracers** | Fading motion trails that follow each object as it moves. |
| **Velocity + prediction** | A heading arrow and a dashed predicted-path line showing **where each object is going**. |
| **MAG-TRK zoom rail** | Magnifier windows **pinned in fixed slots along the top** (they don't jump around following the object). An **amber tether line** runs from each object to its box's crosshair so you can tell which box is which. The header reads `MAG-TRK Tnn` (**MAG**nifier · **TRK** = track · object id), with an X/Y + zoom-factor readout. **Every detected object gets a box** (up to the number of slots that fit across the top; biggest first). Each object keeps its slot until it leaves the screen, so the boxes stay put and steady. Raise/lower how many show with `--max-zoom N`. |
| **Digital zoom** | Zoom into the scene like a photo — **two-finger scroll** (or `+` / `-`) zooms toward your cursor, **right-drag** pans, `0` resets. Crucially, zoom **crops the frame *before* detection**, so the model only sees — and only processes — what you've zoomed into (which also makes far-away objects detect better). A little minimap shows which part of the full frame you're viewing. |
| **Click-to-lock** | Click any object to lock on: a pulsing tracking ring + a big "follow-cam" pinned to the corner that stays on that object. |
| **Activity heatmap** | An accumulated, fading heatmap of where things have been moving. |
| **Stats panel** | A live session tally of how many **unique** objects of each type you've seen. |
| **Sensitivity slider** | Drag from "only the obvious" → "find everything". |
| **Status HUD** | FPS (with a live sparkline), compute device, current confidence, focus mode, and a tally of what it's tracking. |

It can also **record the annotated video to an MP4**, run in **fullscreen**, **mirror** the image, and **focus on one category** (people / vehicles / animals). Your overlay preferences are **remembered between runs**.

## Engine

Ultralytics **YOLOv8 segmentation** (detection + per-pixel outlines) +
**ByteTrack** (stable IDs so trails and zooms stick to the same object). Runs on
the Apple Silicon GPU (Metal/MPS) automatically.

## Setup (once)

```bash
./setup.sh
```

This installs a self-contained Python 3.12 environment and all dependencies
(including PyTorch — a few hundred MB). Your system Python is left untouched.

> The default model weights (`yolov8s-seg.pt`, ~23 MB) ship with the project, so
> it works offline out of the box. If you switch to a model that isn't present
> yet (e.g. `yolov8m-seg.pt`), it downloads once — needs internet that one time.

## Run

```bash
./run.sh                       # default camera
./run.sh --list-cameras        # see which cameras exist
./run.sh --source 1            # use camera #1
./run.sh --gopro               # GoPro Hero 4/5/6/7 over WiFi (see below)
./run.sh --source "rtsp://user:pass@192.168.1.50:554/stream"   # IP / security cam
./run.sh --source clip.mp4     # analyze a video file instead of a camera
./run.sh --model yolov8s-seg.pt   # more accurate, a little slower
```

### Camera permission

The **first** time you run it, macOS will ask the app running your terminal
(Terminal.app / iTerm) for **camera access** — click **Allow**. If you ever
denied it, re-enable under  *System Settings → Privacy & Security → Camera*.

## Use it with a security / IP camera

Most IP and security cameras stream over **RTSP**. Point the tracker at the
camera's stream URL instead of a number:

```bash
./run.sh --source "rtsp://username:password@CAMERA_IP:554/stream"
```

Wrap the URL in quotes. The app automatically uses **TCP**, keeps only the
**newest** frame (so you stay live instead of falling behind), shows a
"Connecting…" screen while it links up, and **auto-reconnects** if the feed drops.

### Finding your camera's URL
- Check the camera / NVR app or its web page — many list the RTSP path under
  *Network → RTSP*.
- It's usually `rtsp://USER:PASS@IP:554/<path>`. The `<path>` depends on brand:
  | Brand | Typical sub-stream path |
  | --- | --- |
  | Hikvision | `/Streaming/Channels/102` |
  | Dahua / Amcrest | `/cam/realmonitor?channel=1&subtype=1` |
  | Reolink | `/h264Preview_01_sub` |
  | UniFi Protect | enable *RTSP* on the camera; it shows you the URL |
  | ONVIF (generic) | `/onvif1` or `/live` |
- If the password has special characters (`@ : / ?`), URL-encode them
  (e.g. `@` → `%40`).

### Test the URL first
Open it in **VLC** → *File ▸ Open Network* and paste the URL. If VLC plays it,
the tracker will too — this rules out URL/credential mistakes fast.

### Tips for smooth tracking
- Use the camera's **sub-stream** (low-res, e.g. 640×480) URL, not the full
  4K main stream — detection runs in real-time and you don't need 4K to spot a
  person or car. This is the single biggest factor in smoothness.
- If TCP stutters on weak WiFi, add `--udp`.
- **HTTP/MJPEG** cameras and phone "IP Webcam" apps work too — just pass their
  `http://…` URL.
- Recorded footage works as a source as well: `./run.sh --source clip.mp4`.

## Use it with a GoPro (Hero 4 / 5 / 6 / 7)

The Hero 7 Black has no HDMI port and isn't supported by GoPro Webcam — but it
can stream live over its own WiFi, and the tracker drives that for you (it tells
the camera to start streaming and keeps the feed alive automatically).

1. **Put the GoPro in app/WiFi mode.** Easiest: pair it once with the **GoPro
   app** (*Connect → pair*). After that, on the camera go to
   *Preferences → Connections → Connect Device → GoPro App*. The camera starts
   broadcasting a WiFi network named like `GPxxxxxxxx`.
2. **Join that WiFi from your Mac** (the password is the one you set when
   pairing). While you're on it your Mac has no internet — that's fine, the
   detection model is already downloaded.
3. **Run:**
   ```bash
   ./run.sh --gopro
   ```

The app sends the start-stream command, pings the camera every 2.5 s so it
doesn't stop, and tracks the live feed. Quit with `Q`.

**Notes**
- It shows a brief **"Connecting…"** screen, then locks on in ~2–3 s.
- The HUD shows the camera's **battery** (`GOPRO BATT 87% [#########-]`) — green
  above 50%, amber under 50%, red under 20%, and `CHG` while charging. It's polled
  from the camera every ~10 s over its WiFi API, so it updates on its own.
- The GoPro's WiFi preview is a lossy UDP feed, so the tracker decodes it through
  a dedicated **FFmpeg** process with error concealment + a large receive buffer.
  That smooths over WiFi packet loss instead of letting it corrupt the picture
  (no more `error while decoding MB` spam). FFmpeg is bundled — nothing to
  install, and it works with no internet.
- The preview feed is low-res (~480–720p) and low-latency — ideal for tracking.
  The window/overlays render at `--width`×`--height` (1080 by default), but since
  the GoPro's live preview is genuinely low-res, that's an **upscale** — the
  picture gets bigger, not more detailed. For a crisper (un-stretched) view use
  `--width 1280 --height 720`; for true full quality, record on the camera and
  run `--source clip.mp4`.
- Different IP? Pass `--gopro-ip 10.5.5.9` (that's the default).
- No video at all? Confirm your Mac is on the `GPxxxx` network (not your home
  WiFi) and the camera is awake. Sanity-check in VLC: run `./run.sh --gopro` once
  to start the stream, then in VLC *Open Network* → `udp://@:8554`.
- Prefer offline / full quality? **Record on the GoPro and analyze the file:**
  `./run.sh --source myclip.mp4`.

## Controls

Click the video window first. **Press `?` any time for the on-screen list.**

| Key | Action | Key | Action |
| --- | --- | --- | --- |
| **Click** | lock onto an object | `X` | clear the lock |
| `D` | outline dots | `B` | bracket targets |
| `T` | tracers | `V` | velocity + prediction |
| `Z` | MAG-TRK zoom insets | `M` | only moving objects |
| `L` | labels | `C` | cycle focus (all/people/vehicles/animals) |
| `H` | activity heatmap | `G` | radar sweep |
| `I` | stats / intel panel | `E` | mirror image |
| scroll / `+` `-` | zoom in/out (toward cursor) | right-drag | pan when zoomed |
| `0` | reset zoom | | |
| `F` | fullscreen | `R` | start/stop recording |
| `P` | save a screenshot | `[` / `]` | less / more sensitive |
| `SPACE` | pause | `?` | help overlay |
| `Q` / `Esc` | quit | | |

You can also drag the **Sensitivity** slider at the top of the window.

## Recording & sessions

- Press `R` to record what you see (overlays included) to a timestamped
  `recording_*.mp4` in the project folder; press `R` again to stop.
- `P` saves a still `capture_*.png`.
- Your toggle/sensitivity/focus/mirror preferences are saved to `settings.json`
  when you quit and restored next time.

## Tuning

- **More/less detections:** the sensitivity slider (or `[` / `]`). Higher
  sensitivity = lower confidence threshold = more (but less certain) objects.
- **Detection model:** **auto-picked** by source — the fast `yolov8n-seg` for
  low-res GoPro feeds (a heavier model barely helps on low-detail video), and
  the more accurate `yolov8s-seg` for cameras and files. The active model is
  shown in the HUD as `NET 8N` / `NET 8S`. Override any time with `--model`
  (e.g. `--model yolov8m-seg.pt` for max accuracy, `--model yolov8n-seg.pt` to
  force speed). Bigger models spot more "obvious" objects, especially
  small/distant ones outdoors.
- **UI size:** the HUD, labels and zoom boxes auto-scale with resolution, so they
  stay readable at 1080p+ instead of shrinking.
- **Capture resolution:** defaults to **1920×1080**. Change with
  `--width 1280 --height 720` (the live value is shown in the telemetry panel).
  Higher capture res makes the zoom insets sharper; it does **not** slow
  detection (the model always runs at `--imgsz`). *(Network/GoPro sources stream
  at their own resolution; this flag only affects local USB/built-in cameras.)*
- **Smoother on a slow machine:** `--imgsz 480` and/or `--max-zoom 2`.
- **Fewer zoom windows:** `--max-zoom 0` turns them off (or press `Z`).

## Detecting small / far-away objects

The reason far-away things get missed isn't your camera's resolution — it's that
YOLO shrinks every frame to `--imgsz` (default 640) before it looks, throwing the
detail away. Three levers fix it (combine them on a strong GPU):

1. **Higher inference resolution** — `--imgsz 1280` (or `1536`). The single
   biggest lever; keeps the detail going into the model. Costs FPS.
2. **Tiled inference** — `--tiles 2` (a 2×2 grid) runs the detector on each tile
   at full res, so small objects are effectively magnified. `--tiles 3` finds the
   tiniest things. Heavy (one inference per tile) — best on a powerful GPU.
3. **Motion layer** — press **`N`** (or `--motion`). Background subtraction flags
   **any moving thing**, however small — even tiny far movers no object-detector
   can see — then crops and classifies them. Best with a **still/tripod camera**
   (it shows movers in magenta, labelled `MOVER` until it can name them). This is
   the big win for *small moving* objects.

Plus a bigger model: `--model yolov8m-seg.pt` (or `yolov8l-seg.pt`). The HUD shows
the active setup, e.g. `NET 8M@1280x4` (model · imgsz · tiles).

You can also just **digital-zoom** into a region (two-finger scroll) — detection
then runs only on that crop, so distant things in it pop right out.

## Files

- `tracker.py` — the whole application.
- `requirements.txt` — Python dependencies.
- `setup.sh` / `run.sh` — install and launch helpers.
