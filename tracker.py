#!/usr/bin/env python3
"""
muc.io tracker — a live "thinking" object tracker for macOS.

Watches a camera (or video file), detects & classifies whatever it sees
(person, car, dog, bird, plane, ... 80 COCO classes) and paints a rich,
analytical overlay on top of the video:

  • outline dots   — each object's silhouette, sampled from its segmentation
                     mask, drawn as glowing dots (what the model is "looking at")
  • smart labels   — the *stabilised* class + confidence, e.g.  dog 87%
  • tracers        — fading motion trails that follow each object
  • velocity/aim   — a heading arrow + a dashed predicted-path showing where the
                     object is going
  • zoom insets    — magnifier windows that follow objects so you can see closer
  • click-to-lock  — click any object to lock on: pulsing ring + a big follow-cam
  • heatmap        — an activity heatmap of where things have been
  • stats          — a live session tally of unique objects seen per class
  • sensitivity    — trade off "find everything" vs "only the obvious"

Engine: Ultralytics YOLOv8 segmentation + ByteTrack for persistent IDs,
running on the Apple-Silicon GPU (Metal/MPS) automatically.

Run:   ./run.sh            ./run.sh --source 1            ./run.sh --list-cameras
Keys:  press  ?  in the window for the full list.
"""

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from collections import Counter, deque

# Quieten OpenCV's bundled FFmpeg decoder spam (set before cv2 is imported).
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")  # fatal only

import cv2
import numpy as np

WINDOW = "muc.io tracker"
HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, "settings.json")

# Only the outdoor moving things we care about — COCO class ids. Everything
# else (cups, chairs, laptops, ...) is filtered out at the detector.
WANTED_IDS = [0,        # person
              2, 3, 5, 7,    # car, motorcycle, bus, truck  (ground vehicles)
              4,        # airplane
              14,       # bird
              15, 16, 17, 18, 19, 20, 21, 22, 23]  # cat dog horse sheep cow
#                                                    elephant bear zebra giraffe

KEYHELP = [
    ("D", "movement dots"), ("B", "object boxes"), ("L", "labels"),
    ("V", "prediction arrows"), ("Z", "zoom boxes (moving objects)"),
    ("Scroll / + -", "zoom the whole frame (toward cursor)"),
    ("Right-drag", "pan when zoomed"), ("0", "reset zoom"),
    ("Click", "lock onto an object (big zoom)"), ("X", "clear lock"),
    ("F", "fullscreen"), ("E", "mirror"), ("R", "record video"),
    ("P", "save photo"), ("[ ]", "less / more sensitive"),
    ("SPACE", "pause"), ("?", "this help"), ("Q / Esc", "quit"),
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def cam_backend():
    """OpenCV capture backend for local cameras on macOS."""
    return cv2.CAP_AVFOUNDATION


def pick_device(requested: str) -> str:
    import torch
    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def list_cameras(max_idx: int = 8):
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cam_backend())
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found.append((i, w, h))
        cap.release()
    return found


# --------------------------------------------------------------------------- #
# Visual theme — phosphor green / cyan / amber palette, crisp monospace text.
# --------------------------------------------------------------------------- #
T_CYAN = (235, 210, 70)        # BGR — primary target colour
T_GREEN = (95, 240, 140)       # phosphor green — telemetry text
T_GREEN_DIM = (80, 200, 115)
T_AMBER = (55, 190, 250)       # tethers / alerts / lock
T_RED = (70, 70, 240)
T_DIM = (120, 150, 120)
T_WHITE = (235, 245, 235)
T_MOVER = (235, 120, 215)      # motion / tiled finds — stand out from YOLO
T_PANEL_TINT = (16, 26, 14)    # dark green-black panel fill
_DOT_GREEN = np.array([90, 255, 120], np.float32)   # movement-dot colour (BGR)
T_PALETTE = [(235, 210, 70), (95, 240, 140), (150, 235, 195),
             (70, 205, 245), (205, 225, 90), (130, 235, 235)]


def tactical_color(idx):
    return T_PALETTE[abs(int(idx)) % len(T_PALETTE)]


# Global UI scale: 1.0 at 720p, grows with resolution so the HUD, labels and
# zoom boxes stay the same *relative* size instead of shrinking at 1080+.
UI = 1.0


def set_ui_scale(h):
    global UI
    UI = max(1.0, h / 720.0)


def S(v):
    """Scale a pixel size by the current UI scale."""
    return max(1, int(round(v * UI)))


def _mono_size(scale):
    return max(int(round(11 * UI)), int(round(scale * 28 * UI)))


def mono_w(s, scale):
    return int(len(s) * _mono_size(scale) * 0.62)   # monospace advance ≈ 0.62em


def lh(scale=0.5):
    return int(_mono_size(scale) * 1.5)             # line height for stacked text


class HUDText:
    """Collects all text draws for a frame and renders them in one pass with a
    crisp monospace font (PIL) for the terminal look. Falls back to OpenCV's
    font if PIL/a mono TTF isn't available."""
    def __init__(self):
        self.items = []
        self.cache = {}
        self.path = self._find_font()

    def _find_font(self):
        cands = []
        try:
            import matplotlib
            cands.append(os.path.join(matplotlib.get_data_path(),
                                      "fonts", "ttf", "DejaVuSansMono.ttf"))
        except Exception:
            pass
        cands += ["/System/Library/Fonts/Menlo.ttc",
                  "/System/Library/Fonts/Monaco.ttf",
                  "/Library/Fonts/Courier New.ttf"]
        return next((c for c in cands if os.path.exists(c)), None)

    def _font(self, size):
        if size not in self.cache:
            from PIL import ImageFont
            try:
                self.cache[size] = ImageFont.truetype(self.path, size)
            except Exception:
                self.cache[size] = ImageFont.load_default()
        return self.cache[size]

    def add(self, s, x, y, scale, color):
        self.items.append((s, int(x), int(y), _mono_size(scale), color))

    def flush(self, frame):
        items, self.items = self.items, []
        if not items:
            return frame
        try:
            from PIL import Image, ImageDraw
            # Feed the BGR array to PIL as-is. Text only *writes* pixels, so the
            # untouched image bytes pass through unchanged; we just supply the
            # fill colour in BGR order. Avoids two full-frame channel swaps.
            img = Image.fromarray(frame)
            d = ImageDraw.Draw(img)
            for s, x, y, size, color in items:
                d.text((x, y - int(size * 0.82)), s, font=self._font(size),
                       fill=(int(color[0]), int(color[1]), int(color[2])))
            return np.asarray(img)
        except Exception:
            for s, x, y, size, color in items:
                cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            size / 28.0, color, 1, cv2.LINE_AA)
            return frame


_HUD = HUDText()


def text(img, s, x, y, scale=0.5, color=T_GREEN, thick=1):
    """Queue monospace text; actually rendered by _HUD.flush() at frame end."""
    _HUD.add(s, x, y, scale, color)


def panel(img, x, y, w, h, alpha=0.5, radius=8, border=None):
    """Dark green-tinted translucent panel with optional border + corner ticks."""
    H, W = img.shape[:2]
    x, y = clamp(x, 0, W - 1), clamp(y, 0, H - 1)
    w, h = clamp(w, 1, W - x), clamp(h, 1, H - y)
    roi = img[y:y + h, x:x + w]
    r = min(radius, w // 2, h // 2)
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    m = mask.astype(bool)
    tint = np.array(T_PANEL_TINT, np.float32)
    roi[m] = (roi[m] * (1.0 - alpha) + tint * alpha).astype(np.uint8)
    if border is not None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, cnts, -1, border, S(1), cv2.LINE_AA)
        t = S(7)                                     # corner accent ticks
        for (cxx, cyy, dx, dy) in ((0, 0, 1, 1), (w, 0, -1, 1),
                                   (0, h, 1, -1), (w, h, -1, -1)):
            cv2.line(roi, (cxx, cyy), (cxx + dx * t, cyy), border, S(2), cv2.LINE_AA)
            cv2.line(roi, (cxx, cyy), (cxx, cyy + dy * t), border, S(2), cv2.LINE_AA)


def dashed_line(img, p1, p2, color, thick=1, dash=9, gap=7):
    p1, p2 = np.array(p1, float), np.array(p2, float)
    d = np.linalg.norm(p2 - p1)
    if d < 1:
        return
    v = (p2 - p1) / d
    for i in range(int(d // (dash + gap)) + 1):
        s = p1 + v * (i * (dash + gap))
        e = p1 + v * min(i * (dash + gap) + dash, d)
        cv2.line(img, tuple(s.astype(int)), tuple(e.astype(int)),
                 color, thick, cv2.LINE_AA)


def draw_brackets(frame, box, color, thick=None):
    """Four corner brackets instead of a full rectangle — cleaner + tactical."""
    x1, y1, x2, y2 = box
    thick = thick or S(2)
    L = max(S(8), min(S(24), (x2 - x1) // 4, (y2 - y1) // 4))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, thick, cv2.LINE_AA)


def draw_reticle(frame, center, color, r=None, gap=None):
    cx, cy = center
    r, gap = r or S(9), gap or S(4)
    cv2.circle(frame, (cx, cy), r, color, S(1), cv2.LINE_AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(frame, (cx + dx * (r + gap + S(5)), cy + dy * (r + gap + S(5))),
                 (cx + dx * (r + gap), cy + dy * (r + gap)), color, S(1), cv2.LINE_AA)


def draw_velocity(frame, center, vel, color, predict=16):
    vx, vy = vel
    if math.hypot(vx, vy) < 0.6:
        return
    cx, cy = center
    cv2.arrowedLine(frame, (cx, cy), (int(cx + vx * 6), int(cy + vy * 6)),
                    color, 1, cv2.LINE_AA, tipLength=0.3)   # thin
    px, py = int(cx + vx * predict), int(cy + vy * predict)
    dashed_line(frame, (cx, cy), (px, py), T_AMBER, 1)      # predicted path
    cv2.circle(frame, (px, py), S(3), T_AMBER, 1, cv2.LINE_AA)


def draw_label(frame, x, y, name, conf, color, track_id, moving, speed=None):
    pct = int(round(conf * 100))
    tag = f"{name.upper()} {pct}%"
    if track_id is not None and track_id >= 0:
        tag += f" T{track_id:02d}"
    w, h = mono_w(tag, 0.55) + S(16), _mono_size(0.55) + S(8)
    ly = max(0, y - h - S(2))
    panel(frame, x, ly, w, h, alpha=0.62, radius=0)
    cv2.rectangle(frame, (x, ly), (x + S(4), ly + h), color, -1)     # colour tab
    text(frame, tag, x + S(10), ly + h - S(6), 0.55, T_WHITE)
    bar_w = int((w - S(4)) * conf)                                   # conf meter
    cv2.line(frame, (x + S(2), ly + h - 1), (x + S(2) + bar_w, ly + h - 1),
             color, S(2))
    if moving:
        cv2.circle(frame, (x + w + S(6), ly + h // 2), S(3), T_AMBER, -1,
                   cv2.LINE_AA)


def fit_into(crop, w, h):
    """Resize `crop` to fit a fixed w×h tile, preserving aspect with letterbox
    padding — so pinned zoom tiles are all the same size and don't jitter."""
    ch, cw = crop.shape[:2]
    s = min(w / cw, h / ch)
    nw, nh = max(1, int(cw * s)), max(1, int(ch * s))
    r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 3), np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = r
    return canvas


def draw_zoom_pinned(frame, clean, target_box, slot, color, header,
                     coast=False, lock=False):
    """A zoom inset pinned at a FIXED screen position (`slot` = x,y,w,h). The box
    does not follow the object; instead an amber tether runs from the object to
    the box's centre crosshair, so you can tell which box is which target."""
    H, W = frame.shape[:2]
    sx, sy, sw, sh = slot
    sx, sy = clamp(sx, 0, W - sw), clamp(sy, 0, H - sh)
    x1, y1, x2, y2 = target_box
    pad_x, pad_y = int((x2 - x1) * 0.12) + 4, int((y2 - y1) * 0.12) + 4
    cx1, cy1 = clamp(x1 - pad_x, 0, W - 1), clamp(y1 - pad_y, 0, H - 1)
    cx2, cy2 = clamp(x2 + pad_x, 1, W), clamp(y2 + pad_y, 1, H)
    crop = clean[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return
    mag = sw / max(1, (cx2 - cx1))
    frame[sy:sy + sh, sx:sx + sw] = fit_into(crop, sw, sh)
    cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), color, S(1), cv2.LINE_AA)
    tk = S(14)
    for (cxx, cyy, dx, dy) in ((sx, sy, 1, 1), (sx + sw, sy, -1, 1),
                               (sx, sy + sh, 1, -1), (sx + sw, sy + sh, -1, -1)):
        cv2.line(frame, (cxx, cyy), (cxx + dx * tk, cyy), color, S(2), cv2.LINE_AA)
        cv2.line(frame, (cxx, cyy), (cxx, cyy + dy * tk), color, S(2), cv2.LINE_AA)
    csx, csy = sx + sw // 2, sy + sh // 2
    draw_reticle(frame, (csx, csy), color)
    bh = _mono_size(0.45) + S(5)
    panel(frame, sx, sy, sw, bh, alpha=0.62, radius=0)
    text(frame, header, sx + S(6), sy + bh - S(4), 0.45, T_AMBER if lock else color)
    tcx, tcy = (x1 + x2) // 2, (y1 + y2) // 2
    panel(frame, sx, sy + sh - bh, sw, bh, alpha=0.62, radius=0)
    text(frame, f"X{tcx} Y{tcy}  {mag:.1f}x", sx + S(6), sy + sh - S(5), 0.42,
         T_GREEN_DIM)
    # tether: object  ->  box crosshair (moves with the object) — thin
    cv2.line(frame, (tcx, tcy), (csx, csy), T_AMBER, 1, cv2.LINE_AA)
    cv2.circle(frame, (tcx, tcy), S(2), T_AMBER, -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Unified source reader for local cameras, network/IP cameras (RTSP/HTTP) and
# video files.
#   • cameras + network streams -> a background thread always holds the *newest*
#     frame (drops stale frames so we stay real-time) and auto-reconnects.
#   • video files               -> read synchronously, frame by frame.
# --------------------------------------------------------------------------- #
STREAM_SCHEMES = ("rtsp://", "rtmps://", "rtmp://", "http://", "https://",
                  "udp://", "tcp://")


class CameraStream:
    def __init__(self, src, width, height, rtsp_udp=False):
        s = str(src)
        self.is_index = s.isdigit()
        self.is_stream = (not self.is_index) and s.lower().startswith(STREAM_SCHEMES)
        self.is_file = not (self.is_index or self.is_stream)
        self.live = self.is_index or self.is_stream      # threaded sources
        self._src = int(s) if self.is_index else src
        self._w, self._h = width, height

        # Tune the FFmpeg backend per protocol. (Must be set before
        # VideoCapture is created; FFmpeg reads the env var at open time.)
        low = s.lower()
        if low.startswith(("rtsp://", "rtmp://", "rtmps://")):
            # Force TCP for RTSP by default — far more reliable than UDP/WiFi.
            transport = "udp" if rtsp_udp else "tcp"
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                                  f"rtsp_transport;{transport}")
        elif low.startswith("udp://"):
            # Big receive buffer + tolerate packet loss (GoPro WiFi preview).
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                                  "fifo_size;5000000|overrun_nonfatal;1")

        self.lock = threading.Lock()
        self.stopped = False
        self.frame = None
        self.cap = self._open()
        self.opened = self.cap.isOpened()
        if self.opened and self.live:
            ok, f = self.cap.read()
            self.frame = f if ok else None
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _backend(self):
        if self.is_index:
            return cam_backend()
        if self.is_stream:
            return cv2.CAP_FFMPEG
        return cv2.CAP_ANY

    def _open(self):
        cap = cv2.VideoCapture(self._src, self._backend())
        if self.is_index:
            if self._w:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
            if self._h:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
        if self.live:
            # keep latency low: don't let decoded frames pile up in the buffer
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass
        return cap

    def _loop(self):
        fails = 0
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                fails += 1
                if self.is_stream and fails >= 25:       # stream dropped: reconnect
                    try:
                        self.cap.release()
                    except cv2.error:
                        pass
                    time.sleep(1.0)
                    self.cap = self._open()
                    fails = 0
                else:
                    time.sleep(0.01)
                continue
            fails = 0
            with self.lock:
                self.frame = f

    def read(self):
        if self.is_file:
            return self.cap.read()
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        time.sleep(0.05)
        try:
            self.cap.release()
        except cv2.error:
            pass


# --------------------------------------------------------------------------- #
# FFmpeg-subprocess reader — used for lossy UDP streams (e.g. the GoPro WiFi
# preview). OpenCV's built-in decoder has no error tolerance, so dropped packets
# turn into a flood of "error while decoding MB" and garbage frames. A dedicated
# FFmpeg process with a large receive buffer + error concealment is far more
# robust and stays quiet. It emits fixed-size BGR frames we read straight into
# numpy. Same read()/release() interface as CameraStream.
# --------------------------------------------------------------------------- #
class FFmpegStream:
    is_file = False
    is_stream = True

    def __init__(self, url, out_w=960, out_h=540):
        self.url = url
        self.w, self.h = out_w, out_h
        self.frame_bytes = out_w * out_h * 3
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.proc = None
        self.opened = self._spawn()
        if self.opened:
            threading.Thread(target=self._loop, daemon=True).start()

    def _ffmpeg_exe(self):
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()

    def _cmd(self):
        url = self.url
        if url.lower().startswith("udp://") and "?" not in url:
            # SMALL buffers + drop-on-overrun => stay near-live (don't queue a
            # backlog) when the detector can't keep up with the camera's fps
            url += "?overrun_nonfatal=1&fifo_size=2000&buffer_size=500000"
        vf = (f"scale={self.w}:{self.h}:force_original_aspect_ratio=decrease,"
              f"pad={self.w}:{self.h}:(ow-iw)/2:(oh-ih)/2")
        return [self._ffmpeg_exe(), "-hide_banner", "-loglevel", "fatal",
                "-fflags", "nobuffer", "-flags", "low_delay",
                "-err_detect", "ignore_err",
                # short analysis window so the live view starts quickly
                "-probesize", "3000000", "-analyzeduration", "3000000",
                "-i", url, "-an", "-vf", vf,
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    def _spawn(self):
        try:
            # tiny pipe buffer so Python never holds a backlog of stale frames
            self.proc = subprocess.Popen(self._cmd(), stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, bufsize=65536)
            return True
        except (OSError, ValueError):
            return False

    def _loop(self):
        while not self.stopped:
            try:
                raw = self.proc.stdout.read(self.frame_bytes)
            except (OSError, ValueError):
                raw = b""
            if not raw or len(raw) < self.frame_bytes:
                if not self.stopped and (self.proc is None
                                         or self.proc.poll() is not None):
                    time.sleep(1.0)          # ffmpeg died -> reconnect
                    self._spawn()
                continue
            f = np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        if self.proc is not None:
            try:
                self.proc.kill()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# GoPro WiFi live-preview helper (Hero 4/5/6/7 legacy API).
#
# Join the camera's WiFi first (GoPro app → Connections → Connect this camera).
# Then we tell the camera to start its UDP preview stream and keep pinging it so
# it doesn't stop. The video itself arrives on  udp://@:8554  which CameraStream
# reads. No HDMI / capture card / GoPro Webcam app required.
# --------------------------------------------------------------------------- #
GOPRO_DEFAULT_IP = "10.5.5.9"
GOPRO_UDP_URL = "udp://0.0.0.0:8554"   # 0.0.0.0 binds reliably on Windows too


class GoProStream:
    KEEPALIVE = b"_GPHD_:0:0:2:0.000000\n"

    def __init__(self, ip=GOPRO_DEFAULT_IP, port=8554):
        self.ip = ip
        self.port = port
        self.stopped = False
        self.sock = None
        self.battery = None        # 0-100 percent, or None until first poll
        self.charging = False

    def start(self):
        """Tell the camera to (re)start streaming and begin the keep-alive loop.
        Raises on failure to reach the camera."""
        import urllib.request
        url = (f"http://{self.ip}/gp/gpControl/execute"
               f"?p1=gpStream&a1=proto_v2&c1=restart")
        urllib.request.urlopen(url, timeout=5).read()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        threading.Thread(target=self._keepalive, daemon=True).start()

    def _keepalive(self):
        tick = 0
        while not self.stopped:
            try:
                self.sock.sendto(self.KEEPALIVE, (self.ip, self.port))
            except OSError:
                pass
            if tick % 4 == 0:                # poll camera status every ~10 s
                self._poll_status()
            tick += 1
            time.sleep(2.5)

    def _poll_status(self):
        """Read battery level from the camera's status endpoint (legacy API:
        status 70 = battery %, status 2 = bars 0-3 / 4=charging)."""
        import json
        import urllib.request
        try:
            raw = urllib.request.urlopen(
                f"http://{self.ip}/gp/gpControl/status", timeout=4).read()
            st = json.loads(raw).get("status", {})
        except Exception:
            return
        pct, bars = st.get("70"), st.get("2")
        if pct is not None:
            self.battery = int(pct)
        elif bars is not None:               # no %: approximate from bars
            self.battery = {0: 5, 1: 25, 2: 55, 3: 95}.get(int(bars), self.battery)
        if bars is not None:
            self.charging = int(bars) == 4

    def stop(self):
        self.stopped = True
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Per-object memory: trail, class-vote history, timing.
# --------------------------------------------------------------------------- #
class TrackState:
    __slots__ = ("trail", "cls_hist", "first_seen", "last_seen", "sbox", "sconf")

    def __init__(self, t):
        self.trail = deque(maxlen=40)
        self.cls_hist = deque(maxlen=20)
        self.first_seen = t
        self.last_seen = t
        self.sbox = None        # EMA-smoothed bounding box [x1,y1,x2,y2]
        self.sconf = None       # EMA-smoothed confidence

    def update_box(self, box, alpha=0.4):
        """Smooth the box to kill jitter — but ADAPTIVELY: when the box jumps a
        lot (a fast mover) snap almost fully to it, so the box rides ON the
        object instead of lagging behind it. Slow/static objects stay smoothed."""
        if self.sbox is None:
            self.sbox = list(map(float, box))
        else:
            move = sum(abs(box[i] - self.sbox[i]) for i in range(4))
            w = max(1.0, self.sbox[2] - self.sbox[0])
            a = clamp(alpha + (move / w) * 1.5, alpha, 0.92)   # big jump → snap
            for i in range(4):
                self.sbox[i] = (1 - a) * self.sbox[i] + a * box[i]
        return tuple(int(round(v)) for v in self.sbox)

    def smoothed(self):
        """Majority-vote class + smoothed confidence — stabilises flickery
        frame-to-frame predictions."""
        votes = Counter(c for c, _ in self.cls_hist)
        best = votes.most_common(1)[0][0]
        confs = [cf for c, cf in self.cls_hist if c == best]
        conf = sum(confs) / len(confs)
        self.sconf = conf if self.sconf is None else 0.8 * self.sconf + 0.2 * conf
        return best, self.sconf

    def velocity(self, k=3):     # short window → registers fast movers quickly
        k = min(k, len(self.trail) - 1)
        if k < 1:
            return (0.0, 0.0)
        ax, ay = self.trail[-1]
        bx, by = self.trail[-1 - k]
        return ((ax - bx) / k, (ay - by) / k)


class Toasts:
    """Transient on-screen notifications."""
    def __init__(self):
        self.items = deque()

    def add(self, msg, dur=2.2):
        self.items.append((msg, time.time() + dur))

    def draw(self, img):
        now = time.time()
        while self.items and self.items[0][1] < now:
            self.items.popleft()
        bh = _mono_size(0.5) + S(12)
        y = img.shape[0] - (_mono_size(0.5) + S(30))
        for msg, _ in list(self.items)[-4:]:
            line = "> " + msg
            panel(img, S(10), y - bh + S(7), mono_w(line, 0.5) + S(22), bh,
                  alpha=0.66, radius=S(4), border=T_AMBER)
            text(img, line, S(18), y, 0.5, T_WHITE)
            y -= bh + S(6)


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #
class TrackerApp:
    DEFAULT_TOGGLES = {
        "dots": True,       # movement dots
        "boxes": True,      # object box outlines
        "labels": True,     # class + confidence
        "velocity": True,   # prediction arrows
        "zoom": True,       # zoom boxes for moving objects
    }

    def __init__(self, args):
        self.args = args
        self.toggles = dict(self.DEFAULT_TOGGLES)
        self.sens = clamp(int(round((0.9 - args.conf) / 0.0085)), 1, 100)
        self.dot_sens = 60          # movement-dot sensitivity (slider 1-100)
        self.zoom_sens = 82         # zoom motion-gate sensitivity (slider 1-100)
        self.mirror = False
        self.fullscreen = False

        self.states = {}                 # id -> TrackState
        self.toasts = Toasts()
        self.selected_id = None
        self.pending_click = None
        self.show_settings = False  # clickable settings panel open?
        self.hotspots = []          # [(x1,y1,x2,y2,action)] clickable regions
        self.quit = False           # set by the Quit button
        self._do_rec = self._do_photo = False
        self.show_help = False
        self.paused = False
        self.frame_idx = 0
        self.fps = 0.0
        self.fps_hist = deque(maxlen=90)
        self.writer = None
        self.rec_name = ""
        self.start_time = time.time()
        self.device = "cpu"
        self.gopro = None
        self.snaps = 0
        self.mouse_pos = None
        self.zoom = 1.0             # digital zoom (1.0 = full view)
        self.zoom_cx = 0.5          # zoom centre, full-frame normalised coords
        self.zoom_cy = 0.5
        self._vw = self._vh = 0     # current view size (set each frame)
        self._pan_last = None
        self.zoom_slots = {}        # track id -> fixed rail slot index (stable)
        self.zoom_hold = {}         # track id -> last frame it was "moving"
        self.prev_gray = None       # previous frame (grayscale) for movement dots
        self.motion_dots = []        # [x, y, life] fading green movement dots
        self.model_name = ""

        self.load_settings()

    # sensitivity slider <-> confidence threshold
    @property
    def conf(self):
        return round(0.9 - (self.sens / 100.0) * 0.85, 3)

    # ---- settings persistence ------------------------------------------- #
    def load_settings(self):
        try:
            with open(SETTINGS_PATH) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        for k, v in data.get("toggles", {}).items():
            if k in self.toggles:
                self.toggles[k] = bool(v)
        self.sens = clamp(int(data.get("sens", self.sens)), 1, 100)
        self.mirror = bool(data.get("mirror", False))

    def save_settings(self):
        try:
            with open(SETTINGS_PATH, "w") as f:
                json.dump({"toggles": self.toggles, "sens": self.sens,
                           "mirror": self.mirror}, f, indent=2)
        except OSError:
            pass

    # ---- mouse selection ------------------------------------------------ #
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pos = (x, y)
            for hx1, hy1, hx2, hy2, action in self.hotspots:   # UI buttons first
                if hx1 <= x <= hx2 and hy1 <= y <= hy2:
                    self._click_action(action)
                    return
            self.pending_click = (x, y)                        # else: lock-on
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._pan_last = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
            if (flags & cv2.EVENT_FLAG_RBUTTON) and self._pan_last:
                self._pan(x - self._pan_last[0], y - self._pan_last[1])
                self._pan_last = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:        # two-finger scroll = zoom
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                delta = flags
            self._zoom_at(x, y, 1.12 if delta > 0 else 1 / 1.12)

    def _click_action(self, action):
        """Dispatch a click on a settings/toolbar button."""
        if action == "settings":
            self.show_settings = not self.show_settings
        elif action.startswith("toggle:"):
            self.toggles[action.split(":", 1)[1]] ^= True
        elif action == "rec":
            self._do_rec = True
        elif action == "photo":
            self._do_photo = True
        elif action == "zoomreset":
            self.zoom, self.zoom_cx, self.zoom_cy = 1.0, 0.5, 0.5
            self.toasts.add("Zoom reset")
        elif action == "mirror":
            self.mirror = not self.mirror
        elif action == "fullscreen":
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif action == "unlock":
            self.selected_id = None
        elif action == "quit":
            self.quit = True

    # ---- digital zoom (crop the frame *before* detection) --------------- #
    def _zoom_at(self, vx, vy, factor):
        """Zoom toward the point (vx,vy) in the displayed view, photo-style."""
        W, H = self._vw, self._vh
        z0 = self.zoom
        z1 = clamp(z0 * factor, 1.0, 8.0)
        if W and z1 != z0:
            cw0 = 1.0 / z0
            nx1 = clamp(self.zoom_cx - cw0 / 2, 0.0, 1.0 - cw0)
            ny1 = clamp(self.zoom_cy - cw0 / 2, 0.0, 1.0 - cw0)
            fx = nx1 + (vx / W) * cw0              # full-frame point under cursor
            fy = ny1 + (vy / H) * cw0
            cw1 = 1.0 / z1
            self.zoom_cx = clamp(fx - (vx / W) * cw1 + cw1 / 2, cw1 / 2, 1 - cw1 / 2)
            self.zoom_cy = clamp(fy - (vy / H) * cw1 + cw1 / 2, cw1 / 2, 1 - cw1 / 2)
        self.zoom = z1
        if z1 <= 1.001:
            self.zoom_cx = self.zoom_cy = 0.5

    def _pan(self, dx, dy):
        if self.zoom <= 1.001 or not self._vw:
            return
        half = 0.5 / self.zoom
        self.zoom_cx = clamp(self.zoom_cx - (dx / self._vw) / self.zoom,
                             half, 1 - half)
        self.zoom_cy = clamp(self.zoom_cy - (dy / self._vh) / self.zoom,
                             half, 1 - half)

    def _apply_zoom(self, frame):
        """Crop to the zoomed region and scale it back up — this is what gets
        fed to the detector, so it only 'sees' what you've zoomed into."""
        if self.zoom <= 1.001:
            return frame
        H, W = frame.shape[:2]
        cw, ch = W / self.zoom, H / self.zoom
        x1 = clamp(self.zoom_cx * W - cw / 2, 0, W - cw)
        y1 = clamp(self.zoom_cy * H - ch / 2, 0, H - ch)
        self.zoom_cx, self.zoom_cy = (x1 + cw / 2) / W, (y1 + ch / 2) / H
        crop = frame[int(y1):int(y1 + ch), int(x1):int(x1 + cw)]
        return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

    def _resolve_model_name(self):
        """Pick a lean *detection* model (no segmentation masks needed for our
        visuals → much faster). GoPro/low-res gets nano; cameras/files get small.
        We only ever report the WANTED_IDS classes, so the rest is ignored."""
        if self.args.model != "auto":
            return self.args.model
        is_gopro = (self.args.gopro
                    or str(self.args.source).lower().startswith("udp://"))
        return "yolov8m.pt"      # best balance: strong AND fast enough (~15fps)
        #                          to actually catch fast movers between frames.
        #                          --model yolov8l.pt / yolov8x.pt = more accurate
        #                          per frame but lower fps (can MISS fast cars).

    # ---- main loop ------------------------------------------------------ #
    def run(self):
        from ultralytics import YOLO

        self.device = pick_device(self.args.device)

        # auto-pick the model + inference size to suit the source
        is_gopro = (self.args.gopro
                    or str(self.args.source).lower().startswith("udp://"))
        self.model_name = self._resolve_model_name()
        if self.args.imgsz is None:              # keep it real-time (fps matters
            self.args.imgsz = 640 if is_gopro else 960   # for tracking fast cars)
        if self.args.model == "auto":
            print(f"Auto-selected model: {self.model_name} @ imgsz "
                  f"{self.args.imgsz}")

        print("Loading model…", flush=True)
        model = YOLO(self.model_name)
        names = model.names
        print(f"Model: {self.model_name}   device: {self.device}")

        # GoPro WiFi mode: start the camera's preview stream + keep-alive, then
        # read it as a UDP source.
        if self.args.gopro:
            self.gopro = GoProStream(self.args.gopro_ip)
            print(f"Starting GoPro stream at {self.args.gopro_ip} …")
            try:
                self.gopro.start()
            except Exception as e:
                sys.exit(
                    f"Couldn't reach the GoPro at {self.args.gopro_ip}.\n"
                    f"  • Turn the camera on and join its WiFi first "
                    f"(GoPro app → Connections, or the camera's Connections "
                    f"menu).\n  • Then your Mac's WiFi should show a network "
                    f"like 'GPxxxxxxxx'.\n  Error: {e}")
            if self.args.source == "0":      # default → use the GoPro feed
                self.args.source = GOPRO_UDP_URL
            print("GoPro stream starting — decoding via FFmpeg (this is "
                  "tolerant of WiFi packet loss; give it a few seconds)…")

        # Lossy UDP (GoPro) → robust FFmpeg reader; everything else → OpenCV.
        if str(self.args.source).lower().startswith("udp://"):
            stream = FFmpegStream(self.args.source,
                                  self.args.width, self.args.height)
        else:
            stream = CameraStream(self.args.source, self.args.width,
                                  self.args.height, rtsp_udp=self.args.udp)
        if not stream.opened:
            hint = ("Check the URL/credentials and that the camera is reachable "
                    "on your network — try opening the same URL in VLC first."
                    if stream.is_stream else "Try --list-cameras.")
            sys.exit(f"Could not open source {self.args.source!r}. {hint}")
        if stream.is_stream and not self.args.gopro:
            print("Connected to network stream. Tip: use the camera's "
                  "low-res 'sub-stream' URL for smoother real-time tracking.")

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Detection", WINDOW, self.sens, 100,
                           lambda v: setattr(self, "sens", max(1, v)))
        cv2.createTrackbar("Dots", WINDOW, self.dot_sens, 100,
                           lambda v: setattr(self, "dot_sens", max(1, v)))
        cv2.createTrackbar("Zoom motion", WINDOW, self.zoom_sens, 100,
                           lambda v: setattr(self, "zoom_sens", max(1, v)))
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        self.toasts.add("Click an object to lock on.  + / - zoom toward the "
                        "mouse.  Click SETTINGS (bottom-left) to toggle things.", 6)

        last_annotated = None
        t_prev = time.time()
        while True:
            if not self.paused:
                ok, frame = stream.read()
                if not ok or frame is None:
                    if stream.is_file:
                        break
                    # live source still connecting / momentary hiccup — show a
                    # placeholder so the window isn't blank, then keep trying
                    ph = np.zeros((self.args.height or 480,
                                   self.args.width or 640, 3), np.uint8)
                    text(ph, "// CONNECTING TO SOURCE ...", 30,
                         ph.shape[0] // 2, 0.8, T_GREEN)
                    cv2.imshow(WINDOW, _HUD.flush(ph))
                    if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
                        break
                    continue
                self.frame_idx += 1
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                self._vw, self._vh = frame.shape[1], frame.shape[0]
                frame = self._apply_zoom(frame)        # crop to the zoom region
                self.sens = max(1, cv2.getTrackbarPos("Detection", WINDOW))
                self.dot_sens = max(1, cv2.getTrackbarPos("Dots", WINDOW))
                self.zoom_sens = max(1, cv2.getTrackbarPos("Zoom motion", WINDOW))
                annotated = self.process(model, names, frame)
                last_annotated = annotated
                if self._do_photo:
                    self._do_photo = False
                    fn = f"capture_{int(time.time())}.png"
                    cv2.imwrite(fn, annotated)
                    self.snaps += 1
                    self.toasts.add(f"Saved {fn}")
                if self._do_rec:
                    self._do_rec = False
                    self._toggle_record(annotated)
                if self.writer is not None:
                    self.writer.write(annotated)
            else:
                annotated = last_annotated.copy() if last_annotated is not None else None
                if annotated is None:
                    break
                text(annotated, "// PAUSED", annotated.shape[1] // 2 - 70,
                     annotated.shape[0] // 2, 1.0, T_AMBER)
                annotated = _HUD.flush(annotated)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
                self.fps_hist.append(self.fps)

            cv2.imshow(WINDOW, annotated)
            key = cv2.waitKey(1) & 0xFF
            if self.quit or not self.handle_key(key, annotated):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

        if self.writer is not None:
            self.writer.release()
        if self.gopro is not None:
            self.gopro.stop()
        self.save_settings()
        stream.release()
        cv2.destroyAllWindows()

    # ---- per-frame processing ------------------------------------------ #
    def process(self, model, names, frame):
        H, W = frame.shape[:2]
        set_ui_scale(H)                  # scale all UI to the frame resolution
        diag = math.hypot(H, W)
        # Zoom-motion slider → how little movement counts as "moving" (gates the
        # zoom boxes). Higher slider = more sensitive = catches fast/slow cars.
        move_thresh = diag * (0.011 - self.zoom_sens * 0.0001)
        clean = frame.copy()
        res = model.track(frame, persist=True, conf=self.conf, iou=0.5,
                          imgsz=self.args.imgsz, device=self.device,
                          classes=WANTED_IDS, verbose=False,
                          tracker="bytetrack.yaml")[0]

        boxes = res.boxes
        polys = res.masks.xy if res.masks is not None else None

        def make_item(tid, box, name, conf, vx, vy, poly, coast):
            x1, y1, x2, y2 = box
            spd = math.hypot(vx, vy)
            return {
                "id": tid, "name": name, "conf": conf, "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2), "poly": poly,
                "area": (x2 - x1) * (y2 - y1), "vel": (vx, vy), "speed": spd,
                "moving": spd > move_thresh, "coast": coast,
            }

        items = []
        present = set()
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            clss = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            ids = (boxes.id.int().cpu().tolist() if boxes.id is not None
                   else [-(i + 1) for i in range(len(clss))])
            for i in range(len(clss)):
                tid = ids[i]
                st = self.states.get(tid)
                if st is None:
                    st = self.states[tid] = TrackState(self.frame_idx)
                st.cls_hist.append((clss[i], confs[i]))
                st.last_seen = self.frame_idx
                sbox = st.update_box(xyxy[i])            # EMA-smoothed box
                cx, cy = (sbox[0] + sbox[2]) // 2, (sbox[1] + sbox[3]) // 2
                st.trail.append((cx, cy))
                s_cls, s_conf = st.smoothed()
                vx, vy = st.velocity()
                name = names.get(s_cls, str(s_cls))
                items.append(make_item(tid, sbox, name, s_conf, vx, vy, None, False))
                present.add(tid)

        # "coast" tracks that briefly dropped out, so labels don't blink
        COAST = 8
        for tid, st in self.states.items():
            if tid in present or tid < 0 or st.sbox is None:
                continue
            miss = self.frame_idx - st.last_seen
            if miss > COAST or len(st.cls_hist) < 4:
                continue
            s_cls, s_conf = st.smoothed()
            vx, vy = st.velocity()
            box = tuple(int(round(st.sbox[j] + (vx if j % 2 == 0 else vy) * miss))
                        for j in range(4))
            items.append(make_item(tid, box, names.get(s_cls, str(s_cls)),
                                   s_conf, vx, vy, None, True))

        # forget long-lost tracks
        for tid in [k for k, s in self.states.items()
                    if self.frame_idx - s.last_seen > 45]:
            del self.states[tid]
            self.zoom_slots.pop(tid, None)
            self.zoom_hold.pop(tid, None)

        # resolve a pending click into a selection (smallest box hit)
        if self.pending_click is not None:
            px, py = self.pending_click
            self.pending_click = None
            hit = None
            for it in items:
                x1, y1, x2, y2 = it["box"]
                if x1 <= px <= x2 and y1 <= py <= y2:
                    if hit is None or it["area"] < hit["area"]:
                        hit = it
            self.selected_id = hit["id"] if hit else None
            self.toasts.add(f"Locked on {hit['name']}" if hit else "Lock cleared")

        # --- fading green movement dots ---
        if self.toggles["dots"]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._draw_movement_dots(frame, gray)
            self.prev_gray = gray
        else:
            self.prev_gray = None
            self.motion_dots.clear()

        items.sort(key=lambda d: d["area"], reverse=True)
        shown = Counter()
        for it in items:
            base = tactical_color(it["id"] if it["id"] >= 0 else hash(it["name"]))
            color = tuple(int(c * 0.5) for c in base) if it["coast"] else base
            x1, y1, x2, y2 = it["box"]
            if self.toggles["boxes"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            if self.toggles["velocity"] and it["moving"] and not it["coast"]:
                draw_velocity(frame, it["center"], it["vel"], color)
            if self.toggles["labels"]:
                draw_label(frame, x1, y1, it["name"], it["conf"], color, None,
                           it["moving"] and not it["coast"])
            shown[it["name"]] += 1

        # zoom boxes — ONLY for moving objects (motion-gated, with hysteresis)
        if self.toggles["zoom"]:
            self._draw_zoom_rail(frame, clean, items, W, H)

        # selection emphasis on top of everything
        sel = next((it for it in items if it["id"] == self.selected_id), None)
        if sel is not None:
            self._draw_selection(frame, clean, sel)
        elif self.selected_id is not None:
            text(frame, f"// ACQUIRING T{self.selected_id:02d} ...", 14, H - 74,
                 0.5, T_AMBER)

        self.hotspots = []           # rebuilt by the toolbar + settings panel
        self._draw_hud(frame, shown)
        self._draw_zoom_minimap(frame)
        if self.writer is not None:
            self._draw_rec_badge(frame)
        self.toasts.draw(frame)
        self._draw_settings_panel(frame)
        return _HUD.flush(frame)

    # ---- small fading green movement dots ------------------------------ #
    def _draw_movement_dots(self, frame, gray):
        """Tiny GREEN dots sampled from the actual moving pixels (they cluster on
        the mover) that fade out by going TRANSPARENT (alpha) — staying green,
        not darkening. Total capped so it never floods."""
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            thresh = clamp(int(55 - self.dot_sens * 0.5), 5, 55)  # Dots slider
            diff = cv2.absdiff(gray, self.prev_gray)
            ys, xs = np.where(diff > thresh)
            n = len(xs)
            if n:
                sel = np.random.choice(n, min(40, n), replace=False)
                for i in sel:
                    self.motion_dots.append([int(xs[i]), int(ys[i]), 1.0])
        r = max(1, S(1))                          # SMALL dots
        H, W = frame.shape[:2]
        amap = np.zeros((H, W), np.float32)        # per-pixel alpha = dot life
        alive = []
        for x, y, life in self.motion_dots:
            life -= 0.10
            if life <= 0.06:
                continue
            cv2.circle(amap, (x, y), r, float(life), -1)
            alive.append([x, y, life])
        self.motion_dots = alive[-220:]
        m = amap > 0.01                            # alpha-blend green where dots are
        if m.any():
            a = amap[m][:, None]
            frame[m] = (frame[m] * (1 - a) + _DOT_GREEN * a).astype(np.uint8)

    # ---- pinned zoom rail ---------------------------------------------- #
    def _draw_zoom_rail(self, frame, clean, items, W, H):
        """MAG-TRK boxes in fixed slots along the top — they don't follow the
        target; an amber tether connects each target to its box crosshair.
        EVERY detected object gets a box (up to the number of slots that fit).
        Each object keeps its slot for as long as it's on screen and slots are
        never reshuffled, so the boxes stay put and steady (no flicker)."""
        # big boxes so they actually magnify (not object-sized), scaled to res
        SW, SH, gap, margin, rail_y = S(248), S(176), S(12), S(12), S(10)
        reserve_left = S(408)                   # keep clear of the telemetry HUD
        nslots = min(self.args.max_zoom,
                     max(0, (W - reserve_left - margin + gap) // (SW + gap)))
        if nslots <= 0:
            return
        cur = {it["id"]: it for it in items
               if it["id"] >= 0 and it["id"] != self.selected_id}
        # MOTION GATE (balanced): a zoom box is for objects that are moving.
        # A ~1s hold means a brief stop doesn't drop the box (no flicker).
        HOLD = 20
        for it in cur.values():
            if it["moving"] and not it["coast"]:
                self.zoom_hold[it["id"]] = self.frame_idx
        wanting = [t for t in cur
                   if self.frame_idx - self.zoom_hold.get(t, -999) <= HOLD]
        for t in list(self.zoom_slots):          # release when no longer wanted
            if t not in wanting:
                del self.zoom_slots[t]
        used = set(self.zoom_slots.values())
        for t in sorted((t for t in wanting if t not in self.zoom_slots),
                        key=lambda tid: cur[tid]["area"], reverse=True):
            free = next((s for s in range(nslots) if s not in used), None)
            if free is None:
                break
            self.zoom_slots[t] = free
            used.add(free)
        total = nslots * SW + (nslots - 1) * gap
        x0 = W - margin - total
        for t, slot in sorted(self.zoom_slots.items(), key=lambda kv: kv[1]):
            if slot >= nslots:
                continue
            it = cur[t]
            rect = (x0 + slot * (SW + gap), rail_y, SW, SH)
            draw_zoom_pinned(frame, clean, it["box"], rect, tactical_color(t),
                             f"MAG-TRK T{t:02d}", coast=it["coast"])

    # ---- lock-on follow-cam -------------------------------------------- #
    def _draw_selection(self, frame, clean, it):
        x1, y1, x2, y2 = it["box"]
        color = T_AMBER
        pulse = int(S(6) + S(4) * math.sin(self.frame_idx * 0.3))
        cx, cy = it["center"]
        rad = int(max(x2 - x1, y2 - y1) * 0.6) + pulse
        cv2.circle(frame, (cx, cy), rad, color, S(2), cv2.LINE_AA)
        draw_brackets(frame, it["box"], color)
        for a in range(0, 360, 90):             # rotating lock ticks
            ang = math.radians(a + self.frame_idx * 2)
            ex, ey = int(cx + math.cos(ang) * rad), int(cy + math.sin(ang) * rad)
            cv2.circle(frame, (ex, ey), S(3), color, -1, cv2.LINE_AA)
        text(frame, "LOCKED", cx - mono_w("LOCKED", 0.5) // 2, cy - rad - S(6),
             0.5, color)
        # big lock-cam pinned lower-right (below the MAG-TRK rail), with tether
        H, W = frame.shape[:2]
        lw, lhgt = S(300), S(214)
        rect = (W - lw - S(14), S(10) + S(176) + S(28), lw, lhgt)
        draw_zoom_pinned(frame, clean, it["box"], rect, color,
                         f"LOCK {it['name'].upper()} {int(it['conf']*100)}%",
                         coast=it.get("coast", False), lock=True)

    # ---- HUD ------------------------------------------------------------ #
    def _model_tag(self):
        b = os.path.basename(self.model_name or "?")
        b = b.replace(".pt", "").replace("yolov", "").replace("-seg", "")
        return b.upper() or "?"

    def _draw_hud(self, frame, shown):
        H, W = frame.shape[:2]
        n = sum(shown.values())
        lines = [
            ("MUC.IO TRACKER", T_GREEN),
            (f"FPS {self.fps:04.1f}   NET {self._model_tag()}   "
             f"{self.device.upper()}   OBJ {n:02d}", T_GREEN_DIM),
        ]
        if shown:
            lines.append(("  " + "  ".join(f"{v}x{k.upper()}"
                          for k, v in shown.most_common(4)), T_CYAN))
        if self.gopro is not None:                 # GoPro battery readout
            pct = self.gopro.battery
            if pct is None:
                msg = ("GOPRO BATT CHG" if self.gopro.charging
                       else "GOPRO BATT --%  (reading…)")
                lines.insert(1, (msg, T_GREEN_DIM))
            else:
                fill = max(0, min(10, round(pct / 10)))
                bar = "[" + "#" * fill + "-" * (10 - fill) + "]"
                chg = " CHG" if self.gopro.charging else ""
                bcol = T_GREEN if pct > 50 else T_AMBER if pct > 20 else T_RED
                lines.insert(1, (f"GOPRO BATT {pct:3d}% {bar}{chg}", bcol))
        row = lh(0.5)
        pw = max(mono_w(s, 0.5) for s, _ in lines) + S(26)
        ph = row * len(lines) + S(22)
        panel(frame, S(8), S(8), pw, ph, alpha=0.55, radius=S(6), border=T_GREEN)
        for i, (s, c) in enumerate(lines):
            text(frame, s, S(18), S(8) + S(20) + i * row, 0.5, c)
        # fps sparkline along the panel's bottom edge
        if len(self.fps_hist) > 2:
            gx, gy, gw, gh = S(18), S(8) + ph - S(4), pw - S(36), S(9)
            hist = list(self.fps_hist)[-gw:]
            mx = max(max(hist), 30.0)
            pts = [(gx + i, int(gy - (v / mx) * gh)) for i, v in enumerate(hist)]
            for a, b in zip(pts, pts[1:]):
                cv2.line(frame, a, b, T_CYAN, S(1), cv2.LINE_AA)
        self._draw_toolbar(frame)

    def _draw_toolbar(self, frame):
        """Clickable bottom toolbar of action buttons (registers hotspots)."""
        H, W = frame.shape[:2]
        bh = _mono_size(0.5) + S(18)
        panel(frame, 0, H - bh, W, bh, alpha=0.6, radius=0)
        cv2.line(frame, (0, H - bh), (W, H - bh), T_GREEN, S(1), cv2.LINE_AA)
        btns = [("settings", "SETTINGS", self.show_settings),
                ("rec", "REC", self.writer is not None),
                ("photo", "PHOTO", False),
                ("zoomreset", f"ZOOM {self.zoom:.1f}x", self.zoom > 1.001),
                ("quit", "QUIT", False)]
        x = S(10)
        y1, y2 = H - bh + S(4), H - S(4)
        for action, lab, active in btns:
            w = mono_w(lab, 0.5) + S(16)
            col = T_AMBER if active else T_GREEN
            cv2.rectangle(frame, (x, y1), (x + w, y2), col, S(1), cv2.LINE_AA)
            text(frame, lab, x + S(8), y2 - S(6), 0.5, col)
            self.hotspots.append((x, y1, x + w, y2, action))
            x += w + S(10)

    def _draw_settings_panel(self, frame):
        """Click-to-toggle settings panel (replaces the old keybinds)."""
        if not self.show_settings:
            return
        H, W = frame.shape[:2]
        rows = [(f"toggle:{k}", lab, self.toggles[k]) for k, lab in (
                ("dots", "Movement dots"), ("boxes", "Object boxes"),
                ("labels", "Labels"), ("velocity", "Prediction arrows"),
                ("zoom", "Zoom boxes (movers)"))]
        rows += [("mirror", "Mirror image", self.mirror),
                 ("fullscreen", "Fullscreen", self.fullscreen),
                 ("unlock", "Clear lock", self.selected_id is not None)]
        rh = lh(0.55)
        pw = S(330)
        ph = S(40) + len(rows) * rh + S(14)
        px, py = S(8), S(118)
        panel(frame, px, py, pw, ph, alpha=0.86, radius=S(8), border=T_GREEN)
        text(frame, "SETTINGS  ( click a row )", px + S(16), py + S(28),
             0.58, T_GREEN)
        y = py + S(46)
        for action, label, on in rows:
            r1, r2 = y, y + rh
            self.hotspots.append((px + S(6), r1, px + pw - S(6), r2, action))
            # checkbox
            bx, bsz = px + S(16), S(16)
            by = r1 + (rh - bsz) // 2
            cv2.rectangle(frame, (bx, by), (bx + bsz, by + bsz),
                          T_GREEN if on else T_DIM, S(1), cv2.LINE_AA)
            if on:
                cv2.rectangle(frame, (bx + S(3), by + S(3)),
                              (bx + bsz - S(3), by + bsz - S(3)), T_GREEN, -1)
            text(frame, label, bx + S(26), r2 - S(7), 0.5,
                 T_WHITE if on else T_GREEN_DIM)
            text(frame, "ON" if on else "OFF", px + pw - S(50), r2 - S(7), 0.48,
                 T_GREEN if on else T_DIM)
            y += rh

    def _draw_zoom_minimap(self, frame):
        """When zoomed, a little map showing which part of the full frame the
        view (and the detector) is currently looking at."""
        if self.zoom <= 1.001:
            return
        H, W = frame.shape[:2]
        mw = S(150)
        mh = max(1, int(mw * H / W))
        mx, my = W - mw - S(12), H - mh - S(44)
        panel(frame, mx, my, mw, mh, alpha=0.5, radius=0, border=T_GREEN)
        cw = 1.0 / self.zoom
        nx1 = clamp(self.zoom_cx - cw / 2, 0, 1 - cw)
        ny1 = clamp(self.zoom_cy - cw / 2, 0, 1 - cw)
        cv2.rectangle(frame, (mx + int(nx1 * mw), my + int(ny1 * mh)),
                      (mx + int((nx1 + cw) * mw), my + int((ny1 + cw) * mh)),
                      T_AMBER, S(2))
        text(frame, f"ZOOM {self.zoom:.1f}x", mx, my - S(6), 0.45, T_AMBER)

    def _draw_rec_badge(self, frame):
        W = frame.shape[1]
        if (self.frame_idx // 8) % 2 == 0:
            cv2.circle(frame, (W - S(30), S(20)), S(7), T_RED, -1, cv2.LINE_AA)
        text(frame, "REC", W - S(72), S(26), 0.6, T_RED)

    # ---- keyboard (settings are clickable; these are the handy ones) ---- #
    def handle_key(self, key, frame):
        if key in (ord("q"), 27):                    # quit
            return False
        if key == ord(" "):                          # pause
            self.paused = not self.paused
        elif key in (ord("="), ord("+")):            # zoom IN toward the cursor
            mx, my = self.mouse_pos or (self._vw // 2, self._vh // 2)
            self._zoom_at(mx, my, 1.25)
        elif key in (ord("-"), ord("_")):            # zoom OUT toward the cursor
            mx, my = self.mouse_pos or (self._vw // 2, self._vh // 2)
            self._zoom_at(mx, my, 1 / 1.25)
        elif key == ord("0"):                        # reset zoom to 1x
            self.zoom, self.zoom_cx, self.zoom_cy = 1.0, 0.5, 0.5
        return True

    def _toggle_record(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            self.rec_name = f"recording_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = clamp(self.fps if self.fps > 1 else 20.0, 10, 30)
            self.writer = cv2.VideoWriter(self.rec_name, fourcc, fps, (w, h))
            self.toasts.add(f"● Recording → {self.rec_name}")
        else:
            self.writer.release()
            self.writer = None
            self.toasts.add(f"Saved {self.rec_name}")


def main():
    p = argparse.ArgumentParser(description="Live 'thinking' object tracker.")
    p.add_argument("--source", default="0",
                   help="camera index (0,1,…), a video file, or a network "
                        "camera URL e.g. rtsp://user:pass@192.168.1.50:554/stream")
    p.add_argument("--udp", action="store_true",
                   help="use UDP instead of TCP for RTSP (only if TCP stutters)")
    p.add_argument("--gopro", action="store_true",
                   help="stream from a GoPro (Hero 4/5/6/7) over its WiFi — join "
                        "the camera's WiFi first, then run with this flag")
    p.add_argument("--gopro-ip", default=GOPRO_DEFAULT_IP,
                   help=f"GoPro IP on its WiFi (default {GOPRO_DEFAULT_IP})")
    p.add_argument("--model", default="auto",
                   help="YOLO detection model, or 'auto' (default): yolov8n for "
                        "low-res GoPro (speed), yolov8s for cameras/files "
                        "(accuracy). Or set one, e.g. yolov8m.pt for more range.")
    p.add_argument("--imgsz", type=int, default=None,
                   help="inference size. Default auto: 960 for cameras/files "
                        "(better at small/distant objects), 640 for the low-res "
                        "GoPro feed. Raise to 1280 for max range; lower for speed.")
    p.add_argument("--conf", type=float, default=0.30,
                   help="initial confidence threshold (sets the Detection "
                        "slider). Lower catches more but risks false positives "
                        "(tree->elephant); higher is stricter.")
    p.add_argument("--width", type=int, default=1920,
                   help="requested camera capture width (default 1920)")
    p.add_argument("--height", type=int, default=1080,
                   help="requested camera capture height (default 1080)")
    p.add_argument("--max-zoom", type=int, default=4,
                   help="max number of follow-along zoom insets")
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--list-cameras", action="store_true",
                   help="probe and print available cameras, then exit")
    args = p.parse_args()

    if args.list_cameras:
        cams = list_cameras()
        if not cams:
            print("No cameras found (or permission not granted to this app).")
        for i, w, h in cams:
            print(f"  camera {i}:  {w}x{h}")
        return

    TrackerApp(args).run()


if __name__ == "__main__":
    main()
