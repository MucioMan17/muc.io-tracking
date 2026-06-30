"""
autocollect.py — automate dataset creation so you don't hand-draw boxes.

Two modes:

  MOTION (default, fully ours, works for ANY object):
      Put the camera down, hold your object, and move it around. Background
      subtraction finds the moving object and boxes it automatically every
      frame — you just press a number key once to say which class it is.

      .venv/bin/python detector/autocollect.py --classes "cup,phone,toy"
      Keys:  1-9 = set class   R = start/stop auto-recording
             B = relearn background   Q = quit

  YOLO (zero effort, for objects YOLO already knows):
      The existing YOLO model detects, boxes AND labels the objects for you.

      .venv/bin/python detector/autocollect.py --mode yolo --classes "cup,bottle,person"
      Keys:  R = start/stop auto-recording   Q = quit

Both write to detector/dataset/ (same format as collect.py). Then:
      .venv/bin/python detector/train.py --data real
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect import (load_classes, save_sample, count_saved, px_to_norm,
                     color, open_source, DATASET)


# ---- the two auto-labelling strategies ------------------------------------ #
def motion_box(bgsub, frame, min_area_frac=0.012):
    """Largest moving blob → a bounding box (x1,y1,x2,y2), or None."""
    h, w = frame.shape[:2]
    fg = bgsub.apply(frame)
    fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]   # drop soft shadows
    k = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_frac * w * h:
        return None
    x, y, bw, bh = cv2.boundingRect(c)
    return (x, y, x + bw, y + bh)


def yolo_boxes(model, frame, name_to_idx):
    """Run YOLO and return [(cls, x1,y1,x2,y2)] for the classes we want."""
    res = model(frame, verbose=False)[0]
    out = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        for i, c in enumerate(res.boxes.cls.int().cpu().tolist()):
            name = res.names[c]
            if name in name_to_idx:
                x1, y1, x2, y2 = xyxy[i]
                out.append((name_to_idx[name], x1, y1, x2, y2))
    return out


def hud(frame, classes, cur, mode, recording, saved, n_boxes):
    f = frame.copy()
    x = 10
    for i, name in enumerate(classes):
        sel = (mode == "motion" and i == cur)
        tag = f"[{i+1}] {name}"
        (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if sel:
            cv2.rectangle(f, (x - 4, 6), (x + tw + 6, 28), color(i), -1)
        cv2.putText(f, tag, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (20, 20, 20) if sel else color(i), 1, cv2.LINE_AA)
        x += tw + 18
    bar = (f"{mode.upper()}  saved:{saved}  boxes:{n_boxes}  "
           f"{'● REC' if recording else 'R=record'}   "
           f"{'1-9=class B=bg ' if mode=='motion' else ''}Q=quit")
    cv2.rectangle(f, (0, f.shape[0] - 26), (f.shape[1], f.shape[0]), (0, 0, 0), -1)
    col = (60, 60, 255) if recording else (200, 255, 200)
    cv2.putText(f, bar, (10, f.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                col, 1, cv2.LINE_AA)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["motion", "yolo"], default="motion")
    ap.add_argument("--classes", default=None, help='e.g. "cup,phone,toy"')
    ap.add_argument("--source", default="0")
    ap.add_argument("--out", default=DATASET)
    ap.add_argument("--stride", type=int, default=5,
                    help="save every Nth frame while recording (avoids dupes)")
    args = ap.parse_args()

    classes = load_classes(args.classes, args.out)
    cur = 0
    model = name_to_idx = bgsub = None
    if args.mode == "yolo":
        from ultralytics import YOLO
        model = YOLO("yolov8s-seg.pt")
        name_to_idx = {c: i for i, c in enumerate(classes)}
        unknown = [c for c in classes if c not in model.names.values()]
        if unknown:
            print(f"⚠ YOLO doesn't know {unknown} — use motion mode for those.")
        print("YOLO auto-label mode. Press R to start recording.")
    else:
        bgsub = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=40,
                                                   detectShadows=False)
        print("Motion mode. Hold an object, pick its class (1-9), press R. "
              "Keep the camera still; let the background learn for ~2s first.")

    cap = open_source(args.source)
    win = "muc.io auto-collect"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    recording = False
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            cv2.waitKey(20)
            continue
        frame_i += 1

        if args.mode == "yolo":
            boxes = yolo_boxes(model, frame, name_to_idx)
        else:
            b = motion_box(bgsub, frame)
            boxes = [(cur, *b)] if b else []

        view = hud(frame, classes, cur, args.mode, recording,
                   count_saved(args.out), len(boxes))
        for (cls, x1, y1, x2, y2) in boxes:
            cv2.rectangle(view, (x1, y1), (x2, y2), color(cls), 2)
            cv2.putText(view, classes[cls], (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color(cls), 1, cv2.LINE_AA)

        if recording and boxes and frame_i % args.stride == 0:
            h, w = frame.shape[:2]
            save_sample(args.out, frame, px_to_norm(
                [(c, x1, y1, x2, y2) for (c, x1, y1, x2, y2) in boxes], w, h))

        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            recording = not recording
        elif key == ord("b") and bgsub is not None:
            bgsub = cv2.createBackgroundSubtractorMOG2(history=400,
                    varThreshold=40, detectShadows=False)
        elif ord("1") <= key <= ord("9") and args.mode == "motion":
            i = key - ord("1")
            if i < len(classes):
                cur = i
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSaved {count_saved(args.out)} labelled images to {args.out}")
    print("Train:  .venv/bin/python detector/train.py --data real")


if __name__ == "__main__":
    main()
