"""
Capture & label your own training data from the camera — for teaching TinyDet
(our from-scratch detector) to find real objects.

    .venv/bin/python detector/collect.py --classes "cup,phone,hand"
    .venv/bin/python detector/collect.py            # reuse existing classes

How it works:
    • A live camera window opens.
    • Press SPACE to FREEZE the current frame, then draw boxes on it.
    • In freeze mode: drag the mouse to draw a box around an object.
      Press a number key (1..9) to choose which class the next box is.
      U = undo last box,  D = clear boxes,  X = discard frame.
      SPACE / ENTER = save the frame + its boxes, then go back to live.
    • Q = quit.

Saves to detector/dataset/  (images/, labels/, classes.txt). Then train with
    .venv/bin/python detector/train.py --data real
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "dataset")
PALETTE = [(80, 200, 255), (120, 255, 140), (255, 160, 90), (200, 120, 255),
           (90, 230, 230), (255, 120, 120), (150, 255, 200), (255, 210, 90),
           (120, 160, 255)]


def color(i):
    return PALETTE[i % len(PALETTE)]


def load_classes(arg, root):
    cf = os.path.join(root, "classes.txt")
    if arg:
        classes = [c.strip() for c in arg.split(",") if c.strip()]
    elif os.path.exists(cf):
        classes = [l.strip() for l in open(cf) if l.strip()]
    else:
        sys.exit('First run needs --classes "cat,dog,cup" (up to 9).')
    os.makedirs(root, exist_ok=True)
    with open(cf, "w") as f:
        f.write("\n".join(classes) + "\n")
    return classes[:9]


def count_saved(root):
    d = os.path.join(root, "images")
    return len([x for x in os.listdir(d) if x.endswith(".jpg")]) if os.path.isdir(d) else 0


def save_sample(root, img, norm_boxes):
    """img: BGR frame. norm_boxes: list of (cls, cx, cy, w, h) normalised."""
    imgs = os.path.join(root, "images")
    labs = os.path.join(root, "labels")
    os.makedirs(imgs, exist_ok=True)
    os.makedirs(labs, exist_ok=True)
    name = f"img_{count_saved(root):05d}_{int(time.time() * 1000) % 100000}"
    cv2.imwrite(os.path.join(imgs, name + ".jpg"), img)
    with open(os.path.join(labs, name + ".txt"), "w") as f:
        for cls, cx, cy, w, h in norm_boxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return name


def px_to_norm(boxes, w, h):
    out = []
    for cls, x1, y1, x2, y2 in boxes:
        out.append((cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                    abs(x2 - x1) / w, abs(y2 - y1) / h))
    return out


class Collector:
    def __init__(self, classes):
        self.classes = classes
        self.cur = 0
        self.frozen = None           # frozen frame (BGR) or None when live
        self.boxes = []              # pixel boxes (cls, x1, y1, x2, y2)
        self.drag = None             # drag start point
        self.cursor = (0, 0)

    def on_mouse(self, event, x, y, flags, param):
        if self.frozen is None:
            return
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            x0, y0 = self.drag
            self.drag = None
            x1, y1, x2, y2 = min(x0, x), min(y0, y), max(x0, x), max(y0, y)
            if x2 - x1 > 5 and y2 - y1 > 5:
                self.boxes.append((self.cur, x1, y1, x2, y2))

    def overlay(self, frame, saved):
        f = frame.copy()
        # class palette / current selection
        x = 10
        for i, name in enumerate(self.classes):
            sel = i == self.cur
            tag = f"[{i+1}] {name}"
            (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if sel:
                cv2.rectangle(f, (x - 4, 6), (x + tw + 6, 28), color(i), -1)
            cv2.putText(f, tag, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (20, 20, 20) if sel else color(i), 1, cv2.LINE_AA)
            x += tw + 18
        if self.frozen is None:
            msg = f"LIVE  saved:{saved}   SPACE=freeze & label   Q=quit"
        else:
            msg = (f"LABEL  boxes:{len(self.boxes)}   drag=box  1-9=class  "
                   f"U=undo D=clear  SPACE=save  X=discard")
            for (cls, x1, y1, x2, y2) in self.boxes:
                cv2.rectangle(f, (x1, y1), (x2, y2), color(cls), 2)
                cv2.putText(f, self.classes[cls], (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color(cls), 1, cv2.LINE_AA)
            if self.drag:
                cv2.rectangle(f, self.drag, self.cursor, color(self.cur), 1)
        cv2.rectangle(f, (0, f.shape[0] - 26), (f.shape[1], f.shape[0]), (0, 0, 0), -1)
        cv2.putText(f, msg, (10, f.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 255, 200), 1, cv2.LINE_AA)
        return f


def open_source(src):
    src = int(src) if str(src).isdigit() else src
    backend = cv2.CAP_AVFOUNDATION if isinstance(src, int) else cv2.CAP_ANY
    cap = cv2.VideoCapture(src, backend)
    if not cap.isOpened():
        sys.exit("Could not open camera. Grant camera access to your terminal "
                 "(System Settings → Privacy & Security → Camera), then retry.")
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default=None, help='e.g. "cup,phone,hand"')
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--out", default=DATASET)
    args = ap.parse_args()

    classes = load_classes(args.classes, args.out)
    print("Classes:", {i + 1: c for i, c in enumerate(classes)})
    cap = open_source(args.source)
    col = Collector(classes)
    win = "muc.io collect"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, col.on_mouse)
    last = None

    while True:
        if col.frozen is None:
            ok, frame = cap.read()
            if not ok:
                cv2.waitKey(20)
                continue
            last = frame
        view = col.overlay(col.frozen if col.frozen is not None else last,
                           count_saved(args.out))
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            if col.frozen is None:
                col.frozen = last.copy()
                col.boxes = []
            else:                                    # save & resume
                h, w = col.frozen.shape[:2]
                save_sample(args.out, col.frozen, px_to_norm(col.boxes, w, h))
                col.frozen = None
        elif key in (13, 10):                        # ENTER also saves
            if col.frozen is not None:
                h, w = col.frozen.shape[:2]
                save_sample(args.out, col.frozen, px_to_norm(col.boxes, w, h))
                col.frozen = None
        elif key == ord("x"):
            col.frozen = None
        elif key == ord("u") and col.boxes:
            col.boxes.pop()
        elif key == ord("d"):
            col.boxes = []
        elif ord("1") <= key <= ord("9"):
            i = key - ord("1")
            if i < len(classes):
                col.cur = i
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSaved {count_saved(args.out)} labelled images to {args.out}")
    print("Now train:  .venv/bin/python detector/train.py --data real")


if __name__ == "__main__":
    main()
