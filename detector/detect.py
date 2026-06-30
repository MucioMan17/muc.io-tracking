"""
Run our trained TinyDet.

    .venv/bin/python detector/detect.py            # on random synthetic samples
    .venv/bin/python detector/detect.py --image foo.png
    .venv/bin/python detector/detect.py --camera   # LIVE on your webcam ⭐

The model's classes come from the checkpoint, so this works whether you trained
on synthetic shapes or on your own collected data.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tinydet import TinyDet, decode, CLASSES, IMG
from data import generate
from train import draw, pick_device, PALETTE


def load_model(weights, device):
    if not os.path.exists(weights):
        sys.exit("No trained weights yet — run detector/train.py first.")
    ckpt = torch.load(weights, map_location=device)
    classes = ckpt.get("classes", CLASSES)
    model = TinyDet(len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes


@torch.no_grad()
def infer(model, device, im, conf):
    x = torch.from_numpy(cv2.resize(im, (IMG, IMG))).permute(2, 0, 1) \
        .float().div(255)[None].to(device)
    return decode(model(x)[0].cpu(), conf=conf)


def run_camera(model, classes, device, conf, source):
    src = int(source) if str(source).isdigit() else source
    backend = cv2.CAP_AVFOUNDATION if isinstance(src, int) else cv2.CAP_ANY
    cap = cv2.VideoCapture(src, backend)
    if not cap.isOpened():
        sys.exit("Could not open camera (grant camera access to your terminal).")
    win = "TinyDet (ours) — live"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Running OUR detector live. Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            cv2.waitKey(20)
            continue
        dets = infer(model, device, frame, conf)
        draw(frame, dets, classes, lambda c: PALETTE[c % len(PALETTE)])
        cv2.putText(frame, "TinyDet (ours)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 140), 2, cv2.LINE_AA)
        cv2.imshow(win, frame)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "tinydet.pt"))
    ap.add_argument("--image", default=None)
    ap.add_argument("--camera", action="store_true", help="run live on the webcam")
    ap.add_argument("--source", default="0")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, classes = load_model(args.weights, device)

    if args.camera:
        run_camera(model, classes, device, args.conf, args.source)
        return

    tiles = []
    for k in range(8):
        im = cv2.resize(cv2.imread(args.image), (IMG, IMG)) if args.image \
            else generate()[0]
        if args.image and im is None:
            sys.exit(f"Could not read {args.image}")
        dets = infer(model, device, im, args.conf)
        vis = im.copy()
        draw(vis, dets, classes, lambda c: PALETTE[c % len(PALETTE)])
        tiles.append(vis)
        print(f"sample {k}: {[(classes[d[1]], round(d[0], 2)) for d in dets]}")
        if args.image:
            break

    out = os.path.join(HERE, "detect_out.png")
    if args.image:
        cv2.imwrite(out, cv2.resize(tiles[0], (IMG * 3, IMG * 3),
                                    interpolation=cv2.INTER_NEAREST))
    else:
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 8, 4)]
        cv2.imwrite(out, np.vstack(rows))
    print("wrote", out)


if __name__ == "__main__":
    main()
