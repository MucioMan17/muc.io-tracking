"""
Synthetic training data for TinyDet — we *generate* labelled images on the fly,
so there's nothing to download or hand-label. Each image has 1–3 shapes
(circle / square / triangle) on a noisy background, and we know each shape's
exact class and bounding box because we drew it.

This is the perfect first dataset for a from-scratch detector: infinite,
balanced, free, and it exercises the full train→detect pipeline. Swapping in
real images later means only replacing `generate()`.
"""

import os
import random
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinydet import IMG, S, C, encode


def _vivid():
    """A random reasonably-saturated BGR colour."""
    base = [random.randint(0, 255) for _ in range(3)]
    base[random.randint(0, 2)] = random.randint(180, 255)
    return tuple(base)


def generate():
    """Return (image HxWx3 uint8, boxes) where boxes is a list of
    (cls, cx, cy, w, h) normalised to [0,1]."""
    bg = np.full((IMG, IMG, 3), random.randint(20, 90), np.uint8)
    bg = (bg.astype(np.int16) +
          np.random.randint(-12, 12, (IMG, IMG, 3))).clip(0, 255).astype(np.uint8)
    img = bg
    boxes = []
    for _ in range(random.randint(1, 3)):
        cls = random.randint(0, C - 1)
        col = _vivid()
        half = random.randint(10, 24)
        cx = random.randint(half + 1, IMG - half - 1)
        cy = random.randint(half + 1, IMG - half - 1)
        if cls == 0:                                   # circle
            cv2.circle(img, (cx, cy), half, col, -1, cv2.LINE_AA)
            w = h = 2 * half
        elif cls == 1:                                 # square
            cv2.rectangle(img, (cx - half, cy - half),
                          (cx + half, cy + half), col, -1)
            w = h = 2 * half
        else:                                          # triangle
            pts = np.array([[cx, cy - half], [cx - half, cy + half],
                            [cx + half, cy + half]], np.int32)
            cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
            w = h = 2 * half
        boxes.append((cls, cx / IMG, cy / IMG, w / IMG, h / IMG))
    return img, boxes


def make_batch(n, device):
    """A batch of n freshly-generated (image, target) pairs, on `device`."""
    imgs = torch.empty(n, 3, IMG, IMG)
    tgts = torch.empty(n, S, S, 5 + C)
    for i in range(n):
        im, boxes = generate()
        imgs[i] = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        tgts[i] = encode(boxes)
    return imgs.to(device), tgts.to(device)


# --------------------------------------------------------------------------- #
# Real data: images + labels you collected from your own camera (see collect.py).
# Layout:  <root>/classes.txt, <root>/images/*.jpg, <root>/labels/*.txt
# Each label line is  "cls cx cy w h"  normalised to [0,1] (same as encode()).
# --------------------------------------------------------------------------- #
def _augment(img, boxes):
    """Cheap, label-safe augmentation — the biggest accuracy lever on small
    real datasets: horizontal flip + brightness/contrast jitter."""
    if random.random() < 0.5:                              # horizontal flip
        img = img[:, ::-1, :].copy()
        boxes = [(c, 1.0 - cx, cy, w, h) for (c, cx, cy, w, h) in boxes]
    a, b = random.uniform(0.75, 1.25), random.randint(-25, 25)
    img = (img.astype(np.float32) * a + b).clip(0, 255).astype(np.uint8)
    return img, boxes


class RealData:
    def __init__(self, root):
        cf = os.path.join(root, "classes.txt")
        if not os.path.exists(cf):
            raise SystemExit(f"No dataset at {root} (run collect.py first).")
        self.classes = [l.strip() for l in open(cf) if l.strip()]
        img_dir = os.path.join(root, "images")
        lab_dir = os.path.join(root, "labels")
        self.items = []
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith((".jpg", ".png")):
                continue
            lab = os.path.join(lab_dir, os.path.splitext(fn)[0] + ".txt")
            if os.path.exists(lab):
                self.items.append((os.path.join(img_dir, fn), lab))
        if not self.items:
            raise SystemExit(f"No labelled images in {root} yet.")

    def __len__(self):
        return len(self.items)

    def sample(self, augment=True):
        img_path, lab_path = random.choice(self.items)
        img = cv2.resize(cv2.imread(img_path), (IMG, IMG))
        boxes = []
        for line in open(lab_path):
            p = line.split()
            if len(p) == 5:
                boxes.append((int(p[0]), float(p[1]), float(p[2]),
                              float(p[3]), float(p[4])))
        if augment:
            img, boxes = _augment(img, boxes)
        return img, boxes


def make_real_batch(ds, n, device, nc):
    imgs = torch.empty(n, 3, IMG, IMG)
    tgts = torch.empty(n, S, S, 5 + nc)
    for i in range(n):
        im, boxes = ds.sample()
        imgs[i] = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        tgts[i] = encode(boxes, S, nc)
    return imgs.to(device), tgts.to(device)
