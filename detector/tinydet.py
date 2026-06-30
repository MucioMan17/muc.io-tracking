"""
TinyDet — a from-scratch object detector, written entirely by us in plain
PyTorch (no Ultralytics, no detection libraries).

It's a single-box-per-cell, YOLO-style grid detector:

  • A small CNN backbone downsamples a 128x128 image to an 8x8 grid.
  • For every grid cell it predicts:  objectness, a box (x,y,w,h), class scores.
  • A cell is "responsible" for an object if the object's *centre* falls in it.

This file contains the whole detector core — the model, the target encoder, the
loss, the decoder, and non-max suppression — so you can read the entire thing
end to end. See data.py (training data) and train.py (training loop).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- problem definition ---------------------------------------------------- #
IMG = 128                                  # input image size (square)
S = 8                                      # output grid is S x S (stride 16)
CLASSES = ["circle", "square", "triangle"]
C = len(CLASSES)


# ---- model ----------------------------------------------------------------- #
def conv(cin, cout, k=3, s=1):
    """Conv → BatchNorm → LeakyReLU, the standard little building block."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.1, inplace=True),
    )


class TinyDet(nn.Module):
    """Our detector. Backbone halves the resolution four times (128→8), then a
    1x1 'head' produces (5 + C) numbers per grid cell."""

    def __init__(self, nc=C):
        super().__init__()
        self.nc = nc
        self.body = nn.Sequential(
            conv(3, 16),  conv(16, 16, s=2),     # 128 -> 64
            conv(16, 32), conv(32, 32, s=2),     # 64  -> 32
            conv(32, 64), conv(64, 64, s=2),     # 32  -> 16
            conv(64, 128), conv(128, 128, s=2),  # 16  -> 8
            conv(128, 256),
        )
        self.head = nn.Conv2d(256, 5 + nc, 1)    # obj, x, y, w, h, class logits…

    def forward(self, x):
        x = self.head(self.body(x))              # (B, 5+nc, S, S)
        return x.permute(0, 2, 3, 1).contiguous()  # (B, S, S, 5+nc)


# ---- target encoding ------------------------------------------------------- #
def encode(boxes, s=S, nc=C):
    """Turn a list of ground-truth boxes (cls, cx, cy, w, h) — all normalised to
    [0,1] — into the (S, S, 5+C) target tensor the loss compares against."""
    t = torch.zeros(s, s, 5 + nc)
    for cls, cx, cy, w, h in boxes:
        col = min(int(cx * s), s - 1)
        row = min(int(cy * s), s - 1)
        t[row, col, 0] = 1.0                     # objectness
        t[row, col, 1] = cx * s - col            # x offset inside the cell [0,1]
        t[row, col, 2] = cy * s - row            # y offset inside the cell [0,1]
        t[row, col, 3] = w                       # width  (fraction of image)
        t[row, col, 4] = h                       # height (fraction of image)
        t[row, col, 5 + int(cls)] = 1.0          # one-hot class
    return t


# ---- loss ------------------------------------------------------------------ #
def detection_loss(pred, target, l_coord=5.0, l_noobj=0.5):
    """YOLO-style loss: localise + classify the responsible cells, and push
    objectness down everywhere else. `pred` is raw (logits); we sigmoid here."""
    obj = target[..., 0] == 1                    # (B,S,S) responsible cells
    noobj = ~obj
    dev = pred.device
    zero = torch.tensor(0.0, device=dev)

    # objectness (use logits directly for numerical stability)
    obj_l = (F.binary_cross_entropy_with_logits(pred[..., 0][obj],
             target[..., 0][obj]) if obj.any() else zero)
    noobj_l = (F.binary_cross_entropy_with_logits(pred[..., 0][noobj],
               target[..., 0][noobj]) if noobj.any() else zero)

    if obj.any():
        pxy = torch.sigmoid(pred[..., 1:3][obj])
        pwh = torch.sigmoid(pred[..., 3:5][obj])
        coord_l = (F.mse_loss(pxy, target[..., 1:3][obj]) +
                   F.mse_loss(torch.sqrt(pwh + 1e-6),
                              torch.sqrt(target[..., 3:5][obj] + 1e-6)))
        cls_l = F.cross_entropy(pred[..., 5:][obj],
                                target[..., 5:][obj].argmax(1))
    else:
        coord_l = cls_l = zero

    total = l_coord * coord_l + obj_l + l_noobj * noobj_l + cls_l
    parts = {k: float(v.detach()) for k, v in
             (("coord", coord_l), ("obj", obj_l),
              ("noobj", noobj_l), ("cls", cls_l))}
    return total, parts


# ---- decoding + NMS (turn predictions into boxes) -------------------------- #
def _iou(a, b):
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, iou_thr=0.4):
    """Greedy non-max suppression. dets: list of (score, cls, x, y, w, h)."""
    dets = sorted(dets, key=lambda d: -d[0])
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets
                if d[1] != best[1] or _iou(best[2:], d[2:]) < iou_thr]
    return keep


def decode(pred, conf=0.3, s=S):
    """Convert one image's raw (S,S,5+C) prediction into a list of detections
    (score, cls, cx, cy, w, h) in normalised coords, after NMS."""
    dets = []
    obj = torch.sigmoid(pred[..., 0])
    cls_prob = torch.softmax(pred[..., 5:], -1)
    cls_score, cls_id = cls_prob.max(-1)
    score = (obj * cls_score)
    for row in range(s):
        for col in range(s):
            sc = score[row, col].item()
            if sc < conf:
                continue
            x = (col + torch.sigmoid(pred[row, col, 1]).item()) / s
            y = (row + torch.sigmoid(pred[row, col, 2]).item()) / s
            w = torch.sigmoid(pred[row, col, 3]).item()
            h = torch.sigmoid(pred[row, col, 4]).item()
            dets.append((sc, int(cls_id[row, col]), x, y, w, h))
    return nms(dets)
