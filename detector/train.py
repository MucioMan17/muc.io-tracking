"""
Train TinyDet — our from-scratch detector — on synthetic shapes.

    .venv/bin/python detector/train.py                # ~1500 steps, a few min
    .venv/bin/python detector/train.py --iters 3000   # train longer = sharper

Saves the trained weights to detector/tinydet.pt and a before/after style
visualisation to detector/result.png (green = ground truth, coloured = our
model's predictions with class + confidence).
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tinydet import TinyDet, detection_loss, decode, _iou, CLASSES, IMG, C
from data import generate, make_batch, make_real_batch, RealData


def pick_device(req):
    if req != "auto":
        return req
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def evaluate(model, device, sample_fn, n=96, conf=0.3):
    """Recall / precision / class-accuracy on fresh samples (IoU>0.5 = a hit)."""
    model.eval()
    hits = correct = gt_total = pred_total = 0
    for _ in range(n):
        im, boxes = sample_fn()
        x = torch.from_numpy(im).permute(2, 0, 1).float().div(255)[None].to(device)
        dets = decode(model(x)[0].cpu(), conf=conf)
        pred_total += len(dets)
        gt_total += len(boxes)
        used = [False] * len(dets)
        for gc, gx, gy, gw, gh in boxes:
            best, bi = 0.5, -1
            for i, (_, c, x_, y_, w_, h_) in enumerate(dets):
                if used[i]:
                    continue
                v = _iou((gx, gy, gw, gh), (x_, y_, w_, h_))
                if v > best:
                    best, bi = v, i
            if bi >= 0:
                used[bi] = True
                hits += 1
                correct += (dets[bi][1] == gc)
    model.train()
    return (hits / max(1, gt_total), hits / max(1, pred_total),
            correct / max(1, hits))


def draw(im, dets, classes, color_fn, labels=True):
    h_im, w_im = im.shape[:2]
    for d in dets:
        if len(d) == 6:
            sc, cls, x, y, w, h = d
        else:
            cls, x, y, w, h = d
            sc = None
        x1, y1 = int((x - w / 2) * w_im), int((y - h / 2) * h_im)
        x2, y2 = int((x + w / 2) * w_im), int((y + h / 2) * h_im)
        cv2.rectangle(im, (x1, y1), (x2, y2), color_fn(cls), 2)
        if labels:
            tag = classes[cls] + (f" {sc:.2f}" if sc is not None else "")
            cv2.putText(im, tag, (x1, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_fn(cls), 1, cv2.LINE_AA)


@torch.no_grad()
def save_result(model, device, sample_fn, classes, path, n=8):
    """Montage of n samples: GT boxes (thin green) + our predictions (coloured)."""
    tiles = []
    model.eval()
    for _ in range(n):
        im, boxes = sample_fn()
        pred = decode(model(torch.from_numpy(im).permute(2, 0, 1)
                            .float().div(255)[None].to(device))[0].cpu(), conf=0.35)
        vis = im.copy()
        draw(vis, boxes, classes, lambda c: (0, 255, 0), labels=False)  # GT green
        draw(vis, pred, classes, lambda c: PALETTE[c % len(PALETTE)])   # preds
        tiles.append(vis)
    model.train()
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, n, 4)]
    cv2.imwrite(path, np.vstack(rows))


PALETTE = [(80, 200, 255), (120, 255, 140), (255, 160, 90), (200, 120, 255),
           (90, 230, 230), (255, 120, 120), (150, 255, 200), (255, 210, 90)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "real"], default="synthetic",
                    help="'real' trains on images you collected with collect.py")
    ap.add_argument("--dataset", default=os.path.join(HERE, "dataset"))
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=os.path.join(HERE, "tinydet.pt"))
    args = ap.parse_args()

    device = pick_device(args.device)
    if args.data == "real":
        ds = RealData(args.dataset)
        classes = ds.classes
        batch_fn = lambda: make_real_batch(ds, args.batch, device, len(classes))
        sample_fn = lambda: ds.sample(augment=False)
        print(f"Real dataset: {len(ds)} images, classes {classes}")
    else:
        classes = CLASSES
        batch_fn = lambda: make_batch(args.batch, device)
        sample_fn = generate
        print(f"Synthetic shapes, classes {classes}")

    nc = len(classes)
    model = TinyDet(nc).to(device)
    print(f"device: {device}   TinyDet parameters: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)

    t0 = time.time()
    run = 0.0
    for it in range(1, args.iters + 1):
        imgs, tgts = batch_fn()
        loss, parts = detection_loss(model(imgs), tgts)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = loss.item() if it == 1 else 0.98 * run + 0.02 * loss.item()
        if it % 100 == 0 or it == args.iters:
            rec, prec, acc = evaluate(model, device, sample_fn, n=48)
            print(f"it {it:5d}/{args.iters}  loss {run:6.3f}  "
                  f"[coord {parts['coord']:.2f} obj {parts['obj']:.2f} "
                  f"cls {parts['cls']:.2f}]  recall {rec:4.0%}  "
                  f"prec {prec:4.0%}  cls-acc {acc:4.0%}")

    torch.save({"model": model.state_dict(), "classes": classes}, args.out)
    save_result(model, device, sample_fn, classes, os.path.join(HERE, "result.png"))
    print(f"\nDone in {time.time()-t0:.0f}s. Saved {args.out} and "
          f"{os.path.join(HERE, 'result.png')}")


if __name__ == "__main__":
    main()
