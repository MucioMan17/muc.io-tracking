# TinyDet — our own detector, from scratch

A complete object detector written by us in plain PyTorch — **no Ultralytics, no
detection libraries**. It's the side experiment to the main tracker (which keeps
using YOLOv8). The point is to actually understand and own every piece.

It's a **YOLO-style single-box-per-cell grid detector**:

- a small CNN backbone turns a 128×128 image into an 8×8 grid (`tinydet.py`);
- each grid cell predicts *objectness*, a *box* (x, y, w, h) and *class scores*;
- a cell is "responsible" for an object whose **centre** lands in it;
- we wrote the target **encoder**, the **loss** (YOLO-style: localise + classify
  responsible cells, suppress objectness elsewhere), the **decoder** and **NMS**.

## What's here

| File | What it is |
| --- | --- |
| `tinydet.py` | The whole detector core: model, encode, loss, decode, NMS. |
| `data.py` | Synthetic shapes dataset — generated on the fly, no downloads. |
| `train.py` | Training loop + live recall/precision/accuracy + a result image. |
| `detect.py` | Run the trained model on samples, an image, or **live webcam**. |
| `collect.py` | **Capture & label your own training data from the camera.** |

## Run it (synthetic shapes)

```bash
.venv/bin/python detector/train.py            # trains in ~1.5 min on Apple GPU
.venv/bin/python detector/detect.py           # see it detect on fresh samples
```

`train.py` writes `tinydet.pt` (weights) and `result.png` (green = ground truth,
coloured = our predictions).

## Teach it YOUR objects (camera) ⭐

This is where it learns *your* world. Three steps:

```bash
# 1) Collect + label frames from your camera (pick your own classes)
.venv/bin/python detector/collect.py --classes "cup,phone,hand"
#    SPACE = freeze a frame, drag = draw a box, 1-9 = pick class,
#    SPACE/ENTER = save, X = discard, Q = quit. Grab ~50-150 varied shots.

# 2) Train our detector on what you collected
.venv/bin/python detector/train.py --data real

# 3) Watch OUR model run live on the webcam
.venv/bin/python detector/detect.py --camera
```

Your data lands in `detector/dataset/` (`images/`, `labels/`, `classes.txt`).
Labels are normalised `cls cx cy w h` (one `.txt` per image). The model's class
list is saved inside the checkpoint, so `detect.py` always knows what it learned.

### Hand-labelling is slow — automate it (`autocollect.py`)

Two modes that **draw the boxes for you**, then `train.py --data real` as usual:

```bash
# MOTION mode (ours, any object): camera still, move the object around;
# background subtraction boxes the moving object automatically. You only press
# a number key once to say what it is, then R to record.
.venv/bin/python detector/autocollect.py --classes "cup,phone,toy"

# YOLO mode (zero effort, for objects YOLO already knows): it detects, boxes
# AND labels for you — just press R and let it run.
.venv/bin/python detector/autocollect.py --mode yolo --classes "cup,bottle,person"
```

- **Motion** works for *any* custom object and stays fully ours, but needs a
  **still camera + fairly static background** (one moving object at a time).
  Let the background learn for ~2 s, press `B` to relearn if it drifts.
- **YOLO** only works for the ~80 things YOLO knows, but needs **no interaction
  at all**. (It's using YOLO to *teach* our model — a legit technique called
  distillation; the trained model is still 100% ours.)
- Both save every Nth frame (`--stride`) so you get variety, not duplicates.
  A 30-second session yields hundreds of labelled images.

**Tips for good results:** capture each object from many angles, distances and
lighting; include some frames with *no* target and some with several; aim for
50+ examples per class. Small datasets train in well under a minute.

## Milestone 1 — done ✅

Trained on synthetic circles/squares/triangles, it reaches **~100% recall,
~90% precision, ~99% class accuracy** in ~1500 steps (84 s on an M-series GPU).
It's only **0.59M parameters** (YOLOv8n is ~3.4M). This proves the full
architecture, loss, training loop and decoding all work end to end.

## Where this goes next

This is a real foundation, but it isn't YOLO yet. The honest road ahead:

1. ~~**Real data** via a camera capture+label tool~~ ✅ **done** (`collect.py`
   + `--data real`). Next data step: a proper **train/val split** so the
   accuracy numbers reflect *unseen* images, not the training set.
2. **Better architecture.** Multiple boxes per cell / **anchor boxes**,
   **multi-scale** detection (detect big and small objects via an FPN), a
   stronger backbone, and a modern box loss (CIoU) instead of MSE.
3. **Data augmentation** (flips, colour jitter, mosaic) — the single biggest
   accuracy lever once we're on real data.
4. **More classes** and a proper **mAP** evaluation (the real benchmark metric).
5. **Plug it into the tracker.** Add an engine switch so `tracker.py` can run
   *our* model live instead of YOLOv8 — once it detects real-world objects.

Each step is a self-contained, understandable upgrade. That's the fun of it:
we'll grow our own detector piece by piece and watch the numbers climb.
