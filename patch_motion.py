"""
One-shot patch to harden the motion layer so it stops flooding the screen with
false 'MOVER' boxes (sensor noise, pieces of an already-detected object) and
garbage low-confidence labels.

Run once:  .venv\\Scripts\\python.exe patch_motion.py   (Windows)
       or  .venv/bin/python patch_motion.py             (mac/Linux)
"""
import io

PAIRS = [
    # 1a) give each supplementary track a "seen" counter
    ('''                self.tracks[bid] = {"box": list(map(float, box)),
                                    "trail": deque(maxlen=32), "cls": cls,
                                    "conf": conf or 0.5, "labelled": -999,
                                    "last": frame_idx}''',
     '''                self.tracks[bid] = {"box": list(map(float, box)),
                                    "trail": deque(maxlen=32), "cls": cls,
                                    "conf": conf or 0.5, "labelled": -999,
                                    "last": frame_idx, "seen": 0}'''),
    # 1b) increment it on every update
    ('''            t = self.tracks[bid]
            t["last"] = frame_idx
            cx = int((t["box"][0] + t["box"][2]) / 2)''',
     '''            t = self.tracks[bid]
            t["last"] = frame_idx
            t["seen"] += 1
            cx = int((t["box"][0] + t["box"][2]) / 2)'''),
    # 2a) less twitchy background subtraction
    ('history=400, varThreshold=40, detectShadows=False)',
     'history=400, varThreshold=60, detectShadows=False)'),
    # 2b) ignore smaller noise blobs
    ('            if a < area * 0.00015 or a > area * 0.25:   # ignore noise & pans',
     '            if a < area * 0.0006 or a > area * 0.25:    # ignore noise & pans'),
    # 3a) drop any blob whose centre sits inside an already-detected object
    ('''        # only keep finds the main full-frame pass missed
        mboxes = [it["box"] for it in main_items]
        cands = [(b, cl, cf) for (b, cl, cf) in cands
                 if not any(iou_xyxy(b, mb) > 0.3 for mb in mboxes)]''',
     '''        # only keep finds the main pass missed (drop any blob whose centre
        # sits inside an already-detected object -> kills the pile-on flood)
        mboxes = [it["box"] for it in main_items]

        def _covered(b):
            bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            return any((mb[0] <= bx <= mb[2] and mb[1] <= by <= mb[3])
                       or iou_xyxy(b, mb) > 0.2 for mb in mboxes)
        cands = [(b, cl, cf) for (b, cl, cf) in cands if not _covered(b)]'''),
    # 3b) only show movers that have persisted a few frames; cap to the biggest
    ('''        tracks = self.supp.update([(b, cl, cf) for (b, cl, cf) in cands],
                                  self.frame_idx)
        items = []
        for tid, t in tracks.items():''',
     '''        self.supp.update([(b, cl, cf) for (b, cl, cf) in cands], self.frame_idx)
        # only movers that have persisted a few frames (filters sensor noise),
        # coasted briefly so they don't flicker, and capped to the biggest few
        steady = [(tid, t) for tid, t in self.supp.tracks.items()
                  if t["seen"] >= 4 and self.frame_idx - t["last"] <= 3]
        steady.sort(key=lambda kt: -(kt[1]["box"][2] - kt[1]["box"][0])
                    * (kt[1]["box"][3] - kt[1]["box"][1]))
        items = []
        for tid, t in steady[:8]:'''),
    # 3c) only show a class label when we're actually confident
    ('            name = names.get(t["cls"], "mover") if t["cls"] is not None else "mover"',
     '''            name = (names.get(t["cls"], "mover")
                    if t["cls"] is not None and t["conf"] >= 0.5 else "mover")'''),
]


def main():
    with io.open("tracker.py", encoding="utf-8") as f:
        src = f.read().replace("\r\n", "\n")      # normalise Windows line endings
    applied = already = missing = 0
    for old, new in PAIRS:
        old = old.replace("\r\n", "\n")           # robust to either line ending
        new = new.replace("\r\n", "\n")
        if old in src:
            src = src.replace(old, new, 1)
            applied += 1
        elif new in src:
            already += 1
        else:
            missing += 1
            print("WARNING: could not find a block to patch:\n   ", repr(old[:70]))
    with io.open("tracker.py", "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    print(f"\napplied {applied}, already-patched {already}, missing {missing}")
    if missing == 0:
        print("OK - motion layer hardened.")


if __name__ == "__main__":
    main()
