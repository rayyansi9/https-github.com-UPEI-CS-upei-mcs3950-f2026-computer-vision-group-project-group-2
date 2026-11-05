"""Hand-label 250 x 250 m patches for the land-cover classes in docs/class_guide.md.

One patch (row, col = top-left pixel, 1 px = 1 m):
    python src/label_patch.py --site stratford --year 1958 --row 1200 --col 800 --labeller rayyan
A list of patches (csv with columns site,year,row,col):
    python src/label_patch.py --patches pilot_patches.csv --labeller rayyan
Minutes per patch from the log:
    python src/label_patch.py --summary

Labels are saved to data/labels/<site>_<year>_r<row>_c<col>.png and each session is
added to data/labels/label_log.csv.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "labels"

PATCH = 250
MARGIN = 25       # context shown around the patch
IDLE_CAP = 120    # seconds; longer pauses are not counted in the time log

UNLABELLED, IGNORE, NODATA = 0, 5, 255
CLASSES = {1: "urban", 2: "agricultural", 3: "vegetated", 4: "water", IGNORE: "ignore"}
COLOURS = {1: (220, 50, 47), 2: (235, 190, 30), 3: (30, 150, 60), 4: (40, 110, 230), IGNORE: (170, 70, 210)}
KEYS = {"1": 1, "2": 2, "3": 3, "4": 4, "i": IGNORE}

HELP = """
keys:  1 urban  2 agricultural  3 vegetated  4 water  i ignore
       left drag = paint, right drag = erase
       b = brush / superpixel mode    [ ] = brush size
       f = fill all unlabelled pixels with the current class
       z = undo   h = hide labels   s = save   q = save and close   esc = close without saving
"""

LOG_FIELDS = ["patch_id", "site", "year", "row", "col", "labeller", "started", "minutes",
              "pct_unlabelled", "pct_urban", "pct_agricultural", "pct_vegetated", "pct_water",
              "pct_ignore", "pct_nodata"]


def patch_id(site, year, row, col):
    return f"{site}_{year}_r{row:05d}_c{col:05d}"


def load_valid(site, year, rgb):
    # use the mask from preprocess.py if it exists, otherwise make it the same way
    path = DATA / "processed" / site / f"{year}_valid.png"
    if path.exists():
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
    print("no saved valid mask, computing it (run src/preprocess.py to save it)")
    invalid = pp.nodata_mask(rgb)
    for (r0, r1), (c0, c1) in pp.KNOWN_GAPS.get((site, year), []):
        invalid[r0:r1, c0:c1] = True
    return ~invalid


class Patch:
    def __init__(self, site, year, row, col, labeller, rgb, valid, out_dir=LABELS):
        self.site, self.year, self.row, self.col = site, year, row, col
        self.labeller = labeller
        self.id = patch_id(site, year, row, col)
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / f"{self.id}.png"

        H, W = rgb.shape[:2]
        if row < 0 or col < 0 or row + PATCH > H or col + PATCH > W:
            raise ValueError(f"{self.id} is outside the {H}x{W} image")
        self.image = rgb[row:row + PATCH, col:col + PATCH]
        self.valid = valid[row:row + PATCH, col:col + PATCH]
        self.r0, self.c0 = max(row - MARGIN, 0), max(col - MARGIN, 0)
        self.context = rgb[self.r0:row + PATCH + MARGIN, self.c0:col + PATCH + MARGIN]

        self.labels = np.zeros((PATCH, PATCH), np.uint8)
        if self.path.exists():
            print("resuming", self.path.name)
            self.labels = cv2.imread(str(self.path), cv2.IMREAD_GRAYSCALE)
        self.labels[self.labels == NODATA] = UNLABELLED
        self.labels[~self.valid] = NODATA

        self.current = 1
        self.brush = False
        self.radius = 4
        self.history = []
        self.make_segments()

        self.started = datetime.now()
        self.last = time.monotonic()
        self.seconds = 0.0

    def make_segments(self):
        from skimage.segmentation import slic
        colour = np.abs(self.image.astype(int) - self.image[..., :1]).max() > 0
        if colour:
            img = self.image
        else:
            img = self.image[..., 0] * (100 / 255)  # so compactness=10 works like for colour
        self.segments = slic(img, n_segments=400, compactness=10, sigma=1, start_label=1,
                             mask=self.valid if self.valid.any() else None,
                             channel_axis=-1 if colour else None, convert2lab=colour)

    def tick(self):
        now = time.monotonic()
        self.seconds += min(now - self.last, IDLE_CAP)
        self.last = now

    def start_stroke(self):
        self.history.append(self.labels.copy())
        self.history = self.history[-30:]
        self.tick()

    def stroke(self, r0, c0, r1, c1):
        """Pixels covered by a mouse move from (r0, c0) to (r1, c1)."""
        line = np.zeros((PATCH, PATCH), np.uint8)
        if self.brush:
            cv2.line(line, (c0, r0), (c1, r1), 1, thickness=2 * self.radius + 1)
            cv2.circle(line, (c1, r1), self.radius, 1, -1)
            return line > 0
        cv2.line(line, (c0, r0), (c1, r1), 1)
        ids = np.unique(self.segments[line > 0])
        return np.isin(self.segments, ids[ids > 0])

    def paint(self, mask, value):
        self.labels[mask & self.valid] = value

    def fill_rest(self):
        self.start_stroke()
        self.labels[self.labels == UNLABELLED] = self.current

    def undo(self):
        if self.history:
            self.labels = self.history.pop()

    def percentages(self):
        pct = {"pct_unlabelled": (self.labels == UNLABELLED).mean(), "pct_nodata": (self.labels == NODATA).mean()}
        for v, name in CLASSES.items():
            pct["pct_" + name] = (self.labels == v).mean()
        return {k: round(100 * float(x), 1) for k, x in pct.items()}

    def save(self):
        self.tick()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.path), self.labels)
        log = self.out_dir / "label_log.csv"
        new = not log.exists()
        with open(log, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"patch_id": self.id, "site": self.site, "year": self.year, "row": self.row,
                        "col": self.col, "labeller": self.labeller,
                        "started": self.started.isoformat(timespec="seconds"),
                        "minutes": round(self.seconds / 60, 2), **self.percentages()})
        # if we keep going after saving, the next log row only has the new time
        self.started, self.seconds = datetime.now(), 0.0
        print(f"saved {self.path.name}, {self.percentages()['pct_unlabelled']}% unlabelled")

    def overlay(self):
        rgba = np.zeros((PATCH, PATCH, 4))
        for v, colour in COLOURS.items():
            rgba[self.labels == v] = (*np.array(colour) / 255, 0.45)
        rgba[self.labels == NODATA] = (0.15, 0.15, 0.15, 0.85)
        if not self.brush:
            from skimage.segmentation import find_boundaries
            edges = find_boundaries(self.segments, mode="inner") & self.valid
            rgba[edges] = (1, 1, 1, 0.4)
        return rgba


def label_gui(p):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # turn off matplotlib's own shortcuts (s, f, q, ... ) so they don't clash with ours
    for k in plt.rcParams:
        if k.startswith("keymap."):
            plt.rcParams[k] = []

    fig, ax = plt.subplots(figsize=(9, 9.5))
    h, w = p.context.shape[:2]
    top, left = p.row - p.r0, p.col - p.c0
    ax.imshow(p.context, extent=(-left - 0.5, w - left - 0.5, h - top - 0.5, -top - 0.5))
    over = ax.imshow(p.overlay(), extent=(-0.5, PATCH - 0.5, PATCH - 0.5, -0.5))
    ax.add_patch(Rectangle((-0.5, -0.5), PATCH, PATCH, fill=False, ec="cyan"))
    ax.set_xticks([])
    ax.set_yticks([])
    state = {"value": None, "last": None, "hide": False}

    def redraw():
        over.set_data(np.zeros((PATCH, PATCH, 4)) if state["hide"] else p.overlay())
        mode = f"brush {p.radius}" if p.brush else "superpixels"
        ax.set_title(f"{p.id}   class: {CLASSES[p.current]}   {mode}\n"
                     f"{p.percentages()['pct_unlabelled']}% unlabelled, {p.seconds / 60:.1f} min")
        fig.canvas.draw_idle()

    def where(event):
        toolbar = fig.canvas.toolbar
        if event.inaxes != ax or (toolbar is not None and toolbar.mode):
            return None
        r, c = int(round(event.ydata)), int(round(event.xdata))
        if 0 <= r < PATCH and 0 <= c < PATCH:
            return r, c
        return None

    def press(event):
        rc = where(event)
        if rc is None or event.button not in (1, 3):
            return
        state["value"] = p.current if event.button == 1 else UNLABELLED
        state["last"] = rc
        p.start_stroke()
        p.paint(p.stroke(*rc, *rc), state["value"])
        redraw()

    def move(event):
        rc = where(event)
        if state["value"] is None or rc is None:
            return
        p.paint(p.stroke(*state["last"], *rc), state["value"])
        state["last"] = rc
        redraw()

    def release(event):
        state["value"] = None
        p.tick()

    def key(event):
        k = event.key
        if k in KEYS:
            p.current = KEYS[k]
        elif k == "b":
            p.brush = not p.brush
        elif k == "[":
            p.radius = max(p.radius - 1, 0)
        elif k == "]":
            p.radius += 1
        elif k == "f":
            p.fill_rest()
        elif k == "z":
            p.undo()
        elif k == "h":
            state["hide"] = not state["hide"]
        elif k == "s":
            p.save()
        elif k == "q":
            p.save()
            plt.close(fig)
            return
        elif k == "escape":
            plt.close(fig)
            return
        redraw()

    fig.canvas.mpl_connect("button_press_event", press)
    fig.canvas.mpl_connect("motion_notify_event", move)
    fig.canvas.mpl_connect("button_release_event", release)
    fig.canvas.mpl_connect("key_press_event", key)
    redraw()
    print(p.id, HELP)
    plt.show()


def summary(out_dir=LABELS):
    log = Path(out_dir) / "label_log.csv"
    if not log.exists():
        print("no label_log.csv yet")
        return
    minutes, left = {}, {}
    with open(log, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["patch_id"], r["labeller"])
            minutes[key] = minutes.get(key, 0) + float(r["minutes"])
            left[key] = float(r["pct_unlabelled"])
    for key in sorted(minutes):
        print(f"{key[0]:34s} {key[1]:10s} {minutes[key]:6.1f} min   {left[key]:5.1f}% unlabelled")
    done = [m for k, m in minutes.items() if left[k] == 0]
    if done:
        print(f"{len(done)} finished patches, mean {np.mean(done):.1f} min "
              f"-> 200 patches = about {200 * np.mean(done) / 60:.0f} hours total")


def main():
    ap = argparse.ArgumentParser(description="Label land-cover patches (see docs/class_guide.md)")
    ap.add_argument("--site", choices=pp.SITES)
    ap.add_argument("--year", type=int, choices=pp.YEARS)
    ap.add_argument("--row", type=int)
    ap.add_argument("--col", type=int)
    ap.add_argument("--patches", help="csv with columns site,year,row,col")
    ap.add_argument("--labeller")
    ap.add_argument("--out", default=LABELS)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.summary:
        summary(args.out)
        return
    if not args.labeller:
        ap.error("--labeller is needed for the time log")
    if args.patches:
        with open(args.patches, newline="") as f:
            todo = [(r["site"], int(r["year"]), int(r["row"]), int(r["col"])) for r in csv.DictReader(f)]
    elif None not in (args.site, args.year, args.row, args.col):
        todo = [(args.site, args.year, args.row, args.col)]
    else:
        ap.error("give --patches, or --site --year --row --col")

    loaded = None
    for i, (site, year, row, col) in enumerate(todo, 1):
        done = Path(args.out) / f"{patch_id(site, year, row, col)}.png"
        if len(todo) > 1 and done.exists() and not (cv2.imread(str(done), 0) == UNLABELLED).any():
            print(f"{i}/{len(todo)} {done.name} already done, skipping")
            continue
        if loaded is None or loaded[0] != (site, year):
            rgb = pp.load_rgb(DATA / site / f"{year}.png")
            loaded = ((site, year), rgb, load_valid(site, year, rgb))
        print(f"{i}/{len(todo)}", end=" ")
        label_gui(Patch(site, year, row, col, args.labeller, loaded[1], loaded[2], args.out))


if __name__ == "__main__":
    main()
