"""Label one 250 x 250 m patch by hand: paint land-cover classes, save a label mask, log the time.

The class rules are in docs/class_guide.md. Read them before labelling.

Usage
-----
One patch (row/col = top-left pixel of the patch; 1 px = 1 m):

    python src/label_patch.py --site stratford --year 1958 --row 1200 --col 800 --labeller rayyan

A list of patches, one after another (CSV with columns site,year,row,col; extra columns ignored):

    python src/label_patch.py --patches results/pilot_patches.csv --labeller rayyan

Time spent per patch (for the pilot timing in the progress report):

    python src/label_patch.py --summary

Input:   data/<site>/<year>.png                      the image
         data/processed/<site>/<year>_valid.png      valid-data mask from src/preprocess.py
                                                     (computed on the fly if missing)
Output:  data/labels/<site>_<year>_r<row>_c<col>.png  uint8 label mask, PATCH x PATCH px
         data/labels/label_log.csv                    one row per labelling session

Label values (see CLASSES below): 0 not yet labelled, 1 urban/peri-urban, 2 agricultural,
3 vegetated, 4 water, 5 ignore (unclear), 255 no data (locked, cannot be painted).
Training and scoring use only 1-4; 0, 5 and 255 are skipped.

Controls (also printed when the window opens)
--------
  1 2 3 4     choose class: urban, agricultural, vegetated, water
  i           choose "ignore" (unclear pixels)
  left drag   paint the current class      right drag   erase (back to unlabelled)
  b           switch superpixel mode (click fills a whole segment) / brush mode
  [ ]         smaller / larger brush       - =   fewer / more superpixels
  f           fill every still-unlabelled pixel with the current class
  z           undo last stroke             g     show / hide segment outlines
  h           hide / show the label overlay (look at the raw image)
  x           contrast stretch on / off    s     save
  q           save and close (next patch in list mode)
  escape      close without saving         Q     save and stop the whole list

The timer counts active time only: a gap of more than IDLE_CAP_S seconds between two actions
counts as IDLE_CAP_S, so a coffee break does not inflate the minutes-per-patch figure.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "labels"

PATCH = 250        # px = m, from the proposal
MARGIN = 25        # px of surrounding context shown around the patch (not labelled)
IDLE_CAP_S = 120   # longest gap between actions that still counts as working time

UNLABELLED, IGNORE, NODATA = 0, 5, 255
# value: (name, key, RGB colour)
CLASSES = {
    1: ("urban", "1", (220, 50, 47)),
    2: ("agricultural", "2", (235, 190, 30)),
    3: ("vegetated", "3", (30, 150, 60)),
    4: ("water", "4", (40, 110, 230)),
    IGNORE: ("ignore", "i", (170, 70, 210)),
}
KEY_TO_CLASS = {key: v for v, (_, key, _) in CLASSES.items()}
LOG_FIELDS = ["patch_id", "site", "year", "row", "col", "labeller", "started", "active_min",
              "wall_min", "pct_unlabelled", "pct_urban", "pct_agricultural", "pct_vegetated",
              "pct_water", "pct_ignore", "pct_nodata"]


def patch_id(site, year, row, col):
    return f"{site}_{year}_r{row:05d}_c{col:05d}"


def load_valid(site, year, rgb):
    """Valid-data mask for the whole image: the saved one if preprocessing has run, else computed."""
    saved = DATA / "processed" / site / f"{year}_valid.png"
    if saved.exists():
        valid = cv2.imread(str(saved), cv2.IMREAD_GRAYSCALE)
        if valid is not None and valid.shape == rgb.shape[:2]:
            return valid > 0
        print(f"warning: {saved} unreadable or wrong size; recomputing")
    print("note: no saved valid mask (run src/preprocess.py); computing it now")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import preprocess as pp
    invalid = pp.nodata_mask(rgb)
    for (r0, r1), (c0, c1) in pp.KNOWN_GAPS.get((site, year), []):
        invalid[r0:r1, c0:c1] = True
    return ~invalid


def stretch(img):
    """2-98 percentile contrast stretch, per channel."""
    out = np.empty_like(img)
    for ch in range(img.shape[2]):
        lo, hi = np.percentile(img[..., ch], (2, 98))
        out[..., ch] = np.clip((img[..., ch].astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255)
    return out


class PatchLabeller:
    """Label state and editing operations for one patch. No GUI code here, so it can be tested."""

    def __init__(self, site, year, row, col, labeller, rgb=None, valid=None, out_dir=LABELS):
        self.site, self.year, self.row, self.col, self.labeller = site, int(year), int(row), int(col), labeller
        self.id = patch_id(site, self.year, self.row, self.col)
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / f"{self.id}.png"

        if rgb is None:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import preprocess as pp
            rgb = pp.load_rgb(DATA / site / f"{self.year}.png")
        if valid is None:
            valid = load_valid(site, self.year, rgb)
        H, W = rgb.shape[:2]
        r, c = self.row, self.col
        if r < 0 or c < 0 or r + PATCH > H or c + PATCH > W:
            raise ValueError(f"{self.id}: patch runs off the {H} x {W} image")

        # context window (patch + margin, clipped to the image) and where the patch sits inside it
        self.ctx_r0, self.ctx_c0 = max(r - MARGIN, 0), max(c - MARGIN, 0)
        ctx_r1, ctx_c1 = min(r + PATCH + MARGIN, H), min(c + PATCH + MARGIN, W)
        self.context = rgb[self.ctx_r0:ctx_r1, self.ctx_c0:ctx_c1].copy()
        self.image = rgb[r:r + PATCH, c:c + PATCH].copy()
        self.valid = valid[r:r + PATCH, c:c + PATCH].copy()
        self.is_colour = bool(np.abs(self.image.astype(np.int16) - self.image[..., :1]).max() > 0)

        self.labels = np.full((PATCH, PATCH), UNLABELLED, np.uint8)
        if self.path.exists():
            old = cv2.imread(str(self.path), cv2.IMREAD_GRAYSCALE)
            if old is not None and old.shape == self.labels.shape:
                self.labels = old
                print(f"resuming {self.path.relative_to(ROOT) if self.path.is_relative_to(ROOT) else self.path}")
        self.labels[~self.valid] = NODATA
        self.labels[self.valid & (self.labels == NODATA)] = UNLABELLED  # mask may have changed since

        self.current = 1
        self.brush_mode = False
        self.brush_radius = 4
        self.n_segments = 400
        self.segments = None
        self.compute_segments()
        self.undo_stack = []
        self.dirty = False

        self.started = datetime.now()
        self.t_start = self.t_last = time.monotonic()
        self.active_s = 0.0

    # ── timing ────────────────────────────────────────────────────────────────
    def tick(self):
        now = time.monotonic()
        self.active_s += min(now - self.t_last, IDLE_CAP_S)
        self.t_last = now

    # ── segments ──────────────────────────────────────────────────────────────
    def compute_segments(self):
        from skimage.segmentation import slic
        if self.is_colour:
            img, kw = self.image, {"channel_axis": -1}
        else:  # grey: scale to 0-100 so the default compactness behaves like it does in Lab
            img, kw = self.image[..., 0].astype(np.float32) * (100 / 255), {"channel_axis": None}
        self.segments = slic(img, n_segments=self.n_segments, compactness=10, sigma=1, start_label=1,
                             mask=self.valid if self.valid.any() else None, convert2lab=self.is_colour,
                             **kw).astype(np.int32)

    # ── editing ───────────────────────────────────────────────────────────────
    def begin_stroke(self):
        self.undo_stack.append(self.labels.copy())
        del self.undo_stack[:-50]
        self.tick()

    def stroke_mask(self, r0, c0, r1=None, c1=None):
        """Pixels covered by a stroke from (r0, c0) to (r1, c1), in the current mode."""
        r1, c1 = (r0, c0) if r1 is None else (r1, c1)
        line = np.zeros((PATCH, PATCH), np.uint8)
        thick = 2 * self.brush_radius + 1 if self.brush_mode else 1
        cv2.line(line, (int(c0), int(r0)), (int(c1), int(r1)), 1, thickness=thick)
        if self.brush_mode:
            cv2.circle(line, (int(c1), int(r1)), self.brush_radius, 1, -1)
            return line.astype(bool)
        ids = np.unique(self.segments[line.astype(bool)])
        ids = ids[ids > 0]
        return np.isin(self.segments, ids)

    def paint(self, mask, value):
        mask = mask & self.valid
        if (self.labels[mask] != value).any():
            self.labels[mask] = value
            self.dirty = True

    def fill_unlabelled(self):
        self.begin_stroke()
        self.paint(self.labels == UNLABELLED, self.current)

    def undo(self):
        if self.undo_stack:
            self.labels = self.undo_stack.pop()
            self.dirty = True
            self.tick()

    # ── output ────────────────────────────────────────────────────────────────
    def fractions(self):
        n = self.labels.size
        out = {"pct_unlabelled": 100 * (self.labels == UNLABELLED).sum() / n,
               "pct_nodata": 100 * (self.labels == NODATA).sum() / n}
        for v, (name, _, _) in CLASSES.items():
            out[f"pct_{name}"] = 100 * (self.labels == v).sum() / n
        return {k: round(float(x), 1) for k, x in out.items()}

    def save(self):
        self.tick()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.path), self.labels):
            raise OSError(f"could not write {self.path}")
        log = self.out_dir / "label_log.csv"
        new = not log.exists()
        with open(log, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"patch_id": self.id, "site": self.site, "year": self.year, "row": self.row,
                        "col": self.col, "labeller": self.labeller,
                        "started": self.started.isoformat(timespec="seconds"),
                        "active_min": round(self.active_s / 60, 2),
                        "wall_min": round((time.monotonic() - self.t_start) / 60, 2), **self.fractions()})
        # the next save of this window logs only the time since this one
        self.started, self.t_start, self.active_s = datetime.now(), time.monotonic(), 0.0
        self.dirty = False
        left = self.fractions()["pct_unlabelled"]
        print(f"saved {self.path.name}" + (f"  ({left:.1f}% still unlabelled)" if left else "  (complete)"))

    # ── display ───────────────────────────────────────────────────────────────
    def overlay_rgba(self, alpha=0.45, outlines=False):
        rgba = np.zeros((PATCH, PATCH, 4), np.float32)
        for v, (_, _, colour) in CLASSES.items():
            m = self.labels == v
            rgba[m, :3] = np.array(colour) / 255
            rgba[m, 3] = alpha
        nd = self.labels == NODATA
        rgba[nd] = (0.15, 0.15, 0.15, 0.85)
        if outlines and not self.brush_mode:
            from skimage.segmentation import find_boundaries
            b = find_boundaries(self.segments, mode="inner") & ~nd
            rgba[b] = (1, 1, 1, 0.5)
        return rgba


# ── GUI ───────────────────────────────────────────────────────────────────────
HELP = "\n" + __doc__.split("Controls (also printed when the window opens)\n--------")[1].split("The timer")[0].rstrip()


def run_gui(lab):
    """Open the labelling window and block until it is closed. Returns 'stop' if Q was pressed."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    for k in list(plt.rcParams):            # free the keys matplotlib uses for its own shortcuts
        if k.startswith("keymap."):
            plt.rcParams[k] = []

    fig, ax = plt.subplots(figsize=(9, 9.6))
    fig.canvas.manager.set_window_title(f"label {lab.id}")
    st = {"drag": None, "last": None, "show": True, "outlines": True, "stretch": False, "result": None}

    m = MARGIN
    top, left = lab.row - lab.ctx_r0, lab.col - lab.ctx_c0
    ch, cw = lab.context.shape[:2]
    extent_ctx = (-left - 0.5, cw - left - 0.5, ch - top - 0.5, -top - 0.5)
    base = ax.imshow(lab.context, extent=extent_ctx, interpolation="nearest")
    over = ax.imshow(lab.overlay_rgba(outlines=True), extent=(-0.5, PATCH - 0.5, PATCH - 0.5, -0.5),
                     interpolation="nearest")
    ax.add_patch(Rectangle((-0.5, -0.5), PATCH, PATCH, fill=False, ec="cyan", lw=1.5))
    ax.set_xlim(-m - 0.5, PATCH + m - 0.5)
    ax.set_ylim(PATCH + m - 0.5, -m - 0.5)
    ax.set_xticks([]), ax.set_yticks([])

    def refresh():
        over.set_data(lab.overlay_rgba(outlines=st["outlines"]) if st["show"] else
                      np.zeros((PATCH, PATCH, 4), np.float32))
        base.set_data(stretch(lab.context) if st["stretch"] else lab.context)
        name = CLASSES[lab.current][0]
        mode = f"brush r={lab.brush_radius}" if lab.brush_mode else f"superpixels ~{lab.n_segments}"
        f = lab.fractions()
        ax.set_title(f"{lab.id}   [{name}]   {mode}\n"
                     f"unlabelled {f['pct_unlabelled']:.0f}%   active {lab.active_s / 60:.1f} min"
                     + ("   *unsaved" if lab.dirty else ""), fontsize=10)
        fig.canvas.draw_idle()

    def pos(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return None
        tb = getattr(fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):    # zoom / pan active: leave the mouse to it
            return None
        r, c = int(round(ev.ydata)), int(round(ev.xdata))
        return (r, c) if 0 <= r < PATCH and 0 <= c < PATCH else None

    def on_press(ev):
        p = pos(ev)
        if p is None or ev.button not in (1, 3):
            return
        st["drag"] = lab.current if ev.button == 1 else UNLABELLED
        st["last"] = p
        lab.begin_stroke()
        lab.paint(lab.stroke_mask(*p), st["drag"])
        refresh()

    def on_motion(ev):
        if st["drag"] is None:
            return
        p = pos(ev)
        if p is None:
            return
        lab.paint(lab.stroke_mask(*st["last"], *p), st["drag"])
        st["last"] = p
        refresh()

    def on_release(ev):
        if st["drag"] is not None:
            st["drag"] = None
            lab.tick()
            refresh()

    def on_key(ev):
        k = ev.key
        if k in KEY_TO_CLASS:
            lab.current = KEY_TO_CLASS[k]
        elif k == "b":
            lab.brush_mode = not lab.brush_mode
        elif k in ("[", "]"):
            lab.brush_radius = int(np.clip(lab.brush_radius + (1 if k == "]" else -1), 0, 30))
        elif k in ("-", "="):
            lab.n_segments = int(np.clip(lab.n_segments * (1.5 if k == "=" else 1 / 1.5), 50, 5000))
            lab.compute_segments()
        elif k == "f":
            lab.fill_unlabelled()
        elif k == "z":
            lab.undo()
        elif k == "g":
            st["outlines"] = not st["outlines"]
        elif k == "h":
            st["show"] = not st["show"]
        elif k == "x":
            st["stretch"] = not st["stretch"]
        elif k == "s":
            lab.save()
        elif k in ("q", "Q"):
            lab.save()
            st["result"] = "stop" if k == "Q" else None
            plt.close(fig)
            return
        elif k == "escape":
            if lab.dirty:
                print(f"closed {lab.id} without saving")
            plt.close(fig)
            return
        else:
            return
        refresh()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.tight_layout()
    refresh()
    print(f"\n{lab.id}: {'colour' if lab.is_colour else 'greyscale'}, "
          f"{100 * (~lab.valid).mean():.0f}% no data{HELP}")
    plt.show()
    return st["result"]


# ── command line ──────────────────────────────────────────────────────────────
def read_patch_list(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    missing = {"site", "year", "row", "col"} - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{path}: missing column(s) {', '.join(sorted(missing))}")
    return [(r["site"].strip(), int(r["year"]), int(r["row"]), int(r["col"])) for r in rows]


def summary(out_dir=LABELS):
    log = Path(out_dir) / "label_log.csv"
    if not log.exists():
        print(f"no log yet at {log}")
        return
    with open(log, newline="") as f:
        rows = list(csv.DictReader(f))
    per = {}
    for r in rows:   # a patch labelled over several sessions: add the sessions up
        key = (r["patch_id"], r["labeller"])
        per.setdefault(key, {"min": 0.0, "left": 0.0})
        per[key]["min"] += float(r["active_min"])
        per[key]["left"] = float(r["pct_unlabelled"])    # last session wins
    print(f"{'patch':34s} {'labeller':10s} {'active min':>10s} {'unlabelled %':>13s}")
    for (pid, who), v in sorted(per.items()):
        print(f"{pid:34s} {who:10s} {v['min']:10.1f} {v['left']:13.1f}")
    done = [v["min"] for v in per.values() if v["left"] == 0]
    if done:
        print(f"\n{len(done)} complete patch(es): mean {np.mean(done):.1f} min, "
              f"median {np.median(done):.1f} min, range {min(done):.1f}-{max(done):.1f} min")
        print(f"at the mean pace, 200 patches = {200 * np.mean(done) / 60:.1f} h in total "
              f"({200 * np.mean(done) / 60 / 3:.1f} h each for three people)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog="Full controls and file formats: see the top of this file.")
    ap.add_argument("--site", choices=["stratford", "charlottetown"])
    ap.add_argument("--year", type=int, choices=[1935, 1958, 1968, 2000, 2020])
    ap.add_argument("--row", type=int, help="top-left pixel row of the patch")
    ap.add_argument("--col", type=int, help="top-left pixel column of the patch")
    ap.add_argument("--patches", type=Path, help="CSV with columns site,year,row,col")
    ap.add_argument("--labeller", help="your name, recorded in the time log")
    ap.add_argument("--out", type=Path, default=LABELS, help=f"label folder (default {LABELS.relative_to(ROOT)})")
    ap.add_argument("--redo", action="store_true", help="in list mode, also open patches already complete")
    ap.add_argument("--summary", action="store_true", help="print minutes per patch from the log and exit")
    a = ap.parse_args()

    if a.summary:
        return summary(a.out)
    if not a.labeller:
        ap.error("--labeller is required (it goes in the time log)")
    if a.patches:
        todo = read_patch_list(a.patches)
    elif None not in (a.site, a.year, a.row, a.col):
        todo = [(a.site, a.year, a.row, a.col)]
    else:
        ap.error("give --patches FILE, or all of --site --year --row --col")

    cache = {}   # full images and masks, so a list with several patches per image loads each once
    for i, (site, year, row, col) in enumerate(todo, 1):
        path = a.out / f"{patch_id(site, year, row, col)}.png"
        if len(todo) > 1 and not a.redo and path.exists():
            done = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if done is not None and not (done == UNLABELLED).any():
                print(f"[{i}/{len(todo)}] {path.name} already complete, skipping (--redo to open it)")
                continue
        if (site, year) not in cache:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import preprocess as pp
            cache.clear()
            rgb = pp.load_rgb(DATA / site / f"{year}.png")
            cache[(site, year)] = (rgb, load_valid(site, year, rgb))
        rgb, valid = cache[(site, year)]
        print(f"[{i}/{len(todo)}]", end=" ")
        lab = PatchLabeller(site, year, row, col, a.labeller, rgb=rgb, valid=valid, out_dir=a.out)
        if run_gui(lab) == "stop":
            break


if __name__ == "__main__":
    main()
