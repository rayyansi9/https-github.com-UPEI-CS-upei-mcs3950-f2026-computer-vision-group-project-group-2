"""Step 1 of the pipeline: greyscale images and valid-data masks for every site and year.

The provided imagery is already co-registered: for each site, all five years share
one 1 m pixel grid (see data/<site>/info.txt), so no resampling or registration is
done here. This script only

  1. converts every year to single-band greyscale (2020 RGB is kept as-is too),
  2. builds a valid-data mask: black no-data areas (water cut-outs, mosaic edges)
     and the known 1958 archive gap in Charlottetown are marked invalid,
  3. writes a summary table so the masks can be checked against info.txt.

Input:   data/<site>/<year>.png
Output:  data/processed/<site>/<year>_grey.png    uint8 greyscale
         data/processed/<site>/<year>_valid.png   uint8, 255 = valid, 0 = no data
         data/processed/<site>/2020_rgb.png       copy of the colour image
         results/preprocess_summary.csv           no-data % per site and year
         results/nodata_masks.png                 downsampled overview figure

Usage:   python src/preprocess.py [--sites stratford charlottetown]
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "processed"
RESULTS = ROOT / "results"

SITES = ["stratford", "charlottetown"]
YEARS = [1935, 1958, 1968, 2000, 2020]

# A pixel is "black" if every channel is at or below this value.
NODATA_MAX = 0
# Black regions smaller than this (px = m^2) are treated as real dark ground
# (deep shadow, dark roofs), not as no-data.
MIN_NODATA_AREA = 500
# Grow no-data regions by this many pixels to drop the dark fringe at mosaic edges.
EDGE_BUFFER = 3

# Areas known to be missing from the archive, from info.txt: (rows, cols), half-open.
KNOWN_GAPS = {
    ("charlottetown", 1958): [((0, 3487), (3804, 7005))],
}


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_grey(rgb):
    """Greyscale for all years. The 1935-2000 images are greyscale stored as RGB,
    so the conversion is lossless for them; for 2020 it is the standard luma."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def nodata_mask(rgb):
    """True where the image has no data."""
    black = (rgb <= NODATA_MAX).all(axis=2).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_NODATA_AREA
    mask = keep[labels].astype(np.uint8)
    if EDGE_BUFFER > 0:
        k = 2 * EDGE_BUFFER + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask.astype(bool)


def process_site(site):
    src, dst = DATA / site, OUT / site
    dst.mkdir(parents=True, exist_ok=True)
    rows, valids, shape = [], {}, None

    for year in YEARS:
        rgb = load_rgb(src / f"{year}.png")
        if shape is None:
            shape = rgb.shape[:2]
        elif rgb.shape[:2] != shape:
            raise ValueError(f"{site} {year}: shape {rgb.shape[:2]} != {shape}; years must share a grid")

        grey = to_grey(rgb)
        invalid = nodata_mask(rgb)
        for (r0, r1), (c0, c1) in KNOWN_GAPS.get((site, year), []):
            invalid[r0:r1, c0:c1] = True
        valid = ~invalid
        valids[year] = valid

        cv2.imwrite(str(dst / f"{year}_grey.png"), grey)
        cv2.imwrite(str(dst / f"{year}_valid.png"), valid.astype(np.uint8) * 255)
        if year == 2020:
            shutil.copyfile(src / "2020.png", dst / "2020_rgb.png")

        if year != 2020:
            spread = int(np.abs(rgb.astype(np.int16) - rgb[..., :1]).max())
            if spread > 0:
                print(f"  warning: {site} {year} channels differ by up to {spread}; not pure greyscale")

        rows.append({"site": site, "year": year, "height": shape[0], "width": shape[1],
                     "nodata_pct": round(100 * invalid.mean(), 1),
                     "valid_ha": round(valid.sum() / 1e4, 1)})
        print(f"  {site} {year}: {rows[-1]['nodata_pct']:5.1f}% no data, {rows[-1]['valid_ha']:7.1f} ha valid")

    all_valid = np.logical_and.reduce(list(valids.values()))
    cv2.imwrite(str(dst / "all_years_valid.png"), all_valid.astype(np.uint8) * 255)
    print(f"  {site}: valid in all five years: {100 * all_valid.mean():.1f}% of the block")
    return rows, valids


def overview_figure(all_valids, path, step=8):
    fig, axs = plt.subplots(len(all_valids), len(YEARS), figsize=(3 * len(YEARS), 2.6 * len(all_valids)),
                            squeeze=False)
    for i, (site, valids) in enumerate(all_valids.items()):
        for j, year in enumerate(YEARS):
            ax = axs[i, j]
            ax.imshow(valids[year][::step, ::step], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{site} {year}\n{100 * (~valids[year]).mean():.0f}% no data", fontsize=9)
            ax.axis("off")
    fig.suptitle("Valid-data masks (white = valid, black = no data)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", default=SITES, choices=SITES)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    rows, all_valids = [], {}
    for site in args.sites:
        print(f"Processing {site}")
        site_rows, all_valids[site] = process_site(site)
        rows += site_rows

    with open(RESULTS / "preprocess_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    overview_figure(all_valids, RESULTS / "nodata_masks.png")
    print(f"Wrote {OUT.relative_to(ROOT)}/, results/preprocess_summary.csv, results/nodata_masks.png")


if __name__ == "__main__":
    main()
