# Labelling guide: classes and rules

**Applies to:** every hand-labelled 250 × 250 m patch, all years, both sites.
**Tool:** `src/label_patch.py` (how to run it is at the bottom).
**Status:** draft for the team to confirm. Rules marked *(proposed)* are the open decisions
from the team plan; change them here, in one place, before the pilot labelling (task 4).

## The classes

| Value | Class | Key | Includes | Does **not** include |
|---|---|---|---|---|
| 1 | **Urban / peri-urban** | `1` | Buildings, paved and gravel roads, parking lots, driveways, rail lines; the **whole lot** of a house or farmyard, lawn and garden included; sports fields, golf courses, cemeteries, school grounds; construction sites and land cleared for new streets | Farm lanes and tracks between fields (→ agricultural) |
| 2 | **Agricultural** | `2` | Crop fields, ploughed or bare fields, hay fields, **pasture**, fallow fields, orchards, farm lanes | The farmhouse, barns and the yard around them (→ urban) |
| 3 | **Vegetated** | `3` | Forest, woodlots, shrubland, hedgerows and shelterbelts **wider than about 10 m**, abandoned fields that are mostly (> 50 %) shrubs or trees, wetland and marsh | Mown lawns (→ urban), pasture (→ agricultural) |
| 4 | **Water** | `4` | Harbour, rivers, ponds, open water you can see in the image | Water already blacked out as no-data (it is locked) |
| 5 | **Ignore** | `i` | Pixels you would only be guessing at: ink marks and scratches, scan damage, deep shadow, cloud, beach and tidal flats | Anything you can classify with reasonable confidence |
| 0 | *unlabelled* | right-drag | Left over when you have not finished. A finished patch has 0 % unlabelled | |
| 255 | *no data* | locked | Set automatically from the preprocessing mask | |

Training and scoring use only classes 1–4. Ignore, unlabelled and no-data pixels are skipped.

## Decision rules for the hard cases

1. **Lawns are urban.** A mown lawn around a house is part of a developed lot. This matters because
   the research question counts land turned urban, and a new subdivision is mostly lawn.
2. **Pasture is agricultural.** That holds even when it looks like rough grass. Use field shape:
   straight edges, fences and a regular outline mean farmland.
3. **Peri-urban = urban** *(proposed)*. Scattered houses along a rural road are labelled urban lot by lot.
   The fields between them stay agricultural. There is no separate peri-urban class, so the U-Net
   doesn't have to learn a fifth class from 200 patches.
4. **Farm buildings are urban** *(proposed)*. Buildings are buildings, so every year is mapped the same way.
   In the change analysis, a farmyard counts as "urban" in both years and adds no change.
5. **Smallest feature: about 5 × 5 m.** Don't outline anything smaller, like a single tree in a field or
   a garden shed. Give it the class of its surroundings. Hedgerows under ~10 m wide go with the field.
6. **Edges:** put the boundary where the cover changes. If a pixel is half and half, pick either side.
   Don't use *ignore* for edges.
7. **1958 forestry ink marks → ignore** *(proposed)*. They don't appear in the no-data mask, and a
   model shouldn't learn them as land cover. If there turn out to be many, preprocessing should mask them instead.
8. **Label each year from its own image only.** Don't copy the 2020 answer onto 1935, even if you
   think you know what was there. The labels have to match what the model can see.
9. **Greyscale years:** when you can't tell pasture from young forest, use texture. Forest is rough
   (separate crowns) and dark; fields are smooth. Press `x` to stretch the contrast.
10. **Unsure after ~10 seconds → ignore.** Keep *ignore* under ~5 % of a patch. A patch that needs
    much more than that is a bad patch; tell the team rather than labelling it.

## Running the tool

The tool shows the patch with 25 m of surrounding context (cyan box = the patch). The class you're painting
and the timer are in the title. It needs only the packages in `requirements.txt`, and it opens a
window, so run it on your own computer, not in Colab.

```bash
# one patch: --row/--col are the top-left pixel of the patch (1 px = 1 m)
python src/label_patch.py --site stratford --year 1958 --row 1200 --col 800 --labeller rayyan

# a list of patches, one after another (CSV with columns site,year,row,col)
python src/label_patch.py --patches results/pilot_patches.csv --labeller rayyan

# minutes per patch, for the pilot timing in the progress report
python src/label_patch.py --summary
```

**Fast workflow:** first label the smaller classes with superpixel clicks (one click fills a segment).
Press `b` for the brush to fix edges. Then pick the class that covers most of the patch and press
`f` to fill everything left. `z` undoes, `q` saves and moves to the next patch. All the keys are
printed in the terminal when the window opens.

**Output:** `data/labels/<site>_<year>_r<row>_c<col>.png` (uint8, values as in the table)
and `data/labels/label_log.csv`, one row per session with active minutes. A pause of more than 2
minutes is not counted. Re-opening a patch resumes it.

**Back the labels up.** `data/` is git-ignored, so labels saved there exist only on your laptop.
They are days of hand work. Until the team decides where they live (see below), copy
`data/labels/` to the shared Drive after every session.

## Why this tool

We compared QGIS, Label Studio, GIMP layers and a small Python tool. The Python tool won for four
reasons. It reads our no-data masks and won't let anyone label them. It writes masks straight in
the format the training code reads, so there's no export step. It logs minutes per patch
automatically, which the pilot (task 4) has to report. And it needs no install beyond
`requirements.txt`. The cost is that it runs on a desktop only, not in Colab, and it has no zoom
beyond the matplotlib toolbar.

## Still to decide (team)

- Rules marked *(proposed)* above: peri-urban, farm buildings, ink marks.
- Where labels are stored. Option: commit them under `labels/` (a 250 × 250 label PNG is a few KB)
  with a `.gitignore` exception. They would then be versioned and backed up with the code.
- Patch list format for task 3 (Mohid): the tool reads a CSV with `site,year,row,col` (top-left pixel).
  Any extra columns are ignored.
