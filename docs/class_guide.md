# Class guide for labelling

Use this for every patch so we all label the same way. If you disagree with a rule, change it here
before the pilot labelling, not halfway through.

## Classes

| Value | Class | What goes in it |
|---|---|---|
| 1 | urban / peri-urban | buildings, roads, parking lots, driveways, rail lines, house lots including the lawn and garden, farmyards, sports fields, golf courses, cemeteries, construction sites |
| 2 | agricultural | crop fields, ploughed or bare fields, hay, pasture, fallow fields, orchards, farm lanes |
| 3 | vegetated | forest, woodlots, shrubs, hedgerows wider than ~10 m, old fields mostly grown over with trees, wetland |
| 4 | water | harbour, rivers, ponds that are visible (not the parts already blacked out) |
| 5 | ignore | things we can't classify: ink marks, scratches, scan damage, very dark shadow, beach / tidal flats |
| 0 | unlabelled | not done yet, a finished patch should have none |
| 255 | no data | set automatically from the preprocessing mask, can't be painted |

Only 1-4 are used for training and scoring.

## Rules for the hard cases

- **Lawns = urban.** A lawn around a house is part of the lot. New subdivisions are mostly lawn, so this matters for the change numbers.
- **Pasture = agricultural**, even if it looks like rough grass. Straight edges and fences usually mean farmland.
- **Peri-urban = urban** (to confirm). Houses along a rural road are labelled urban lot by lot. The fields between them stay agricultural. No separate class.
- **Farm buildings = urban** (to confirm). That way every year is labelled the same.
- **Don't outline small things** (under ~5 x 5 m), like a single tree or a shed. Give them the class around them.
- **Edges:** draw the line where the cover changes. Don't use ignore for edge pixels.
- **1958 ink marks = ignore** (to confirm). If there are a lot of them, it's better to mask them in preprocessing.
- **Label each year from its own image.** Don't copy what you know from 2020 onto older years.
- **Grey years:** forest looks dark and rough, fields smooth. Use that when it's hard to tell pasture from young trees.
- If you're still unsure after ~10 s, use ignore, but keep it under ~5% of the patch.

## Tool

We're using `src/label_patch.py` instead of QGIS / Label Studio / GIMP. It locks the no-data areas,
saves masks in the format the training code needs, and records minutes per patch (we need that for
the pilot). It opens a window, so run it locally, not in Colab.

```bash
python src/label_patch.py --site stratford --year 1958 --row 1200 --col 800 --labeller rayyan
python src/label_patch.py --patches pilot_patches.csv --labeller rayyan   # csv: site,year,row,col
python src/label_patch.py --summary                                       # minutes per patch
```

Quickest way: click superpixels for the smaller classes, press `b` for the brush to fix edges, then
choose the main class and press `f` to fill the rest. `q` saves. All keys are printed in the terminal.

Labels go to `data/labels/`, which isn't in git, so copy that folder to the Drive after labelling
until we decide where labels should be kept.
