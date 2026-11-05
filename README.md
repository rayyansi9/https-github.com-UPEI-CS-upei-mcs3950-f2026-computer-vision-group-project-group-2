# Urban Growth and Land-Cover Change in Charlottetown and Stratford

**Team:** Sajid Oyon, Mohammad Rayyan, Mohid Tahir  
**Project option:** 4 — Charlottetown Urban Growth  
**Research questions:**

1. How many hectares of agricultural and vegetated land were converted to
   urban/peri-urban land in Charlottetown and Stratford between 1935 and 2020?
2. How much does classification accuracy decrease when colour is removed from
   the 2020 imagery, particularly when separating agricultural and vegetated land?

The full plan is in [`docs/Group 2 -Project Proposal.pdf`](docs/Group%202%20-Project%20Proposal.pdf)
(revised after instructor feedback).

---

## Status

<!-- Keep this current. Two lines. It is the first thing anyone reads. -->
Proposal revised and submitted. Now setting up the data layout and labelling the first
patches; the pipeline in `src/` has **not been written yet**, so nothing below runs end to end.

Last updated: 2026-11-04

---

## Project overview

| | |
|---|---|
| **Years** | 1935, 1958, 1968, 2000, 2020 |
| **Imagery** | 1935–2000 single-band greyscale; 2020 RGB colour |
| **Classes** | urban/peri-urban, agricultural, vegetated, water (where the imagery supports them) |
| **Baseline** | Random Forest on image intensity + GLCM texture features |
| **Main model** | Compact U-Net, fine-tuned if label quality, validation results and compute time allow |
| **Colour experiment** | RGB vs greyscale U-Net on identical 2020 patches, labels, splits and training settings |
| **Labels** | 40 hand-labelled 250 × 250 m patches per year across both sites → 200 patches |
| **Splits** | Spatially separate 60 / 20 / 20 train / val / test; every year of a location stays in the same split |
| **Metrics** | Mean IoU and per-class F1; change in agricultural/vegetated F1 for the colour experiment |
| **Change analysis** | Transition matrices (hectares agri/veg → urban), using only pixels valid in both years being compared |
| **Fallback** | If the U-Net is unreliable: Random Forest for the main analysis and matched RGB/greyscale RF models for the colour comparison |

**No-data handling.** Missing areas in 1958 and water cutouts in 1935 and 1968 are masked
and excluded from training, evaluation and change calculations.

---

## How to run this

> This section is graded. A reader who has never seen the project should be
> able to reproduce the figures from it.

### What you need

```bash
# Python 3.10+ (Google Colab works out of the box)
pip install -r requirements.txt
```

- Training the U-Net needs a GPU; a Colab GPU runtime is enough. The Random Forest
  baseline runs on CPU.
- PyTorch is not pinned in `requirements.txt`; install it for your platform from
  pytorch.org and record the version there.

### What data it expects

Planned layout (to be confirmed once preprocessing is written):

```
data/
  raw/
    1935/   greyscale scans as provided
    1958/   greyscale scans (contain missing areas)
    1968/   greyscale scans
    2000/   greyscale orthophotos
    2020/   RGB orthophotos
  aligned/  all years resampled to a common projected grid (written by preprocessing)
  labels/   hand-labelled 250 × 250 m patch masks, one file per patch and year
```

**The imagery is not in this repository**; see [Data](#data) below.

### The pipeline, end to end

**Not implemented yet.** The planned order is:

1. **Preprocess.** Align every year to a common projected grid and build no-data masks.
2. **Extract patches.** Cut the 250 × 250 m labelled patches and assign the spatial train/val/test split.
3. **Random Forest baseline.** Compute intensity + GLCM features, then train and evaluate.
4. **U-Net.** Fine-tune and evaluate it, and run the RGB vs greyscale 2020 comparison.
5. **Change analysis.** Classify each year and build the transition matrices in hectares.

Script names and run times will be added here as each step lands in `src/`.

### Which script produces which figure

<!-- Fill this in as you write the report, not the night before. -->

| Figure / table in the report | Produced by | Notes |
|---|---|---|
| Classification maps per year | *TBD* | |
| RF vs U-Net: mean IoU and per-class F1 | *TBD* | |
| RGB vs greyscale 2020 comparison | *TBD* | |
| Transition matrices (ha converted to urban) | *TBD* | |

---

## Repository layout

```
data/           imagery: NOT committed, see below
notebooks/      exploration and figure generation (forest_demo.ipynb is the in-class demo)
src/            the pipeline (empty so far)
results/        small output artefacts (numbers, plots) that the report uses
docs/           proposal, progress report, final report
```

---

## Data

The aerial imagery is **not committed to this repository**: it is large and its
redistribution is restricted. Only code, configuration and small result
artefacts are committed.

- **Source:** *TODO: where the 1935–2020 Charlottetown/Stratford imagery came from*
- **To obtain it:** *TODO: e.g. the course shared Google Drive folder, and its path*
- **Preprocessing already applied:** none yet. Alignment to a common projected grid and
  no-data masking will be done in code under `src/`, not by hand.

---

## Milestones

| Milestone | Due | Done |
|---|---|---|
| Team formation & project ranking | Fri Sept 25 | ✅ |
| Project proposal (1 page) | Mon Oct 5 | ✅ (revised Oct 20 after feedback) |
| Progress report (2–3 pages + repo link) | Mon Nov 9 | [ ] |
| Poster session — Day 1 | Mon Dec 7 | [ ] |
| Poster session — Day 2 | Wed Dec 9 | [ ] |
| Final report (6–8 pp, IEEE format) | Wed Dec 16 | [ ] |
| Repository development history | Wed Dec 16 | [ ] |

---

## How this repository is graded

<!-- Leave this section in place. It is here so nobody can say they didn't know. -->

The repository is worth **10% of the project mark** and is graded on *how the
work developed*, not only on what it ends up containing. A repository whose
entire history is one commit made the night before the deadline scores poorly
no matter how good the final code is.

| Expectation | What it means |
|---|---|
| Sustained activity | Commits spread across the project period, not bunched at the end |
| Meaningful commit messages | `fix registration drift on 1958 tiles` — not `update`, `stuff`, `asdf` |
| Visible iteration | Things tried, failed, changed. Dead ends are expected — delete them in a later commit rather than hiding that they happened |
| All members contributing | Every member has commits under their own identity |
| Working state at milestones | It runs at the progress report checkpoint, or the README says plainly what does not |
| Reproducibility | The "How to run this" section above, filled in properly |

**Before your first commit:** set your Git identity, or your work will be
credited to somebody else.

```bash
git config user.name  "Your Name"
git config user.email "your@email.com"
```

If you pair-program, say so in the commit message (`co-authored with ...`) so
the history reflects what actually happened.

---

## Use of generative AI

AI use is permitted throughout this project, for code and for prose. **What is
required is disclosure.** Undisclosed use is academic dishonesty; disclosed use
is not.

| Tool | Used for | Where it shows up |
|---|---|---|
| ChatGPT | Revising proposal wording and LaTeX formatting | Revised proposal in `docs/` |
| Claude Code | Summarising the proposal into this README and restructuring its sections | `README.md` (Nov 4) |
