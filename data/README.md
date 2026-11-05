# data/

**Nothing in this folder is committed.** The `.gitignore` excludes everything
here except this file — the imagery is large and its redistribution is
restricted, so it lives on your machine, not in the repository.

## What to put here

Whatever your project option needs, in a layout you describe in the top-level
README under *"What data it expects"*. A reader who follows that description
should end up with exactly the structure your code assumes. For example:

```
data/
  1935/     scanned panchromatic tiles
  1968/     scanned panchromatic tiles
  2020/     colour orthophoto tiles
```

## Two things worth doing now

**Write down where your files came from,** in the top-level README. In December
you will not remember which of three similar folders on your laptop was the one
the good results came from.

**Record your preprocessing as code, not as mouse clicks.** If you resample,
crop, or register the imagery by hand in QGIS or Photoshop, the result is not
reproducible and your teammates cannot regenerate it. Put those steps in a
script under `src/` that reads from here and writes somewhere else.
