# Project structure

`data/images/` holds original micrographs. `data/masks/` holds matching
training labels. Those labels can be exact masks, or corrected overlays when
validation is run with `--with-overlays`. Generated clean masks go in
`data/validated_masks/`.

`data/test/images/` holds images used for checking a trained model without
adding them to training. Prediction output normally goes to `outputs/`.

`src/` contains shared label, dataset, model, loss, and validation code.

`statistics/` contains small analysis scripts for results already produced by
the current project.

# Flow

```text
paint masks
    -> validate_data.py
    -> train.py
    -> checkpoints/final.pt
    -> predict.py
    -> outputs/
    -> statistics/get-statistics.py
```

Training uses image patches and produces a model checkpoint. Prediction loads
that checkpoint and produces a four-color mask plus an overlay.

Statistics read the four-color prediction masks and count pixels per class.
This replaces only the useful statistics part of the old `Entrega01.ipynb`
notebook from `C:\lolo\dev\micrography-imgpro`; it does not bring over the
legacy processing modules.

# Control scripts

`validate_data.py`

Checks that every training image has a same-name PNG mask, that image and mask
sizes match, and that masks use the expected label colors. It can write cleaned
masks to `data/validated_masks/`. With `--with-overlays`, mixed exact masks and
corrected overlays are allowed in `data/masks/`; overlays are converted back to
exact masks before being written to validated data.

`train.py`

Validates the training data, trains the segmentation model, and writes
checkpoints into `checkpoints/`. It writes `latest.pt` every epoch and
`final.pt` at the end.

`predict.py`

Runs one image through a trained checkpoint. It writes a color mask and overlay
to `outputs/` unless another output folder is passed.

`predict-configurable.py`

Runs prediction on one image or every supported image directly inside a folder.
It uses the constants at the top of the file for checkpoint, output folder,
tile size, and overlap. Its default output folder is `outputs/test/`.
s
`statistics/get-statistics.py`

Calculates `pores`, `fibers`, `resin`, `undefined`, and `sumcheck` percentages
from exact-color segmentation masks. If given a folder, it prefers `*-mask`
files so overlays are ignored. It writes `statistics/statistics.csv` by
default and supports `--crop-bottom 290` when old microscope labels need to be
excluded.

# Important

Do not train until the masks are manually painted and validated.
`python train.py` starts fresh by default. Use
`python train.py --resume checkpoints/latest.pt --epochs N` to continue, where
`N` is the final epoch number, not extra epochs.
The legacy `example-microg-processed/` folder is not used by the pipeline.


# TODO

- Desktop batch analysis keeps generated results only in memory until the user discards them or closes the app; downloads create a ZIP without clearing the batch. It has a 25-image limit but no byte-based memory limit yet. Add one if larger batches create memory pressure.

- a way to know that images have been uploaded or files selected.

- an actuall progress bar while processing

- Let users click a batch overlay to open an in-app modal. Show the original
  beside the overlay and the four composition cards; after the original is no
  longer available, show the overlay and its statistics alone.

- Add four box-and-strip plots after the batch composition cards and before the
  image table. Plot each image's fiber, resin, pore, and unidentified
  percentage to show spread, median, and outliers. Wait for sample or
  magnification grouping before adding further comparative analytics.
