# Fiberglass micrograph segmentation

This project trains a neural network to color each micrograph pixel as fiber,
resin, pore, or unidentified.

## Data

Original images belong in `data/images/`. Paint a same-size PNG mask for each
image and save it in `data/masks/` with the same basename:

```text
data/images/micrography-example-0.jpg
data/masks/micrography-example-0.png
```

`data/masks/` can also contain corrected overlays when validation is run with
`--with-overlays`. Exact masks pass through the normal validation path; overlays
are converted back into exact masks before being written to
`data/validated_masks/`.

Masks must use solid colors:

| Class | RGB color |
| --- | --- |
| Fiber | `(255, 0, 0)` red |
| Resin | `(0, 255, 0)` green |
| Pore | `(0, 0, 255)` blue |
| Unidentified/artifact | `(0, 0, 0)` black |

Other colors are reported and replaced with the most common surrounding
fiber, resin, or pore color in `data/validated_masks/`. Black unidentified
pixels are excluded from that decision. Invalid regions with no reachable
colored neighbors remain black.

## Setup

```bash
source .ven/bin/activate
pip install -r requirements.txt
```

## Commands

Check all masks before training:

```bash
python validate_data.py
```

If `data/masks/` contains a mix of exact masks and corrected overlays, run:

```bash
python validate_data.py --with-overlays
```

Train only after the masks have been reviewed:

```bash
python train.py
```

By default, `python train.py` starts a fresh training run. To continue from the
latest checkpoint, pass `--resume` and set `--epochs` to the final epoch number:

```bash
python train.py --resume checkpoints/latest.pt --epochs 60
```

For example, if the checkpoint is already at epoch 30, `--epochs 60` trains
through epoch 60, not 60 extra epochs. The first run downloads pretrained
MobileNetV2 weights. Training automatically uses a CUDA GPU when available and
otherwise uses the CPU.

Segment a new image:

```bash
python predict.py path/to/image.jpg --checkpoint checkpoints/final.pt
```

This writes an exact-color mask and a visual overlay to `outputs/`.

Process one test image with `predict.py`:

```bash
python predict.py "data/test/images/Experiment-1693.jpg" \
  --checkpoint checkpoints/final.pt \
  --output-dir outputs/test
```

Process one test image or the whole test image folder with
`predict-configurable.py`:

```bash
python predict-configurable.py "data/test/images/Experiment-1693.jpg"
python predict-configurable.py data/test/images
```

`predict-configurable.py` uses `checkpoints/final.pt` and writes to
`outputs/test/` by default. Change the constants at the top of that file if you
need a different checkpoint, output folder, tile size, or overlap.

Calculate statistics and make mask reports with a statistics panel:

```bash
python statistics/get-statistics.py outputs/test \
  --output outputs/test/statistics/statistics.csv \
  --report-dir outputs/test/statistics
```

Create grouped statistical summaries and charts from that CSV. By default,
groups are inferred from magnifications in the image names, such as `10X`:

```bash
python statistics/analyze-statistics.py outputs/test/statistics/statistics.csv
```

This writes a Markdown report, source and grouped-summary CSV files, boxplots,
and mean comparison charts in `statistics/analysis/`.

Run tests:

```bash
python -m unittest discover -s tests
```

The legacy `example-microg-processed/` folder is not read by validation or
training.
