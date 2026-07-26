# Fiberglass micrograph segmentation

`controller.py` is the simple way to run this project:

```bash
python controller.py
```

It prints every command and the folders to use. First run
`python controller.py setup`; then use `python controller.py all` for the
complete workflow.

## Folders

- `training/images/`: original pictures used to train the model.
- `training/masks/`: same-name painted PNG masks. Fiber is red, resin is green,
  pore is blue, and unidentified pixels are black.
- `training/validated_masks/`: cleaned masks created by validation.
- `predicting/images/`: pictures to segment with a trained model.
- `predicting/outputs/`: generated masks, overlays, statistics, and charts.
- `checkpoints/`: trained model files shared by training and prediction.
- `common/`: shared Python code; do not run these files directly.

## Controller commands

```bash
python controller.py setup
python controller.py validate
python controller.py validate --with-overlays
python controller.py train
python controller.py train --with-overlays
python controller.py predict
python controller.py predict path/to/image-or-folder
python controller.py statistics
python controller.py analysis
python controller.py all
python controller.py all --with-overlays
python controller.py test
```

`all` validates the training data, trains a fresh model, predicts every image
in `predicting/images/`, then makes statistics and analysis reports. It stops
when a step fails. Add `--with-overlays` when your training masks include
corrected overlays.

## Advanced direct use

Use the configurable prediction script for its easy fixed defaults:

```bash
python -m predicting.predict_configurable predicting/images
```

Use `predict.py` when you want to choose the settings yourself:

```bash
python -m predicting.predict path/to/image.jpg \
  --checkpoint checkpoints/final.pt \
  --output-dir predicting/outputs \
  --tile-size 512 \
  --overlap 128
```

For advanced options on any script, use `python -m MODULE --help`, for example
`python -m training.train --help`.
