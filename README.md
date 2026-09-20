# Lunar Surface Classification — Depth vs Rise
### The Pareidolia Paradox

Binary classifier for 256x256 grayscale lunar surface images: distinguishing
craters/depressions (Class 0 — Depth) from mounds/rocks/boulders (Class 1 — Rise).

## Methodology: handling `sun_azimuth_angle`

Optical shadows on the lunar surface flip depending on sun direction — the
same crater can look like a mound (and vice versa) if the shadow falls on
the "wrong" side. To normalize this, every image is rotated by
`-sun_azimuth_angle` degrees before training and inference, so the sun's
direction is consistent across the whole dataset. To avoid introducing black
corners from the rotation (which a CNN can latch onto as a spurious shortcut
correlated with the azimuth value), the image is reflect-padded before
rotating and cropped back to its original size afterward — so the full
native resolution is preserved with no artifacts.

During training, horizontal flips are never used as augmentation, since a
horizontal flip would mirror the sun axis and silently swap the crater/mound
appearance while keeping the label unchanged. Only vertical flips (which
preserve the left-right sun axis) are used.

We verified the rotation direction (`physics_check.py`) with a lightweight,
non-CNN sanity check before any training: measuring brightness asymmetry
along the sun axis, before vs. after rotation, on a sample of images.

## Result

**Final OOF balanced accuracy: 0.6524** (ensemble of ResNet18 + ResNet34, 5-fold each)

`submission.csv` is fully reproducible from the saved checkpoints via
`inference.py --stack` (verified: predictions match exactly).

## Files

- **`physics_check.py`** — ~30s sanity check that the rotation direction is
  correct (no training).
- **`inspect_data.py`** — visual + statistical check: azimuth units, label
  balance, file matching, sample image grid.
- **`train.py`** — main training script. ResNet18/34 (ImageNet pretrained),
  5-fold CV, reflect-pad rotation normalization, vertical-flip-only
  augmentation, threshold tuned on OOF predictions, TTA at inference. Saves
  per-fold checkpoints, OOF/test probabilities, and the tuned threshold.
- **`inference.py`** — loads saved checkpoints and runs prediction only, no
  training required. `--tag <name>` for a single model, `--stack` to combine
  multiple models via the saved ensemble strategy (reproduces `submission.csv`).
- **`cnn_xgb.py`** — alternative approach tried: CNN embeddings + XGBoost.
  Underperformed the CNN's own head (0.56 vs 0.65+), not used in the final result.
- **`ensemble.py`** — simple average of multiple saved runs.
- **`stack.py`** — combines multiple runs, choosing between a logistic-regression
  meta-learner and simple averaging (whichever scores higher on OOF), and
  saves the chosen strategy to `stack_model.joblib` for `inference.py` to reuse.
- **`finalize.py`** — compares all saved runs and builds a submission from
  the best single one, if needed.

## Usage

Train (5-fold, saves checkpoints + OOF/test probabilities under a tag):
```bash
python train.py --train_dir ./train_images --train_meta train_metadata.csv \
    --test_dir ./test_images --test_meta test_metadata.csv \
    --folds 5 --epochs 12 --tag mymodel
```

Combine multiple runs into a final ensemble:
```bash
python stack.py --tags model1 model2 --test_meta test_metadata.csv --out submission.csv
```

Reproduce `submission.csv` from saved checkpoints only (no training):
```bash
python inference.py --test_dir ./test_images --test_meta test_metadata.csv \
    --stack --out submission.csv
```

## Model weights

`model_resnet18_final_fold{0-4}.pt` and `model_resnet34_final_fold{0-4}.pt`
(5-fold each, 12 epochs) plus `stack_model.joblib` (the ensembling strategy) —
download: **[[LINK](https://drive.google.com/drive/folders/1i73eBbDFNr2qUK7OgwL7C6PSqsWpyb1h?usp=drive_link)]**


These are the exact weights used to produce `submission.csv`; running
`inference.py --stack` with these files reproduces it exactly.

## Hardware requirements

- GPU strongly recommended (trained on a single consumer NVIDIA GPU, ~25-30
  min per 5-fold training run at 12 epochs). CPU inference works but is slower.
- ~2GB disk for model checkpoints, ~1GB for the dataset.

## Requirements

```bash
pip install -r requirements.txt
```