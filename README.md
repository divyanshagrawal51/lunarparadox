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

**Final OOF balanced accuracy: 0.6564** (stacked ensemble of 5 ResNet18/34 runs)

| Stage | Balanced accuracy |
|---|---|
| Initial baseline | 0.604 |
| Best single model (ResNet18) | 0.651 |
| Simple ensemble (5 models, averaged) | 0.654 |
| Stacked ensemble (logistic regression meta-learner) | **0.656** |

## Files

- **`physics_check.py`** — ~30s sanity check that the rotation direction is
  correct (no training).
- **`inspect_data.py`** — visual + statistical check: azimuth units, label
  balance, file matching, sample image grid.
- **`train.py`** — main training script. ResNet18/34 (ImageNet pretrained),
  5-fold CV, reflect-pad rotation normalization, vertical-flip-only
  augmentation, threshold tuned on OOF predictions, TTA at inference. Saves
  per-fold checkpoints, OOF/test probabilities, and the tuned threshold.
- **`inference.py`** — loads saved checkpoints for a given `--tag` and runs
  prediction only, no training required.
- **`cnn_xgb.py`** — alternative approach: CNN embeddings + XGBoost.
  Underperformed the CNN's own head (0.563 vs 0.65+).
- **`ensemble.py`** — simple average of multiple saved runs.
- **`stack.py`** — logistic-regression meta-learner over saved runs; the
  final submission.
- **`finalize.py`** — compares all saved runs and builds a submission from
  the best one or an ensemble.

## Usage

Train (5-fold, saves checkpoints + OOF/test probabilities under a tag):
```bash
python train.py --train_dir ./train_images --train_meta train_metadata.csv \
    --test_dir ./test_images --test_meta test_metadata.csv \
    --folds 5 --epochs 12 --tag mymodel
```

Run inference only, from saved checkpoints:
```bash
python inference.py --test_dir ./test_images --test_meta test_metadata.csv \
    --tag mymodel --out submission.csv
```

Combine multiple runs:
```bash
python ensemble.py --tags mymodel othermodel --test_meta test_metadata.csv
python stack.py --test_meta test_metadata.csv
```

## Model weights

`model_resnet18_v3_fold{0-4}.pt` (5-fold ResNet18, 18 epochs) — download:
**[[LINK](https://drive.google.com/drive/folders/1i73eBbDFNr2qUK7OgwL7C6PSqsWpyb1h?usp=drive_link)]**

Note: the final submission is a stacked ensemble of 5 separate training
runs, but checkpoint-saving was only added partway through experimentation —
so only this one run's weights are available here. This run alone scores
0.6447 balanced accuracy (via `inference.py`); the full ensemble (0.6564)
requires re-running `train.py` for the other backbone/epoch variants.

## Requirements

```bash
pip install -r requirements.txt
```