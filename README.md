# Lunar Surface Classification — Depth vs Rise

Binary classifier for 256x256 grayscale lunar surface images: distinguishing
craters/depressions (Class 0 — Depth) from mounds/rocks/boulders (Class 1 — Rise).

Since optical shadows on the lunar surface depend on sun direction, every
image is rotated by `-sun_azimuth_angle` before training/inference, so the
sun direction is consistent across the dataset and shadow-based features
aren't flipped between samples.

## Result

**Final OOF balanced accuracy: 0.6564** (stacked ensemble of 5 ResNet18/34 runs)

| Stage | Balanced accuracy |
|---|---|
| Initial baseline | 0.604 |
| Best single model (ResNet18) | 0.651 |
| Simple ensemble (5 models, averaged) | 0.654 |
| Stacked ensemble (logistic regression meta-learner) | **0.656** |

## Pipeline

1. **`physics_check.py`** — quick (~30s, no training) sanity check that the
   `-sun_azimuth_angle` rotation direction is correct, by measuring brightness
   asymmetry along the sun axis before/after rotation.
2. **`inspect_data.py`** — visual + statistical sanity check: azimuth column
   units, label balance, file-matching, and a sample image grid (raw vs
   rotation-normalized).
3. **`train_predict.py`** — main training script. ResNet18/34 (ImageNet
   pretrained), 5-fold CV, vertical-flip-only augmentation (horizontal flip
   would mirror the sun axis and swap crater↔mound), reflect-pad rotation to
   avoid black corners, threshold tuned on OOF predictions for balanced
   accuracy, TTA at inference.
4. **`cnn_xgb.py`** — alternative: extract CNN embeddings, train XGBoost on
   top. Underperformed the CNN's own head in this case (0.563).
5. **`ensemble.py`** — simple average of multiple saved runs.
6. **`stack.py`** — logistic-regression meta-learner over saved runs;
   slightly beat simple averaging.
7. **`finalize.py`** — compares all saved runs and builds the final
   submission from the best one (or an ensemble).

## Usage

```bash
# 1. sanity check the rotation direction
python physics_check.py --images ./train_images --meta train_metadata.csv

# 2. train (5-fold, saves OOF/test probabilities under a tag for ensembling)
python train_predict.py --train_dir ./train_images --train_meta train_metadata.csv \
    --test_dir ./test_images --test_meta test_metadata.csv \
    --folds 5 --epochs 12 --tag mymodel

# 3. (optional) train more variants with different --backbone / --epochs / --tag,
#    then combine them
python ensemble.py --tags mymodel othermodel --test_meta test_metadata.csv
python stack.py --test_meta test_metadata.csv
```

## Model weights

`model_resnet18_v3_fold{0-4}.pt` — the single ResNet18 run (5-fold, 18 epochs)
whose checkpoints were saved to disk. The final submission is a stacked
ensemble of 5 separate training runs, but weight-saving was only added
partway through — so only this run's weights are included here. It scores
0.6447 solo; the full ensemble (0.6564) can't be exactly reproduced from
these weights alone, only re-run from scratch via `train_predict.py`.

## Requirements

```bash
pip install -r requirements.txt
```
