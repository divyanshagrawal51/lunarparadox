import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score

CROP = 180      # 256/sqrt(2) ~ 181, keeps us inside the rotation bounds
ROT_SIGN = -1   # flip to +1 if results come out inverted


def find_col(df, candidates):
    low = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    raise KeyError(f"None of {candidates} found in {list(df.columns)}")


def normalized(path, az):
    im = Image.open(path).convert("L")
    im = im.rotate(ROT_SIGN * float(az), resample=Image.BILINEAR)
    w, h = im.size
    l, t = (w - CROP) // 2, (h - CROP) // 2
    return im.crop((l, t, l + CROP, t + CROP))


def asymmetry_score(im):
    a = np.asarray(im, dtype=np.float32)
    a -= a.mean()
    a /= (a.std() + 1e-6)

    x = np.linspace(-1, 1, a.shape[1], dtype=np.float32)[None, :]
    y = np.linspace(-1, 1, a.shape[0], dtype=np.float32)[:, None]
    mask = ((x ** 2 + y ** 2) < 1.0).astype(np.float32)

    return float((a * x * mask).sum() / mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()

    df = pd.read_csv(args.meta)
    id_c = find_col(df, ["image_id", "id", "filename", "image", "file_name"])
    az_c = find_col(df, ["sun_azimuth_angle", "sun_azimuth", "azimuth", "solar_azimuth_angle"])
    lb_c = find_col(df, ["label", "class", "target", "y"])

    df = df.sample(min(args.n, len(df)), random_state=0)

    raw_scores, rot_scores, labels = [], [], []
    for _, r in df.iterrows():
        p = os.path.join(args.images, str(r[id_c]))
        try:
            rot_scores.append(asymmetry_score(normalized(p, r[az_c])))

            im0 = Image.open(p).convert("L")
            l = (im0.size[0] - CROP) // 2
            raw_scores.append(asymmetry_score(im0.crop((l, l, l + CROP, l + CROP))))

            labels.append(int(r[lb_c]))
        except FileNotFoundError:
            continue

    labels = np.array(labels)
    for name, scores in [("raw (no rotation)", np.array(raw_scores)),
                         ("rotation-normalized", np.array(rot_scores))]:
        pred = (scores > np.median(scores)).astype(int)
        b = balanced_accuracy_score(labels, pred)
        flag = "  -> inverted, flip ROT_SIGN" if b < 0.5 else ""
        print(f"{name:22s} bal-acc = {max(b, 1 - b):.4f}   (raw {b:.4f}{flag})")

    print(f"\nIf rotation-normalized clearly beats raw, ROT_SIGN={ROT_SIGN} is correct.")


if __name__ == "__main__":
    main()