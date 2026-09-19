import argparse, os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

CROP = 180
ROT_SIGN = -1


def find_col(df, cands):
    low = {c.strip().lower(): c for c in df.columns}
    for c in cands:
        if c in low:
            return low[c]
    raise KeyError(f"None of {cands} in {list(df.columns)}")


def crop_center(im, c):
    w, h = im.size
    l, t = (w - c) // 2, (h - c) // 2
    return im.crop((l, t, l + c, t + c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="sanity_grid.png")
    args = ap.parse_args()

    df = pd.read_csv(args.meta)
    id_c = find_col(df, ["image_id", "id", "filename", "image", "file_name"])
    az_c = find_col(df, ["sun_azimuth_angle", "sun_azimuth", "azimuth", "solar_azimuth_angle"])
    lb_c = find_col(df, ["label", "class", "target", "y"])

    print("columns:", list(df.columns))
    print(f"\nazimuth ('{az_c}') stats:")
    print(df[az_c].describe())
    print(f"dtype: {df[az_c].dtype}")
    if df[az_c].max() <= 6.5:
        print("max <= 6.5 -> looks like radians, not degrees. convert before rotating.")
    else:
        print("looks like degrees, good.")

    print(f"\nlabel balance:")
    print(df[lb_c].value_counts())

    missing = 0
    for _, r in df.head(20).iterrows():
        p = os.path.join(args.images, str(r[id_c]))
        if not os.path.exists(p):
            missing += 1
    print(f"\nfile check: {missing}/20 missing (0 = good, nonzero = filename mismatch)")

    rows = []
    for cls in df[lb_c].unique():
        sub = df[df[lb_c] == cls].sample(min(args.n, (df[lb_c] == cls).sum()), random_state=1)
        rows.append(sub)
    sample = pd.concat(rows).reset_index(drop=True)

    cell = 220
    cols = 4
    grid = Image.new("RGB", (cell * cols, cell * len(sample)), "white")
    draw = ImageDraw.Draw(grid)

    for i, r in sample.iterrows():
        p = os.path.join(args.images, str(r[id_c]))
        if not os.path.exists(p):
            continue
        raw = Image.open(p).convert("L")
        az = float(r[az_c])
        rot = raw.rotate(ROT_SIGN * az, resample=Image.BILINEAR)

        raw_c = crop_center(raw, CROP).resize((cell - 20, cell - 20)).convert("RGB")
        rot_c = crop_center(rot, CROP).resize((cell - 20, cell - 20)).convert("RGB")

        y = i * cell
        grid.paste(raw_c, (10, y + 10))
        grid.paste(rot_c, (cell + 10, y + 10))

        label_txt = f"id={r[id_c]}\nlabel={int(r[lb_c])}\naz={az:.1f}"
        draw.text((cell * 2 + 10, y + 10), label_txt, fill="black")

    grid.save(args.out)
    print(f"\nwrote {args.out}")
    print("col1 = raw, col2 = rotation-normalized, col3 = label/azimuth")
    print("check: do label=0 rows look like pits, label=1 like bumps?")
    print("does the shadow direction in col2 look consistent within a class?")


if __name__ == "__main__":
    main()