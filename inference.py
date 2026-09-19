import argparse, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train import LunarDS, find_col, make_model, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--test_meta", required=True)
    ap.add_argument("--tag", required=True, help="tag used when training, e.g. resnet18_v3")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()

    te = pd.read_csv(a.test_meta)
    id_c = find_col(te, ["image_id", "id", "filename", "image", "file_name"])
    az_c = find_col(te, ["sun_azimuth_angle", "sun_azimuth", "azimuth", "solar_azimuth_angle"])

    loader = DataLoader(LunarDS(te, a.test_dir, id_c, az_c), batch_size=a.bs,
                        shuffle=False, num_workers=2, pin_memory=True)

    ckpts = sorted(glob.glob(f"model_{a.tag}_fold*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints found matching model_{a.tag}_fold*.pt")
    print(f"found {len(ckpts)} fold checkpoints for tag '{a.tag}'")

    test_p = np.zeros(len(te))
    for path in ckpts:
        ckpt = torch.load(path, map_location="cpu")
        model = make_model(ckpt["backbone"])
        model.load_state_dict(ckpt["state_dict"])
        p, _ = predict(model, loader, has_y=False)
        test_p += p
        print(f"  {path}  (fold val_bal_acc={ckpt.get('val_bal_acc', 'n/a')})")
    test_p /= len(ckpts)

    try:
        with open(f"threshold_{a.tag}.txt") as fh:
            thr = float(fh.read().strip())
        print(f"using saved threshold {thr:.3f}")
    except FileNotFoundError:
        thr = 0.5
        print("no saved threshold found, defaulting to 0.5")

    sub = pd.DataFrame({"image_id": te[id_c].astype(str).values,
                        "label": (test_p > thr).astype(int)})
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(sub)} rows, predicted balance={np.bincount(sub.label)})")


if __name__ == "__main__":
    main()
