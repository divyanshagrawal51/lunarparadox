import argparse, glob
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader

from train import LunarDS, find_col, make_model, predict


def load_tag_predictions(tag, loader):
    ckpts = sorted(glob.glob(f"model_{tag}_fold*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints found matching model_{tag}_fold*.pt")
    test_p = None
    for path in ckpts:
        ckpt = torch.load(path, map_location="cpu")
        model = make_model(ckpt["backbone"])
        model.load_state_dict(ckpt["state_dict"])
        p, _ = predict(model, loader, has_y=False)
        test_p = p if test_p is None else test_p + p
        print(f"  {path}  (fold val_bal_acc={ckpt.get('val_bal_acc', 'n/a')})")
    return test_p / len(ckpts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--test_meta", required=True)
    ap.add_argument("--tag", default=None, help="single model tag, e.g. resnet18_final")
    ap.add_argument("--stack", action="store_true",
                    help="use stack_model.joblib to combine multiple tags (reproduces the final submission)")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()

    te = pd.read_csv(a.test_meta)
    id_c = find_col(te, ["image_id", "id", "filename", "image", "file_name"])
    az_c = find_col(te, ["sun_azimuth_angle", "sun_azimuth", "azimuth", "solar_azimuth_angle"])
    loader = DataLoader(LunarDS(te, a.test_dir, id_c, az_c), batch_size=a.bs,
                        shuffle=False, num_workers=2, pin_memory=True)

    if a.stack:
        bundle = joblib.load("stack_model.joblib")
        tags, method, thr = bundle["tags"], bundle["method"], bundle["threshold"]
        print(f"combining tags: {tags} (method={method})")
        test_mat = np.stack([load_tag_predictions(t, loader) for t in tags], axis=1)
        if method == "stacking":
            test_p = bundle["model"].predict_proba(test_mat)[:, 1]
        else:
            test_p = test_mat.mean(axis=1)
    else:
        if not a.tag:
            raise ValueError("pass --tag <name> for a single model, or --stack to use stack_model.joblib")
        print(f"single model tag: {a.tag}")
        test_p = load_tag_predictions(a.tag, loader)
        try:
            with open(f"threshold_{a.tag}.txt") as fh:
                thr = float(fh.read().strip())
        except FileNotFoundError:
            thr = 0.5
            print("no saved threshold found, defaulting to 0.5")

    print(f"using threshold {thr:.3f}")
    sub = pd.DataFrame({"image_id": te[id_c].astype(str).values,
                        "label": (test_p > thr).astype(int)})
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(sub)} rows, predicted balance={np.bincount(sub.label)})")


if __name__ == "__main__":
    main()