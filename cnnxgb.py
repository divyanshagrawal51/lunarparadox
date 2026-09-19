import argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier
from tqdm import tqdm

from train_predict import LunarDS, find_col, make_model, DEV


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(y, (p > t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    feats = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        f = model(x.to(DEV))
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True); ap.add_argument("--train_meta", required=True)
    ap.add_argument("--test_dir", required=True);  ap.add_argument("--test_meta", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--out", default="submission_cnn_xgb.csv")
    a = ap.parse_args()

    tr = pd.read_csv(a.train_meta); te = pd.read_csv(a.test_meta)
    id_c = find_col(tr, ["image_id", "id", "filename", "image", "file_name"])
    az_c = find_col(tr, ["sun_azimuth_angle", "sun_azimuth", "azimuth", "solar_azimuth_angle"])
    lb_c = find_col(tr, ["label", "class", "target", "y"])
    te_id = find_col(te, ["image_id", "id", "filename", "image", "file_name"])
    te_az = find_col(te, ["sun_azimuth_angle", "sun_azimuth", "azimuth", "solar_azimuth_angle"])

    y_all = tr[lb_c].astype(int).values
    n_splits = max(2, a.folds)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    test_loader = DataLoader(LunarDS(te, a.test_dir, te_id, te_az), batch_size=a.bs,
                             shuffle=False, num_workers=2, pin_memory=True)

    feat_dim = 512
    oof_feats = np.zeros((len(tr), feat_dim), dtype=np.float32)
    test_feats_sum = np.zeros((len(te), feat_dim), dtype=np.float32)

    for f, (itr, iva) in enumerate(skf.split(tr, y_all)):
        dl_tr = DataLoader(LunarDS(tr.iloc[itr], a.train_dir, id_c, az_c, lb_c, train=True),
                           batch_size=a.bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        dl_va = DataLoader(LunarDS(tr.iloc[iva], a.train_dir, id_c, az_c, lb_c),
                           batch_size=a.bs, shuffle=False, num_workers=2, pin_memory=True)

        model = make_model(a.backbone)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl_tr))
        scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))
        crit = nn.BCEWithLogitsLoss()

        for ep in range(a.epochs):
            model.train()
            pbar = tqdm(dl_tr, desc=f"fold{f} ep{ep+1}/{a.epochs}", leave=False)
            for x, y in pbar:
                x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
                    loss = crit(model(x).squeeze(1), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.fc = nn.Identity()
        oof_feats[iva] = extract_features(model, dl_va)
        test_feats_sum += extract_features(model, test_loader)
        del model; torch.cuda.empty_cache()
        print(f"fold{f} done")

    test_feats = test_feats_sum / n_splits

    print("\ntraining XGBoost on CNN embeddings...")
    xgb_oof = np.zeros(len(tr))
    xgb_test = np.zeros(len(te))
    for f, (itr, iva) in enumerate(skf.split(oof_feats, y_all)):
        spw = (y_all[itr] == 0).sum() / (y_all[itr] == 1).sum()
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                            eval_metric="logloss", n_jobs=-1, random_state=0)
        clf.fit(oof_feats[itr], y_all[itr])
        xgb_oof[iva] = clf.predict_proba(oof_feats[iva])[:, 1]
        xgb_test += clf.predict_proba(test_feats)[:, 1] / n_splits

    thr, sc = best_threshold(y_all, xgb_oof)
    print(f"\nCNN-features + XGBoost OOF balanced accuracy = {sc:.4f} (thr={thr:.3f})")

    sub = pd.DataFrame({"image_id": te[te_id].astype(str).values,
                        "label": (xgb_test > thr).astype(int)})
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(sub)} rows, predicted balance={np.bincount(sub.label)})")

    np.save("oof_cnnxgb.npy", xgb_oof)
    np.save("testprob_cnnxgb.npy", xgb_test)
    np.save("ylabels_cnnxgb.npy", y_all)


if __name__ == "__main__":
    main()