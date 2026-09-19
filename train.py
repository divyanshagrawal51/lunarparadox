import argparse, os, numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

CROP, SIZE = 180, 224
ROT_SIGN = -1   # keep in sync with physics_check.py
DEV = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_col(df, cands):
    low = {c.strip().lower(): c for c in df.columns}
    for c in cands:
        if c in low:
            return low[c]
    raise KeyError(f"None of {cands} in {list(df.columns)}")


def rotate_no_black_corners(im, angle):
    # reflect-pad before rotating so corners don't go black, keeps full res
    arr = np.array(im)
    w, h = arr.shape[1], arr.shape[0]
    pad = int(0.42 * max(w, h))
    arr = np.pad(arr, pad, mode="reflect")
    im2 = Image.fromarray(arr).rotate(angle, resample=Image.BILINEAR)
    w2, h2 = im2.size
    l, t = (w2 - w) // 2, (h2 - h) // 2
    return im2.crop((l, t, l + w, t + h))


class LunarDS(Dataset):
    def __init__(self, df, root, id_c, az_c, lb_c=None, train=False, aug_rot=8):
        self.df, self.root = df.reset_index(drop=True), root
        self.id_c, self.az_c, self.lb_c, self.train = id_c, az_c, lb_c, train
        self.aug_rot = aug_rot

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(os.path.join(self.root, str(r[self.id_c]))).convert("L")
        im = rotate_no_black_corners(im, ROT_SIGN * float(r[self.az_c]))

        if self.train:
            # never horizontal-flip: it mirrors the sun axis and swaps crater<->mound
            if self.aug_rot > 0:
                ang = np.random.uniform(-self.aug_rot, self.aug_rot)
                im = rotate_no_black_corners(im, ang)
            c = 230
        else:
            c = 256

        w, h = im.size
        l, t = (w - c) // 2, (h - c) // 2
        if self.train:
            l += np.random.randint(-8, 9); t += np.random.randint(-8, 9)
            l, t = max(0, min(w - c, l)), max(0, min(h - c, t))
        im = im.crop((l, t, l + c, t + c)).resize((SIZE, SIZE), Image.BILINEAR)

        a = np.asarray(im, dtype=np.float32) / 255.0
        if self.train:
            if np.random.rand() < 0.5:
                a = a[::-1].copy()
            a = np.clip(a * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.04, 0.04), 0, 1)

        rgb = np.stack([a, a, a], axis=0)
        rgb = (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        x = torch.from_numpy(rgb)
        if self.lb_c is None:
            return x
        return x, torch.tensor(float(r[self.lb_c]))


def make_model(backbone="resnet18"):
    if backbone == "resnet34":
        m = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    else:
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m.to(DEV)


@torch.no_grad()
def predict(model, loader, has_y=False, tta=True):
    model.eval()
    ps, ys = [], []
    for batch in loader:
        x = batch[0] if has_y else batch
        x = x.to(DEV)
        logit = model(x).squeeze(1)
        if tta:
            logit = (logit + model(torch.flip(x, dims=[2])).squeeze(1)) / 2
        ps.append(torch.sigmoid(logit).float().cpu().numpy())
        if has_y:
            ys.append(batch[1].numpy())
    return np.concatenate(ps), (np.concatenate(ys) if has_y else None)


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(y, (p > t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True); ap.add_argument("--train_meta", required=True)
    ap.add_argument("--test_dir", required=True);  ap.add_argument("--test_meta", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aug_rot", type=float, default=8)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--freeze_epochs", type=int, default=2)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    tr = pd.read_csv(a.train_meta); te = pd.read_csv(a.test_meta)
    id_c = find_col(tr, ["image_id", "id", "filename", "image", "file_name"])
    az_c = find_col(tr, ["sun_azimuth_angle", "sun_azimuth", "azimuth", "solar_azimuth_angle"])
    lb_c = find_col(tr, ["label", "class", "target", "y"])
    te_id = find_col(te, ["image_id", "id", "filename", "image", "file_name"])
    te_az = find_col(te, ["sun_azimuth_angle", "sun_azimuth", "azimuth", "solar_azimuth_angle"])

    y_all = tr[lb_c].astype(int).values
    print(f"train={len(tr)}  test={len(te)}  class balance={np.bincount(y_all)}  device={DEV}")

    test_loader = DataLoader(LunarDS(te, a.test_dir, te_id, te_az), batch_size=a.bs,
                             shuffle=False, num_workers=4, pin_memory=True)

    n_splits = max(2, a.folds)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(tr)); test_p = np.zeros(len(te)); used = 0

    for f, (itr, iva) in enumerate(skf.split(tr, y_all)):
        if a.folds == 1 and f > 0:
            break
        dl_tr = DataLoader(LunarDS(tr.iloc[itr], a.train_dir, id_c, az_c, lb_c, train=True, aug_rot=a.aug_rot),
                           batch_size=a.bs, shuffle=True, num_workers=4,
                           pin_memory=True, drop_last=True)
        dl_va = DataLoader(LunarDS(tr.iloc[iva], a.train_dir, id_c, az_c, lb_c),
                           batch_size=a.bs, shuffle=False, num_workers=4, pin_memory=True)

        model = make_model(a.backbone)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                    total_steps=a.epochs * len(dl_tr))
        scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))

        n_pos = float(y_all[itr].sum()); n_neg = float(len(itr) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEV)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_score, best_state = -1.0, None
        for p in model.parameters():
            p.requires_grad = (a.freeze_epochs == 0)
        for p in model.fc.parameters():
            p.requires_grad = True

        for ep in range(a.epochs):
            if ep == a.freeze_epochs:
                for p in model.parameters():
                    p.requires_grad = True
                print(f"  fold{f}: unfreezing backbone at epoch {ep+1}")

            model.train(); tot = 0.0
            pbar = tqdm(dl_tr, desc=f"fold{f} ep{ep+1}/{a.epochs}", leave=False)
            for x, y in pbar:
                x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
                    loss = crit(model(x).squeeze(1), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item() * len(y)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            pv, yv = predict(model, dl_va, has_y=True, tta=False)
            vscore = balanced_accuracy_score(yv, (pv > 0.5).astype(int))
            print(f"  fold{f} ep{ep+1}/{a.epochs} loss={tot/len(itr):.4f} "
                  f"val_bal_acc@0.5={vscore:.4f}"
                  f"{'  <- best so far' if vscore > best_score else ''}")
            if vscore > best_score:
                best_score = vscore
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        if a.tag:
            ckpt_path = f"model_{a.tag}_fold{f}.pt"
            torch.save({"state_dict": best_state, "backbone": a.backbone,
                       "fold": f, "val_bal_acc": best_score}, ckpt_path)
            print(f"  saved {ckpt_path}")

        pv, yv = predict(model, dl_va, has_y=True)
        print(f"  fold{f} best val_bal_acc (TTA)={balanced_accuracy_score(yv, (pv>0.5).astype(int)):.4f}")

        oof[iva] = pv
        test_p += predict(model, test_loader)[0]; used += 1
        del model; torch.cuda.empty_cache()

    test_p /= used
    if a.folds == 1:
        mask = oof > 0
        thr, sc = best_threshold(y_all[mask], oof[mask])
    else:
        thr, sc = best_threshold(y_all, oof)
    print(f"\nOOF balanced accuracy = {sc:.4f} at threshold {thr:.3f}")

    sub = pd.DataFrame({"image_id": te[te_id].astype(str).values,
                        "label": (test_p > thr).astype(int)})
    assert len(sub) == len(te), "row count mismatch"
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(sub)} rows, predicted balance={np.bincount(sub.label)})")

    if a.tag:
        np.save(f"oof_{a.tag}.npy", oof)
        np.save(f"testprob_{a.tag}.npy", test_p)
        np.save(f"ylabels_{a.tag}.npy", y_all)
        with open(f"threshold_{a.tag}.txt", "w") as fh:
            fh.write(str(thr))
        print(f"saved oof_{a.tag}.npy, testprob_{a.tag}.npy, ylabels_{a.tag}.npy, threshold_{a.tag}.txt")


if __name__ == "__main__":
    main()
