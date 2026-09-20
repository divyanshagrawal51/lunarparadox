import argparse, glob, re
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(y, (p > t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--test_meta", required=True)
    ap.add_argument("--out", default="submission_stacked.csv")
    a = ap.parse_args()

    tags = a.tags or sorted(re.match(r"oof_(.+)\.npy", p).group(1) for p in glob.glob("oof_*.npy"))
    tags = [t for t in tags if t != "cnnxgb"]
    print("using:", tags)

    oof_mat = np.stack([np.load(f"oof_{t}.npy") for t in tags], axis=1)
    test_mat = np.stack([np.load(f"testprob_{t}.npy") for t in tags], axis=1)
    y = np.load(f"ylabels_{tags[0]}.npy")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    stack_oof = np.zeros(len(y))
    for itr, iva in skf.split(oof_mat, y):
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(oof_mat[itr], y[itr])
        stack_oof[iva] = lr.predict_proba(oof_mat[iva])[:, 1]

    thr_stack, sc_stack = best_threshold(y, stack_oof)
    print(f"\nstacked OOF bal-acc = {sc_stack:.4f} (thr={thr_stack:.3f})")

    simple_avg_oof = oof_mat.mean(axis=1)
    thr_avg, sc_avg = best_threshold(y, simple_avg_oof)
    print(f"simple average OOF bal-acc = {sc_avg:.4f} (thr={thr_avg:.3f})")

    final_lr = LogisticRegression(C=1.0, max_iter=1000)
    final_lr.fit(oof_mat, y)

    if sc_stack > sc_avg:
        print("stacking wins, using it")
        method, model, threshold = "stacking", final_lr, thr_stack
        test_pred = final_lr.predict_proba(test_mat)[:, 1]
    else:
        print("simple average was as good or better, using that")
        method, model, threshold = "average", None, thr_avg
        test_pred = test_mat.mean(axis=1)

    joblib.dump({"method": method, "model": model, "tags": tags, "threshold": threshold},
               "stack_model.joblib")
    print(f"saved stack_model.joblib (method={method}, threshold={threshold:.3f})")

    te = pd.read_csv(a.test_meta)
    id_c = [c for c in te.columns if c.lower() in ("image_id", "id", "filename", "image", "file_name")][0]
    sub = pd.DataFrame({"image_id": te[id_c].astype(str).values,
                        "label": (test_pred > threshold).astype(int)})
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(sub)} rows, predicted balance={np.bincount(sub.label)})")


if __name__ == "__main__":
    main()