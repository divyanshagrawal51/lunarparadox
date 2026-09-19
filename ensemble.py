import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(y, (p > t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--test_meta", required=True)
    ap.add_argument("--out", default="submission_ensemble.csv")
    a = ap.parse_args()

    oofs = [np.load(f"oof_{t}.npy") for t in a.tags]
    tests = [np.load(f"testprob_{t}.npy") for t in a.tags]
    y = np.load(f"ylabels_{a.tags[0]}.npy")

    for t, o in zip(a.tags, oofs):
        thr, sc = best_threshold(y, o)
        print(f"{t:12s} solo OOF bal-acc = {sc:.4f} (thr={thr:.3f})")

    oof_avg = np.mean(oofs, axis=0)
    test_avg = np.mean(tests, axis=0)
    thr, sc = best_threshold(y, oof_avg)
    print(f"\nensemble OOF bal-acc = {sc:.4f} (thr={thr:.3f})")

    te = pd.read_csv(a.test_meta)
    id_c = [c for c in te.columns if c.lower() in ("image_id", "id", "filename", "image", "file_name")][0]
    sub = pd.DataFrame({"image_id": te[id_c].astype(str).values,
                        "label": (test_avg > thr).astype(int)})
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(sub)} rows, predicted balance={np.bincount(sub.label)})")


if __name__ == "__main__":
    main()