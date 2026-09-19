import argparse, glob, re
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
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--test_meta", required=True)
    ap.add_argument("--pick", default=None)
    ap.add_argument("--out", default="submission_FINAL.csv")
    a = ap.parse_args()

    tags = a.tags or sorted(re.match(r"oof_(.+)\.npy", p).group(1) for p in glob.glob("oof_*.npy"))
    if not tags:
        print("no saved oof_*.npy files found"); return

    results = {}
    for t in tags:
        try:
            oof = np.load(f"oof_{t}.npy"); y = np.load(f"ylabels_{t}.npy")
        except FileNotFoundError:
            print(f"  skipping {t}: files not found"); continue
        thr, sc = best_threshold(y, oof)
        results[t] = (thr, sc)
        print(f"{t:20s} OOF bal-acc = {sc:.4f}  (thr={thr:.3f})")

    if not results:
        print("nothing to compare"); return

    best_tag = max(results, key=lambda t: results[t][1])
    print(f"\nbest single run: {best_tag} ({results[best_tag][1]:.4f})")

    pick = a.pick or best_tag
    te = pd.read_csv(a.test_meta)
    id_c = [c for c in te.columns if c.lower() in ("image_id", "id", "filename", "image", "file_name")][0]

    if pick == "ensemble":
        test_avg = np.mean([np.load(f"testprob_{t}.npy") for t in results], axis=0)
        oof_avg = np.mean([np.load(f"oof_{t}.npy") for t in results], axis=0)
        y = np.load(f"ylabels_{tags[0]}.npy")
        thr, sc = best_threshold(y, oof_avg)
        print(f"ensemble of {list(results)} -> OOF bal-acc = {sc:.4f} (thr={thr:.3f})")
        test_p, use_thr = test_avg, thr
    else:
        test_p = np.load(f"testprob_{pick}.npy")
        use_thr = results[pick][0]

    sub = pd.DataFrame({"image_id": te[id_c].astype(str).values,
                        "label": (test_p > use_thr).astype(int)})
    sub.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} using '{pick}'  ({len(sub)} rows, "
          f"predicted balance={np.bincount(sub.label)})")


if __name__ == "__main__":
    main()