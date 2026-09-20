"""
One-command reproduction: downloads model weights from Google Drive,
then runs inference to produce submission.csv.

Usage:
  python setup_and_infer.py --test_dir ./test_images --test_meta test_metadata.csv
"""
import argparse, subprocess, sys, os

DRIVE_FOLDER_ID = "1i73eBbDFNr2qUK7OgwL7C6PSqsWpyb1h"  # from the shareable link


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--test_meta", required=True)
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()

    if not os.path.exists("stack_model.joblib"):
        print("downloading model weights from Google Drive...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        subprocess.run(["gdown", "--folder", DRIVE_FOLDER_ID, "-O", "."], check=True)
    else:
        print("weights already present, skipping download")

    subprocess.run([sys.executable, "inference.py",
                    "--test_dir", a.test_dir, "--test_meta", a.test_meta,
                    "--stack", "--out", a.out], check=True)


if __name__ == "__main__":
    main()
