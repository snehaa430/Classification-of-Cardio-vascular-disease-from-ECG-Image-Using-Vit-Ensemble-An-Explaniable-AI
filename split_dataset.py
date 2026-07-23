"""
split_dataset.py
-----------------
Splits the raw ECG image folders into a stratified 80/10/10
train/val/test ImageFolder-style structure (Table 3.1 in the proposal).

Usage:
    python split_dataset.py --src ecg_data --dst dataset_split --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

# Maps your raw folder names -> clean class names used everywhere downstream
CLASS_MAP = {
    "normal_ecg_images": "Normal",
    "myocardial_infarction_ecg_images": "MI",
    "abnormal_heartbeat_ecg_images": "Abnormal_Heartbeat",
    "post_mi_history_ecg_images": "History_of_MI",
}

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def split_list(files, seed):
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to raw ecg_data folder")
    ap.add_argument("--dst", required=True, help="Output path for split dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true",
                     help="Copy files instead of symlinking (default: copy)")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    summary = {}

    for raw_name, clean_name in CLASS_MAP.items():
        class_dir = src_root / raw_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} not found, skipping")
            continue

        files = sorted([f for f in class_dir.iterdir()
                         if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        train, val, test = split_list(files, args.seed)
        summary[clean_name] = (len(train), len(val), len(test))

        for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
            out_dir = dst_root / split_name / clean_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                dest = out_dir / f.name
                shutil.copy2(f, dest)

    print("\nSplit summary (train / val / test):")
    total = [0, 0, 0]
    for cls, counts in summary.items():
        print(f"  {cls:20s}: {counts[0]:4d} / {counts[1]:4d} / {counts[2]:4d}")
        for i in range(3):
            total[i] += counts[i]
    print(f"  {'TOTAL':20s}: {total[0]:4d} / {total[1]:4d} / {total[2]:4d}")
    print(f"\nDone. Output written to: {dst_root.resolve()}")


if __name__ == "__main__":
    main()