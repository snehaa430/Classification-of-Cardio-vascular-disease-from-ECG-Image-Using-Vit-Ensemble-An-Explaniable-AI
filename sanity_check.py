"""
sanity_check.py
----------------
Run this ONCE, right after split_dataset.py and before train.py.

It uses the exact ECGDataset / get_train_transforms / get_eval_transforms /
DataLoader pattern from the pipeline to:
  1. Print dataset sizes per split (confirms split_dataset.py worked)
  2. Print a batch's tensor shape/dtype/value range (confirms preprocessing works)
  3. Save a grid image comparing: raw image -> CLAHE+resize (val/test) ->
     several augmented train versions of the SAME image, so you can visually
     confirm the augmentation is realistic and not destroying the waveform.

Usage:
    python sanity_check.py --data_dir dataset_split --out sanity_check.png
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ecg_dataset import (ECGDataset, get_train_transforms, get_eval_transforms,
                          apply_clahe, IMG_SIZE, CLASS_NAMES)


def denormalize(tensor_img):
    """Undo ImageNet normalization for display purposes."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor_img.permute(1, 2, 0).numpy()
    img = (img * std) + mean
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset_split")
    ap.add_argument("--out", default="sanity_check.png")
    args = ap.parse_args()

    # ---- 1. dataset sizes ----
    train_ds = ECGDataset(f"{args.data_dir}/train", transform=get_train_transforms())
    val_ds = ECGDataset(f"{args.data_dir}/val", transform=get_eval_transforms())
    test_ds = ECGDataset(f"{args.data_dir}/test", transform=get_eval_transforms())

    print("Dataset sizes:")
    print(f"  train: {len(train_ds)}")
    print(f"  val  : {len(val_ds)}")
    print(f"  test : {len(test_ds)}")

    # class balance check
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        counts = {c: 0 for c in CLASS_NAMES}
        for _, label in ds.samples:
            counts[CLASS_NAMES[label]] += 1
        print(f"  {name} class counts: {counts}")

    # ---- 2. batch shape sanity check ----
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {tuple(images.shape)}  dtype: {images.dtype}")
    print(f"Pixel value range (normalized): [{images.min():.3f}, {images.max():.3f}]")
    print(f"Labels in this batch: {[CLASS_NAMES[l] for l in labels.tolist()]}")

    # ---- 3. visual grid: raw -> CLAHE -> 4 augmented versions ----
    raw_path, label = train_ds.samples[0]
    raw_bgr = cv2.imread(str(raw_path))
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    resized_raw = cv2.resize(raw_rgb, (IMG_SIZE, IMG_SIZE))
    clahe_img = cv2.resize(apply_clahe(raw_rgb), (IMG_SIZE, IMG_SIZE))

    aug_transform = get_train_transforms()
    n_aug = 4
    fig, axes = plt.subplots(1, 2 + n_aug, figsize=(4 * (2 + n_aug), 4))

    axes[0].imshow(resized_raw)
    axes[0].set_title("Raw (resized only)")
    axes[1].imshow(clahe_img)
    axes[1].set_title("+ CLAHE (preprocessing)")

    for i in range(n_aug):
        augmented = aug_transform(image=raw_rgb)["image"]
        axes[2 + i].imshow(denormalize(augmented))
        axes[2 + i].set_title(f"Augmented sample {i+1}")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Class: {CLASS_NAMES[label]}  |  File: {raw_path.name}")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nSaved visual sanity check to: {args.out}")


if __name__ == "__main__":
    main()