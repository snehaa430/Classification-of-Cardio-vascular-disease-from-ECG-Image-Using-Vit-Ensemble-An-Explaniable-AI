"""
offline_augment.py
-------------------
Generates 12 augmented versions of EACH original image in the training
set, using: rotation, flipping, zooming, brightness adjustment, cropping.
Saves all outputs to disk as actual image files (unlike the on-the-fly
augmentation in ecg_dataset.py, which regenerates random augmentations
every epoch instead of saving them).

WARNING: Horizontal/vertical flips invert ECG waveform direction and lead
order, which doesn't correspond to any real physiological signal. It's
included here because it was explicitly requested, but you can disable
it by setting ENABLE_FLIP = False below if your validation results look
worse than expected.

Only run this on the TRAINING split. Never augment val/test — they must
stay untouched, real, unmodified images for a fair evaluation.

Usage:
    python offline_augment.py --src dataset_split/train --dst dataset_split_augmented/train --n_aug 12
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import albumentations as A

ENABLE_FLIP = True  # set False to skip flipping (recommended for ECG waveforms)

IMG_SIZE = 224


def build_single_augmentation():
    """One random augmentation combining all 5 requested operations."""
    ops = [
        A.Rotate(limit=15, p=0.8),
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), p=0.8),  # zoom + crop combined
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.15, p=0.8),
    ]
    if ENABLE_FLIP:
        ops.append(A.HorizontalFlip(p=0.5))

    return A.Compose([
        *ops,
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to dataset_split/train")
    ap.add_argument("--dst", required=True, help="Where to save augmented images")
    ap.add_argument("--n_aug", type=int, default=12, help="Augmented copies per original image")
    ap.add_argument("--include_original", action="store_true",
                     help="Also copy the original (unaugmented) image into the output folder")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    class_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class subfolders found in {src_root}")

    total_generated = 0

    for class_dir in class_dirs:
        out_class_dir = dst_root / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        images = [f for f in class_dir.iterdir()
                  if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

        for img_path in images:
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if args.include_original:
                orig_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
                out_path = out_class_dir / f"{img_path.stem}_orig.jpg"
                cv2.imwrite(str(out_path), cv2.cvtColor(orig_resized, cv2.COLOR_RGB2BGR))

            for i in range(args.n_aug):
                transform = build_single_augmentation()
                augmented = transform(image=img_rgb)["image"]
                out_path = out_class_dir / f"{img_path.stem}_aug{i+1}.jpg"
                cv2.imwrite(str(out_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                total_generated += 1

        print(f"  {class_dir.name}: {len(images)} originals -> "
              f"{len(images) * args.n_aug} augmented images")

    print(f"\nDone. Total augmented images generated: {total_generated}")
    print(f"Saved to: {dst_root.resolve()}")
    if ENABLE_FLIP:
        print("\nNote: flipping was ENABLED. If model performance looks off, "
              "set ENABLE_FLIP = False at the top of this script and rerun.")


if __name__ == "__main__":
    main()