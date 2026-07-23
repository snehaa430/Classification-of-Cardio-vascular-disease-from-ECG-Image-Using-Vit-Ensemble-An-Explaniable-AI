"""
ecg_dataset.py
--------------
PyTorch Dataset for the ECG image classification task.

Implements exactly the pipeline described in Section 3.2/3.3 of the proposal:
  Preprocessing : Resize (224x224, bilinear) -> CLAHE -> ImageNet normalize
  Augmentation  : rotation, translation, zoom, brightness, shear (TRAIN ONLY,
                  applied on-the-fly every epoch, never written to disk)

Requires:
    pip install albumentations opencv-python-headless torch torchvision
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = ["Normal", "MI", "Abnormal_Heartbeat", "History_of_MI"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def apply_clahe(image_rgb):
    """
    Contrast Limited Adaptive Histogram Equalization.
    Applied on the L channel in LAB space so waveform lines gain contrast
    against the ECG grid without distorting color balance.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def get_train_transforms():
    """Preprocessing + augmentation for the TRAINING split only."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
        # --- augmentation (kept mild so ECG diagnostic morphology survives) ---
        A.Affine(
            rotate=(-10, 10),          # rotation
            translate_percent=(0.0, 0.05),  # translation
            scale=(0.95, 1.05),        # zoom
            shear=(-5, 5),             # shear
            p=0.8,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms():
    """Deterministic preprocessing for VAL/TEST splits — no augmentation."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class ECGDataset(Dataset):
    """
    Expects an ImageFolder-style directory:
        root/Normal/*.jpg
        root/MI/*.jpg
        root/Abnormal_Heartbeat/*.jpg
        root/History_of_MI/*.jpg
    """

    def __init__(self, root, transform=None, use_clahe=True):
        self.root = Path(root)
        self.transform = transform
        self.use_clahe = use_clahe
        self.samples = []

        for cls_name, cls_idx in CLASS_TO_IDX.items():
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for f in cls_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((f, cls_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root}. "
                                f"Expected subfolders: {list(CLASS_TO_IDX.keys())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image_bgr = cv2.imread(str(path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.use_clahe:
            image_rgb = apply_clahe(image_rgb)

        if self.transform:
            image = self.transform(image=image_rgb)["image"]
        else:
            image = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        return image, label


if __name__ == "__main__":
    # quick sanity check
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset_split/train"
    ds = ECGDataset(root, transform=get_train_transforms())
    print(f"Loaded {len(ds)} images from {root}")
    img, label = ds[0]
    print("Sample tensor shape:", img.shape, "label:", label, CLASS_NAMES[label])