"""
train.py
--------
Fine-tunes ONE Vision Transformer backbone (ViT-B/16, DeiT-Base, or Swin-Tiny)
on the ECG dataset. Run this three times, once per model (Section 3.4 of the
proposal). Saves the best checkpoint (by validation macro F1) plus the
validation softmax probabilities, which compute_ensemble.py later needs for
validation-based weight computation (Eq. 3.1) and weighted soft voting.

Usage:
    python train.py --model vit_base_patch16_224 --data_dir dataset_split --epochs 15
    python train.py --model deit_base_patch16_224 --data_dir dataset_split --epochs 15
    python train.py --model swin_tiny_patch4_window7_224 --data_dir dataset_split --epochs 15

Requires:
    pip install -r requirements.txt   (torch, timm, albumentations, scikit-learn ...)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import timm

from ecg_dataset import ECGDataset, get_train_transforms, get_eval_transforms, CLASS_NAMES

MODEL_CHOICES = [
    "vit_base_patch16_224",
    "deit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
]


def build_model(model_name, num_classes=4):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model


def run_epoch(model, loader, device, optimizer=None):
    """One pass over `loader`. If optimizer is given, trains; else, evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    torch.set_grad_enabled(is_train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)

        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs)

    torch.set_grad_enabled(True)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, macro_f1, all_probs, all_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODEL_CHOICES)
    ap.add_argument("--data_dir", default="dataset_split")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ECGDataset(Path(args.data_dir) / "train", transform=get_train_transforms())
    val_ds = ECGDataset(Path(args.data_dir) / "val", transform=get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_val_probs, best_val_labels = None, None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc, val_f1, val_probs, val_labels = run_epoch(model, val_loader, device, None)
        scheduler.step()

        print(f"[{args.model}] Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_val_probs, best_val_labels = val_probs, val_labels
            torch.save(model.state_dict(), out_dir / "best.pt")
            with open(out_dir / "val_summary.json", "w") as f:
                json.dump({"model": args.model, "best_val_macro_f1": best_f1,
                           "best_epoch": epoch}, f, indent=2)
            print(f"  -> new best checkpoint saved (val_f1={best_f1:.4f})")

    # Save validation softmax probabilities for the ensemble weighting step (Eq. 3.1)
    np.savez(out_dir / "val_probs.npz", probs=best_val_probs, labels=best_val_labels)
    print(f"\nDone. Best val macro F1 for {args.model}: {best_f1:.4f}")
    print(f"Checkpoint + val probs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()