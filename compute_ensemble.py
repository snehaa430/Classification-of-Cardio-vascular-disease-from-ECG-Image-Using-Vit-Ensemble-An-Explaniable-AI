"""
compute_ensemble.py
--------------------
Implements Section 3.4.4 (Validation-Based Weight Computation) and
Section 3.4.5 (Weighted Soft Voting), then evaluates every individual
model AND the ensemble on the held-out test set (Section 3.6 / Table 3.2),
matching Section 4.4's requirement to also report an equal-weight baseline
for comparison.
 
Run this AFTER train.py has produced checkpoints for all three models:
    checkpoints/vit_base_patch16_224/best.pt        + val_summary.json
    checkpoints/deit_base_patch16_224/best.pt       + val_summary.json
    checkpoints/swin_tiny_patch4_window7_224/best.pt + val_summary.json
 
Usage:
    python compute_ensemble.py --data_dir dataset_split --ckpt_dir checkpoints
"""
 
import argparse
import json
import time
from pathlib import Path
 
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import label_binarize
import timm
 
from ecg_dataset import ECGDataset, get_eval_transforms, CLASS_NAMES
 
MODELS = [
    "vit_base_patch16_224",
    "deit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
]
 
 
def load_model(model_name, ckpt_path, device, num_classes=4):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model
 
 
@torch.no_grad()
def get_test_probs(model, loader, device):
    all_probs, all_labels = [], []
    start = time.time()
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    elapsed = time.time() - start
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    ms_per_image = (elapsed / len(all_labels)) * 1000
    return all_probs, all_labels, ms_per_image
 
 
def report_metrics(name, probs, labels, num_classes=4):
    preds = probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
 
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    try:
        auc = roc_auc_score(labels_bin, probs, average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")  # can happen if a class is missing from the test split
 
    cm = confusion_matrix(labels, preds)
 
    print(f"\n=== {name} ===")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall   : {rec:.4f}")
    print(f"Macro F1       : {f1:.4f}")
    print(f"Macro AUC-ROC  : {auc:.4f}")
    print(f"Confusion Matrix (rows=true, cols=pred), classes={CLASS_NAMES}:")
    print(cm)
 
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset_split")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
 
    test_ds = ECGDataset(Path(args.data_dir) / "test", transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
 
    # ---- Step 1: read each model's validation macro F1 (Eq. 3.1 numerator) ----
    val_f1s = {}
    for m in MODELS:
        summary_path = ckpt_dir / m / "val_summary.json"
        with open(summary_path) as f:
            val_f1s[m] = json.load(f)["best_val_macro_f1"]
 
    total_f1 = sum(val_f1s.values())
    weights = {m: val_f1s[m] / total_f1 for m in MODELS}  # Eq. 3.1
 
    print("Validation macro F1 (per model):")
    for m in MODELS:
        print(f"  {m:35s}: F1={val_f1s[m]:.4f}  ->  weight={weights[m]:.4f}")
    print(f"  Sum of weights: {sum(weights.values()):.4f}  (should be 1.0)")
 
    # ---- Step 2: run inference on the test set for each model ----
    all_test_probs = {}
    ms_timings = {}
    results_table = {}
 
    for m in MODELS:
        model = load_model(m, ckpt_dir / m / "best.pt", device)
        probs, labels, ms = get_test_probs(model, test_loader, device)
        all_test_probs[m] = probs
        ms_timings[m] = ms
        results_table[m] = report_metrics(m, probs, labels)
        results_table[m]["inference_ms"] = ms
        del model
        torch.cuda.empty_cache()
 
    # ---- Step 3: weighted soft voting ensemble (Eq. 3.3 / 3.4) ----
    ensemble_probs = sum(weights[m] * all_test_probs[m] for m in MODELS)
    results_table["Ensemble (weighted)"] = report_metrics(
        "Ensemble (Validation-Weighted Soft Voting)", ensemble_probs, labels)
 
    # ---- Step 4: equal-weight baseline, for the Section 4.4 comparison ----
    equal_probs = sum(all_test_probs[m] for m in MODELS) / len(MODELS)
    results_table["Ensemble (equal-weight baseline)"] = report_metrics(
        "Ensemble (Equal-Weight Baseline)", equal_probs, labels)
 
    # ---- Final comparison table ----
    print("\n\n================ FINAL COMPARISON (Table for Section 4.4) ================")
    header = f"{'Model':38s}{'Acc':>8s}{'Prec':>8s}{'Recall':>8s}{'F1':>8s}{'AUC':>8s}"
    print(header)
    for name, r in results_table.items():
        print(f"{name:38s}{r['accuracy']:8.4f}{r['precision']:8.4f}"
              f"{r['recall']:8.4f}{r['f1']:8.4f}{r['auc']:8.4f}")
 
    with open(ckpt_dir / "final_comparison.json", "w") as f:
        json.dump({"weights": weights, "results": results_table}, f, indent=2)
    print(f"\nSaved full results to {ckpt_dir / 'final_comparison.json'}")
 
 
if __name__ == "__main__":
    main()
 