#!/usr/bin/env python3
"""
Batch-evaluate all best_<vuln>.pt checkpoints on their corresponding test sets.

Usage:
  python -m src.models.eval_all \
    --models-dir src/models \
    --datasets-root datasets/vudenc/prepared \
    --outdir test_results \
    --threshold 0.5
  # (or) use per-task JSON thresholds if you have them:
  # --threshold-json "src/models/val_best_threshold_{vuln}.json"
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    precision_recall_fscore_support, confusion_matrix
)
import matplotlib.pyplot as plt

from .graphcodebert_cls import GCBertClassifier
from ..data.dataset import VulnerabilityDataset 


# IO helpers 
def infer_vuln_from_ckpt(ckpt: Path) -> Optional[str]:
    m = re.match(r"best_(.+?)\.pt$", ckpt.name)
    return m.group(1) if m else None

def find_test_file(datasets_root: Path, vuln: str) -> Optional[Path]:
    d = datasets_root / vuln
    candidates = [
        d / "test.jsonl",                
        d / "test.json",
        d / f"{vuln}_test.jsonl",
        d / f"{vuln}_test.json",
    ]
    if not any(p.exists() for p in candidates):
        candidates += sorted(d.glob("*test.jsonl")) + sorted(d.glob("*test.json"))
        if not any(p.exists() for p in candidates):
            candidates += sorted(datasets_root.rglob(f"{vuln}*test.jsonl")) \
                       +  sorted(datasets_root.rglob(f"{vuln}*test.json"))
    for p in candidates:
        if p.exists():
            return p
    return None


def maybe_load_threshold(pattern: Optional[str], vuln: str, fixed: Optional[float]) -> float:
    if fixed is not None:
        return float(fixed)
    if pattern:
        p = Path(pattern.format(vuln=vuln))
        if p.is_file():
            with open(p, "r") as f:
                obj = json.load(f)
            if isinstance(obj, (int, float)):
                return float(obj)
            for k in ("threshold", "best_threshold", "val_best_threshold", "t", "tau"):
                if k in obj and isinstance(obj[k], (int, float)):
                    return float(obj[k])
    return 0.5 # default


# Data / Model 
def build_loader(test_file: Path, batch_size: int, workers: int) -> DataLoader:
    ds = VulnerabilityDataset(json_path=str(test_file), split="test")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)

def load_model(ckpt: Path, device: torch.device) -> torch.nn.Module:
    model = GCBertClassifier(num_labels=2)
    state = torch.load(str(ckpt), map_location="cpu")
    state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


# Eval
@torch.no_grad()
def infer(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = (batch.get("labels") or batch.get("label")).to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)  # [B,2]
        probs1 = torch.softmax(logits, dim=-1)[:, 1]
        ys.append(labels.cpu().numpy())
        ps.append(probs1.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict:
    precs, recs, _ = precision_recall_curve(y_true, y_score)
    fprs, tprs, _ = roc_curve(y_true, y_score)
    pr_auc = auc(recs, precs)
    roc_auc = auc(fprs, tprs)

    y_pred = (y_score >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "threshold": float(thr),
        "confusion_matrix": {"labels": [0, 1], "matrix": cm},
        "pr_curve": {"precision": precs.tolist(), "recall": recs.tolist()},
        "roc_curve": {"fpr": fprs.tolist(), "tpr": tprs.tolist()},
    }

def plot_pr(recall: List[float], precision: List[float], pr_auc: float, save: Path, dot: Optional[Tuple[float,float]]):
    plt.figure()
    plt.plot(recall, precision, lw=2, label=f"PR-AUC={pr_auc:.3f}")
    if dot:
        plt.scatter([dot[0]], [dot[1]], s=30)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Test Precision–Recall")
    plt.legend(loc="lower left"); plt.tight_layout(); plt.savefig(save, dpi=180); plt.close()

def plot_roc(fpr: List[float], tpr: List[float], roc_auc: float, save: Path):
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC-AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1], lw=1, linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("Test ROC")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(save, dpi=180); plt.close()

def plot_conf(cm: List[List[int]], save: Path):
    import numpy as np
    cm = np.array(cm)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
    for (i,j), v in np.ndenumerate(cm): plt.text(j, i, str(v), ha="center", va="center")
    plt.title("Test Confusion Matrix"); plt.tight_layout(); plt.savefig(save, dpi=180); plt.close()


# Main 
def main():
    ap = argparse.ArgumentParser(description="Batch test all best_<vuln>.pt models.")
    ap.add_argument("--models-dir", default="src/models", help="Where the best_*.pt live")
    ap.add_argument("--datasets-root", default="datasets/vudenc/prepared", help="Root for test sets")
    ap.add_argument("--outdir", default="test_results", help="Output root")
    ap.add_argument("--threshold", type=float, default=None, help="Global threshold (fallback 0.5)")
    ap.add_argument("--threshold-json", default=None,
                    help="Optional pattern, e.g. 'src/models/val_best_threshold_{vuln}.json'")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    data_root = Path(args.datasets_root)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(models_dir.glob("best_*.pt"))
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {models_dir} (expected best_<vuln>.pt).")

    device = torch.device(args.device)
    summary: List[Dict] = []

    for ckpt in ckpts:
        vuln = infer_vuln_from_ckpt(ckpt)
        if not vuln:
            print(f"[skip] {ckpt.name}: cannot infer vuln name")
            continue

        test_file = find_test_file(data_root, vuln)
        if not test_file:
            print(f"[skip] {vuln}: no test file found under {data_root}")
            continue

        thr = maybe_load_threshold(args.threshold_json, vuln, args.threshold)
        outdir = out_root / vuln
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"[*] {vuln}: model={ckpt.name} test={test_file} thr={thr}")
        loader = build_loader(test_file, args.batch_size, args.num_workers)
        model = load_model(ckpt, device)
        y_true, y_score = infer(model, loader, device)
        m = compute_metrics(y_true, y_score, thr)

        # save metrics json
        payload = {
            "task": vuln,
            "n": int(len(y_true)),
            "metrics": {
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "threshold": m["threshold"],
                "confusion_matrix": m["confusion_matrix"],
            },
            "paths": {"model": str(ckpt), "test": str(test_file)}
        }
        with open(outdir / "test_metrics.json", "w") as f:
            json.dump(payload, f, indent=2)

        # operating-point dot for visualsation
        tp = ((y_score >= m["threshold"]) & (y_true == 1)).sum()
        fp = ((y_score >= m["threshold"]) & (y_true == 0)).sum()
        fn = ((y_score <  m["threshold"]) & (y_true == 1)).sum()
        dot = (
            (tp / (tp + fn)) if (tp + fn) > 0 else 0.0,  # recall
            (tp / (tp + fp)) if (tp + fp) > 0 else 0.0   # precision
        )

        # plots
        plot_pr(m["pr_curve"]["recall"], m["pr_curve"]["precision"], m["pr_auc"], outdir / "test_pr_curve.png", dot)
        plot_roc(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"], m["roc_auc"], outdir / "test_roc_curve.png")
        plot_conf(m["confusion_matrix"]["matrix"], outdir / "test_confusion_matrix.png")

        print(f"[✓] {vuln}: PR-AUC={m['pr_auc']:.3f} ROC-AUC={m['roc_auc']:.3f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

        summary.append({
            "task": vuln, "P": m["precision"], "R": m["recall"], "F1": m["f1"],
            "PR_AUC": m["pr_auc"], "ROC_AUC": m["roc_auc"], "Threshold": m["threshold"],
            "N": int(len(y_true))
        })

    # write summary
    with open(out_root / "summary.tsv", "w") as f:
        f.write("Task\tP\tR\tF1\tPR_AUC\tROC_AUC\tThreshold\tN\n")
        for r in sorted(summary, key=lambda x: x["task"]):
            f.write("{task}\t{P}\t{R}\t{F1}\t{PR_AUC}\t{ROC_AUC}\t{Threshold}\t{N}\n".format(**r))
    print(f"\n[i] Wrote {out_root}/summary.tsv")

if __name__ == "__main__":
    main()
