#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from types import SimpleNamespace
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    precision_recall_fscore_support, confusion_matrix
)
import matplotlib.pyplot as plt

# progress (fallback to no-op if tqdm not installed)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

from models.graphcodebert_cls import GCBertClassifier
from data.dataset import VulnerabilityDataset

# helpers 
def infer_vuln_from_ckpt(ckpt: Path) -> Optional[str]:
    m = re.match(r"best_(.+?)\.pt$", ckpt.name)
    return m.group(1) if m else None

def find_test_file(datasets_root: Path, vuln: str) -> Optional[Path]:
    d = datasets_root / vuln
    cands = [d / "test.jsonl", d / "test.json", d / f"{vuln}_test.jsonl", d / f"{vuln}_test.json"]
    if not any(p.exists() for p in cands):
        cands += sorted(d.glob("*test.jsonl")) + sorted(d.glob("*test.json"))
        if not any(p.exists() for p in cands):
            cands += sorted(datasets_root.rglob(f"{vuln}*test.jsonl")) + sorted(datasets_root.rglob(f"{vuln}*test.json"))
    for p in cands:
        if p.exists():
            return p
    return None

def maybe_load_threshold(pattern: Optional[str], vuln: str, fixed: Optional[float]) -> float:
    if fixed is not None:
        return float(fixed)
    if vuln in OVERRIDE_THRESHOLDS:
        return float(OVERRIDE_THRESHOLDS[vuln])
    if pattern:
        p = Path(pattern.format(vuln=vuln))
        if p.is_file():
            with open(p) as f:
                obj = json.load(f)
            if isinstance(obj, (int, float)):
                return float(obj)
            for k in ("threshold", "best_threshold", "val_best_threshold", "t", "tau"):
                if k in obj and isinstance(obj[k], (int, float)):
                    return float(obj[k])
    return 0.5

def collate_tokens_only(batch):
    import torch
    input_ids      = torch.stack([b["input_ids"]      for b in batch], dim=0)          # [B,L]
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0).long()   # [B,L]
    labels         = torch.stack([b["labels"]         for b in batch], dim=0)

    # GraphCodeBERT position indices: 0 on pad; tokens start at 3 (2 is reserved)
    pos = torch.cumsum(attention_mask, dim=1) + 2
    position_idx = pos * attention_mask

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "position_idx": position_idx,
    }

def build_loader(test_file: Path, bs: int, nw: int) -> DataLoader:
    ds = VulnerabilityDataset(path=str(test_file))
    return DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=True, persistent_workers=(nw > 0),
        collate_fn=collate_tokens_only
    )

def load_model(ckpt: Path, device: torch.device, hf_model_name: str) -> torch.nn.Module:
    cfg = SimpleNamespace(
        model_name_or_path=hf_model_name,
        num_labels=2,
        hidden_dropout_prob=None,
        classifier_dropout=None,
    )
    model = GCBertClassifier(cfg)
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

@torch.no_grad()
def infer(model: torch.nn.Module, loader: DataLoader, device: torch.device, desc: str) -> Tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    t0, seen = perf_counter(), 0
    use_cuda = (device.type == "cuda") and torch.cuda.is_available()
    for batch in tqdm(loader, desc=desc, total=len(loader)):
        ids  = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        pos  = batch["position_idx"].to(device, non_blocking=True)

        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=mask, position_idx=pos)
        else:
            logits = model(input_ids=ids, attention_mask=mask, position_idx=pos)

        probs1 = torch.softmax(logits, dim=-1)[:, 1]
        ys.append(batch["labels"].cpu().numpy())
        ps.append(probs1.float().cpu().numpy())

        seen += ids.size(0)
        if seen % max(loader.batch_size or 32, 1600) == 0:
            spd = seen / max(1e-6, perf_counter() - t0)
            print(f"[spd] {desc}: {spd:.1f} samples/s")
    return np.concatenate(ys), np.concatenate(ps)

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict:
    precs, recs, _ = precision_recall_curve(y_true, y_score)
    fprs, tprs, _  = roc_curve(y_true, y_score)
    pr_auc  = auc(recs, precs)
    roc_auc = auc(fprs, tprs)

    y_pred = (y_score >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "pr_auc": float(pr_auc), "roc_auc": float(roc_auc),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "threshold": float(thr),
        "confusion_matrix": {"labels": [0, 1], "matrix": cm},
        "pr_curve": {"precision": precs.tolist(), "recall": recs.tolist()},
        "roc_curve": {"fpr": fprs.tolist(), "tpr": tprs.tolist()},
    }

def parse_args():
    ap = argparse.ArgumentParser(description="Batch test best_<vuln>.pt models.")
    ap.add_argument("--hf_model_name", default="microsoft/graphcodebert-base",
                    help="HF model name/path used during training")
    ap.add_argument("--models-dir", default="src/models")
    ap.add_argument("--datasets-root", default="datasets/vudenc/prepared")
    ap.add_argument("--outdir", default="test_results")
    ap.add_argument("--threshold", type=float, default=None, help="Global threshold (overrides all)")
    ap.add_argument("--threshold-json", default=None, help="Pattern like 'src/models/val_best_threshold_{vuln}.json'")
    ap.add_argument("--only", default=None, help="Comma-separated task names to evaluate")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def main():
    args = parse_args()

    models_dir = Path(args.models_dir)
    data_root  = Path(args.datasets_root)
    out_root   = Path(args.outdir); out_root.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(models_dir.glob("best_*.pt"))
    if args.only:
        allow = {s.strip() for s in args.only.split(",") if s.strip()}
        ckpts = [ck for ck in ckpts if infer_vuln_from_ckpt(ck) in allow]
        print(f"[*] Filtering to tasks: {sorted(allow)}")
    if not ckpts:
        raise SystemExit(f"No checkpoints matching selection in {models_dir}")

    device = torch.device(args.device)
    summary: List[Dict] = []

    for ckpt in ckpts:
        vuln = infer_vuln_from_ckpt(ckpt) or "unknown"
        test_file = find_test_file(data_root, vuln)
        if not test_file:
            print(f"[skip] {vuln}: no test set under {data_root}")
            continue

        thr = maybe_load_threshold(args.threshold_json, vuln, args.threshold)
        outdir = out_root / vuln; outdir.mkdir(parents=True, exist_ok=True)

        loader = build_loader(test_file, args.batch_size, args.num_workers)
        print(f"[*] {vuln}: model={ckpt.name} test={test_file} thr={thr} | N={len(loader.dataset)} steps={len(loader)} bs={args.batch_size}")

        model  = load_model(ckpt, device, args.hf_model_name)
        y_true, y_score = infer(model, loader, device, desc=f"{vuln} infer")
        m = compute_metrics(y_true, y_score, thr)

        pos_at_thr = int((y_score >= m["threshold"]).sum())
        print(f"[dist] {vuln}: min={y_score.min():.3f} mean={y_score.mean():.3f} max={y_score.max():.3f} | pos@thr={pos_at_thr}/{len(y_score)}")

        with open(outdir / "test_metrics.json", "w") as f:
            json.dump({
                "task": vuln, "n": int(len(y_true)),
                "metrics": {
                    "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                    "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                    "threshold": m["threshold"], "confusion_matrix": m["confusion_matrix"],
                },
                "paths": {"model": str(ckpt), "test": str(test_file)}
            }, f, indent=2)

        # plots
        def plot_pr(recall: List[float], precision: List[float], pr_auc: float, save: Path, dot: Tuple[float, float]):
            plt.figure(); plt.plot(recall, precision, lw=2, label=f"PR-AUC={pr_auc:.3f}")
            if dot: plt.scatter([dot[0]], [dot[1]], s=30)
            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Test Precision–Recall")
            plt.legend(loc="lower left"); plt.tight_layout(); plt.savefig(save, dpi=180); plt.close()
        def plot_roc(fpr: List[float], tpr: List[float], roc_auc: float, save: Path):
            plt.figure(); plt.plot(fpr, tpr, lw=2, label=f"ROC-AUC={roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], "--", lw=1); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("Test ROC"); plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(save, dpi=180); plt.close()
        def plot_conf(cm: List[List[int]], save: Path):
            import numpy as np
            cm = np.array(cm); plt.figure(); plt.imshow(cm, interpolation="nearest")
            plt.xticks([0, 1], ["Pred 0", "Pred 1"]); plt.yticks([0, 1], ["True 0", "True 1"])
            for (i, j), v in np.ndenumerate(cm): plt.text(j, i, str(v), ha="center", va="center")
            plt.title("Test Confusion Matrix"); plt.tight_layout(); plt.savefig(save, dpi=180); plt.close()

        tp = ((y_score >= m["threshold"]) & (y_true == 1)).sum()
        fp = ((y_score >= m["threshold"]) & (y_true == 0)).sum()
        fn = ((y_score <  m["threshold"]) & (y_true == 1)).sum()
        dot = ((tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
               (tp / (tp + fp)) if (tp + fp) > 0 else 0.0)

        plot_pr(m["pr_curve"]["recall"], m["pr_curve"]["precision"], m["pr_auc"], outdir / "test_pr_curve.png", dot)
        plot_roc(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"], m["roc_auc"], outdir / "test_roc_curve.png")
        plot_conf(m["confusion_matrix"]["matrix"], outdir / "test_confusion_matrix.png")

        print(f"[✓] {vuln}: PR-AUC={m['pr_auc']:.3f} ROC-AUC={m['roc_auc']:.3f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

        summary.append({
            "task": vuln, "P": m["precision"], "R": m["recall"], "F1": m["f1"],
            "PR_AUC": m["pr_auc"], "ROC_AUC": m["roc_auc"],
            "Threshold": m["threshold"], "N": int(len(y_true))
        })

    with open(out_root / "summary.tsv", "w") as f:
        f.write("Task\tP\tR\tF1\tPR_AUC\tROC_AUC\tThreshold\tN\n")
        for r in sorted(summary, key=lambda x: x["task"]):
            f.write("{task}\t{P}\t{R}\t{F1}\t{PR_AUC}\t{ROC_AUC}\t{Threshold}\t{N}\n".format(**r))
    print(f"[i] Wrote {out_root}/summary.tsv")

if __name__ == "__main__":
    main()
