"""
Evaluate trained classifier on held-out test set.
Produces robustness curve by sequence divergence level.
Compares against BLAST baseline.

Run after train.py.
Reads:  AIxBIO_submission/models/best_classifier.pt
        AIxBIO_submission/models/best_classifier_natural_only.pt  (if present)
        AIxBIO_submission/data/embeddings/test_embeddings.npy
        AIxBIO_submission/data/splits/test.csv
Writes: AIxBIO_submission/models/test_results.json
        AIxBIO_submission/models/test_predictions.csv

Usage:
    python evaluate.py
"""

import json
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# ----------------------------------------------------------------
# config
# ----------------------------------------------------------------

BASE_DIR      = Path("AIxBIO_submission/data")
SPLITS_DIR    = BASE_DIR / "splits"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
MODELS_DIR    = Path("AIxBIO_submission/models")

EMBEDDING_DIM = 1280   # esm2_t33_650M_UR50D, layer 33
HIDDEN_DIM    = 256
DROPOUT       = 0.3
BATCH_SIZE    = 64
SEED          = 42

DIVERGENCE_LEVELS = ["natural", "high_sim", "med_sim", "low_sim"]

# ----------------------------------------------------------------
# model — must match train.py exactly
# ----------------------------------------------------------------

class ToxinClassifier(nn.Module):
    def __init__(self, input_dim=EMBEDDING_DIM,
                 hidden_dim=HIDDEN_DIM,
                 dropout=DROPOUT):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

# ----------------------------------------------------------------
# dataset
# ----------------------------------------------------------------

class EvalDataset(Dataset):
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

# ----------------------------------------------------------------
# metrics
# ----------------------------------------------------------------

def compute_metrics(predictions: np.ndarray,
                    labels: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute full set of classification metrics."""
    if len(np.unique(labels)) < 2:
        return {
            "auc_roc":   None,
            "auc_pr":    None,
            "precision": None,
            "recall":    None,
            "f1":        None,
            "fpr":       None,
            "n_positive": int(labels.sum()),
            "n_total":    len(labels)
        }

    binary_preds = (predictions >= threshold).astype(int)

    auc_roc = roc_auc_score(labels, predictions)
    auc_pr  = average_precision_score(labels, predictions)

    report = classification_report(
        labels, binary_preds,
        target_names=["non_toxin", "toxin"],
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(labels, binary_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "auc_roc":   float(auc_roc),
        "auc_pr":    float(auc_pr),
        "precision": float(report["toxin"]["precision"]),
        "recall":    float(report["toxin"]["recall"]),
        "f1":        float(report["toxin"]["f1-score"]),
        "fpr":       float(fpr),
        "accuracy":  float(report["accuracy"]),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "n_positive": int(labels.sum()),
        "n_total":    len(labels)
    }

# ----------------------------------------------------------------
# blast baseline
# ----------------------------------------------------------------

def build_blast_db(train_df: pd.DataFrame) -> Path | None:
    """
    Build BLAST database from training positive sequences.
    Includes both natural and augmented positives — the classifier
    also trained on both, so the comparison is fair.
    """
    db_fasta = MODELS_DIR / "blast_db.fasta"
    db_path  = MODELS_DIR / "blast_db"

    positives = train_df[train_df["label"] == 1]
    with open(db_fasta, "w") as f:
        for _, row in positives.iterrows():
            f.write(f">{row['seq_id']}\n{row['sequence']}\n")

    try:
        subprocess.run([
            "makeblastdb",
            "-in",     str(db_fasta),
            "-dbtype", "prot",
            "-out",    str(db_path)
        ], check=True, capture_output=True)
        print(f"  BLAST database built from {len(positives)} sequences "
              f"({int(positives['is_augmented'].sum())} augmented)")
        return db_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  makeblastdb not available, skipping BLAST baseline")
        return None


def run_blast_baseline(test_df: pd.DataFrame,
                       toxin_db_path: Path) -> np.ndarray:
    """
    Run BLAST against training toxin sequences.
    Returns binary predictions (1 = hit found, 0 = no hit).
    Returns zeros if BLAST is not available.
    """
    try:
        subprocess.run(
            ["blastp", "-version"],
            capture_output=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  BLAST not available, skipping baseline")
        return np.zeros(len(test_df))

    tmp_query = MODELS_DIR / "tmp_blast_query.fasta"
    with open(tmp_query, "w") as f:
        for _, row in test_df.iterrows():
            f.write(f">{row['seq_id']}\n{row['sequence']}\n")

    blast_out = MODELS_DIR / "tmp_blast_results.txt"
    cmd = [
        "blastp",
        "-query",       str(tmp_query),
        "-db",          str(toxin_db_path),
        "-evalue",      "0.001",
        "-outfmt",      "6 qseqid sseqid pident evalue",
        "-out",         str(blast_out),
        "-num_threads", "4"
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  BLAST failed: {e}")
        return np.zeros(len(test_df))

    flagged_ids = set()
    if blast_out.exists() and blast_out.stat().st_size > 0:
        blast_df = pd.read_csv(
            blast_out, sep="\t",
            names=["qseqid", "sseqid", "pident", "evalue"]
        )
        hits = blast_df[
            (blast_df["evalue"] < 0.001) &
            (blast_df["pident"] > 30.0)
        ]
        flagged_ids = set(hits["qseqid"].unique())

    blast_preds = np.array([
        1.0 if row["seq_id"] in flagged_ids else 0.0
        for _, row in test_df.iterrows()
    ])

    print(f"  BLAST flagged {int(blast_preds.sum())}/{len(blast_preds)} sequences")

    tmp_query.unlink(missing_ok=True)
    blast_out.unlink(missing_ok=True)

    return blast_preds

# ----------------------------------------------------------------
# robustness curve
# ----------------------------------------------------------------

def compute_robustness_curve(test_df: pd.DataFrame,
                              classifier_preds: np.ndarray,
                              blast_preds: np.ndarray,
                              label: str = "") -> dict:
    """
    Detection rate (recall), FPR, and F1 at each divergence level
    for classifier vs BLAST. This is the core result figure —
    classifier maintains detection where BLAST degrades.
    """
    prefix = f"[{label}] " if label else ""
    results = {}

    print(f"\n  {prefix}ROBUSTNESS CURVE")
    print(
        f"  {'Level':12s} | {'N':5s} | {'N+':4s} | "
        f"{'CLF Rec':8s} {'CLF F1':8s} {'CLF FPR':8s} | "
        f"{'BLAST Rec':10s} {'BLAST F1':8s} | "
        f"{'Rec Adv':8s}"
    )
    print("  " + "-" * 82)

    for level in DIVERGENCE_LEVELS:
        mask = test_df["divergence_level"].values == level
        if mask.sum() == 0:
            continue

        level_labels     = test_df["label"].values[mask]
        level_clf_preds  = classifier_preds[mask]
        level_blast_preds = blast_preds[mask]

        n_positive = int(level_labels.sum())
        if n_positive == 0:
            continue

        clf_metrics   = compute_metrics(level_clf_preds,   level_labels)
        blast_metrics = compute_metrics(level_blast_preds, level_labels)

        clf_rec   = clf_metrics["recall"]   or 0.0
        clf_f1    = clf_metrics["f1"]       or 0.0
        clf_fpr   = clf_metrics["fpr"]      or 0.0
        blast_rec = blast_metrics["recall"] or 0.0
        blast_f1  = blast_metrics["f1"]     or 0.0
        adv       = clf_rec - blast_rec

        results[level] = {
            "n_total":                    int(mask.sum()),
            "n_positive":                 n_positive,
            "classifier_recall":          float(clf_rec),
            "classifier_f1":              float(clf_f1),
            "classifier_fpr":             float(clf_fpr),
            "blast_recall":               float(blast_rec),
            "blast_f1":                   float(blast_f1),
            "recall_advantage_over_blast": float(adv),
        }

        print(
            f"  {level:12s} | {mask.sum():5d} | {n_positive:4d} | "
            f"{clf_rec:8.3f} {clf_f1:8.3f} {clf_fpr:8.3f} | "
            f"{blast_rec:10.3f} {blast_f1:8.3f} | "
            f"{adv:+8.3f}"
        )

    return results

# ----------------------------------------------------------------
# segment breakdown
# ----------------------------------------------------------------

def compute_segment_breakdown(test_df: pd.DataFrame,
                               classifier_preds: np.ndarray,
                               blast_preds: np.ndarray,
                               label: str = "") -> dict:
    prefix = f"[{label}] " if label else ""

    quadrants = {
        "natural_toxic":      (~test_df["is_augmented"]) & (test_df["label"] == 1),
        "natural_nontoxic":   (~test_df["is_augmented"]) & (test_df["label"] == 0),
        "synthetic_toxic":    test_df["is_augmented"]    & (test_df["label"] == 1),
        "synthetic_nontoxic": test_df["is_augmented"]    & (test_df["label"] == 0),
    }

    results = {}

    print(f"\n  {prefix}SEGMENT BREAKDOWN (four quadrants)")
    print(
        f"  {'Segment':20s} | {'N':5s} | {'N+':4s} | "
        f"{'CLF Recall':10s} {'CLF FPR':8s} {'CLF AvgScore':12s} | "
        f"{'BLAST Recall':12s} {'BLAST FPR':9s}"
    )
    print("  " + "-" * 90)

    for name, mask in quadrants.items():
        mask_vals = mask.values
        n = int(mask_vals.sum())
        if n == 0:
            results[name] = {"n": 0, "n_positive": 0}
            continue

        seg_labels      = test_df["label"].values[mask_vals]
        seg_clf_preds   = classifier_preds[mask_vals]
        seg_blast_preds = blast_preds[mask_vals]
        n_positive      = int(seg_labels.sum())
        threshold       = 0.5

        # --- classifier ---
        clf_binary = (seg_clf_preds >= threshold).astype(int)
        if n_positive == n:
            # all positive: report recall only
            clf_recall = float(clf_binary.sum()) / n
            clf_fpr    = float("nan")
        elif n_positive == 0:
            # all negative: report FPR only
            clf_recall = float("nan")
            clf_fpr    = float(clf_binary.sum()) / n
        else:
            clf_recall = float((clf_binary[seg_labels == 1]).sum()) / n_positive
            tn_fp      = (seg_labels == 0).sum()
            clf_fpr    = float((clf_binary[seg_labels == 0]).sum()) / tn_fp

        clf_avg_score = float(seg_clf_preds.mean())

        # --- blast ---
        blast_binary = (seg_blast_preds >= threshold).astype(int)
        if n_positive == n:
            blast_recall = float(blast_binary.sum()) / n
            blast_fpr    = float("nan")
        elif n_positive == 0:
            blast_recall = float("nan")
            blast_fpr    = float(blast_binary.sum()) / n
        else:
            blast_recall = float((blast_binary[seg_labels == 1]).sum()) / n_positive
            tn_fp        = (seg_labels == 0).sum()
            blast_fpr    = float((blast_binary[seg_labels == 0]).sum()) / tn_fp

        results[name] = {
            "n":              n,
            "n_positive":     n_positive,
            "clf_recall":     clf_recall,
            "clf_fpr":        clf_fpr,
            "clf_avg_score":  clf_avg_score,
            "blast_recall":   blast_recall,
            "blast_fpr":      blast_fpr,
        }

        def fmt(v):
            return f"{v:.3f}" if not (isinstance(v, float) and v != v) else "  N/A"

        print(
            f"  {name:20s} | {n:5d} | {n_positive:4d} | "
            f"{fmt(clf_recall):10s} {fmt(clf_fpr):8s} {clf_avg_score:12.3f} | "
            f"{fmt(blast_recall):12s} {fmt(blast_fpr):9s}"
        )

    return results

# ----------------------------------------------------------------
# error analysis
# ----------------------------------------------------------------

def error_analysis(test_df: pd.DataFrame,
                   predictions: np.ndarray,
                   label: str = "",
                   threshold: float = 0.5) -> tuple:
    """
    False negative and false positive breakdown by divergence level
    and augmentation status. Reports the most confident errors —
    cases where the model was most wrong.
    """
    prefix = f"[{label}] " if label else ""

    df = test_df.copy()
    df["predicted_score"] = predictions
    df["predicted_label"] = (predictions >= threshold).astype(int)

    fn = df[(df["label"] == 1) & (df["predicted_label"] == 0)].copy()
    fp = df[(df["label"] == 0) & (df["predicted_label"] == 1)].copy()

    # ---- false negatives ----
    print(f"\n  {prefix}FALSE NEGATIVES (missed toxins): {len(fn)}")
    if len(fn) > 0:
        print(f"  {'Level':12s} | {'Augmented':10s} | {'Missed':6s} / {'Total':6s}")
        print("  " + "-" * 40)
        for level in DIVERGENCE_LEVELS:
            for is_aug in [False, True]:
                total_mask = (
                    (df["label"] == 1) &
                    (df["divergence_level"] == level) &
                    (df["is_augmented"] == is_aug)
                )
                missed_mask = (
                    (fn["divergence_level"] == level) &
                    (fn["is_augmented"] == is_aug)
                )
                total  = int(total_mask.sum())
                missed = int(missed_mask.sum())
                if total > 0:
                    aug_str = "augmented" if is_aug else "natural  "
                    print(f"  {level:12s} | {aug_str:10s} | "
                          f"{missed:6d} / {total:6d}  "
                          f"({100*missed/total:.1f}%)")

        # most confident false negatives — lowest predicted score
        # (model was most certain these were non-toxic)
        print(f"\n  {prefix}Most confident false negatives (lowest score):")
        worst_fn = fn.nsmallest(5, "predicted_score")[
            ["seq_id", "divergence_level", "is_augmented",
             "source", "predicted_score"]
        ]
        print(worst_fn.to_string(index=False))

    # ---- false positives ----
    print(f"\n  {prefix}FALSE POSITIVES (incorrect flags): {len(fp)}")
    if len(fp) > 0:
        print(f"  {'Level':12s} | {'Augmented':10s} | {'FP count':8s}")
        print("  " + "-" * 36)
        for level in DIVERGENCE_LEVELS:
            for is_aug in [False, True]:
                fp_mask = (
                    (fp["divergence_level"] == level) &
                    (fp["is_augmented"] == is_aug)
                )
                n = int(fp_mask.sum())
                if n > 0:
                    aug_str = "augmented" if is_aug else "natural  "
                    print(f"  {level:12s} | {aug_str:10s} | {n:8d}")

        # most confident false positives — highest predicted score
        # (model was most certain these were toxic when they weren't)
        print(f"\n  {prefix}Most confident false positives (highest score):")
        worst_fp = fp.nlargest(5, "predicted_score")[
            ["seq_id", "divergence_level", "is_augmented",
             "source", "predicted_score"]
        ]
        print(worst_fp.to_string(index=False))

    return fn, fp

# ----------------------------------------------------------------
# inference
# ----------------------------------------------------------------

def get_predictions(model, embeddings: np.ndarray, device) -> np.ndarray:
    dataset = EvalDataset(embeddings)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds)


def load_checkpoint(path: Path, device) -> tuple:
    """Load model from checkpoint. Returns (model, epoch, val_metrics)."""
    checkpoint = torch.load(path, map_location=device)
    model = ToxinClassifier(
        input_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["epoch"], checkpoint["val_metrics"]


def evaluate_checkpoint(checkpoint_path: Path,
                         label: str,
                         test_embeddings: np.ndarray,
                         test_df: pd.DataFrame,
                         blast_preds: np.ndarray,
                         blast_metrics: dict,
                         device) -> dict:
    """
    Full evaluation pipeline for a single checkpoint.
    Returns results dict for inclusion in test_results.json.
    """
    print(f"\n{'='*60}")
    print(f"{label.upper()}")
    print(f"{'='*60}")

    model, epoch, val_metrics_at_save = load_checkpoint(
        checkpoint_path, device
    )
    print(f"  Checkpoint: {checkpoint_path.name}")
    print(f"  Saved at epoch {epoch}, "
          f"val AUC-ROC={val_metrics_at_save['auc_roc']:.4f}")

    classifier_preds = get_predictions(model, test_embeddings, device)

    print("\nOVERALL TEST METRICS")
    print("-"*60)
    overall = compute_metrics(classifier_preds, test_df["label"].values)
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nBLAST COMPARISON")
    print("-"*60)
    print(f"  Classifier AUC-ROC: {overall['auc_roc']:.4f}  "
          f"BLAST: {blast_metrics.get('auc_roc', 'N/A')}")
    print(f"  Classifier recall:  {overall['recall']:.4f}  "
          f"BLAST: {blast_metrics.get('recall', 'N/A')}")
    print(f"  Classifier FPR:     {overall['fpr']:.4f}  "
          f"BLAST: {blast_metrics.get('fpr', 'N/A')}")

    print("\nROBUSTNESS CURVE BY DIVERGENCE LEVEL")
    print("-"*60)
    robustness = compute_robustness_curve(
        test_df, classifier_preds, blast_preds, label=label
    )

    print("\nSEGMENT BREAKDOWN")
    print("-"*60)
    segments = compute_segment_breakdown(
        test_df, classifier_preds, blast_preds, label=label
    )

    print("\nERROR ANALYSIS")
    print("-"*60)
    fn, fp = error_analysis(test_df, classifier_preds, label=label)

    return {
        "checkpoint":          checkpoint_path.name,
        "epoch":               epoch,
        "val_auc_at_save":     val_metrics_at_save["auc_roc"],
        "overall":             overall,
        "robustness_curve":    robustness,
        "segment_breakdown":   segments,
        "error_analysis": {
            "n_false_negatives": len(fn),
            "n_false_positives": len(fp),
            "fn_by_divergence_and_augmentation": {
                f"{level}_{'aug' if aug else 'nat'}": int(
                    ((fn["divergence_level"] == level) &
                     (fn["is_augmented"] == aug)).sum()
                )
                for level in DIVERGENCE_LEVELS
                for aug in [False, True]
            }
        },
        "classifier_predictions": classifier_preds,  # kept in memory only
    }

# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Embedding dim: {EMBEDDING_DIM}")

    # ----------------------------------------------------------------
    # verify inputs
    # ----------------------------------------------------------------

    required = [
        EMBEDDINGS_DIR / "test_embeddings.npy",
        SPLITS_DIR / "test.csv",
        SPLITS_DIR / "train.csv",
        MODELS_DIR / "best_classifier.pt",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. "
                f"Run data pipeline, embedding extraction, "
                f"and training first."
            )

    # ----------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------

    print("\nLoading test data...")
    test_embeddings = np.load(EMBEDDINGS_DIR / "test_embeddings.npy")
    test_df  = pd.read_csv(SPLITS_DIR / "test.csv")
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")

    for df in [test_df, train_df]:
        df["confidence"]       = df["confidence"].fillna("structural_proxy")
        df["divergence_level"] = df["divergence_level"].fillna("natural")
        df["is_augmented"]     = df["is_augmented"].fillna(False)

    print(f"  Test sequences: {len(test_df)}")
    print(f"  Positives:      {int(test_df['label'].sum())}")
    print(f"  Negatives:      {int((test_df['label'] == 0).sum())}")
    n_aug_test = int(test_df["is_augmented"].sum())
    print(f"  Augmented:      {n_aug_test}")

    assert test_embeddings.shape[0] == len(test_df), (
        f"Embedding rows ({test_embeddings.shape[0]}) != "
        f"CSV rows ({len(test_df)}). Re-run extract_embeddings.py."
    )
    assert test_embeddings.shape[1] == EMBEDDING_DIM, (
        f"Embedding dim ({test_embeddings.shape[1]}) != "
        f"expected ({EMBEDDING_DIM}). Check ESM model config."
    )

    # ----------------------------------------------------------------
    # blast baseline — built once, used for all checkpoints
    # ----------------------------------------------------------------

    print("\nBLAST BASELINE")
    print("="*60)
    blast_db = build_blast_db(train_df)
    if blast_db is not None:
        blast_preds  = run_blast_baseline(test_df, blast_db)
        blast_metrics = compute_metrics(blast_preds, test_df["label"].values)
        print(f"  BLAST AUC-ROC: {blast_metrics['auc_roc']:.4f}")
        print(f"  BLAST recall:  {blast_metrics['recall']:.4f}")
        print(f"  BLAST F1:      {blast_metrics['f1']:.4f}")
        print(f"  BLAST FPR:     {blast_metrics['fpr']:.4f}")
    else:
        blast_preds   = np.zeros(len(test_df))
        blast_metrics = {}

    # ----------------------------------------------------------------
    # evaluate checkpoints
    # ----------------------------------------------------------------

    checkpoints = [
        (MODELS_DIR / "best_classifier.pt",
         "Experiment 1: full training set (natural + augmented)"),
    ]
    nat_only_path = MODELS_DIR / "best_classifier_natural_only.pt"
    if nat_only_path.exists():
        checkpoints.append((
            nat_only_path,
            "Experiment 2: natural sequences only"
        ))
    else:
        print(f"\nNote: {nat_only_path.name} not found — "
              f"skipping experiment 2 evaluation.")

    all_results = {}
    all_clf_preds = {}

    for ckpt_path, ckpt_label in checkpoints:
        result = evaluate_checkpoint(
            checkpoint_path  = ckpt_path,
            label            = ckpt_label,
            test_embeddings  = test_embeddings,
            test_df          = test_df,
            blast_preds      = blast_preds,
            blast_metrics    = blast_metrics,
            device           = device,
        )
        # pop predictions out of result before JSON serialization
        preds = result.pop("classifier_predictions")
        key   = ckpt_path.stem  # "best_classifier" or "best_classifier_natural_only"
        all_results[key]    = result
        all_clf_preds[key]  = preds

    # ----------------------------------------------------------------
    # save results
    # ----------------------------------------------------------------

    output = {
        "blast_baseline": blast_metrics,
        **all_results,
    }

    with open(MODELS_DIR / "test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # predictions CSV — one column per checkpoint
    test_df["blast_prediction"] = blast_preds
    for key, preds in all_clf_preds.items():
        test_df[f"predicted_score_{key}"]  = preds
        test_df[f"predicted_label_{key}"]  = (preds >= 0.5).astype(int)

    test_df.to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"  Results:     {MODELS_DIR / 'test_results.json'}")
    print(f"  Predictions: {MODELS_DIR / 'test_predictions.csv'}")

    # ----------------------------------------------------------------
    # experiment comparison summary (if both checkpoints ran)
    # ----------------------------------------------------------------

    if len(all_results) == 2:
        keys = list(all_results.keys())
        e1   = all_results[keys[0]]["overall"]
        e2   = all_results[keys[1]]["overall"]

        print(f"\nEXPERIMENT COMPARISON SUMMARY")
        print(f"  {'Metric':12s} | {'Exp1 (full)':12s} | {'Exp2 (natural)':14s} | {'Delta':8s}")
        print("  " + "-" * 56)
        for metric in ["auc_roc", "recall", "f1", "fpr", "precision"]:
            v1    = e1.get(metric) or 0.0
            v2    = e2.get(metric) or 0.0
            delta = v1 - v2
            print(f"  {metric:12s} | {v1:12.4f} | {v2:14.4f} | {delta:+8.4f}")