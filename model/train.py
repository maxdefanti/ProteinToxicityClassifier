"""
Train binary toxin classifier on precomputed ESM-2 embeddings.

Run after extract_embeddings.py.
Reads:  AIxBIO_submission/data/embeddings/{train,val}_embeddings.npy
        AIxBIO_submission/data/splits/{train,val}.csv
Writes: AIxBIO_submission/models/best_classifier.pt
        AIxBIO_submission/models/best_classifier_natural_only.pt
        AIxBIO_submission/models/training_history.csv
        AIxBIO_submission/models/training_history_natural_only.csv
        AIxBIO_submission/models/training_config.json

Usage:
    python train.py

Runtime: 10-20 minutes on CPU
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import classification_report
from pathlib import Path

# ----------------------------------------------------------------
# config
# ----------------------------------------------------------------

BASE_DIR = Path("AIxBIO_submission/data")
SPLITS_DIR = BASE_DIR / "splits"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
MODELS_DIR = Path("AIxBIO_submission/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 1280   # esm2_t33_650M_UR50D, layer 33
HIDDEN_DIM = 256
DROPOUT = 0.3

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
N_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------
# dataset
# ----------------------------------------------------------------

class ToxinDataset(Dataset):
    """
    Dataset wrapping precomputed ESM-2 embeddings.

    Confidence weights downweight augmented sequences
    to account for label noise — we are less certain that
    ProteinMPNN redesigns are genuinely toxic than we are
    about verified Tox-Prot sequences.
    """

    CONFIDENCE_WEIGHTS = {
        "verified": 1.0,
        "verified_negative": 1.0,
        "structural_proxy": 0.7,
        "uncertain": 0.4
    }

    def __init__(self, embeddings: np.ndarray, df: pd.DataFrame):
        assert len(embeddings) == len(df), (
            f"Mismatch: {len(embeddings)} embeddings, {len(df)} rows"
        )

        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(
            df["label"].values, dtype=torch.float32
        )

        confidence = df["confidence"].fillna("structural_proxy").values
        self.weights = torch.tensor(
            [self.CONFIDENCE_WEIGHTS.get(c, 0.7) for c in confidence],
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx],
            "weight": self.weights[idx]
        }

# ----------------------------------------------------------------
# model
# ----------------------------------------------------------------

class ToxinClassifier(nn.Module):
    """
    Two-layer MLP classifier on top of frozen ESM-2 embeddings.

    Deliberately simple — ESM-2 has already done the heavy
    representational work. The MLP just learns the decision
    boundary in embedding space.
    """

    def __init__(self, input_dim: int = EMBEDDING_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

# ----------------------------------------------------------------
# loss
# ----------------------------------------------------------------

def weighted_bce_loss(predictions: torch.Tensor,
                      labels: torch.Tensor,
                      weights: torch.Tensor,
                      pos_weight: float) -> torch.Tensor:
    """
    Binary cross entropy with:
    - per-sample weights for label noise handling
    - positive class weight for class imbalance
    """
    eps = 1e-7
    predictions = predictions.clamp(eps, 1 - eps)

    bce = -(
        labels * torch.log(predictions) +
        (1 - labels) * torch.log(1 - predictions)
    )

    class_weights = torch.where(
        labels == 1,
        torch.tensor(pos_weight, device=predictions.device),
        torch.tensor(1.0, device=predictions.device)
    )
    bce = bce * class_weights
    bce = bce * weights

    return bce.mean()

# ----------------------------------------------------------------
# metrics
# ----------------------------------------------------------------

def compute_metrics(predictions: np.ndarray,
                    labels: np.ndarray,
                    threshold: float = 0.5) -> dict:
    binary_preds = (predictions >= threshold).astype(int)

    try:
        auc_roc = roc_auc_score(labels, predictions)
    except ValueError:
        auc_roc = 0.0

    try:
        auc_pr = average_precision_score(labels, predictions)
    except ValueError:
        auc_pr = 0.0

    report = classification_report(
        labels, binary_preds,
        target_names=["non_toxin", "toxin"],
        output_dict=True,
        zero_division=0
    )

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "precision": report["toxin"]["precision"],
        "recall": report["toxin"]["recall"],
        "f1": report["toxin"]["f1-score"],
        "accuracy": report["accuracy"],
        "fpr": 1 - report["non_toxin"]["recall"]
    }

# ----------------------------------------------------------------
# training loop
# ----------------------------------------------------------------

def run_epoch(model, loader, device, pos_weight,
              optimizer=None) -> tuple[float, np.ndarray, np.ndarray]:
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch in loader:
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)
            weights = batch["weight"].to(device)

            if is_training:
                optimizer.zero_grad()

            predictions = model(embeddings)
            loss = weighted_bce_loss(
                predictions, labels, weights, pos_weight
            )

            if is_training:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

            total_loss += loss.item()
            all_preds.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def train(model, train_loader, val_loader, pos_weight,
          device, save_path: Path) -> list[dict]:
    """
    Full training loop with early stopping.
    Saves best checkpoint to save_path based on validation AUC-ROC.
    """
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max",
        factor=0.5, patience=3,
        verbose=True
    )

    best_val_auc = 0.0
    patience_counter = 0
    history = []

    print(f"\n  Epochs:            {N_EPOCHS}")
    print(f"  Early stopping:    {EARLY_STOPPING_PATIENCE} epochs")
    print(f"  Learning rate:     {LEARNING_RATE}")
    print(f"  Pos class weight:  {pos_weight:.2f}")
    print(f"  Checkpoint:        {save_path.name}")
    print()

    for epoch in range(N_EPOCHS):
        train_loss, train_preds, train_labels = run_epoch(
            model, train_loader, device, pos_weight, optimizer
        )
        train_metrics = compute_metrics(train_preds, train_labels)

        val_loss, val_preds, val_labels = run_epoch(
            model, val_loader, device, pos_weight
        )
        val_metrics = compute_metrics(val_preds, val_labels)

        scheduler.step(val_metrics["auc_roc"])

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_auc_roc": train_metrics["auc_roc"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_loss,
            "val_auc_roc": val_metrics["auc_roc"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "val_f1": val_metrics["f1"],
            "val_fpr": val_metrics["fpr"]
        }
        history.append(row)

        print(
            f"  Epoch {epoch+1:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_auc={val_metrics['auc_roc']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_fpr={val_metrics['fpr']:.4f}"
        )

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": {
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT,
                    "esm_model": "esm2_t33_650M_UR50D",
                    "esm_layer": 33,
                }
            }, save_path)
            print(f"           ✓ saved best model (val_auc={best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\n  Best val AUC-ROC: {best_val_auc:.4f}")
    return history

# ----------------------------------------------------------------
# logistic regression baseline
# ----------------------------------------------------------------

def fit_lr_baseline(train_embeddings: np.ndarray,
                    train_labels: np.ndarray,
                    val_embeddings: np.ndarray,
                    val_labels: np.ndarray,
                    label: str = "") -> dict:
    """
    Fit logistic regression on ESM-2 embeddings as baseline.
    MLP should outperform this — if it doesn't, something is
    wrong with MLP training.
    """
    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}Fitting logistic regression baseline...")
    print(f"  {prefix}Train: {len(train_labels)} sequences "
          f"({int(train_labels.sum())} positive)")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_embeddings)
    X_val   = scaler.transform(val_embeddings)

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        C=1.0
    )
    lr.fit(X_train, train_labels)
    val_preds = lr.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(val_preds, val_labels)

    print(f"  {prefix}LR val AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  {prefix}LR val recall:  {metrics['recall']:.4f}")
    print(f"  {prefix}LR val F1:      {metrics['f1']:.4f}")
    return metrics

# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------

def print_split_composition(name: str, df: pd.DataFrame) -> None:
    n_total    = len(df)
    n_pos      = int(df["label"].sum())
    n_neg      = n_total - n_pos
    n_aug      = int(df["is_augmented"].sum())
    n_nat_pos  = int((df["label"] == 1) & ~df["is_augmented"]).sum() \
                 if "is_augmented" in df.columns else n_pos

    print(f"  {name}:")
    print(f"    Total:             {n_total}")
    print(f"    Positive:          {n_pos}  "
          f"(natural: {n_nat_pos}, augmented: {n_aug})")
    print(f"    Negative:          {n_neg}")

    if n_aug > 0 and "divergence_level" in df.columns:
        aug_df = df[df["is_augmented"]]
        for level in ["high_sim", "med_sim", "low_sim"]:
            n = int((aug_df["divergence_level"] == level).sum())
            print(f"    Augmented {level}:  {n}")


def make_loader(embeddings: np.ndarray, df: pd.DataFrame,
                shuffle: bool) -> DataLoader:
    dataset = ToxinDataset(embeddings, df)
    return DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=shuffle, num_workers=0
    )

# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------

if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Embedding dim: {EMBEDDING_DIM}")

    # ----------------------------------------------------------------
    # verify embeddings exist
    # ----------------------------------------------------------------

    for split in ["train", "val"]:
        path = EMBEDDINGS_DIR / f"{split}_embeddings.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run extract_embeddings.py first."
            )

    # ----------------------------------------------------------------
    # load embeddings and metadata
    # ----------------------------------------------------------------

    print("\nLoading embeddings and metadata...")

    train_embeddings = np.load(EMBEDDINGS_DIR / "train_embeddings.npy")
    val_embeddings   = np.load(EMBEDDINGS_DIR / "val_embeddings.npy")

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val.csv")

    for df in [train_df, val_df]:
        df["confidence"]       = df["confidence"].fillna("structural_proxy")
        df["divergence_level"] = df["divergence_level"].fillna("natural")
        df["is_augmented"]     = df["is_augmented"].fillna(False)

    assert train_embeddings.shape[1] == EMBEDDING_DIM, (
        f"Embedding dim mismatch: expected {EMBEDDING_DIM}, "
        f"got {train_embeddings.shape[1]}. "
        f"Check ESM_MODEL_NAME in extract_embeddings.py."
    )

    # ----------------------------------------------------------------
    # dataset composition
    # ----------------------------------------------------------------

    print("\nDataset composition:")
    print_split_composition("train", train_df)
    print_split_composition("val",   val_df)

    # ----------------------------------------------------------------
    # experiment 1: full augmented training set
    # ----------------------------------------------------------------

    print("\n" + "="*60)
    print("EXPERIMENT 1: Full training set (natural + augmented)")
    print("="*60)

    n_pos_1    = int(train_df["label"].sum())
    n_neg_1    = len(train_df) - n_pos_1
    pos_weight_1 = n_neg_1 / n_pos_1

    print(f"\n  Positive: {n_pos_1}  Negative: {n_neg_1}  "
          f"Pos weight: {pos_weight_1:.2f}")

    print("\nBASELINE (Logistic Regression — full train)")
    print("-"*60)
    baseline_metrics_1 = fit_lr_baseline(
        train_embeddings, train_df["label"].values,
        val_embeddings,   val_df["label"].values,
        label="full"
    )

    print("\nTRAINING (MLP — full train)")
    print("-"*60)
    model_1 = ToxinClassifier(
        input_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)

    history_1 = train(
        model_1,
        make_loader(train_embeddings, train_df, shuffle=True),
        make_loader(val_embeddings,   val_df,   shuffle=False),
        pos_weight_1,
        device,
        save_path=MODELS_DIR / "best_classifier.pt"
    )

    pd.DataFrame(history_1).to_csv(
        MODELS_DIR / "training_history.csv", index=False
    )

    # ----------------------------------------------------------------
    # experiment 2: natural sequences only
    # ----------------------------------------------------------------

    print("\n" + "="*60)
    print("EXPERIMENT 2: Natural sequences only (no augmentation)")
    print("="*60)

    nat_mask         = ~train_df["is_augmented"].values
    train_df_nat     = train_df[nat_mask].reset_index(drop=True)
    train_emb_nat    = train_embeddings[nat_mask]

    n_pos_2      = int(train_df_nat["label"].sum())
    n_neg_2      = len(train_df_nat) - n_pos_2
    pos_weight_2 = n_neg_2 / n_pos_2

    print(f"\n  Positive: {n_pos_2}  Negative: {n_neg_2}  "
          f"Pos weight: {pos_weight_2:.2f}")

    print("\nBASELINE (Logistic Regression — natural only)")
    print("-"*60)
    baseline_metrics_2 = fit_lr_baseline(
        train_emb_nat,  train_df_nat["label"].values,
        val_embeddings, val_df["label"].values,
        label="natural"
    )

    print("\nTRAINING (MLP — natural only)")
    print("-"*60)
    model_2 = ToxinClassifier(
        input_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)

    history_2 = train(
        model_2,
        make_loader(train_emb_nat, train_df_nat, shuffle=True),
        make_loader(val_embeddings, val_df,       shuffle=False),
        pos_weight_2,
        device,
        save_path=MODELS_DIR / "best_classifier_natural_only.pt"
    )

    pd.DataFrame(history_2).to_csv(
        MODELS_DIR / "training_history_natural_only.csv", index=False
    )

    # ----------------------------------------------------------------
    # save config
    # ----------------------------------------------------------------

    config = {
        "embedding_dim":           EMBEDDING_DIM,
        "esm_model":               "esm2_t33_650M_UR50D",
        "esm_layer":               33,
        "hidden_dim":              HIDDEN_DIM,
        "dropout":                 DROPOUT,
        "batch_size":              BATCH_SIZE,
        "learning_rate":           LEARNING_RATE,
        "weight_decay":            WEIGHT_DECAY,
        "n_epochs":                N_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "experiment_1": {
            "train_size":          len(train_df),
            "pos_weight":          pos_weight_1,
            "lr_baseline_val_auc": baseline_metrics_1["auc_roc"],
        },
        "experiment_2": {
            "train_size":          len(train_df_nat),
            "pos_weight":          pos_weight_2,
            "lr_baseline_val_auc": baseline_metrics_2["auc_roc"],
        },
    }

    with open(MODELS_DIR / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Experiment 1 checkpoint: best_classifier.pt")
    print(f"  Experiment 2 checkpoint: best_classifier_natural_only.pt")
    print(f"  Run evaluate.py for test set results")