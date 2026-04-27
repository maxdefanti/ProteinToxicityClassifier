"""
Extract ESM-2 embeddings for all splits and cache to disk.

Run once after run_data_pipeline.py completes.
Reads:  AIxBIO_submission/scripts/splits/{train,val,test}.csv
Writes: AIxBIO_submission/data/embeddings/{train,val,test}_embeddings.npy
        AIxBIO_submission/data/embeddings/{train,val,test}_ids.json

Usage:
    python extract_embeddings.py

Runtime on CPU (MacBook Air M-series):
    150M model: ~1-2 hours for full dataset
    650M model: ~4-6 hours, not recommended without GPU
"""

import json
import numpy as np
import pandas as pd
import torch
import esm
from pathlib import Path

# ----------------------------------------------------------------
# config
# ----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # AIxBIO_submission/
BASE_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = BASE_DIR / "splits"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# 150M model is practical on CPU
# swap to esm2_t33_650M_UR50D if you have GPU
# 150M: layer=30, dim=640
# 650M: layer=33, dim=1280
ESM_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM_LAYER = 33
EMBEDDING_DIM = 1280

# small batch size for CPU memory
BATCH_SIZE = 32

# ----------------------------------------------------------------
# load model once
# ----------------------------------------------------------------

def load_esm_model():
    """
    Load ESM-2 model and alphabet.
    Called once and reused across all splits.
    """
    print(f"Loading {ESM_MODEL_NAME}...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()

    # move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    return model, alphabet, device

# ----------------------------------------------------------------
# embedding extraction
# ----------------------------------------------------------------

def extract_embeddings_for_split(sequences: list[str],
                                  seq_ids: list[str],
                                  split_name: str,
                                  model,
                                  alphabet,
                                  device) -> np.ndarray:
    """
    Extract mean-pooled final layer ESM-2 representations
    for a list of sequences.

    Args:
        sequences:   list of amino acid sequences as strings
        seq_ids:     list of sequence IDs matching sequences
        split_name:  "train", "val", or "test" for cache naming
        model:       loaded ESM-2 model
        alphabet:    ESM-2 alphabet for tokenization
        device:      torch device

    Returns:
        embeddings: np.ndarray shape (n_sequences, EMBEDDING_DIM)
    """
    cache_path = EMBEDDINGS_DIR / f"{split_name}_embeddings.npy"
    ids_cache_path = EMBEDDINGS_DIR / f"{split_name}_ids.json"

    # return cached embeddings if they exist
    if cache_path.exists() and ids_cache_path.exists():
        print(f"  {split_name}: loading from cache {cache_path}")
        cached_embeddings = np.load(cache_path)

        # verify cache matches current sequences
        with open(ids_cache_path) as f:
            cached_ids = json.load(f)
        if cached_ids == seq_ids:
            print(f"  {split_name}: cache valid, {len(cached_embeddings)} embeddings")
            return cached_embeddings
        else:
            print(f"  {split_name}: cache mismatch, recomputing")

    print(f"  {split_name}: extracting {len(sequences)} embeddings...")

    batch_converter = alphabet.get_batch_converter()
    all_embeddings = []
    failed = 0

    for i in range(0, len(sequences), BATCH_SIZE):
        batch_seqs = sequences[i:i + BATCH_SIZE]
        batch_ids = seq_ids[i:i + BATCH_SIZE]

        # ESM expects list of (label, sequence) tuples
        data = list(zip(batch_ids, batch_seqs))

        try:
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            with torch.no_grad():
                results = model(
                    tokens,
                    repr_layers=[ESM_LAYER],
                    return_contacts=False
                )

            # shape: (batch, seq_len + 2, embedding_dim)
            # +2 for BOS and EOS special tokens
            token_representations = results["representations"][ESM_LAYER]

            for j, (_, seq) in enumerate(data):
                # slice off BOS at position 0 and EOS at position seq_len+1
                # then mean pool over actual sequence positions
                seq_len = len(seq)
                embedding = token_representations[j, 1:seq_len + 1].mean(0)
                all_embeddings.append(embedding.cpu().numpy())

        except RuntimeError as e:
            # most common failure: sequence too long for memory
            print(f"\n  Warning: batch starting at {i} failed: {e}")
            print(f"  Filling {len(batch_seqs)} sequences with zeros")
            for _ in batch_seqs:
                all_embeddings.append(np.zeros(EMBEDDING_DIM))
            failed += len(batch_seqs)

        # progress indicator
        n_done = min(i + BATCH_SIZE, len(sequences))
        pct = 100 * n_done / len(sequences)
        print(f"  {split_name}: {n_done}/{len(sequences)} ({pct:.1f}%)", end="\r")

    print(f"\n  {split_name}: complete. Failed: {failed}/{len(sequences)}")

    embeddings = np.array(all_embeddings)
    assert embeddings.shape == (len(sequences), EMBEDDING_DIM), (
        f"Expected shape ({len(sequences)}, {EMBEDDING_DIM}), "
        f"got {embeddings.shape}"
    )

    # save to cache
    np.save(cache_path, embeddings)
    with open(ids_cache_path, "w") as f:
        json.dump(seq_ids, f)
    print(f"  {split_name}: saved to {cache_path}")

    return embeddings

# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------

if __name__ == "__main__":


    # verify splits exist
    for split in ["train", "val", "test"]:
        path = SPLITS_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. "
                f"Run run_data_pipeline.py first."
            )

    # load splits
    print("\nLoading splits...")
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    print(f"  Train: {len(train_df)} sequences")
    print(f"  Val:   {len(val_df)} sequences")
    print(f"  Test:  {len(test_df)} sequences")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} sequences")

    # load ESM-2 model once
    model, alphabet, device = load_esm_model()

    # extract embeddings for each split
    print("\nExtracting embeddings...")

    for split_name, df in [("train", train_df),
                            ("val", val_df),
                            ("test", test_df)]:
        embeddings = extract_embeddings_for_split(
            sequences=df["sequence"].tolist(),
            seq_ids=df["seq_id"].tolist(),
            split_name=split_name,
            model=model,
            alphabet=alphabet,
            device=device
        )
        print(f"  {split_name} embeddings shape: {embeddings.shape}")

    print("\nEmbedding extraction complete")
    print(f"Embeddings saved to {EMBEDDINGS_DIR}")

    # verify all files exist
    print("\nVerifying output files:")
    for split in ["train", "val", "test"]:
        emb_path = EMBEDDINGS_DIR / f"{split}_embeddings.npy"
        ids_path = EMBEDDINGS_DIR / f"{split}_ids.json"
        emb = np.load(emb_path)
        print(f"  {split}_embeddings.npy: {emb.shape} ✓")