import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from Bio import SeqIO

from scripts.config import (
    POSITIVES_FILTERED,
    NEGATIVES_FILTERED,
    POSITIVES_CLUSTER_MAP,
    NEGATIVES_CLUSTER_MAP,
    TRAIN_POSITIVES_FOR_AUG,
    TRAIN_POSITIVE_IDS,
    TRAIN_NATURAL_CSV,
    VAL_NATURAL_CSV,
    TEST_NATURAL_CSV,
    REDESIGNS_ALL_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    SPLIT_SUMMARY,
    TEMPERATURES,
    TM_SCORE_THRESHOLD,
    CLUSTER_IDENTITY,
    SPLIT_FRACTIONS,
    ensure_dirs,
)

# ----------------------------------------------------------------
# CRITICAL: split must happen before ProteinMPNN augmentation
# to prevent leakage. Toxin structures used for redesign must
# only come from the training split.
# ----------------------------------------------------------------

@dataclass
class SequenceRecord:
    seq_id: str
    sequence: str
    label: int  # 1 = toxin, 0 = non-toxin
    source: str
    confidence: str  # "verified", "structural_proxy", "uncertain"
    divergence_level: str  # "natural", "high_sim", "med_sim", "low_sim"

def load_fasta_as_records(path: Path, label: int,
                           source: str) -> List[SequenceRecord]:
    records = []
    for r in SeqIO.parse(path, "fasta"):
        records.append(SequenceRecord(
            seq_id=r.id,
            sequence=str(r.seq),
            label=label,
            source=source,
            confidence="verified" if label == 1 else "verified_negative",
            divergence_level="natural"
        ))
    return records

def _split_by_cluster(records: List[SequenceRecord],
                      cluster_map: dict,
                      train_frac: float, val_frac: float,
                      seed: int) -> Tuple[List, List, List]:
    """Split records so that whole clusters go to a single split.

    Any record whose seq_id isn't in cluster_map gets its own singleton cluster
    (shouldn't happen if preprocess.py was run, but keeps the function safe).
    """
    rng = random.Random(seed)

    # group records by cluster id
    clusters: dict = {}
    next_singleton = max(cluster_map.values(), default=-1) + 1
    for r in records:
        cid = cluster_map.get(r.seq_id)
        if cid is None:
            cid = next_singleton
            next_singleton += 1
        clusters.setdefault(cid, []).append(r)

    cluster_ids = list(clusters.keys())
    rng.shuffle(cluster_ids)

    # walk through shuffled clusters, fill train then val then test
    target_train = train_frac * len(records)
    target_val = (train_frac + val_frac) * len(records)

    train, val, test = [], [], []
    for cid in cluster_ids:
        members = clusters[cid]
        if len(train) < target_train:
            train.extend(members)
        elif len(train) + len(val) < target_val:
            val.extend(members)
        else:
            test.extend(members)
    return train, val, test


def stratified_split(records: List[SequenceRecord],
                     seed: int = 42) -> Tuple[List, List, List]:
    """Cluster-level stratified split honoring per-class SPLIT_FRACTIONS.

    Positives and negatives are split independently (preserves class balance)
    and within each class whole CD-HIT clusters go to one split. This prevents
    homology leakage: any two sequences across splits share <CLUSTER_IDENTITY
    identity.
    """
    for cls, fracs in SPLIT_FRACTIONS.items():
        total = fracs["train"] + fracs["val"] + fracs["test"]
        assert abs(total - 1.0) < 1e-6, f"SPLIT_FRACTIONS[{cls}] must sum to 1.0"

    pos_clusters = json.loads(POSITIVES_CLUSTER_MAP.read_text())
    neg_clusters = json.loads(NEGATIVES_CLUSTER_MAP.read_text())

    positives = [r for r in records if r.label == 1]
    negatives = [r for r in records if r.label == 0]

    pos_fracs = SPLIT_FRACTIONS["positive"]
    neg_fracs = SPLIT_FRACTIONS["negative"]
    pos_train, pos_val, pos_test = _split_by_cluster(
        positives, pos_clusters, pos_fracs["train"], pos_fracs["val"], seed)
    neg_train, neg_val, neg_test = _split_by_cluster(
        negatives, neg_clusters, neg_fracs["train"], neg_fracs["val"], seed + 1)

    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    print(f"Cluster-level split at {CLUSTER_IDENTITY:.0%} identity:")
    print(f"  toxin fractions:    {pos_fracs}")
    print(f"  harmless fractions: {neg_fracs}")
    print(f"  Train: {len(train)} ({sum(r.label for r in train)} positive)")
    print(f"  Val:   {len(val)} ({sum(r.label for r in val)} positive)")
    print(f"  Test:  {len(test)} ({sum(r.label for r in test)} positive)")

    return train, val, test


def records_to_df(records: List[SequenceRecord]) -> pd.DataFrame:
    return pd.DataFrame([{
        "seq_id": r.seq_id,
        "sequence": r.sequence,
        "label": r.label,
        "source": r.source,
        "confidence": r.confidence,
        "divergence_level": r.divergence_level,
        "tm_score": None,
        "is_augmented": False
    } for r in records])


def split_natural() -> None:
    """
    Stage 3: split natural sequences into train/val/test and emit the
    artifacts that augment.py consumes (training-positive IDs and the
    FASTA of training toxin sequences for structure retrieval).

    val.csv and test.csv are written here in their final form -- they
    contain only natural sequences and are never touched again.
    train_natural.csv is an intermediate that finalize_splits() reads
    after augmentation to produce the final train.csv.
    """
    positives = load_fasta_as_records(POSITIVES_FILTERED, label=1, source="toxprot_dbeth")
    negatives = load_fasta_as_records(NEGATIVES_FILTERED, label=0, source="swissprot")

    all_natural = positives + negatives
    train_natural, val_natural, test_natural = stratified_split(all_natural)

    # the IDs of training positives -- these are the ONLY sequences whose
    # structures may be used for ProteinMPNN augmentation
    train_positive_ids = sorted({r.seq_id for r in train_natural if r.label == 1})

    # FASTA of training toxin sequences for structure retrieval
    with open(TRAIN_POSITIVES_FOR_AUG, "w") as f:
        for r in train_natural:
            if r.label == 1:
                f.write(f">{r.seq_id}\n{r.sequence}\n")

    # JSON list of UniProt IDs consumed by augment.py
    with open(TRAIN_POSITIVE_IDS, "w") as f:
        json.dump(train_positive_ids, f)

    print(f"\nTraining positive IDs saved: {len(train_positive_ids)}")
    print("These are the only sequences whose structures can be used for redesign")

    # write natural-only intermediates — finalize_splits() produces the finals
    records_to_df(train_natural).to_csv(TRAIN_NATURAL_CSV, index=False)
    records_to_df(val_natural).to_csv(VAL_NATURAL_CSV, index=False)
    records_to_df(test_natural).to_csv(TEST_NATURAL_CSV, index=False)


def finalize_splits() -> None:
    """
    Stage 5: merge ProteinMPNN redesigns into all three splits.

    REDESIGNS_ALL_CSV has a 'split' column ('train'/'val'/'test') derived from
    which natural split the parent structure's sequence belongs to. This keeps
    redesigns of any given parent entirely within one split, preventing leakage.
    """
    natural = {
        "train": pd.read_csv(TRAIN_NATURAL_CSV),
        "val":   pd.read_csv(VAL_NATURAL_CSV),
        "test":  pd.read_csv(TEST_NATURAL_CSV),
    }

    needed = ["seq_id", "sequence", "label", "divergence_level", "tm_score", "split"]
    if REDESIGNS_ALL_CSV.exists():
        all_redesigns = pd.read_csv(REDESIGNS_ALL_CSV)
        has_needed = all_redesigns is not None and not all_redesigns.empty and \
                     all(c in all_redesigns.columns for c in needed)
    else:
        has_needed = False

    out_dfs = {}
    for split_name, nat_df in natural.items():
        if has_needed:
            redesigns = all_redesigns[all_redesigns["split"] == split_name].copy()
            aug = redesigns[["seq_id", "sequence", "label",
                             "divergence_level", "tm_score"]].copy()
            aug["source"] = "proteinmpnn_redesign"
            aug["confidence"] = "structural_proxy"
            aug["is_augmented"] = True
            combined = pd.concat([nat_df, aug], ignore_index=True)
        else:
            combined = nat_df.copy()
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        out_dfs[split_name] = combined

    out_dfs["train"].to_csv(TRAIN_CSV, index=False)
    out_dfs["val"].to_csv(VAL_CSV, index=False)
    out_dfs["test"].to_csv(TEST_CSV, index=False)

    def _stats(df):
        aug = df.get("is_augmented", pd.Series(False, index=df.index))
        return {
            "total": len(df),
            "positive": int(df.label.sum()),
            "negative": int((df.label == 0).sum()),
            "natural_positive": int(((df.label == 1) & (~aug)).sum()),
            "augmented_positive": int(((df.label == 1) & aug).sum()),
            "augmented_by_divergence": {
                level: int(((df.get("divergence_level", pd.Series()) == level) & aug).sum())
                for level in TEMPERATURES.keys()
            },
        }

    summary = {s: _stats(df) for s, df in out_dfs.items()}
    summary["tm_score_threshold"] = TM_SCORE_THRESHOLD
    summary["augmentation_temperatures"] = TEMPERATURES

    with open(SPLIT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal dataset summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ensure_dirs()
    split_natural()
