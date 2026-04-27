from pathlib import Path

# repo layout:
#   <repo>/scripts/   - this package (pipeline code)
#   <repo>/data/      - generated artifacts (created on first run)
# Anchored to __file__ so paths resolve correctly regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = REPO_ROOT / "data"
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
STRUCTURES_DIR = BASE_DIR / "structures"
REDESIGNS_DIR = BASE_DIR / "redesigns"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
SPLITS_DIR = BASE_DIR / "splits"

# external tools
PROTEINMPNN_DIR = Path("ProteinMPNN")  # path to cloned ProteinMPNN repo

# augmentation parameters
TEMPERATURES = {
    "high_sim": 0.1,   # ~85% identity to original
    "med_sim": 0.5,    # ~55-65% identity
    "low_sim": 1.0,    # ~35-50% identity
}
N_SEQUENCES_PER_TEMP = 3   # per parent per temperature (so 9 redesigns per parent)

# Per-class train/val/test split fractions. Each row must sum to 1.0.
# Set toxin and harmless rows differently if you want, e.g., a more
# aggressive test split for toxins to stress-test detection of novel toxins.
SPLIT_FRACTIONS = {
    "positive": {"train": 0.70, "val": 0.15, "test": 0.15},
    "negative": {"train": 0.70, "val": 0.15, "test": 0.15},
}


# Per-split, per-class parent selection. For each split, what fraction of
# the available PDB-having sequences in that split should be used as parents
# for ProteinMPNN redesigns. Set None to fall back to TARGET_*_PDBS as a
# global cap (split-proportional).
#
# Example: {"train": {"positive": 0.30, "negative": 0.10}, ...}
#   -> in the train split, 30% of PDB-having toxins and 10% of PDB-having
#      harmless sequences become parents for synthetic children.
PARENT_FRACTIONS = {
    "train": {"positive": 0.30, "negative": 0.30},
    "val":   {"positive": 0.30, "negative": 0.30},
    "test":  {"positive": 0.30, "negative": 0.30},
}

# Optional global caps -- only used if PARENT_FRACTIONS is None for a split.
TARGET_TOXIN_PDBS = 100
TARGET_HARMLESS_PDBS = 100

PDB_LOOKUP_CACHE = "data/processed/pdb_lookup_cache.json"

# Target synthetic-to-natural ratio per class, used as a sanity check.
# The actual synthetic count = parents * N_SEQUENCES_PER_TEMP * len(TEMPERATURES).
#
# Recommended ranges:
#   positive (toxin):    0.3 - 1.0   (augmentation is the point for minority class)
#   negative (harmless): 0.1 - 0.3   (mainly to prevent "synthetic = positive" shortcut)
TARGET_SYNTHETIC_RATIO = {
    "positive": 0.4,
    "negative": 0.2,
}

TM_SCORE_THRESHOLD = 0.70  # legacy -- ESMFold/TM-score filter is now off by default

# raw inputs
TOXPROT_RAW = RAW_DIR / "toxprot.fasta"
DBETH_RAW = RAW_DIR / "dbeth.fasta"
NEGATIVES_RAW = RAW_DIR / "negatives.fasta"

# preprocess outputs
POSITIVES_MERGED = PROCESSED_DIR / "positives_merged.fasta"
POSITIVES_DEDUP = PROCESSED_DIR / "positives_dedup.fasta"
POSITIVES_FILTERED = PROCESSED_DIR / "positives_filtered.fasta"
NEGATIVES_DEDUP = PROCESSED_DIR / "negatives_dedup.fasta"
NEGATIVES_FILTERED = PROCESSED_DIR / "negatives_filtered.fasta"

# Cluster assignments from CD-HIT at CLUSTER_IDENTITY -- one JSON per class.
# Used by splits.py to split whole clusters (no homology leakage between
# train/val/test).
CLUSTER_IDENTITY = 0.50  # any pair across splits will share <50% identity
POSITIVES_CLUSTERS_FASTA = PROCESSED_DIR / "positives_clusters.fasta"
NEGATIVES_CLUSTERS_FASTA = PROCESSED_DIR / "negatives_clusters.fasta"
POSITIVES_CLUSTER_MAP = PROCESSED_DIR / "positives_cluster_map.json"
NEGATIVES_CLUSTER_MAP = PROCESSED_DIR / "negatives_cluster_map.json"

# splits / augmentation handoff artifacts
TRAIN_POSITIVES_FOR_AUG = PROCESSED_DIR / "train_positives_for_augmentation.fasta"
TRAIN_POSITIVE_IDS = PROCESSED_DIR / "train_positive_ids.json"
TRAIN_NATURAL_CSV = PROCESSED_DIR / "train_natural.csv"
VAL_NATURAL_CSV = PROCESSED_DIR / "val_natural.csv"
TEST_NATURAL_CSV = PROCESSED_DIR / "test_natural.csv"

# augment outputs — one CSV covering all splits (has a "split" column)
REDESIGNS_ALL_CSV = PROCESSED_DIR / "redesigns_all.csv"

# final split outputs
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"
SPLIT_SUMMARY = SPLITS_DIR / "split_summary.json"


def ensure_dirs() -> None:
    for d in [RAW_DIR, PROCESSED_DIR, STRUCTURES_DIR, REDESIGNS_DIR,
              EMBEDDINGS_DIR, SPLITS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
