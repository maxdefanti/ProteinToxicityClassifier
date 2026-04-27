import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List

from Bio import SeqIO

from scripts.config import (
    TOXPROT_RAW,
    DBETH_RAW,
    NEGATIVES_RAW,
    POSITIVES_MERGED,
    POSITIVES_DEDUP,
    POSITIVES_FILTERED,
    NEGATIVES_DEDUP,
    NEGATIVES_FILTERED,
    CLUSTER_IDENTITY,
    POSITIVES_CLUSTERS_FASTA,
    NEGATIVES_CLUSTERS_FASTA,
    POSITIVES_CLUSTER_MAP,
    NEGATIVES_CLUSTER_MAP,
    ensure_dirs,
)

# ----------------------------------------------------------------
# 2a. Merge positive class and deduplicate at 90% identity
# This prevents data leakage from near-identical sequences
# appearing in both train and test splits
# ----------------------------------------------------------------

def merge_fasta(paths: List[Path], output_path: Path) -> Path:
    """Merge multiple fasta files, adding source label to description."""
    records = []
    for path in paths:
        source = path.stem
        for record in SeqIO.parse(path, "fasta"):
            record.description = f"{record.description} source={source}"
            records.append(record)
    SeqIO.write(records, output_path, "fasta")
    print(f"Merged {len(records)} sequences to {output_path}")
    return output_path

def _word_size_for_identity(identity: float) -> int:
    """CD-HIT word size recommendations -- below 0.4 isn't reliable."""
    if identity >= 0.70:
        return 5
    if identity >= 0.60:
        return 4
    if identity >= 0.50:
        return 3
    return 2  # 0.40-0.50 (CD-HIT minimum supported)


def run_cdhit(input_path: Path, output_path: Path,
              identity: float = 0.90) -> Path:
    """
    Cluster sequences at given identity threshold and keep
    one representative per cluster. Prevents the classifier
    from overfitting to near-duplicate sequences.

    Requires cd-hit installed: conda install -c bioconda cd-hit
    """
    cmd = [
        "cd-hit",
        "-i", str(input_path),
        "-o", str(output_path),
        "-c", str(identity),
        "-n", str(_word_size_for_identity(identity)),
        "-M", "4000",  # memory limit MB
        "-T", "4",     # threads
        "-d", "0",     # full sequence description in output
    ]
    print(f"Running CD-HIT at {identity} identity threshold...")
    subprocess.run(cmd, check=True, capture_output=True)

    n = sum(1 for _ in SeqIO.parse(output_path, "fasta"))
    print(f"After deduplication: {n} sequences")
    return output_path


def parse_cdhit_clusters(clstr_path: Path) -> Dict[str, int]:
    """Parse a CD-HIT .clstr file and return {seq_id: cluster_id}.

    The .clstr format looks like:
        >Cluster 0
        0\t234aa, >sp|P12345|TOXIN_X... *
        1\t230aa, >sp|P67890|TOXIN_Y... at 75.43%
    """
    seq_to_cluster: Dict[str, int] = {}
    current_cluster = -1
    pattern = re.compile(r">([^.]+)\.\.\.")
    with open(clstr_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
            else:
                m = pattern.search(line)
                if m:
                    seq_to_cluster[m.group(1)] = current_cluster
    return seq_to_cluster


def cluster_for_splitting(input_fasta: Path, output_fasta: Path,
                          cluster_map_json: Path,
                          identity: float = CLUSTER_IDENTITY) -> Dict[str, int]:
    """Cluster at `identity` (typically 0.5) and dump {seq_id: cluster_id}.

    The output FASTA holds cluster representatives (unused by splits.py); the
    JSON is what splits.py reads to assign each sequence to a cluster.
    """
    run_cdhit(input_fasta, output_fasta, identity=identity)
    clstr_path = output_fasta.with_suffix(output_fasta.suffix + ".clstr")
    cluster_map = parse_cdhit_clusters(clstr_path)
    cluster_map_json.write_text(json.dumps(cluster_map, indent=2))
    n_clusters = len(set(cluster_map.values()))
    print(f"  -> {len(cluster_map)} sequences in {n_clusters} clusters "
          f"(saved to {cluster_map_json.name})")
    return cluster_map

def filter_length(input_path: Path, output_path: Path,
                  min_len: int = 50, max_len: int = 1000) -> Path:
    """
    Filter sequences by length. ESM-2 handles long sequences
    poorly without chunking, and very short sequences have
    unreliable embeddings. Toxins tend to be short-to-medium
    length so this filter is not very aggressive.
    """
    records = [
        r for r in SeqIO.parse(input_path, "fasta")
        if min_len <= len(r.seq) <= max_len
    ]
    SeqIO.write(records, output_path, "fasta")
    print(f"After length filter ({min_len}-{max_len}): {len(records)} sequences")
    return output_path


if __name__ == "__main__":
    ensure_dirs()

    # positives: merge → dedup at 90% → length filter → cluster at 50%
    merge_fasta([TOXPROT_RAW, DBETH_RAW], POSITIVES_MERGED)
    run_cdhit(POSITIVES_MERGED, POSITIVES_DEDUP, identity=0.90)
    filter_length(POSITIVES_DEDUP, POSITIVES_FILTERED)
    cluster_for_splitting(POSITIVES_FILTERED, POSITIVES_CLUSTERS_FASTA,
                          POSITIVES_CLUSTER_MAP)

    # negatives: dedup at 90% → length filter → cluster at 50%
    run_cdhit(NEGATIVES_RAW, NEGATIVES_DEDUP, identity=0.90)
    filter_length(NEGATIVES_DEDUP, NEGATIVES_FILTERED)
    cluster_for_splitting(NEGATIVES_FILTERED, NEGATIVES_CLUSTERS_FASTA,
                          NEGATIVES_CLUSTER_MAP)
