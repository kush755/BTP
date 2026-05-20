import os
from pathlib import Path
import pandas as pd

# =========================
# CONFIG
# =========================
DATA_DIR = Path("data1")
OUTPUT_FILE = DATA_DIR / "dataset.csv"

MAX_LEN = 150
MIN_LEN = 3
PAD_TOKEN = "X"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

FILES = [
    "dbaasp.fasta",
    "dbamp.fasta",
    "general_amps.fasta"
]

# =========================
# FASTA READER
# =========================
def read_fasta(file_path):
    sequences = []
    seq = ""

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line

        if seq:
            sequences.append(seq)

    return sequences


# =========================
# CLEANING
# =========================
def clean_sequence(seq):
    seq = seq.upper()
    return "".join([c for c in seq if c in VALID_AA])


def process_sequence(seq):
    if len(seq) < MIN_LEN or len(seq) > MAX_LEN:
        return None

    return seq 


# =========================
# MAIN PIPELINE
# =========================
def load_sequences():
    all_seqs = []

    for file in FILES:
        path = DATA_DIR / file

        if not path.exists():
            print(f"Warning: {file} not found")
            continue

        print(f"Loading: {file}")
        seqs = read_fasta(path)
        print(f"  → {len(seqs)} sequences")

        all_seqs.extend(seqs)

    print(f"\nTotal raw sequences: {len(all_seqs)}")
    return all_seqs


def preprocess():
    # Load
    seqs = load_sequences()

    # Clean
    seqs = [clean_sequence(s) for s in seqs]
    seqs = [s for s in seqs if len(s) > 0]
    print(f"After cleaning: {len(seqs)}")

    # Deduplicate
    seqs = list(set(seqs))
    print(f"After deduplication: {len(seqs)}")

    # Filter + Pad
    processed = []
    for s in seqs:
        p = process_sequence(s)
        if p:
            processed.append(p)

    print(f"Final dataset size: {len(processed)}")

    return processed


# =========================
# SAVE
# =========================
def save_dataset(seqs):
    df = pd.DataFrame({"sequence": seqs})
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved dataset to: {OUTPUT_FILE}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    dataset = preprocess()
    save_dataset(dataset)