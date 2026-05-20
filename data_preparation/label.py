import os
import numpy as np
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
processed_dir = "data1/processed_v4"
dataset_path = "data1/dataset.csv"
inactive_path = "data1/inactive.csv"

# -----------------------------
# Load active & inactive sets
# -----------------------------
active_df = pd.read_csv(dataset_path, header=None)
active_set = set(active_df[0].dropna().str.strip())

inactive_df = pd.read_csv(inactive_path)
inactive_set = set(inactive_df["sequence"].dropna().str.strip())

print("Active:", len(active_set))
print("Inactive:", len(inactive_set))


# -----------------------------
# 3-letter → 1-letter mapping
# -----------------------------
aa_map = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

def convert_3_to_1(seq_array):
    return ''.join([aa_map.get(res, 'X') for res in seq_array])


# -----------------------------
# Label function
# -----------------------------
def get_label(seq_3):
    seq_1 = convert_3_to_1(seq_3)

    if seq_1 in active_set:
        return 1
    elif seq_1 in inactive_set:
        return 0
    else:
        return -1


# -----------------------------
# Process all .npz files
# -----------------------------
output_dir = "data1/processed_v4_labeled"
os.makedirs(output_dir, exist_ok=True)

label_counts = {1: 0, 0: 0, -1: 0}

for file in os.listdir(processed_dir):
    if not file.endswith(".npz"):
        continue

    file_path = os.path.join(processed_dir, file)
    data = np.load(file_path, allow_pickle=True)

    # Extract peptide sequence
    peptide_seq = data["peptide_seq"]  # array of 3-letter AA

    # Get label
    label = get_label(peptide_seq)
    label_counts[label] += 1

    # Save new file with label added
    save_path = os.path.join(output_dir, file)

    np.savez(
        save_path,
        peptide_coords=data["peptide_coords"],
        peptide_seq=data["peptide_seq"],
        backbone_torsions=data["backbone_torsions"],
        sidechain_torsions=data["sidechain_torsions"],
        receptor_coords=data["receptor_coords"],
        receptor_types=data["receptor_types"],
        peptide_mask=data["peptide_mask"],
        activity_label=label   # ✅ NEW FIELD
    )

print("\nLabel distribution:")
print(label_counts)