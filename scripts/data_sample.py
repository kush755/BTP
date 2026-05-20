import numpy as np
import os

folder = "data1/processed_v4"

# 🔥 total samples
files = [f for f in os.listdir(folder) if f.endswith(".npz")]
print("Total samples:", len(files))


# -----------------------------
# Load one sample
# -----------------------------
file = os.path.join(folder, "1a2x_B.npz")

data = np.load(file)

print("\nKeys:", data.files)

print("peptide_coords:", data["peptide_coords"].shape)
print("backbone:", data["backbone_torsions"].shape)
print("sidechain:", data["sidechain_torsions"].shape)
print("receptor:", data["receptor_coords"].shape)
print("types:", data["receptor_types"].shape)
print("mask:", data["peptide_mask"].shape)

print("\nSequence:", data["peptide_seq"])
print("\nFirst 5 coords:\n", data["peptide_coords"][:5])

np.set_printoptions(precision=3, suppress=True, linewidth=120)

print("\nBackbone torsions (phi, psi):")
for i, row in enumerate(data["backbone_torsions"]):
    print(f"Residue {i}: {row}")

print("\nSidechain torsions (chi1–chi4):")
for i, row in enumerate(data["sidechain_torsions"]):
    print(f"Residue {i}: {row}")

print("\nReceptor atom types:")
print(data["receptor_types"][:100])

print("\nSequence:")
print(" ".join(data["peptide_seq"]))