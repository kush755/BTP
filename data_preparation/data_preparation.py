import os
import numpy as np
from glob import glob
import mdtraj as md

ATOM_TYPES = ["C", "N", "O", "S"]
ATOM_MAP = {a: i for i, a in enumerate(ATOM_TYPES)}

RECEPTOR_CUTOFF = 12.0


# -----------------------------
# Extract peptide data (FIXED)
# -----------------------------
def extract_peptide_data(pdb_file):

    traj = md.load(pdb_file)

    peptide_coords = traj.xyz[0]

    # ✅ FIX: convert generator → list
    residues = list(traj.topology.residues)
    peptide_seq = [res.name for res in residues]
    L = len(residues)

    # -------------------------
    # Backbone torsions (FIXED)
    # -------------------------
    phi_idx, phi_val = md.compute_phi(traj)
    psi_idx, psi_val = md.compute_psi(traj)

    backbone = np.zeros((L, 2), dtype=np.float32)

    # ✅ FIX: correct residue mapping
    for i, atom_indices in enumerate(phi_idx):
        atom = traj.topology.atom(atom_indices[1])
        res_id = atom.residue.index
        if res_id < L:
            backbone[res_id, 0] = phi_val[0][i]

    for i, atom_indices in enumerate(psi_idx):
        atom = traj.topology.atom(atom_indices[1])
        res_id = atom.residue.index
        if res_id < L:
            backbone[res_id, 1] = psi_val[0][i]

    # -------------------------
    # Sidechain torsions (FIXED)
    # -------------------------
    sidechain = np.zeros((L, 4), dtype=np.float32)

    chi_functions = [
        md.compute_chi1,
        md.compute_chi2,
        md.compute_chi3,
        md.compute_chi4
    ]

    for chi_i, fn in enumerate(chi_functions):
        try:
            chi_idx, chi_val = fn(traj)

            for j, atom_indices in enumerate(chi_idx):
                atom = traj.topology.atom(atom_indices[1])
                res_id = atom.residue.index

                if res_id < L:
                    sidechain[res_id, chi_i] = chi_val[0][j]

        except Exception:
            continue

    return peptide_coords, np.array(peptide_seq), backbone, sidechain


# -----------------------------
# Extract receptor data
# -----------------------------
def extract_receptor_data(pdb_file, peptide_coords):

    traj = md.load(pdb_file)

    coords = traj.xyz[0]

    types = []
    for atom in traj.topology.atoms:
        element = atom.element.symbol if atom.element else "C"
        types.append(ATOM_MAP.get(element, 0))

    coords = np.array(coords)
    types = np.array(types)

    # 🔥 filter receptor
    center = peptide_coords.mean(axis=0)
    dist = np.linalg.norm(coords - center, axis=1)

    mask = dist < RECEPTOR_CUTOFF

    return coords[mask], types[mask]


# -----------------------------
# Build one sample
# -----------------------------
def build_sample(folder):
    pep_files = glob(os.path.join(folder, "*pep*.pdb"))
    rec_files = glob(os.path.join(folder, "*rec*.pdb"))

    if len(pep_files) == 0 or len(rec_files) == 0:
        raise ValueError("Missing peptide/receptor file")

    pep_file = pep_files[0]
    rec_file = rec_files[0]

    # peptide
    peptide_coords, peptide_seq, backbone, sidechain = extract_peptide_data(pep_file)

    if len(peptide_coords) == 0:
        raise ValueError("Empty peptide")

    # receptor
    receptor_coords, receptor_types = extract_receptor_data(rec_file, peptide_coords)

    # -----------------------------
    # NORMALIZATION
    # -----------------------------
    center = peptide_coords.mean(axis=0)

    peptide_coords -= center
    receptor_coords -= center

    # -----------------------------
    # compression
    # -----------------------------
    peptide_coords = peptide_coords.astype(np.float16)
    receptor_coords = receptor_coords.astype(np.float16)

    peptide_mask = np.ones(len(peptide_seq), dtype=np.float32)

    return {
        "peptide_coords": peptide_coords,
        "peptide_seq": peptide_seq,
        "backbone_torsions": backbone,
        "sidechain_torsions": sidechain,
        "receptor_coords": receptor_coords,
        "receptor_types": receptor_types,
        "peptide_mask": peptide_mask
    }


# -----------------------------
# Build dataset
# -----------------------------
def build_dataset(root_dir, save_folder):
    folders = sorted(glob(os.path.join(root_dir, "*")))

    print(f"Found {len(folders)} folders")

    os.makedirs(save_folder, exist_ok=True)

    for folder in folders:
        try:
            sample = build_sample(folder)

            save_path = os.path.join(
                save_folder,
                os.path.basename(folder) + ".npz"
            )

            np.savez_compressed(save_path, **sample)

            print(f"Saved {save_path}")

        except Exception as e:
            print(f"Skipping {folder}: {e}")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    build_dataset(
        root_dir="data1/pepbdb-20200318/pepbdb",
        save_folder="data1/processed_v4"
    )