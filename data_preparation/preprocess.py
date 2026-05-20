import os
import numpy as np
import pandas as pd
import mdtraj as md

# ==========================
# CONFIG
# ==========================
CSV_FILE = "metadata/pepbdb.csv"
PDB_FOLDER = "raw_pdb"
SAVE_FOLDER = "processed_v2"

DISTANCE_CUTOFF = 8.0

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ==========================
# ATOM TYPE ENCODING
# ==========================
ATOM_TYPES = ["C", "N", "O", "S"]
ATOM_MAP = {a: i for i, a in enumerate(ATOM_TYPES)}

# ==========================
# LOAD CSV
# ==========================
metadata = pd.read_csv(CSV_FILE)

# ==========================
# MAIN LOOP
# ==========================
for idx, row in metadata.iterrows():

    pdb_id = row["PDB_ID"]
    peptide_chain = row["PEPTIDE_CHAIN"]
    receptor_chain = row["RECEPTOR_CHAIN"]

    pdb_file = os.path.join(PDB_FOLDER, f"{pdb_id}.pdb")

    if not os.path.exists(pdb_file):
        continue

    try:
        traj = md.load(pdb_file)
        topology = traj.topology

        # ==========================
        # SELECT ATOMS
        # ==========================
        pep_atoms = topology.select(f"chainid {peptide_chain}")
        rec_atoms = topology.select(f"chainid {receptor_chain}")

        pep_traj = traj.atom_slice(pep_atoms)
        rec_traj = traj.atom_slice(rec_atoms)

        # ==========================
        # PEPTIDE FEATURES
        # ==========================
        peptide_coords = pep_traj.xyz[0]   # (num_atoms, 3)

        peptide_seq = []
        residue_indices = []

        for res in pep_traj.topology.residues:
            peptide_seq.append(res.name)
            residue_indices.append(res.index)

        L = len(residue_indices)

        # ==========================
        # BACKBONE TORSIONS
        # ==========================
        phi = md.compute_phi(pep_traj)[1]
        psi = md.compute_psi(pep_traj)[1]

        backbone = np.zeros((L, 2))
        backbone[:len(phi), 0] = phi[0]
        backbone[:len(psi), 1] = psi[0]

        # ==========================
        # SIDECHAIN TORSIONS
        # ==========================
        chi_list = []

        for fn in [md.compute_chi1, md.compute_chi2, md.compute_chi3, md.compute_chi4]:
            try:
                chi = fn(pep_traj)[1]
                arr = np.zeros(L)
                arr[:len(chi)] = chi[0]
                chi_list.append(arr)
            except:
                chi_list.append(np.zeros(L))

        sidechain = np.stack(chi_list, axis=1)  # (L, 4)

        # ==========================
        # RECEPTOR FEATURES
        # ==========================
        receptor_coords = rec_traj.xyz[0]

        receptor_types = []

        for atom in rec_traj.topology.atoms:
            element = atom.element.symbol if atom.element else "C"
            receptor_types.append(ATOM_MAP.get(element, 0))

        receptor_types = np.array(receptor_types)

        # ==========================
        # DISTANCE MAP
        # ==========================
        distance_map = np.linalg.norm(
            peptide_coords[:, None, :] - receptor_coords[None, :, :],
            axis=-1
        )

        contact_mask = (distance_map < DISTANCE_CUTOFF).astype(np.float32)

        # ==========================
        # MASKS
        # ==========================
        peptide_mask = np.ones(L)

        # ==========================
        # SAVE
        # ==========================
        save_path = os.path.join(SAVE_FOLDER, f"{pdb_id}.npz")

        np.savez(
            save_path,
            peptide_coords=peptide_coords,
            peptide_seq=np.array(peptide_seq),
            backbone_torsions=backbone,
            sidechain_torsions=sidechain,
            receptor_coords=receptor_coords,
            receptor_types=receptor_types,
            distance_map=distance_map,
            contact_mask=contact_mask,
            peptide_mask=peptide_mask
        )

        print(f"Saved {pdb_id}")

    except Exception as e:
        print(f"Error {pdb_id}: {e}")