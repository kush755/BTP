import os
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_L = 30
MAX_N = 128

class PeptideDatasetV2(Dataset):

    def __init__(self, folder):
        self.files = os.listdir(folder)
        self.folder = folder

    def pad(self, arr, shape):
        out = np.zeros(shape)
        slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
        out[slices] = arr[slices]
        return out

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = np.load(os.path.join(self.folder, self.files[idx]), allow_pickle=True)

        peptide_coords = self.pad(data["peptide_coords"], (MAX_L, 3))
        backbone = self.pad(data["backbone_torsions"], (MAX_L, 2))
        sidechain = self.pad(data["sidechain_torsions"], (MAX_L, 4))
        peptide_mask = self.pad(data["peptide_mask"], (MAX_L,))

        receptor_coords = self.pad(data["receptor_coords"], (MAX_N, 3))
        receptor_types = self.pad(data["receptor_types"], (MAX_N,))

        distance_map = self.pad(data["distance_map"], (MAX_L, MAX_N))
        contact_mask = self.pad(data["contact_mask"], (MAX_L, MAX_N))

        return {
            "peptide_coords": torch.tensor(peptide_coords, dtype=torch.float32),
            "backbone_torsions": torch.tensor(backbone, dtype=torch.float32),
            "sidechain_torsions": torch.tensor(sidechain, dtype=torch.float32),
            "peptide_mask": torch.tensor(peptide_mask, dtype=torch.float32),

            "receptor_coords": torch.tensor(receptor_coords, dtype=torch.float32),
            "receptor_types": torch.tensor(receptor_types, dtype=torch.long),

            "distance_map": torch.tensor(distance_map, dtype=torch.float32),
            "contact_mask": torch.tensor(contact_mask, dtype=torch.float32)
        }