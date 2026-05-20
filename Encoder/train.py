import os
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from encode import Encoder


# ======================================
# Amino Acid Vocabulary
# ======================================

AA_TO_ID = {
    'A':0, 'C':1, 'D':2, 'E':3,
    'F':4, 'G':5, 'H':6, 'I':7,
    'K':8, 'L':9, 'M':10, 'N':11,
    'P':12, 'Q':13, 'R':14, 'S':15,
    'T':16, 'V':17, 'W':18, 'Y':19,
    'X':20
}


# ======================================
# Dataset
# ======================================

class PeptideDataset(Dataset):

    def __init__(self, data_dir):

        self.files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npz")
        ]


    def __len__(self):
        return len(self.files)


    def encode_sequence(self, seq):

        ids = [
            AA_TO_ID.get(s, 20)
            for s in seq
        ]

        return torch.tensor(ids, dtype=torch.long)


    def __getitem__(self, idx):

        data = np.load(
            self.files[idx],
            allow_pickle=True
        )

        # ==========================
        # peptide sequence
        # ==========================

        peptide_seq = str(data["peptide_seq"])

        peptide_seq_ids = self.encode_sequence(
            peptide_seq
        )

        # ==========================
        # coordinates
        # ==========================

        peptide_coords = torch.tensor(
            data["peptide_coords"],
            dtype=torch.float32
        )

        receptor_coords = torch.tensor(
            data["receptor_coords"],
            dtype=torch.float32
        )

        # ==========================
        # torsions
        # ==========================

        backbone = torch.tensor(
            data["backbone_torsions"],
            dtype=torch.float32
        )

        sidechain = torch.tensor(
            data["sidechain_torsions"],
            dtype=torch.float32
        )

        # combine peptide features
        peptide_features = torch.cat(
            [
                peptide_coords,
                backbone,
                sidechain
            ],
            dim=-1
        )

        # ==========================
        # receptor types
        # ==========================

        receptor_types = torch.tensor(
            data["receptor_types"],
            dtype=torch.long
        )

        # ==========================
        # peptide mask
        # ==========================

        peptide_mask = torch.tensor(
            data["peptide_mask"],
            dtype=torch.float32
        )

        # ==========================
        # dummy targets
        # ==========================

        # create contact map from distance
        dist_map = torch.cdist(
            peptide_coords,
            receptor_coords
        )

        contact_map = (
            dist_map < 8.0
        ).float()

        return {

            "peptide_seq": peptide_seq_ids,

            "pep_features": peptide_features,

            "rec_features": receptor_coords,

            "receptor_types": receptor_types,

            "contact_mask": contact_map,

            "distance_map": dist_map,

            "peptide_mask": peptide_mask
        }


# ======================================
# Collate Function
# ======================================

def collate_fn(batch):

    pep_feat = [
        x["pep_features"]
        for x in batch
    ]

    rec_feat = [
        x["rec_features"]
        for x in batch
    ]

    pep_seq = [
        x["peptide_seq"]
        for x in batch
    ]

    contact = [
        x["contact_mask"]
        for x in batch
    ]

    dist = [
        x["distance_map"]
        for x in batch
    ]

    peptide_mask = [
        x["peptide_mask"]
        for x in batch
    ]

    # ==========================
    # pad sequences
    # ==========================

    pep_feat = pad_sequence(
        pep_feat,
        batch_first=True
    )

    rec_feat = pad_sequence(
        rec_feat,
        batch_first=True
    )

    pep_seq = pad_sequence(
        pep_seq,
        batch_first=True
    )

    peptide_mask = pad_sequence(
        peptide_mask,
        batch_first=True
    )

    # ==========================
    # pad matrices
    # ==========================

    max_p = max(
        x.shape[0]
        for x in contact
    )

    max_r = max(
        x.shape[1]
        for x in contact
    )

    padded_contact = []
    padded_dist = []

    for c, d in zip(contact, dist):

        cp = torch.zeros(max_p, max_r)
        dp = torch.zeros(max_p, max_r)

        cp[:c.shape[0], :c.shape[1]] = c
        dp[:d.shape[0], :d.shape[1]] = d

        padded_contact.append(cp)
        padded_dist.append(dp)

    return {

        "peptide_seq": pep_seq,

        "pep_features": pep_feat,

        "rec_features": rec_feat,

        "contact_mask": torch.stack(
            padded_contact
        ),

        "distance_map": torch.stack(
            padded_dist
        ),

        "peptide_mask": peptide_mask
    }


# ======================================
# Device
# ======================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)


# ======================================
# Loader
# ======================================

dataset = PeptideDataset(
    "../data1/processed_v4"
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)


# ======================================
# Model
# ======================================

encoder = Encoder().to(device)

optimizer = torch.optim.Adam(
    encoder.parameters(),
    lr=1e-3
)


# ======================================
# Training
# ======================================

for epoch in range(50):

    encoder.train()

    total_loss = 0

    for batch in dataloader:

        for k in batch:
            batch[k] = batch[k].to(device)

        # forward
        h_pep, h_rec = encoder(batch)

        # interaction
        pred_contact = torch.sigmoid(
            torch.matmul(
                h_pep,
                h_rec.transpose(1, 2)
            )
        )

        pred_dist = torch.matmul(
            h_pep,
            h_rec.transpose(1, 2)
        )

        # losses
        contact_loss = F.binary_cross_entropy(
            pred_contact,
            batch["contact_mask"]
        )

        dist_loss = F.mse_loss(
            pred_dist,
            batch["distance_map"]
        )

        loss = (
            contact_loss +
            0.1 * dist_loss
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1} "
        f"Loss: {total_loss/len(dataloader):.4f}"
    )