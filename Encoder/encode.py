import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
        num_layers=3,
        num_heads=8
    ):

        super().__init__()

        # =====================================
        # sequence embedding
        # =====================================

        self.seq_embedding = nn.Embedding(
            21,
            hidden_dim
        )

        # =====================================
        # peptide feature projection
        # input:
        # backbone(2) + sidechain(4)
        # =====================================

        self.pep_proj = nn.Linear(
            6,
            hidden_dim
        )

        # =====================================
        # receptor projection
        # xyz coords -> hidden
        # =====================================

        self.rec_proj = nn.Linear(
            3,
            hidden_dim
        )

        # =====================================
        # peptide encoder
        # =====================================

        pep_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.pep_encoder = nn.TransformerEncoder(
            pep_layer,
            num_layers=num_layers
        )

        # =====================================
        # receptor encoder
        # =====================================

        rec_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.rec_encoder = nn.TransformerEncoder(
            rec_layer,
            num_layers=num_layers
        )

    def forward(self, batch):

        # =====================================
        # peptide inputs
        # =====================================

        peptide_seq = batch["peptide_seq"]

        pep_features = batch["pep_features"]

        peptide_mask = batch["peptide_mask"]

        # =====================================
        # receptor inputs
        # =====================================

        receptor_coords = batch["rec_features"]

        # =====================================
        # embeddings
        # =====================================

        seq_embed = self.seq_embedding(
            peptide_seq
        )

        feat_embed = self.pep_proj(
            pep_features
        )

        h_pep = seq_embed + feat_embed

        h_rec = self.rec_proj(
            receptor_coords
        )

        # =====================================
        # masks
        # =====================================

        pep_padding_mask = (
            peptide_mask == 0
        )

        rec_padding_mask = (
            receptor_coords.abs().sum(dim=-1) == 0
        )

        # =====================================
        # transformer encoding
        # =====================================

        h_pep = self.pep_encoder(
            h_pep,
            src_key_padding_mask=pep_padding_mask
        )

        h_rec = self.rec_encoder(
            h_rec,
            src_key_padding_mask=rec_padding_mask
        )

        return h_pep, h_rec