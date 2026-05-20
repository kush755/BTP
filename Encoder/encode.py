import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Basic blocks
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.ffn = MLP(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ffn(x))
        return x


# -----------------------------
# Distance-aware Cross Attention
# -----------------------------
class CrossAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)

    def forward(self, h_pep, h_rec, distance_map):

        Q = self.q(h_pep)             # (L, d)
        K = self.k(h_rec)             # (N, d)
        V = self.v(h_rec)             # (N, d)

        attn_logits = Q @ K.transpose(0, 1) / (Q.shape[-1] ** 0.5)

        # distance bias
        distance_bias = -distance_map / 5.0
        attn_logits = attn_logits + distance_bias

        attn = torch.softmax(attn_logits, dim=-1)

        context = attn @ V

        return h_pep + context


# -----------------------------
# Full Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, d=128):

        super().__init__()

        # embeddings
        self.seq_embed = nn.Embedding(20, d)
        self.atom_embed = nn.Embedding(4, d)

        self.coord_embed = MLP(3, d)
        self.torsion_embed = MLP(12, d)

        # encoders
        self.pep_encoder = SelfAttentionBlock(d)
        self.rec_encoder = SelfAttentionBlock(d)

        self.cross_attn = CrossAttention(d)

    def forward(self, batch):

        # unpack
        peptide_coords = batch["peptide_coords"]
        receptor_coords = batch["receptor_coords"]
        seq_ids = batch["seq_ids"]
        torsions = batch["torsions"]
        atom_types = batch["receptor_types"]

        # embeddings
        h_pep = (
            self.seq_embed(seq_ids) +
            self.coord_embed(peptide_coords) +
            self.torsion_embed(torsions)
        )

        h_rec = (
            self.atom_embed(atom_types) +
            self.coord_embed(receptor_coords)
        )

        # self encoding
        h_pep = self.pep_encoder(h_pep)
        h_rec = self.rec_encoder(h_rec)

        # compute distance map ON-THE-FLY
        D = torch.cdist(peptide_coords, receptor_coords)

        # cross attention
        h_pep = self.cross_attn(h_pep, h_rec, D)

        return h_pep, h_rec