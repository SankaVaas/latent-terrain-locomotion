"""
models/encoder.py
Two encoders:

ObsEncoder
──────────
Maps the 49-dim proprioceptive observation into a dense embedding
fed into the RSSM posterior network.

Architecture: Linear → LayerNorm → ELU (×N layers)
No CNN needed — observation is a flat vector, not an image.

TerrainEncoder
──────────────
Maps terrain probe heights (16-dim subset of obs) into a compact
latent code z_terrain (32-dim) that conditions the RSSM dynamics.

This is the novel contribution: z_terrain gives the world model
terrain-type awareness without ever seeing raw physics parameters
(friction, restitution) — those are not observable on real hardware.

Training objectives for z_terrain:
  1. Implicit: conditioning the RSSM dynamics forces z_terrain to
     encode whatever information predicts terrain-specific motion.
  2. Auxiliary classification loss: cross-entropy predicting terrain
     label from z_terrain. Prevents posterior collapse.
  3. NT-Xent contrastive loss: same-terrain embeddings attract,
     different-terrain embeddings repel. Enforces disentanglement
     as a training objective (not just hoped for implicitly).

Linear probe evaluation:
  A frozen z_terrain + linear classifier should achieve >90% terrain
  type accuracy if disentanglement is working. This is the metric
  that replaces t-SNE as a *quantitative* disentanglement measure.
  (t-SNE is visual only — a linear probe is falsifiable.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List


def _mlp_block(in_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ELU(),
        nn.Dropout(p=dropout),
    )


class ObsEncoder(nn.Module):
    """
    Encodes the 49-dim proprioceptive observation into a dense embedding.

    The embedding is fed to the RSSM posterior network q(z_t | h_t, embed).

    Args:
        obs_dim:     input observation dimension (49)
        embed_dim:   output embedding dimension (256)
        hidden_sizes: hidden layer widths
        dropout:     dropout rate (small — obs encoder is stable)
    """

    def __init__(
        self,
        obs_dim: int = 49,
        embed_dim: int = 256,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(_mlp_block(prev, h, dropout=dropout))
            prev = h
        layers.append(nn.Linear(prev, embed_dim))

        self.net = nn.Sequential(*layers)
        self.embed_dim = embed_dim

    def forward(self, obs: Tensor) -> Tensor:
        """
        Args:
            obs: (batch, obs_dim) or (batch, T, obs_dim)
        Returns:
            embed: same leading dims, last dim = embed_dim
        """
        return self.net(obs)


class TerrainEncoder(nn.Module):
    """
    Encodes terrain probe heights into a latent terrain code z_terrain.

    Input: 16 terrain probe heights (relative to robot base) — the
           last 16 dimensions of the 49-dim observation vector.
           These are observable on real hardware via foot contact
           sensors or LiDAR sweeps, making this sim-to-real ready.

    Output: z_terrain of shape (batch, terrain_latent_dim)

    Training signals (three layers of supervision):

      1. Implicit dynamics conditioning — RSSM loss backprops through
         z_terrain, forcing it to encode terrain-predictive information.

      2. Auxiliary terrain classification — cross-entropy loss on
         terrain type labels. Prevents z_terrain from collapsing.
         (terrain_classify_loss method)

      3. NT-Xent contrastive loss — InfoNCE-style: anchor from terrain A,
         positive = another sample from terrain A, negatives = all other
         terrains in the batch. Temperature-scaled cosine similarity.
         Enforces disentanglement as a *hard training objective*.
         (contrastive_loss method)

    Args:
        probe_dim:          number of terrain probe heights (16)
        terrain_latent_dim: z_terrain output dimension (32)
        hidden_sizes:       hidden layer widths
        num_terrain_types:  number of terrain classes for aux loss (5)
        temperature:        NT-Xent temperature (0.07 is standard)
        dropout:            dropout rate
    """

    def __init__(
        self,
        probe_dim: int = 16,
        terrain_latent_dim: int = 32,
        hidden_sizes: Optional[List[int]] = None,
        num_terrain_types: int = 5,
        temperature: float = 0.07,
        dropout: float = 0.05,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        self.terrain_latent_dim = terrain_latent_dim
        self.temperature = temperature
        self.num_terrain_types = num_terrain_types

        # Main encoder: probe heights → z_terrain
        layers = []
        prev = probe_dim
        for h in hidden_sizes:
            layers.append(_mlp_block(prev, h, dropout=dropout))
            prev = h
        layers.append(nn.Linear(prev, terrain_latent_dim))
        self.encoder = nn.Sequential(*layers)

        # Auxiliary classification head: z_terrain → terrain type logits
        # Used for the classification auxiliary loss only — not at inference
        self.classifier_head = nn.Linear(terrain_latent_dim, num_terrain_types)

        # Projection head for NT-Xent contrastive loss
        # Maps z_terrain to a normalised contrastive embedding
        # (separate from the z_terrain used for conditioning)
        self.proj_head = nn.Sequential(
            nn.Linear(terrain_latent_dim, terrain_latent_dim),
            nn.ReLU(),
            nn.Linear(terrain_latent_dim, terrain_latent_dim),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """
        Extract terrain probes from obs and encode to z_terrain.

        Args:
            obs: (..., 49) full observation vector
                 Terrain probes are at indices [33:49]
        Returns:
            z_terrain: (..., terrain_latent_dim)
        """
        probes = obs[..., 33:49]   # last 16 dims — terrain probe heights
        return self.encoder(probes)

    def encode_probes(self, probes: Tensor) -> Tensor:
        """
        Encode raw probe heights directly.
        Args:
            probes: (..., probe_dim)
        Returns:
            z_terrain: (..., terrain_latent_dim)
        """
        return self.encoder(probes)

    # ── Auxiliary loss 1: classification ──────────────────────────────────

    def terrain_classify_loss(
        self,
        z_terrain: Tensor,
        terrain_labels: Tensor,
    ) -> Tensor:
        """
        Cross-entropy loss predicting terrain type from z_terrain.
        Prevents posterior collapse — forces z_terrain to actually
        encode terrain-discriminative information.

        Args:
            z_terrain:      (batch, terrain_latent_dim)
            terrain_labels: (batch,) integer class labels [0, num_terrain_types)

        Returns:
            scalar loss
        """
        logits = self.classifier_head(z_terrain)
        return F.cross_entropy(logits, terrain_labels)

    def classify(self, z_terrain: Tensor) -> Tensor:
        """
        Predict terrain type from z_terrain.
        Used for the linear probe evaluation (quantitative disentanglement).

        Returns:
            logits: (batch, num_terrain_types)
        """
        return self.classifier_head(z_terrain)

    # ── Auxiliary loss 2: NT-Xent contrastive ─────────────────────────────

    def contrastive_loss(
        self,
        z_terrain: Tensor,
        terrain_labels: Tensor,
    ) -> Tensor:
        """
        NT-Xent (InfoNCE) contrastive loss on terrain embeddings.

        For each anchor sample i:
          - Positive: sample j with the same terrain label
          - Negatives: all other samples in the batch

        L = -log( exp(sim(i,j)/τ) / Σ_k exp(sim(i,k)/τ) )

        This enforces disentanglement as a hard training objective:
        same-terrain embeddings cluster together, different-terrain
        embeddings are pushed apart in the projection space.

        τ (temperature=0.07) is the standard SimCLR value.

        Args:
            z_terrain:      (batch, terrain_latent_dim)
            terrain_labels: (batch,) integer labels

        Returns:
            scalar NT-Xent loss
        """
        batch = z_terrain.shape[0]
        device = z_terrain.device

        # Project to contrastive embedding space and L2-normalise
        proj = self.proj_head(z_terrain)
        proj = F.normalize(proj, dim=-1)   # (batch, terrain_latent_dim)

        # Cosine similarity matrix: (batch, batch)
        sim = torch.mm(proj, proj.T) / self.temperature

        # Mask: True where labels match (positive pairs)
        labels_col = terrain_labels.unsqueeze(1)   # (batch, 1)
        labels_row = terrain_labels.unsqueeze(0)   # (1, batch)
        positive_mask = (labels_col == labels_row)  # (batch, batch)

        # Exclude self-similarity from positives and denominator
        eye = torch.eye(batch, dtype=torch.bool, device=device)
        positive_mask = positive_mask & ~eye

        # If a sample has no positive pair in the batch, skip it
        has_positive = positive_mask.any(dim=1)
        if not has_positive.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # For numerical stability, subtract row max
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # Log-softmax denominator: all pairs except self
        denom_mask = ~eye
        exp_sim = torch.exp(sim) * denom_mask.float()
        log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)  # (batch,)

        # Mean log-probability of positive pairs
        # Sum over all positives in the row, normalised by count
        pos_log_sum = (sim * positive_mask.float()).sum(dim=1)
        pos_count = positive_mask.float().sum(dim=1).clamp(min=1.0)
        log_prob = pos_log_sum / pos_count - log_denom   # (batch,)

        # Only average over samples that had a positive pair
        loss = -log_prob[has_positive].mean()
        return loss

    # ── Linear probe evaluation ───────────────────────────────────────────

    @torch.no_grad()
    def linear_probe_accuracy(
        self,
        z_terrain: Tensor,
        terrain_labels: Tensor,
    ) -> float:
        """
        Evaluate linear separability of z_terrain using the classifier head.

        A frozen z_terrain achieving >90% linear probe accuracy indicates
        the latent space is linearly disentangled by terrain type.
        This is a stronger and quantifiable claim than t-SNE visualisation.

        Returns:
            accuracy in [0, 1]
        """
        logits = self.classifier_head(z_terrain)
        preds = logits.argmax(dim=-1)
        return (preds == terrain_labels).float().mean().item()