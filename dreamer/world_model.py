"""
dreamer/world_model.py
Full DreamerV3-lite world model — ties together RSSM, encoders, and decoders
into a single trainable module with one unified training step.

What happens in one training step
───────────────────────────────────
Given a batch of (obs, action, reward, terminated) sequences from the
replay buffer:

  1. ObsEncoder:      obs_seq → embed_seq          (batch, T, embed_dim)
  2. TerrainEncoder:  obs_seq → z_terrain           (batch, terrain_dim)
  3. RSSM.observe_sequence:
                      embed_seq, actions, z_terrain
                      → states (h,z), kl_stats      (batch, T, *)
  4. RewardDecoder:   states.combined → pred_reward  loss vs actual reward
  5. ContinueDecoder: states.combined → pred_continue loss vs terminated
  6. RSSM.kl_loss:    prior vs posterior logits
  7. TerrainEncoder auxiliary losses:
       - classification loss (prevent posterior collapse)
       - NT-Xent contrastive loss (enforce disentanglement)
  8. Total loss = reward_loss + continue_loss + kl_loss
                + cls_weight * classify_loss
                + ctr_weight * contrastive_loss
  9. Backprop + gradient clip + optimizer step

Terrain labels
──────────────
The replay buffer stores a terrain_id integer alongside each transition.
This is used for the auxiliary classification and contrastive losses.
It is NOT used by the RSSM itself — the world model never sees terrain
labels during dynamics learning, only during the auxiliary losses on
the terrain encoder. This preserves the "learned from proprioception
only" property of z_terrain.

Loss weights
────────────
  reward_loss:      1.0   (main signal)
  continue_loss:    1.0
  kl_loss:          1.0   (β in ELBO, set in RSSM config)
  classify_loss:    0.1   (auxiliary — too high overwhelms dynamics)
  contrastive_loss: 0.1   (auxiliary — enforces disentanglement)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from models.rssm import RSSM, RSSMState
from models.encoder import ObsEncoder, TerrainEncoder
from models.decoder import RewardDecoder, ContinueDecoder


@dataclass
class WorldModelLosses:
    """All loss components for logging."""
    total: float
    reward: float
    continue_: float
    kl: float
    classify: float
    contrastive: float


class WorldModel(nn.Module):
    """
    Full DreamerV3-lite world model.

    Bundles RSSM + encoders + decoders into one module.
    Exposes a single train_step() method for the training loop.

    Args:
        cfg: dict from configs/default.yaml (world_model section)
        obs_dim:        49
        action_dim:     12
        num_terrains:   5
        device:         torch.device
    """

    def __init__(
        self,
        cfg: dict,
        obs_dim: int = 49,
        action_dim: int = 12,
        num_terrains: int = 5,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device

        wm = cfg.get("world_model", {})
        latent_dim        = wm.get("latent_dim", 256)
        stoch_dim         = wm.get("stoch_dim", 32)
        stoch_classes     = wm.get("stoch_classes", 32)
        terrain_latent    = wm.get("terrain_latent_dim", 32)
        encoder_hidden    = wm.get("encoder_hidden", [256, 256])
        decoder_hidden    = wm.get("decoder_hidden", [256, 256])
        kl_weight         = wm.get("kl_weight", 1.0)
        kl_free_nats      = wm.get("kl_free_nats", 1.0)
        self.imagine_horizon = wm.get("imagine_horizon", 15)

        embed_dim = 256

        # ── Sub-modules ───────────────────────────────────────────────────
        self.obs_encoder = ObsEncoder(
            obs_dim=obs_dim,
            embed_dim=embed_dim,
            hidden_sizes=encoder_hidden,
        )

        self.terrain_encoder = TerrainEncoder(
            probe_dim=16,
            terrain_latent_dim=terrain_latent,
            hidden_sizes=[128, 128],
            num_terrain_types=num_terrains,
        )

        self.rssm = RSSM(
            obs_dim=embed_dim,
            action_dim=action_dim,
            terrain_latent_dim=terrain_latent,
            latent_dim=latent_dim,
            stoch_dim=stoch_dim,
            stoch_classes=stoch_classes,
            hidden_sizes=[256, 256],
            kl_weight=kl_weight,
            kl_free_nats=kl_free_nats,
        )

        state_dim = latent_dim + stoch_dim * stoch_classes

        self.reward_decoder = RewardDecoder(
            state_dim=state_dim,
            hidden_sizes=decoder_hidden,
        )

        self.continue_decoder = ContinueDecoder(
            state_dim=state_dim,
            hidden_sizes=decoder_hidden,
        )

        # ── Optimizer (single optimizer over all WM parameters) ───────────
        opt_cfg = cfg.get("optimizer", {})
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=opt_cfg.get("lr", 1e-4),
            eps=opt_cfg.get("eps", 1e-8),
        )
        self.grad_clip = opt_cfg.get("grad_clip", 1000.0)

        # Loss weights for auxiliary terrain losses
        self.classify_weight    = 0.1
        self.contrastive_weight = 0.1

        self.to(device)

    # ── Main training step ────────────────────────────────────────────────

    def train_step(self, batch: Dict[str, Tensor]) -> WorldModelLosses:
        """
        One world model training step on a batch of experience sequences.

        Args:
            batch: dict with keys:
                obs         (B, T, 49)   float32
                actions     (B, T, 12)   float32
                rewards     (B, T)       float32
                terminated  (B, T)       float32  (1.0 if episode ended)
                terrain_id  (B,)         int64    terrain class label

        Returns:
            WorldModelLosses dataclass with all loss values for logging
        """
        obs        = batch["obs"].to(self.device)
        actions    = batch["actions"].to(self.device)
        rewards    = batch["rewards"].to(self.device)
        terminated = batch["terminated"].to(self.device)
        terrain_id = batch["terrain_id"].to(self.device)

        self.optimizer.zero_grad()

        # 1. Encode observations → embeddings
        embed_seq = self.obs_encoder(obs)              # (B, T, embed_dim)

        # 2. Encode terrain from first obs in sequence
        #    (terrain is constant within an episode)
        z_terrain = self.terrain_encoder(obs[:, 0])    # (B, terrain_dim)

        # 3. RSSM: process full sequence with posterior updates
        states, kl_stats = self.rssm.observe_sequence(
            obs_embeds=embed_seq,
            actions=actions,
            z_terrain=z_terrain,
        )
        state_combined = states.combined               # (B, T, state_dim)

        # 4. Reward prediction loss
        reward_loss = self.reward_decoder.loss(state_combined, rewards)

        # 5. Continue prediction loss
        continue_loss = self.continue_decoder.loss(state_combined, terminated)

        # 6. KL divergence loss (balanced, with free nats)
        kl_loss = self.rssm.kl_loss(
            kl_stats["prior_logits"],
            kl_stats["posterior_logits"],
        )

        # 7. Terrain encoder auxiliary losses
        classify_loss = self.terrain_encoder.terrain_classify_loss(
            z_terrain, terrain_id
        )
        contrastive_loss = self.terrain_encoder.contrastive_loss(
            z_terrain, terrain_id
        )

        # 8. Total loss
        total_loss = (
            reward_loss
            + continue_loss
            + kl_loss
            + self.classify_weight    * classify_loss
            + self.contrastive_weight * contrastive_loss
        )

        # 9. Backprop + gradient clip + step
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        return WorldModelLosses(
            total=total_loss.item(),
            reward=reward_loss.item(),
            continue_=continue_loss.item(),
            kl=kl_loss.item(),
            classify=classify_loss.item(),
            contrastive=contrastive_loss.item(),
        )

    # ── Inference helpers (used by agent during collection + imagination) ──

    @torch.no_grad()
    def encode(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode a single observation.
        Returns (obs_embed, z_terrain) — both (batch, dim).
        Used during environment interaction (data collection).
        """
        embed = self.obs_encoder(obs)
        z_terrain = self.terrain_encoder(obs)
        return embed, z_terrain

    @torch.no_grad()
    def step_posterior(
        self,
        state: RSSMState,
        obs: Tensor,
        action: Tensor,
        z_terrain: Tensor,
    ) -> RSSMState:
        """
        Update RSSM state using a real observation (posterior step).
        Called every env step during data collection.

        Args:
            state:     current RSSMState
            obs:       current observation (batch, 49)
            action:    last action taken (batch, 12)
            z_terrain: terrain latent (batch, terrain_dim)

        Returns:
            updated RSSMState
        """
        embed = self.obs_encoder(obs)
        next_state, _ = self.rssm.observe_step(
            prev_state=state,
            prev_action=action,
            obs_embed=embed,
            z_terrain=z_terrain,
        )
        return next_state

    def imagine(
        self,
        init_state: RSSMState,
        actor: nn.Module,
        z_terrain: Tensor,
        horizon: Optional[int] = None,
    ):
        """
        Roll out imagined trajectories for actor-critic training.
        Gradients flow through this — do NOT use torch.no_grad().

        Returns:
            states:  list of RSSMState (length=horizon)
            actions: list of action tensors
        """
        h = horizon or self.imagine_horizon
        return self.rssm.imagine_sequence(
            init_state=init_state,
            actor=actor,
            z_terrain=z_terrain,
            horizon=h,
        )

    @torch.no_grad()
    def epistemic_uncertainty(
        self,
        state: RSSMState,
        action: Tensor,
        z_terrain: Tensor,
        n_samples: int = 10,
    ) -> float:
        """
        Estimate epistemic uncertainty for the current (state, terrain) pair.
        High uncertainty → robot is in OOD terrain conditions.
        Returns scalar float.
        """
        return self.rssm.epistemic_uncertainty(
            state, action, z_terrain, n_samples
        ).item()

    @torch.no_grad()
    def linear_probe_accuracy(
        self,
        obs_batch: Tensor,
        terrain_labels: Tensor,
    ) -> float:
        """
        Evaluate terrain latent disentanglement via linear probe.
        Call this periodically during training to track disentanglement.
        Target: >90% accuracy = latents are linearly separable by terrain.
        """
        obs_batch = obs_batch.to(self.device)
        terrain_labels = terrain_labels.to(self.device)
        z_terrain = self.terrain_encoder(obs_batch)
        return self.terrain_encoder.linear_probe_accuracy(z_terrain, terrain_labels)

    def initial_state(self, batch_size: int) -> RSSMState:
        """Return zeroed initial RSSM state."""
        return self.rssm.initial_state(batch_size, self.device)

    @property
    def state_dim(self) -> int:
        return self.rssm.state_dim