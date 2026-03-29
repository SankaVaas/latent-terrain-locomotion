"""
models/decoder.py
World model decoders — predict reward and episode continuation from latent state.

We use two decoders (no observation reconstruction decoder):

  RewardDecoder   — predicts symlog(reward) from (h_t, z_t)
  ContinueDecoder — predicts P(episode continues) from (h_t, z_t)

Why no observation reconstruction decoder?
───────────────────────────────────────────
DreamerV3 on image observations uses a CNN decoder to reconstruct
pixel observations — this provides a rich training signal.

For proprioceptive observations (49-dim vectors), reconstruction
is trivial and adds almost no useful learning pressure.
Instead, the reward and continue decoders provide all the
supervision needed to shape the latent space for control.

This is also more compute-efficient on T4.

Symlog reward prediction (two-hot encoding):
────────────────────────────────────────────
DreamerV3 predicts reward using a two-hot encoded categorical
distribution over a fixed set of symlog-spaced bins.

Why categorical over regression?
  - Avoids mean-seeking behaviour (regression predicts the mean,
    ignoring multimodal reward distributions)
  - Naturally handles heavy-tailed reward distributions
  - More stable gradients than MSE on sparse rewards

Two-hot encoding: for a target value v falling between bins b_i
and b_{i+1}, the target distribution has weight (b_{i+1}-v)/(b_{i+1}-b_i)
at bin i and (v-b_i)/(b_{i+1}-b_i) at bin i+1, zeros elsewhere.

For our proprioceptive case we use a simpler MSE on symlog(reward)
(DreamerV3's fallback for scalar scalar reward) — this is T4-friendly
and still captures the scale invariance benefit of symlog.

Continue decoder:
─────────────────
Predicts P(not terminated | h_t, z_t) as a Bernoulli.
Used during imagination to weight future returns:
  value = Σ_t γ^t · P(continue)_t · reward_t
So the critic naturally discounts trajectories heading toward
predicted termination (fall, out-of-bounds).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
from models.rssm import symlog, symexp


def _mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class RewardDecoder(nn.Module):
    """
    Predicts symlog-transformed reward from combined latent state (h ‖ z).

    Loss: MSE( predicted_symlog_reward, symlog(actual_reward) )

    At inference, apply symexp to get back the actual reward scale.

    Args:
        state_dim:    dimension of combined latent (h_dim + stoch_flat_dim)
        hidden_sizes: MLP hidden layer widths
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        self.net = _mlp(state_dim, hidden_sizes, out_dim=1)

    def forward(self, state_combined: Tensor) -> Tensor:
        """
        Args:
            state_combined: (batch, [T,] state_dim)
        Returns:
            pred_symlog_reward: (batch, [T,] 1)
        """
        return self.net(state_combined)

    def loss(self, state_combined: Tensor, rewards: Tensor) -> Tensor:
        """
        MSE loss in symlog space.

        Args:
            state_combined: (batch, T, state_dim)
            rewards:        (batch, T) actual rewards from environment

        Returns:
            scalar loss
        """
        pred = self.forward(state_combined).squeeze(-1)   # (batch, T)
        target = symlog(rewards)
        return F.mse_loss(pred, target)

    def predict_reward(self, state_combined: Tensor) -> Tensor:
        """
        Predict reward in original scale (applies symexp).
        Used during imagination rollouts for critic targets.
        """
        return symexp(self.net(state_combined))


class ContinueDecoder(nn.Module):
    """
    Predicts P(episode continues | h_t, z_t) as a Bernoulli probability.

    Used during imagination rollouts to weight future returns:
      effective_discount = gamma * P(continue)
    The critic learns to naturally discount trajectories heading
    toward predicted termination (fall, boundary violation).

    Loss: Binary cross-entropy( predicted_continue, actual_continue )
    where actual_continue = 1 - terminated (1 if episode still running)

    Args:
        state_dim:    dimension of combined latent (h_dim + stoch_flat_dim)
        hidden_sizes: MLP hidden layer widths
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        self.net = _mlp(state_dim, hidden_sizes, out_dim=1)

    def forward(self, state_combined: Tensor) -> Tensor:
        """
        Args:
            state_combined: (batch, [T,] state_dim)
        Returns:
            continue_prob: (batch, [T,] 1) in [0, 1]
        """
        return torch.sigmoid(self.net(state_combined))

    def loss(self, state_combined: Tensor, terminated: Tensor) -> Tensor:
        """
        Binary cross-entropy loss.

        Args:
            state_combined: (batch, T, state_dim)
            terminated:     (batch, T) float — 1.0 if episode ended, else 0.0

        Returns:
            scalar loss
        """
        logits = self.net(state_combined).squeeze(-1)      # (batch, T)
        # continue = 1 - terminated
        continue_targets = 1.0 - terminated.float()
        return F.binary_cross_entropy_with_logits(logits, continue_targets)

    def predict_continue(self, state_combined: Tensor) -> Tensor:
        """
        Return continue probability. Used in imagination rollout discounting.
        Shape: same as state_combined minus last dim (squeezed to scalar per step).
        """
        return torch.sigmoid(self.net(state_combined)).squeeze(-1)