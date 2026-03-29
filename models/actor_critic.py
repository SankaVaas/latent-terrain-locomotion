"""
models/actor_critic.py
Actor and Critic networks operating entirely in the RSSM latent space.

Key design decisions
─────────────────────

Actor (policy):
  Input:  combined latent state (h_t ‖ z_t)
  Output: mean + log_std of a Gaussian in action space
          → tanh-squashed to [-1, 1]

  Why continuous Gaussian + tanh?
  The A1 joint targets are continuous (12-dim). A Gaussian gives
  a differentiable distribution we can backprop through during
  imagination rollouts (reparameterisation trick). Tanh squashing
  bounds outputs to the action range without hard clipping.

  Entropy regularisation:
  Actor loss = -E[value] - β_entropy * H(π)
  The entropy term prevents premature determinism. Without it, the
  actor collapses to a deterministic policy early in training and
  never recovers from local optima in the imagined landscape.
  β_entropy = 3e-4 (DreamerV3 default).

Critic (value function):
  Input:  combined latent state (h_t ‖ z_t)
  Output: scalar value estimate V(s)

  λ-returns (TD(λ)):
  V_target = Σ_{n=1}^{T} (λγ)^{n-1} (1-λ) G^{(n)} + (λγ)^T V(s_T)

  λ=0.95 biases toward long-horizon signal while keeping variance
  manageable. λ=0 → pure TD(0), high bias. λ=1 → Monte Carlo,
  high variance.

  The critic is trained on IMAGINED trajectories — no env steps
  needed during policy training. The world model provides free
  rollouts. This is the efficiency advantage of model-based RL.

  Critic EMA target network:
  A slow exponential moving average copy of the critic is used
  to compute λ-return targets. This breaks the deadly triad
  (bootstrapping + off-policy + function approximation) that
  destabilises naive value learning.

Gradient clipping:
  Both actor and critic use hard gradient norm clipping (100.0).
  Imagination rollouts can produce extremely large gradients —
  clipping is essential for stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple
import copy


def _mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Stochastic policy π(a | h_t, z_t) — Gaussian with tanh squashing.

    Maps the combined RSSM latent state to a distribution over actions.
    Trained to maximise expected λ-returns computed over imagined trajectories.

    Args:
        state_dim:    h_dim + stoch_flat_dim
        action_dim:   12 (A1 joint targets)
        hidden_sizes: MLP hidden widths
        min_std:      minimum std dev for numerical stability
        init_std:     initial std dev (controls early exploration)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Optional[List[int]] = None,
        min_std: float = 0.1,
        init_std: float = 1.0,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.action_dim = action_dim
        self.min_std = min_std

        # Outputs mean and log_std (2 * action_dim)
        self.net = _mlp(state_dim, hidden_sizes, action_dim * 2)

        # Initialise last layer with small weights for stable early actions
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.uniform_(self.net[-1].bias, -1e-3, 1e-3)

    def forward(self, state_combined: Tensor) -> Tensor:
        """
        Sample an action (used during imagination rollouts — differentiable).

        Uses reparameterisation trick: a = tanh(mu + eps * sigma)
        so gradients flow through the sample to the actor parameters.

        Args:
            state_combined: (batch, state_dim)
        Returns:
            action: (batch, action_dim) in [-1, 1]
        """
        mean, log_std = self._get_dist_params(state_combined)
        std = F.softplus(log_std) + self.min_std
        eps = torch.randn_like(mean)
        action_pretanh = mean + eps * std
        return torch.tanh(action_pretanh)

    def get_dist(
        self, state_combined: Tensor
    ) -> Tuple[torch.distributions.Distribution, Tensor]:
        """
        Return the action distribution and its entropy.
        Used to compute the entropy regularisation term in actor loss.

        Returns:
            dist:    TransformedDistribution (Gaussian + tanh)
            entropy: (batch,) entropy of the distribution
        """
        mean, log_std = self._get_dist_params(state_combined)
        std = F.softplus(log_std) + self.min_std

        base_dist = torch.distributions.Normal(mean, std)
        # TanhTransform squashes to [-1, 1] — correct log_prob accounting
        dist = torch.distributions.TransformedDistribution(
            base_dist,
            [torch.distributions.transforms.TanhTransform(cache_size=1)],
        )
        # Entropy of the squashed distribution (approximate via base entropy)
        entropy = base_dist.entropy().sum(dim=-1)  # (batch,)
        return dist, entropy

    def actor_loss(
        self,
        imagined_values: Tensor,
        imagined_states: Tensor,
        entropy_scale: float = 3e-4,
    ) -> Tensor:
        """
        Actor loss = -E[value] - entropy_scale * H(π)

        Maximise expected value over imagined trajectories while
        keeping the policy sufficiently stochastic (entropy reg).

        Args:
            imagined_values: (batch, T) value targets from critic
            imagined_states: (batch, T, state_dim) imagined latent states
            entropy_scale:   weight on entropy bonus

        Returns:
            scalar actor loss
        """
        # Flatten batch and time dimensions
        B, T, D = imagined_states.shape
        flat_states = imagined_states.view(B * T, D)

        _, entropy = self.get_dist(flat_states)
        entropy = entropy.view(B, T)

        # Normalise values for stable actor gradients (DreamerV3)
        values = imagined_values.detach()
        val_std = values.std().clamp(min=1.0)
        values_norm = values / val_std

        loss = -(values_norm + entropy_scale * entropy).mean()
        return loss

    def _get_dist_params(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.net(state)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-10.0, 2.0)
        return mean, log_std


class Critic(nn.Module):
    """
    Value function V(h_t, z_t) — scalar estimate over imagined trajectories.

    Trained with λ-returns computed from world model reward and continue
    predictions. Uses an EMA target network for stable bootstrapping.

    Args:
        state_dim:    h_dim + stoch_flat_dim
        hidden_sizes: MLP hidden widths
        gamma:        discount factor
        lambda_:      TD(λ) interpolation coefficient
        ema_decay:    exponential moving average decay for target network
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Optional[List[int]] = None,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        ema_decay: float = 0.98,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.gamma = gamma
        self.lambda_ = lambda_
        self.ema_decay = ema_decay

        self.net = _mlp(state_dim, hidden_sizes, out_dim=1)

        # EMA target network — updated via soft copy, not backprop
        self.target_net = copy.deepcopy(self.net)
        for p in self.target_net.parameters():
            p.requires_grad_(False)

    def forward(self, state_combined: Tensor) -> Tensor:
        """
        Args:
            state_combined: (batch, [T,] state_dim)
        Returns:
            value: (batch, [T,])
        """
        return self.net(state_combined).squeeze(-1)

    def target_value(self, state_combined: Tensor) -> Tensor:
        """Value from the EMA target network (used for λ-return bootstrapping)."""
        return self.target_net(state_combined).squeeze(-1)

    def lambda_returns(
        self,
        rewards: Tensor,
        values: Tensor,
        continues: Tensor,
    ) -> Tensor:
        """
        Compute λ-returns for a sequence of imagined steps.

        G_t^λ = r_t + γ·c_t·[ (1-λ)·V(s_{t+1}) + λ·G_{t+1}^λ ]

        where c_t = P(continue | s_t) from the continue decoder.

        The recursion is computed backwards from t=T-1 to t=0.

        Args:
            rewards:   (batch, T)   predicted rewards from reward decoder
            values:    (batch, T+1) value estimates (T imagined + 1 bootstrap)
            continues: (batch, T)   continue probabilities

        Returns:
            targets: (batch, T) λ-return targets for critic training
        """
        T = rewards.shape[1]
        targets = torch.zeros_like(rewards)

        # Bootstrap from the last value estimate
        last = values[:, -1]

        for t in reversed(range(T)):
            td_target = rewards[:, t] + self.gamma * continues[:, t] * values[:, t + 1]
            last = td_target + self.gamma * continues[:, t] * self.lambda_ * (last - values[:, t + 1])
            targets[:, t] = last

        return targets.detach()

    def critic_loss(
        self,
        state_combined: Tensor,
        lambda_targets: Tensor,
    ) -> Tensor:
        """
        MSE loss between predicted values and λ-return targets.

        Args:
            state_combined:  (batch, T, state_dim)
            lambda_targets:  (batch, T) from lambda_returns()

        Returns:
            scalar critic loss
        """
        predicted = self.forward(state_combined)    # (batch, T)
        return F.mse_loss(predicted, lambda_targets)

    def update_target(self):
        """
        Soft EMA update of target network.
        Call once per training step after the critic optimizer step.

        θ_target ← ema_decay * θ_target + (1 - ema_decay) * θ
        """
        for p_target, p in zip(self.target_net.parameters(), self.net.parameters()):
            p_target.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)