"""
models/rssm.py
Recurrent State Space Model (RSSM) — DreamerV3-lite world model core.

Architecture
────────────
The RSSM factorises the latent state into two parts:

  h_t  — deterministic recurrent state (GRU hidden state).
          Carries temporal memory. Computed without sampling.

  z_t  — stochastic discrete latent state.
          Shape: (batch, stoch_dim * stoch_classes) after flattening
          the (stoch_dim, stoch_classes) categorical distribution.

Two distributions over z_t are maintained at all times:

  Prior  p(z_t | h_t)           — predicted WITHOUT seeing o_t.
                                   Used during imagination rollouts.

  Posterior q(z_t | h_t, o_t)  — refined AFTER seeing o_t.
                                   Used during representation learning.

The ELBO loss:
  L = E_q[ log p(o_t|h_t,z_t) + log p(r_t|h_t,z_t) ]
      - β · KL[ q(z_t|h_t,o_t) ‖ p(z_t|h_t) ]

KL balancing (DreamerV3):
  KL_loss = 0.8 · KL(sg(posterior) ‖ prior)
           + 0.2 · KL(posterior ‖ sg(prior))
  sg = stop_gradient (detach). Stabilises training.

Free nats:
  KL_loss = max(KL_loss, free_nats)
  Prevents the model wasting capacity on trivial latent variation.

Straight-through gradients:
  Discrete sampling uses argmax (non-differentiable). We use the
  straight-through estimator: forward uses hard one-hot, backward
  uses soft categorical probabilities.

Terrain conditioning (novel contribution):
  z_terrain (32-dim) concatenated to the GRU input at every step.
  Conditions the dynamics on terrain type — separate imagined
  dynamics per terrain emerge implicitly in the latent space.

MC Dropout for epistemic UQ:
  Dropout inside prior/posterior MLPs. At test time, calling
  rssm.train() and running N forward passes gives epistemic
  uncertainty estimates via prediction variance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class RSSMState:
    """
    Container for a single RSSM latent state (h_t, z_t).
    """
    h: Tensor   # deterministic state  — shape (..., latent_dim)
    z: Tensor   # stochastic state     — shape (..., stoch_dim * stoch_classes)

    def detach(self) -> "RSSMState":
        return RSSMState(self.h.detach(), self.z.detach())

    def to(self, device) -> "RSSMState":
        return RSSMState(self.h.to(device), self.z.to(device))

    @property
    def combined(self) -> Tensor:
        """Concatenated (h, z) — input to decoder/actor/critic."""
        return torch.cat([self.h, self.z], dim=-1)


def symlog(x: Tensor) -> Tensor:
    """
    Symmetric log transform (DreamerV3).
    symlog(x) = sign(x) * log(|x| + 1)
    Compresses large reward values without zeroing small ones.
    Handles reward scales varying by 1000x without explicit normalisation.
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: Tensor) -> Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def _build_mlp(
    in_dim: int,
    hidden_sizes: List[int],
    out_dim: int,
    dropout: float = 0.1,
) -> nn.Sequential:
    """MLP with LayerNorm + ELU + Dropout per hidden layer."""
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [
            nn.Linear(prev, h),
            nn.LayerNorm(h),
            nn.ELU(),
            nn.Dropout(p=dropout),
        ]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class RSSM(nn.Module):
    """
    Recurrent State Space Model — DreamerV3-lite with terrain conditioning.

    Args:
        obs_dim:            encoder output embedding dimension
        action_dim:         action dimension (12 for A1 joints)
        terrain_latent_dim: z_terrain dimension from terrain encoder
        latent_dim:         GRU hidden size (h_t)
        stoch_dim:          number of categorical distributions
        stoch_classes:      number of classes per categorical
        hidden_sizes:       MLP hidden layers for prior/posterior
        dropout:            MC Dropout rate (epistemic UQ at test time)
        kl_weight:          beta in ELBO KL term
        kl_free_nats:       free nats threshold
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        terrain_latent_dim: int,
        latent_dim: int = 256,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
        kl_weight: float = 1.0,
        kl_free_nats: float = 1.0,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.latent_dim = latent_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.stoch_flat_dim = stoch_dim * stoch_classes
        self.kl_weight = kl_weight
        self.kl_free_nats = kl_free_nats

        # GRU input: [z_{t-1}, a_{t-1}, z_terrain]
        gru_input_dim = self.stoch_flat_dim + action_dim + terrain_latent_dim

        self.input_norm = nn.LayerNorm(gru_input_dim)
        self.gru = nn.GRUCell(gru_input_dim, latent_dim)

        # Prior p(z_t | h_t) — no observation
        self.prior_net = _build_mlp(
            in_dim=latent_dim,
            hidden_sizes=hidden_sizes,
            out_dim=self.stoch_flat_dim,
            dropout=dropout,
        )

        # Posterior q(z_t | h_t, obs_embed) — uses observation
        self.posterior_net = _build_mlp(
            in_dim=latent_dim + obs_dim,
            hidden_sizes=hidden_sizes,
            out_dim=self.stoch_flat_dim,
            dropout=dropout,
        )

    # ── Core API ──────────────────────────────────────────────────────────

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """Zeroed initial state for a new episode."""
        return RSSMState(
            h=torch.zeros(batch_size, self.latent_dim, device=device),
            z=torch.zeros(batch_size, self.stoch_flat_dim, device=device),
        )

    def observe_step(
        self,
        prev_state: RSSMState,
        prev_action: Tensor,
        obs_embed: Tensor,
        z_terrain: Tensor,
    ) -> Tuple[RSSMState, Dict[str, Tensor]]:
        """
        One posterior step using a real observation.
        Called during representation learning on collected experience.

        Returns:
            next_state:  posterior RSSMState at t
            stats:       prior_logits, posterior_logits for KL loss
        """
        h_t = self._gru_step(prev_state, prev_action, z_terrain)
        prior_logits = self.prior_net(h_t)
        posterior_logits = self.posterior_net(torch.cat([h_t, obs_embed], dim=-1))
        z_t = self._straight_through_sample(posterior_logits)

        return RSSMState(h=h_t, z=z_t), {
            "prior_logits": prior_logits,
            "posterior_logits": posterior_logits,
        }

    def imagine_step(
        self,
        prev_state: RSSMState,
        action: Tensor,
        z_terrain: Tensor,
    ) -> RSSMState:
        """
        One prior step — no observation used.
        Called during actor-critic imagination rollouts.
        """
        h_t = self._gru_step(prev_state, action, z_terrain)
        prior_logits = self.prior_net(h_t)
        z_t = self._straight_through_sample(prior_logits)
        return RSSMState(h=h_t, z=z_t)

    def observe_sequence(
        self,
        obs_embeds: Tensor,
        actions: Tensor,
        z_terrain: Tensor,
        init_state: Optional[RSSMState] = None,
    ) -> Tuple[RSSMState, Dict[str, Tensor]]:
        """
        Process a full sequence (batch, T, dim) of obs embeddings and actions.
        Called on experience batches during world model training.

        Args:
            obs_embeds:  (batch, T, obs_dim)
            actions:     (batch, T, action_dim)
            z_terrain:   (batch, terrain_latent_dim) — constant per episode
            init_state:  optional RSSMState (zeros if None)

        Returns:
            states:  RSSMState with h, z of shape (batch, T, dim)
            stats:   prior_logits, posterior_logits of shape (batch, T, stoch_flat_dim)
        """
        batch, T, _ = obs_embeds.shape
        device = obs_embeds.device

        state = init_state or self.initial_state(batch, device)

        h_list, z_list, prior_list, post_list = [], [], [], []

        for t in range(T):
            prev_action = (
                torch.zeros(batch, actions.shape[-1], device=device)
                if t == 0 else actions[:, t - 1]
            )
            state, stats = self.observe_step(
                prev_state=state,
                prev_action=prev_action,
                obs_embed=obs_embeds[:, t],
                z_terrain=z_terrain,
            )
            h_list.append(state.h)
            z_list.append(state.z)
            prior_list.append(stats["prior_logits"])
            post_list.append(stats["posterior_logits"])

        return (
            RSSMState(
                h=torch.stack(h_list, dim=1),
                z=torch.stack(z_list, dim=1),
            ),
            {
                "prior_logits": torch.stack(prior_list, dim=1),
                "posterior_logits": torch.stack(post_list, dim=1),
            },
        )

    def imagine_sequence(
        self,
        init_state: RSSMState,
        actor: nn.Module,
        z_terrain: Tensor,
        horizon: int = 15,
    ) -> Tuple[List[RSSMState], List[Tensor]]:
        """
        Roll out imagined trajectories for actor-critic training.
        Actor produces actions; RSSM imagines forward using the prior.

        Returns:
            states:   list of T RSSMState objects
            actions:  list of T action tensors (batch, action_dim)
        """
        state = init_state
        states, actions = [], []
        for _ in range(horizon):
            action = actor(state.combined)
            state = self.imagine_step(state, action, z_terrain)
            states.append(state)
            actions.append(action)
        return states, actions

    # ── KL Loss ───────────────────────────────────────────────────────────

    def kl_loss(
        self,
        prior_logits: Tensor,
        posterior_logits: Tensor,
    ) -> Tensor:
        """
        Balanced KL divergence with free nats (DreamerV3).

        KL_balanced = 0.8 · KL(sg(post) ‖ prior)
                    + 0.2 · KL(post ‖ sg(prior))

        Clamped to free_nats minimum to prevent trivial solutions.

        Logit shapes: (batch, [T,] stoch_dim * stoch_classes)
        """
        shape = prior_logits.shape[:-1]

        def reshape(x):
            return x.view(*shape, self.stoch_dim, self.stoch_classes)

        p_logits = reshape(prior_logits)
        q_logits = reshape(posterior_logits)

        prior_dist = torch.distributions.Categorical(logits=p_logits)
        post_dist = torch.distributions.Categorical(logits=q_logits)
        post_sg = torch.distributions.Categorical(logits=q_logits.detach())
        prior_sg = torch.distributions.Categorical(logits=p_logits.detach())

        kl_prior = torch.distributions.kl_divergence(post_sg, prior_dist)
        kl_post = torch.distributions.kl_divergence(post_dist, prior_sg)

        kl = 0.8 * kl_prior + 0.2 * kl_post   # (*, stoch_dim)
        kl = kl.clamp(min=self.kl_free_nats / self.stoch_dim)
        return self.kl_weight * kl.mean()

    # ── Epistemic Uncertainty (MC Dropout) ────────────────────────────────

    @torch.no_grad()
    def epistemic_uncertainty(
        self,
        prev_state: RSSMState,
        action: Tensor,
        z_terrain: Tensor,
        n_samples: int = 10,
    ) -> Tensor:
        """
        Estimate epistemic uncertainty via MC Dropout.
        Runs N stochastic forward passes through the prior network
        (dropout active) and returns variance of softmax predictions.

        High variance → model uncertain about this (state, terrain) pair.
        Used to detect out-of-distribution terrain at test time.

        Returns:
            scalar uncertainty estimate
        """
        was_training = self.training
        self.train()  # activate dropout

        z_samples = []
        for _ in range(n_samples):
            h_t = self._gru_step(prev_state, action, z_terrain)
            logits = self.prior_net(h_t)
            probs = F.softmax(
                logits.view(-1, self.stoch_dim, self.stoch_classes), dim=-1
            )
            z_samples.append(probs)

        if not was_training:
            self.eval()

        z_stack = torch.stack(z_samples, dim=0)  # (N, batch, stoch_dim, stoch_classes)
        return z_stack.var(dim=0).mean()

    # ── Private ───────────────────────────────────────────────────────────

    def _gru_step(
        self,
        prev_state: RSSMState,
        prev_action: Tensor,
        z_terrain: Tensor,
    ) -> Tensor:
        gru_input = torch.cat([prev_state.z, prev_action, z_terrain], dim=-1)
        gru_input = self.input_norm(gru_input)
        return self.gru(gru_input, prev_state.h)

    def _straight_through_sample(self, logits: Tensor) -> Tensor:
        """
        Categorical straight-through estimator.
        Forward:  hard one-hot (discrete, non-differentiable)
        Backward: soft probabilities (differentiable)
        Allows gradients to flow through discrete latent sampling.
        """
        batch = logits.shape[0]
        logits_3d = logits.view(batch, self.stoch_dim, self.stoch_classes)
        probs = F.softmax(logits_3d, dim=-1)
        indices = probs.argmax(dim=-1)
        hard = F.one_hot(indices, self.stoch_classes).float()
        z_3d = hard - probs.detach() + probs   # straight-through trick
        return z_3d.view(batch, self.stoch_flat_dim)

    @property
    def state_dim(self) -> int:
        """Total dimension of combined latent state (h ‖ z)."""
        return self.latent_dim + self.stoch_flat_dim