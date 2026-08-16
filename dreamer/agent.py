"""
dreamer/agent.py
DreamerV3-lite agent — replay buffer + data collection + actor-critic training.

Three responsibilities
───────────────────────

1. ReplayBuffer
   Stores (obs, action, reward, terminated, terrain_id) transitions.
   Samples random sequences of length batch_length for world model training.
   Circular buffer — oldest episodes overwritten when capacity reached.

2. collect_episode()
   Runs one full episode in the environment using the current policy.
   Stores transitions into the replay buffer.
   During collection: RSSM posterior is updated at every step using
   real observations — this gives the policy a good latent state estimate.

3. ActorCriticTrainer.train_step()
   Samples a starting state from the replay buffer's stored RSSM states.
   Rolls out imagined trajectories (horizon=15) using the world model.
   Computes λ-returns from imagined rewards and continue probabilities.
   Updates actor and critic with gradient clipping.

Why train actor-critic on imagined trajectories?
─────────────────────────────────────────────────
Each imagined rollout is free — no env step needed. The world model
provides thousands of virtual experience trajectories per real env step.
This is the core efficiency advantage of model-based RL over PPO/SAC.

The trade-off: imagined trajectories have compounding model error.
Horizon=15 is empirically stable for DreamerV3. Longer horizons amplify
model errors and destabilise training.
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from collections import deque
import random

from models.rssm import RSSMState
from models.actor_critic import Actor, Critic
from dreamer.world_model import WorldModel


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular replay buffer storing full episodes as sequences.

    Stores raw transitions. Samples random (batch_size, batch_length)
    sub-sequences for world model training.

    Each transition stores:
        obs         (49,)   float32
        action      (12,)   float32
        reward      ()      float32
        terminated  ()      bool
        terrain_id  ()      int     terrain class label (0-4)

    Args:
        capacity:     max number of transitions to store
        obs_dim:      49
        action_dim:   12
        batch_length: sequence length per training sample
    """

    def __init__(
        self,
        capacity: int = 500_000,
        obs_dim: int = 49,
        action_dim: int = 12,
        batch_length: int = 64,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_length = batch_length

        # Store episodes as a list of dicts
        # Each episode: dict of numpy arrays
        self._episodes: deque = deque()
        self._total_steps = 0

    def add_episode(self, episode: Dict[str, np.ndarray]):
        """
        Add a complete episode to the buffer.

        Args:
            episode: dict with keys:
                obs        (T, 49)  float32
                actions    (T, 12)  float32
                rewards    (T,)     float32
                terminated (T,)     float32
                terrain_id int      scalar terrain label
        """
        T = len(episode["rewards"])
        self._total_steps += T
        self._episodes.append(episode)

        # Evict oldest episodes if over capacity
        while self._total_steps > self.capacity and self._episodes:
            old = self._episodes.popleft()
            self._total_steps -= len(old["rewards"])

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        """
        Sample a batch of random sub-sequences.

        Returns dict with tensors of shape:
            obs        (B, T, 49)
            actions    (B, T, 12)
            rewards    (B, T)
            terminated (B, T)
            terrain_id (B,)
        """
        assert len(self._episodes) > 0, "Replay buffer is empty"

        obs_list, act_list, rew_list, term_list, tid_list = [], [], [], [], []

        for _ in range(batch_size):
            # Pick a random episode long enough for batch_length
            valid = [ep for ep in self._episodes
                     if len(ep["rewards"]) >= self.batch_length]
            if not valid:
                # If no episode is long enough, use the longest available
                ep = max(self._episodes, key=lambda e: len(e["rewards"]))
                T = len(ep["rewards"])
                start = 0
                end = T
            else:
                ep = random.choice(valid)
                T = len(ep["rewards"])
                start = random.randint(0, T - self.batch_length)
                end = start + self.batch_length

            sl = slice(start, end)
            obs_list.append(ep["obs"][sl])
            act_list.append(ep["actions"][sl])
            rew_list.append(ep["rewards"][sl])
            term_list.append(ep["terminated"][sl])
            tid_list.append(ep["terrain_id"])

        # Pad sequences to batch_length if needed
        def pad(arr_list, dim):
            out = np.zeros((batch_size, self.batch_length, dim), dtype=np.float32)
            for i, arr in enumerate(arr_list):
                L = min(len(arr), self.batch_length)
                out[i, :L] = arr[:L]
            return out

        def pad1d(arr_list):
            out = np.zeros((batch_size, self.batch_length), dtype=np.float32)
            for i, arr in enumerate(arr_list):
                L = min(len(arr), self.batch_length)
                out[i, :L] = arr[:L]
            return out

        return {
            "obs":        torch.tensor(pad(obs_list, self.obs_dim)),
            "actions":    torch.tensor(pad(act_list, self.action_dim)),
            "rewards":    torch.tensor(pad1d(rew_list)),
            "terminated": torch.tensor(pad1d(term_list)),
            "terrain_id": torch.tensor(np.array(tid_list, dtype=np.int64)),
        }

    @property
    def size(self) -> int:
        return self._total_steps

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)


# ── Data collection ───────────────────────────────────────────────────────────

TERRAIN_NAME_TO_ID = {
    "flat": 0, "sand": 1, "ice": 2, "rock": 3, "regolith": 4
}


def collect_episode(
    env,
    world_model: WorldModel,
    actor: Actor,
    device: torch.device,
    random_action: bool = False,
    terrain_name: str = "flat",
) -> Dict[str, np.ndarray]:
    """
    Run one episode in the environment, collecting transitions.

    During collection, the RSSM posterior is updated at every step
    using real observations. The actor receives the current latent
    state to produce actions.

    Args:
        env:           A1Env instance
        world_model:   WorldModel (used for encoding + posterior updates)
        actor:         Actor network
        device:        torch device
        random_action: if True, sample random actions (for initial collection)
        terrain_name:  terrain type string for labelling

    Returns:
        episode dict ready for ReplayBuffer.add_episode()
    """
    obs_list, act_list, rew_list, term_list = [], [], [], []

    obs_np, _ = env.reset(terrain_name=terrain_name)
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

    # Initial RSSM state and terrain encoding
    state = world_model.initial_state(batch_size=1)
    _, z_terrain = world_model.encode(obs)

    prev_action = torch.zeros(1, 12, device=device)
    done = False

    while not done:
        # Update RSSM with current observation (posterior step)
        state = world_model.step_posterior(state, obs, prev_action, z_terrain)

        # Get action from actor (or random during warmup)
        if random_action:
            action_np = env.action_space.sample()
            action = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            with torch.no_grad():
                action = actor(state.combined)   # (1, 12)
            action_np = action.squeeze(0).cpu().numpy()

        # Step environment
        next_obs_np, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        obs_list.append(obs_np)
        act_list.append(action_np)
        rew_list.append(float(reward))
        term_list.append(float(terminated))

        obs_np = next_obs_np
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        prev_action = action

    terrain_id = TERRAIN_NAME_TO_ID.get(terrain_name, 0)

    return {
        "obs":        np.array(obs_list, dtype=np.float32),
        "actions":    np.array(act_list, dtype=np.float32),
        "rewards":    np.array(rew_list, dtype=np.float32),
        "terminated": np.array(term_list, dtype=np.float32),
        "terrain_id": terrain_id,
    }


# ── Actor-Critic trainer ──────────────────────────────────────────────────────

class ActorCriticTrainer:
    """
    Trains actor and critic on imagined trajectories from the world model.

    One train_step():
      1. Sample a batch of RSSM states from replay buffer sequences
         (encode a mini-batch → get posterior states → use as init)
      2. Imagine forward horizon=15 steps using actor + world model prior
      3. Decode imagined rewards and continue probabilities
      4. Compute λ-returns from imagined rewards
      5. Update critic: MSE(predicted_value, lambda_targets)
      6. Update actor: -E[value] - entropy_scale * H(π)
      7. Update critic EMA target network

    Args:
        world_model:    WorldModel instance
        actor:          Actor network
        critic:         Critic network
        cfg:            config dict (actor_critic section)
        device:         torch device
    """

    def __init__(
        self,
        world_model: WorldModel,
        actor: Actor,
        critic: Critic,
        cfg: dict,
        device: torch.device,
    ):
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.device = device

        ac_cfg = cfg.get("actor_critic", {})
        self.gamma         = ac_cfg.get("gamma", 0.99)
        self.lambda_       = ac_cfg.get("lambda_", 0.95)
        self.entropy_scale = ac_cfg.get("entropy_scale", 3e-4)
        self.grad_clip     = ac_cfg.get("grad_clip", 100.0)

        self.actor_opt = torch.optim.Adam(
            actor.parameters(),
            lr=ac_cfg.get("actor_lr", 3e-5),
            eps=1e-8,
        )
        self.critic_opt = torch.optim.Adam(
            critic.parameters(),
            lr=ac_cfg.get("critic_lr", 3e-5),
            eps=1e-8,
        )

    def train_step(
        self,
        replay: ReplayBuffer,
        batch_size: int = 16,
    ) -> Dict[str, float]:
        """
        One actor-critic training step on imagined trajectories.

        Returns dict of loss values for logging.
        """
        # 1. Sample a batch from replay buffer
        batch = replay.sample(batch_size)
        obs     = batch["obs"].to(self.device)        # (B, T, 49)
        actions = batch["actions"].to(self.device)    # (B, T, 12)

        # 2. Get posterior RSSM states for the sampled sequences
        #    (we need a good starting state for imagination)
        with torch.no_grad():
            embed_seq = self.world_model.obs_encoder(obs)
            z_terrain = self.world_model.terrain_encoder(obs[:, 0])
            states, _ = self.world_model.rssm.observe_sequence(
                embed_seq, actions, z_terrain
            )

        # Use the midpoint state as imagination starting point
        # (avoids using initial zeros or final states near termination)
        T = states.h.shape[1]
        mid = T // 2
        init_state = RSSMState(
            h=states.h[:, mid].detach(),
            z=states.z[:, mid].detach(),
        )

        # 3. Imagine forward from init_state
        #    Gradients flow through imagination for actor update
        img_states, img_actions = self.world_model.imagine(
            init_state=init_state,
            actor=self.actor,
            z_terrain=z_terrain.detach(),
            horizon=self.world_model.imagine_horizon,
        )

        # Stack imagined states → (B, H, state_dim)
        img_combined = torch.stack(
            [s.combined for s in img_states], dim=1
        )   # (B, H, state_dim)

        # 4. Decode imagined rewards and continue probs
        with torch.no_grad():
            img_rewards = self.world_model.reward_decoder.predict_reward(
                img_combined
            ).squeeze(-1)                             # (B, H)

            img_continues = self.world_model.continue_decoder.predict_continue(
                img_combined
            )                                         # (B, H)

        # 5. Bootstrap value at the end of the imagination horizon
        with torch.no_grad():
            # Append one more value estimate for bootstrapping
            last_value = self.critic.target_value(
                img_states[-1].combined
            ).unsqueeze(1)                            # (B, 1)

            all_values_target = torch.cat([
                self.critic.target_value(img_combined),
                last_value,
            ], dim=1)                                 # (B, H+1)

        # 6. Compute λ-returns
        lambda_targets = self.critic.lambda_returns(
            rewards=img_rewards,
            values=all_values_target,
            continues=img_continues,
        )                                             # (B, H)

        # ── Critic update ─────────────────────────────────────────────────
        self.critic_opt.zero_grad()
        critic_loss = self.critic.critic_loss(
            img_combined.detach(),
            lambda_targets,
        )
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()
        self.critic.update_target()

        # ── Actor update ──────────────────────────────────────────────────
        self.actor_opt.zero_grad()
        # Re-compute values with gradient flow to actor
        actor_values = self.critic(img_combined)      # (B, H) — gradients flow
        actor_loss = self.actor.actor_loss(
            imagined_values=actor_values,
            imagined_states=img_combined,
            entropy_scale=self.entropy_scale,
        )
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        return {
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "mean_value":  lambda_targets.mean().item(),
            "mean_reward": img_rewards.mean().item(),
        }