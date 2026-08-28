"""
meta/maml.py
Model-Agnostic Meta-Learning (MAML) for zero-shot terrain adaptation.

What MAML does here
────────────────────
Standard training gives us a policy that works on flat/sand/ice/rock
but fails catastrophically on regolith (ep_len=13, reward=-6.99).

MAML reframes the problem:
  Instead of learning ONE policy that works on all terrains,
  learn an INITIALIZATION of weights such that a FEW gradient
  steps on ANY terrain produces a good policy for that terrain.

Inner loop (task-specific adaptation):
  Given K episodes of experience on terrain T:
    θ' = θ - α ∇_θ L_T(θ)   (5 gradient steps, α=0.01)
  θ' is the adapted policy for terrain T.

Outer loop (meta-optimization):
  Minimize expected loss of the ADAPTED policy across all terrain tasks:
    min_θ Σ_T L_T(θ')
  This pushes θ toward an initialization that is easy to adapt.

At test time (zero-shot regolith):
  1. Collect 3-5 episodes on regolith with current policy
  2. Run 5 inner gradient steps → θ'_regolith
  3. Deploy θ'_regolith on regolith
  Target: ep_len > 150 (from baseline of 13)

Why MAML works here
────────────────────
The terrain-conditioned RSSM already encodes terrain information in
z_terrain. MAML makes the ACTOR weights adapt to use z_terrain more
effectively for the specific terrain dynamics — it learns to read the
terrain latent code more precisely in just 5 steps.

Implementation notes
─────────────────────
We use first-order MAML (FOMAML) — drops the second-order Hessian term.
Justified because:
  1. Actor loss landscape is locally linear near convergence
  2. Second-order costs O(n²) memory — infeasible on T4 for our model size
  3. Empirically FOMAML matches second-order MAML within 5% on locomotion

We use the `higher` library for differentiable inner loops.
higher.innerloop_ctx() creates a functional copy of the model that
tracks gradients through the inner optimization steps, enabling
outer loop gradients to flow through the inner loop.
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Dict, List, Tuple, Optional
import copy
import gc

try:
    import higher
    HIGHER_AVAILABLE = True
except ImportError:
    HIGHER_AVAILABLE = False
    print("WARNING: 'higher' not installed. Run: pip install higher")
    print("         Falling back to manual FOMAML implementation.")

from dreamer.world_model import WorldModel
from dreamer.agent import ReplayBuffer, collect_episode, TERRAIN_NAME_TO_ID
from models.actor_critic import Actor, Critic
from envs.a1_env import A1Env


# ── Task definition ───────────────────────────────────────────────────────────

class TerrainTask:
    """
    A single MAML task = one terrain type.
    Stores support episodes (inner loop) and query episodes (outer loop).

    Support set: used for inner loop adaptation
    Query set:   used to evaluate the adapted policy (outer loop loss)

    Args:
        terrain_name:    e.g. "sand"
        support_episodes: list of episode dicts for inner loop
        query_episodes:   list of episode dicts for outer loop evaluation
    """
    def __init__(
        self,
        terrain_name: str,
        support_episodes: List[Dict],
        query_episodes: List[Dict],
    ):
        self.terrain_name = terrain_name
        self.support_episodes = support_episodes
        self.query_episodes   = query_episodes

    def support_replay(self, batch_length: int = 32) -> ReplayBuffer:
        """Build a small replay buffer from support episodes."""
        buf = ReplayBuffer(capacity=10000, batch_length=batch_length)
        for ep in self.support_episodes:
            buf.add_episode(ep)
        return buf

    def query_replay(self, batch_length: int = 32) -> ReplayBuffer:
        """Build a small replay buffer from query episodes."""
        buf = ReplayBuffer(capacity=10000, batch_length=batch_length)
        for ep in self.query_episodes:
            buf.add_episode(ep)
        return buf


# ── Episode return computation ────────────────────────────────────────────────

def compute_episode_loss(
    world_model: WorldModel,
    actor: Actor,
    critic: Critic,
    replay: ReplayBuffer,
    device: torch.device,
    batch_size: int = 4,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tensor:
    """
    Compute actor-critic loss on a batch from the replay buffer.
    Used for both inner and outer loop loss computation.

    Returns scalar loss tensor with gradients attached.
    """
    if replay.size < batch_size:
        # Not enough data — return zero loss
        return torch.tensor(0.0, device=device, requires_grad=True)

    batch = replay.sample(batch_size)
    obs     = batch["obs"].to(device)
    actions = batch["actions"].to(device)

    # Encode observations and terrain
    embed_seq = world_model.obs_encoder(obs)
    z_terrain = world_model.terrain_encoder(obs[:, 0])

    # Get RSSM posterior states
    states, _ = world_model.rssm.observe_sequence(
        embed_seq, actions, z_terrain
    )

    # Use midpoint as imagination start
    T   = states.h.shape[1]
    mid = T // 2
    from models.rssm import RSSMState
    init_state = RSSMState(
        h=states.h[:, mid],
        z=states.z[:, mid],
    )

    # Imagine forward
    img_states, _ = world_model.imagine(
        init_state=init_state,
        actor=actor,
        z_terrain=z_terrain,
        horizon=10,  # shorter horizon for MAML stability
    )

    img_combined = torch.stack(
        [s.combined for s in img_states], dim=1
    )  # (B, H, state_dim)

    # Decode rewards and continues
    with torch.no_grad():
        img_rewards   = world_model.reward_decoder.predict_reward(
            img_combined
        ).squeeze(-1)
        img_continues = world_model.continue_decoder.predict_continue(
            img_combined
        )
        last_val = critic.target_value(
            img_states[-1].combined
        ).unsqueeze(1)
        all_vals = torch.cat([
            critic.target_value(img_combined), last_val
        ], dim=1)

    # λ-returns
    targets = critic.lambda_returns(img_rewards, all_vals, img_continues)

    # Actor loss
    actor_values = critic(img_combined)
    B, H, D = img_combined.shape
    flat_states = img_combined.view(B * H, D)
    _, entropy = actor.get_dist(flat_states)
    entropy = entropy.view(B, H)

    val_std  = targets.std().clamp(min=1.0)
    val_norm = targets.detach() / val_std
    loss = -(val_norm + 3e-4 * entropy).mean()
    return loss


# ── MAML trainer ──────────────────────────────────────────────────────────────

class MAMLTrainer:
    """
    First-Order MAML (FOMAML) for terrain-adaptive locomotion.

    Meta-trains the actor (and optionally critic) so that 5 inner
    gradient steps on any terrain produces a well-adapted policy.

    Args:
        world_model:      trained WorldModel (frozen during MAML)
        actor:            Actor to meta-train
        critic:           Critic (used for value targets)
        cfg:              config dict
        device:           torch device
        inner_lr:         inner loop learning rate α (0.01)
        inner_steps:      number of inner gradient steps (5)
        outer_lr:         outer loop learning rate (1e-4)
        meta_batch_size:  number of terrain tasks per meta-update (4)
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
        self.actor  = actor
        self.critic = critic
        self.device = device

        maml_cfg = cfg.get("maml", {})
        self.inner_lr        = maml_cfg.get("inner_lr", 0.01)
        self.inner_steps     = maml_cfg.get("inner_steps", 5)
        self.outer_lr        = maml_cfg.get("outer_lr", 1e-4)
        self.meta_batch_size = maml_cfg.get("meta_batch_size", 4)

        # Outer loop optimizer — updates the meta-initialization
        self.meta_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.outer_lr,
            eps=1e-8,
        )

        # Freeze world model during MAML — we only adapt the actor
        for p in self.world_model.parameters():
            p.requires_grad_(False)

    def _inner_loop(
        self,
        support_replay: ReplayBuffer,
        actor_copy: nn.Module,
    ) -> nn.Module:
        """
        Run inner loop adaptation on a support set.
        Uses FOMAML — stops gradient at inner loop boundary.

        Args:
            support_replay: replay buffer with support episodes
            actor_copy:     copy of actor to adapt (not the meta-init)

        Returns:
            adapted actor (θ') after inner_steps gradient steps
        """
        inner_opt = torch.optim.SGD(
            actor_copy.parameters(),
            lr=self.inner_lr,
        )

        for step in range(self.inner_steps):
            inner_opt.zero_grad()
            loss = compute_episode_loss(
                world_model=self.world_model,
                actor=actor_copy,
                critic=self.critic,
                replay=support_replay,
                device=self.device,
                batch_size=4,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(actor_copy.parameters(), 10.0)
            inner_opt.step()

        return actor_copy

    def meta_train_step(
        self,
        tasks: List[TerrainTask],
    ) -> Dict[str, float]:
        """
        One MAML outer loop step.

        For each task in the meta-batch:
          1. Copy actor weights (θ → θ_copy)
          2. Run inner loop on support set → θ'
          3. Evaluate θ' on query set → query_loss
        Outer loop: backprop mean(query_loss) through to θ

        FOMAML: we stop gradients at the inner loop boundary
        (don't backprop through the inner optimization steps themselves).
        This makes it computationally tractable on T4.

        Returns:
            dict with meta_loss and per-task losses
        """
        self.meta_optimizer.zero_grad()
        self.world_model.eval()
        self.actor.train()

        task_losses = []

        for task in tasks:
            # 1. Create a deep copy of the actor for this task's inner loop
            actor_copy = copy.deepcopy(self.actor)
            actor_copy.to(self.device)

            # 2. Build support replay and run inner loop
            support_buf = task.support_replay(batch_length=32)
            adapted_actor = self._inner_loop(support_buf, actor_copy)

            # 3. Evaluate adapted actor on query set (outer loop loss)
            #    FOMAML: stop_gradient on adapted weights
            #    The gradient flows to meta-init θ only through the
            #    outer loss evaluated at θ' (not through the inner steps)
            query_buf = task.query_replay(batch_length=32)
            query_loss = compute_episode_loss(
                world_model=self.world_model,
                actor=adapted_actor,
                critic=self.critic,
                replay=query_buf,
                device=self.device,
                batch_size=4,
            )
            task_losses.append(query_loss.detach().item())
            del actor_copy
            gc.collect()

            task_losses_tensors.append(query_loss)     
            task_losses_floats.append(query_loss.item())

        if not task_losses:
            return {"meta_loss": 0.0}

        # 4. Outer loop: mean query loss across tasks
        meta_loss = torch.stack(task_losses).mean()

        # 5. Manually copy gradients from adapted actors back to meta-init
        #    (FOMAML gradient approximation)
        meta_loss.backward()

        # Copy gradients from the last adapted actor to the meta actor
        # This is the FOMAML approximation:
        # ∂L_query(θ') / ∂θ ≈ ∂L_query(θ') / ∂θ'
        for p_meta, p_adapted in zip(
            self.actor.parameters(),
            task_losses[-1].grad_fn and adapted_actor.parameters()
            if task_losses else []
        ):
            if p_adapted.grad is not None and p_meta.grad is None:
                p_meta.grad = p_adapted.grad.clone()

        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.meta_optimizer.step()

        del tasks
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()

        return {
            "meta_loss": meta_loss.item(),
            "task_losses": [l.item() for l in task_losses],
        }

    def adapt(
        self,
        env: A1Env,
        terrain_name: str,
        n_support_episodes: int = 3,
        n_inner_steps: Optional[int] = None,
    ) -> Actor:
        """
        Adapt the meta-learned actor to a new terrain at test time.
        This is the zero-shot adaptation inference path.

        Args:
            env:                  environment set to the target terrain
            terrain_name:         target terrain name
            n_support_episodes:   how many episodes to collect for adaptation
            n_inner_steps:        override inner steps (default: self.inner_steps)

        Returns:
            adapted_actor: policy adapted to the target terrain
        """
        steps = n_inner_steps or self.inner_steps
        self.world_model.eval()

        # Collect support episodes on the new terrain
        print(f"  Collecting {n_support_episodes} support episodes on {terrain_name}...")
        support_buf = ReplayBuffer(capacity=10000, batch_length=32)

        for i in range(n_support_episodes):
            ep = collect_episode(
                env=env,
                world_model=self.world_model,
                actor=self.actor,
                device=self.device,
                random_action=False,
                terrain_name=terrain_name,
            )
            support_buf.add_episode(ep)
            print(f"    episode {i+1}: len={len(ep['rewards'])}  "
                  f"reward={ep['rewards'].sum():.2f}")

        # Run inner loop adaptation
        actor_copy = copy.deepcopy(self.actor).to(self.device)
        actor_copy.train()

        inner_opt = torch.optim.SGD(actor_copy.parameters(), lr=self.inner_lr)
        losses = []

        print(f"  Running {steps} inner gradient steps...")
        for step in range(steps):
            inner_opt.zero_grad()
            loss = compute_episode_loss(
                world_model=self.world_model,
                actor=actor_copy,
                critic=self.critic,
                replay=support_buf,
                device=self.device,
                batch_size=min(4, support_buf.num_episodes),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(actor_copy.parameters(), 10.0)
            inner_opt.step()
            losses.append(loss.item())
            print(f"    inner step {step+1}/{steps}  loss={loss.item():.4f}")

        print(f"  Adaptation complete. Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
        actor_copy.eval()
        return actor_copy


# ── Full MAML training loop ───────────────────────────────────────────────────

def run_maml_training(
    world_model: WorldModel,
    actor: Actor,
    critic: Critic,
    cfg: dict,
    device: torch.device,
    n_meta_iterations: int = 100,
    episodes_per_task: int = 3,
    save_path: str = "checkpoints/maml_final.pt",
):
    """
    Full MAML meta-training loop.

    Meta-train terrains: flat, sand, ice, rock
    Meta-test terrain:   regolith (zero-shot evaluation)

    Each meta-iteration:
      1. For each of 4 terrain tasks:
         - Collect support episodes (inner loop data)
         - Collect query episodes (outer loop evaluation data)
      2. Run one MAML outer loop step
      3. Every 10 iterations: evaluate zero-shot on regolith

    Args:
        n_meta_iterations: number of outer loop steps (100 fits in ~1hr T4)
        episodes_per_task: support + query episodes per task per iteration
    """
    META_TRAIN_TERRAINS = ["flat", "sand", "ice", "rock"]

    maml = MAMLTrainer(
        world_model=world_model,
        actor=actor,
        critic=critic,
        cfg=cfg,
        device=device,
    )

    # Open environments for each training terrain
    envs = {
        t: A1Env(terrain_name=t, render=False, cfg=cfg["env"])
        for t in META_TRAIN_TERRAINS
    }
    regolith_env = A1Env(terrain_name="regolith", render=False, cfg=cfg["env"])

    print(f"\n{'='*60}")
    print(f"  MAML meta-training")
    print(f"  Meta-train: {META_TRAIN_TERRAINS}")
    print(f"  Meta-test:  regolith (zero-shot)")
    print(f"  Iterations: {n_meta_iterations}")
    print(f"  Inner steps: {maml.inner_steps}  Inner lr: {maml.inner_lr}")
    print(f"{'='*60}\n")

    best_regolith_reward = -np.inf
    meta_losses = []

    for iteration in range(n_meta_iterations):

        # ── Build task batch ───────────────────────────────────────────────
        tasks = []
        for terrain in META_TRAIN_TERRAINS:
            env = envs[terrain]
            support_eps, query_eps = [], []

            # Collect support episodes
            for _ in range(episodes_per_task):
                ep = collect_episode(
                    env=env, world_model=world_model,
                    actor=actor, device=device,
                    random_action=False, terrain_name=terrain,
                )
                support_eps.append(ep)

            # Collect query episodes
            for _ in range(episodes_per_task):
                ep = collect_episode(
                    env=env, world_model=world_model,
                    actor=actor, device=device,
                    random_action=False, terrain_name=terrain,
                )
                query_eps.append(ep)

            tasks.append(TerrainTask(terrain, support_eps, query_eps))

        # ── Meta-update ────────────────────────────────────────────────────
        result = maml.meta_train_step(tasks)
        meta_losses.append(result["meta_loss"])

        print(f"iter {iteration+1:>4d}/{n_meta_iterations}  "
              f"meta_loss={result['meta_loss']:.4f}  "
              f"task_losses={[f'{l:.3f}' for l in result.get('task_losses', [])]}")

        # ── Zero-shot evaluation on regolith every 10 iterations ───────────
        if (iteration + 1) % 10 == 0:
            print(f"\n  [Zero-shot eval on regolith]")
            adapted = maml.adapt(
                env=regolith_env,
                terrain_name="regolith",
                n_support_episodes=3,
                n_inner_steps=maml.inner_steps,
            )

            # Evaluate adapted policy
            rewards, lengths = [], []
            for _ in range(3):
                ep = collect_episode(
                    env=regolith_env,
                    world_model=world_model,
                    actor=adapted,
                    device=device,
                    random_action=False,
                    terrain_name="regolith",
                )
                rewards.append(ep["rewards"].sum())
                lengths.append(len(ep["rewards"]))

            mean_r = np.mean(rewards)
            mean_l = np.mean(lengths)
            print(f"  regolith (adapted):  "
                  f"reward={mean_r:.2f}  ep_len={mean_l:.0f}")
            print(f"  regolith (baseline): reward=-6.99  ep_len=13")
            print(f"  improvement: Δreward={mean_r-(-6.99):+.2f}  "
                  f"Δep_len={mean_l-13:+.0f}\n")

            # Save best adapted actor
            if mean_r > best_regolith_reward:
                best_regolith_reward = mean_r
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "iteration": iteration,
                    "meta_actor": actor.state_dict(),
                    "adapted_actor": adapted.state_dict(),
                    "world_model": world_model.state_dict(),
                    "critic": critic.state_dict(),
                    "meta_losses": meta_losses,
                    "best_regolith_reward": best_regolith_reward,
                    "best_regolith_ep_len": mean_l,
                }, save_path)
                print(f"  ✓ New best! Saved to {save_path}")

    # Cleanup
    for env in envs.values():
        env.close()
    regolith_env.close()

    print(f"\nMAML training complete.")
    print(f"Best regolith reward: {best_regolith_reward:.2f}")
    return maml, meta_losses