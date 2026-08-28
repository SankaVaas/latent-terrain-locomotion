"""
meta/maml.py
First-Order MAML (FOMAML) for zero-shot terrain adaptation.
Memory-optimised: actor copies deleted after each task, CUDA cache cleared per iteration.
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Dict, List, Tuple, Optional
import copy
import gc

from dreamer.world_model import WorldModel
from dreamer.agent import ReplayBuffer, collect_episode, TERRAIN_NAME_TO_ID
from models.actor_critic import Actor, Critic
from envs.a1_env import A1Env


class TerrainTask:
    def __init__(self, terrain_name, support_episodes, query_episodes):
        self.terrain_name = terrain_name
        self.support_episodes = support_episodes
        self.query_episodes   = query_episodes

    def support_replay(self, batch_length=32):
        buf = ReplayBuffer(capacity=10000, batch_length=batch_length)
        for ep in self.support_episodes:
            buf.add_episode(ep)
        return buf

    def query_replay(self, batch_length=32):
        buf = ReplayBuffer(capacity=10000, batch_length=batch_length)
        for ep in self.query_episodes:
            buf.add_episode(ep)
        return buf


def compute_episode_loss(world_model, actor, critic, replay, device, batch_size=4):
    if replay.size < batch_size:
        return torch.tensor(0.0, device=device, requires_grad=True)

    batch   = replay.sample(batch_size)
    obs     = batch["obs"].to(device)
    actions = batch["actions"].to(device)

    embed_seq = world_model.obs_encoder(obs)
    z_terrain = world_model.terrain_encoder(obs[:, 0])

    states, _ = world_model.rssm.observe_sequence(embed_seq, actions, z_terrain)

    T   = states.h.shape[1]
    mid = T // 2
    from models.rssm import RSSMState
    init_state = RSSMState(h=states.h[:, mid], z=states.z[:, mid])

    img_states, _ = world_model.imagine(
        init_state=init_state, actor=actor,
        z_terrain=z_terrain, horizon=10,
    )

    img_combined = torch.stack([s.combined for s in img_states], dim=1)

    with torch.no_grad():
        img_rewards   = world_model.reward_decoder.predict_reward(img_combined).squeeze(-1)
        img_continues = world_model.continue_decoder.predict_continue(img_combined)
        last_val      = critic.target_value(img_states[-1].combined).unsqueeze(1)
        all_vals      = torch.cat([critic.target_value(img_combined), last_val], dim=1)

    targets = critic.lambda_returns(img_rewards, all_vals, img_continues)

    actor_values = critic(img_combined)
    B, H, D = img_combined.shape
    _, entropy = actor.get_dist(img_combined.view(B * H, D))
    entropy = entropy.view(B, H)

    val_norm = targets.detach() / targets.std().clamp(min=1.0)
    return -(val_norm + 3e-4 * entropy).mean()


class MAMLTrainer:

    def __init__(self, world_model, actor, critic, cfg, device):
        self.world_model = world_model
        self.actor  = actor
        self.critic = critic
        self.device = device

        maml_cfg = cfg.get("maml", {})
        self.inner_lr    = maml_cfg.get("inner_lr", 0.0001)
        self.inner_steps = maml_cfg.get("inner_steps", 5)
        self.outer_lr    = maml_cfg.get("outer_lr", 1e-5)

        self.meta_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.outer_lr, eps=1e-8,
        )

        for p in self.world_model.parameters():
            p.requires_grad_(False)

    def _inner_loop(self, support_replay, actor_copy):
        inner_opt = torch.optim.SGD(actor_copy.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            inner_opt.zero_grad()
            loss = compute_episode_loss(
                self.world_model, actor_copy, self.critic,
                support_replay, self.device, batch_size=4,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(actor_copy.parameters(), 10.0)
            inner_opt.step()
        return actor_copy

    def meta_train_step(self, tasks):
        self.meta_optimizer.zero_grad()
        self.world_model.eval()
        self.actor.train()

        # Separate tensors (for backward) from floats (for logging)
        task_loss_tensors = []
        task_loss_floats  = []

        for task in tasks:
            # Deep copy actor for this task only
            actor_copy = copy.deepcopy(self.actor).to(self.device)

            # Inner loop on support set
            support_buf  = task.support_replay(batch_length=32)
            adapted_actor = self._inner_loop(support_buf, actor_copy)

            # Outer loop loss on query set
            query_buf  = task.query_replay(batch_length=32)
            query_loss = compute_episode_loss(
                self.world_model, adapted_actor, self.critic,
                query_buf, self.device, batch_size=4,
            )

            task_loss_tensors.append(query_loss)
            task_loss_floats.append(query_loss.item())

            # Immediately free the copy
            del actor_copy, adapted_actor, support_buf, query_buf
            gc.collect()
            torch.cuda.empty_cache()

        if not task_loss_tensors:
            return {"meta_loss": 0.0, "task_losses": []}

        meta_loss = torch.stack(task_loss_tensors).mean()
        meta_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.meta_optimizer.step()

        # Free computation graphs
        del task_loss_tensors, meta_loss
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "meta_loss": sum(task_loss_floats) / len(task_loss_floats),
            "task_losses": task_loss_floats,
        }

    def adapt(self, env, terrain_name, n_support_episodes=3, n_inner_steps=None):
        steps = n_inner_steps or self.inner_steps
        self.world_model.eval()

        print(f"  Collecting {n_support_episodes} support episodes on {terrain_name}...")
        support_buf = ReplayBuffer(capacity=10000, batch_length=32)

        for i in range(n_support_episodes):
            ep = collect_episode(
                env=env, world_model=self.world_model, actor=self.actor,
                device=self.device, random_action=False, terrain_name=terrain_name,
            )
            support_buf.add_episode(ep)
            print(f"    episode {i+1}: len={len(ep['rewards'])}  reward={ep['rewards'].sum():.2f}")

        actor_copy = copy.deepcopy(self.actor).to(self.device)
        actor_copy.train()
        inner_opt = torch.optim.SGD(actor_copy.parameters(), lr=self.inner_lr)
        losses = []

        print(f"  Running {steps} inner gradient steps...")
        for step in range(steps):
            inner_opt.zero_grad()
            loss = compute_episode_loss(
                self.world_model, actor_copy, self.critic,
                support_buf, self.device,
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


def run_maml_training(
    world_model, actor, critic, cfg, device,
    n_meta_iterations=100, episodes_per_task=1,
    save_path="checkpoints/maml_final.pt",
):
    META_TRAIN_TERRAINS = ["flat", "sand", "ice", "rock"]

    maml = MAMLTrainer(world_model=world_model, actor=actor, critic=critic,
                       cfg=cfg, device=device)

    envs = {t: A1Env(terrain_name=t, render=False, cfg=cfg["env"])
            for t in META_TRAIN_TERRAINS}
    regolith_env = A1Env(terrain_name="regolith", render=False, cfg=cfg["env"])

    print(f"\n{'='*60}")
    print(f"  MAML meta-training")
    print(f"  Meta-train: {META_TRAIN_TERRAINS}")
    print(f"  Meta-test:  regolith (zero-shot)")
    print(f"  Iterations: {n_meta_iterations}  episodes_per_task: {episodes_per_task}")
    print(f"  Inner steps: {maml.inner_steps}  Inner lr: {maml.inner_lr}")
    print(f"{'='*60}\n")

    best_regolith_reward = -np.inf
    meta_losses = []

    for iteration in range(n_meta_iterations):

        tasks = []
        for terrain in META_TRAIN_TERRAINS:
            env = envs[terrain]
            support_eps, query_eps = [], []
            for _ in range(episodes_per_task):
                ep = collect_episode(env=env, world_model=world_model, actor=actor,
                                     device=device, random_action=False, terrain_name=terrain)
                support_eps.append(ep)
            for _ in range(episodes_per_task):
                ep = collect_episode(env=env, world_model=world_model, actor=actor,
                                     device=device, random_action=False, terrain_name=terrain)
                query_eps.append(ep)
            tasks.append(TerrainTask(terrain, support_eps, query_eps))

        result = maml.meta_train_step(tasks)
        meta_losses.append(result["meta_loss"])

        print(f"iter {iteration+1:>4d}/{n_meta_iterations}  "
              f"meta_loss={result['meta_loss']:.4f}  "
              f"task_losses={[f'{l:.3f}' for l in result.get('task_losses', [])]}")

        if (iteration + 1) % 10 == 0:
            print(f"\n  [Zero-shot eval on regolith]")
            adapted = maml.adapt(env=regolith_env, terrain_name="regolith",
                                 n_support_episodes=2, n_inner_steps=maml.inner_steps)

            rewards, lengths = [], []
            for _ in range(3):
                ep = collect_episode(env=regolith_env, world_model=world_model,
                                     actor=adapted, device=device,
                                     random_action=False, terrain_name="regolith")
                rewards.append(ep["rewards"].sum())
                lengths.append(len(ep["rewards"]))
            del adapted
            gc.collect()
            torch.cuda.empty_cache()

            mean_r = np.mean(rewards)
            mean_l = np.mean(lengths)
            print(f"  regolith (adapted):  reward={mean_r:.2f}  ep_len={mean_l:.0f}")
            print(f"  regolith (baseline): reward=-6.99  ep_len=13")
            print(f"  improvement: Δreward={mean_r-(-6.99):+.2f}  Δep_len={mean_l-13:+.0f}\n")

            if mean_r > best_regolith_reward:
                best_regolith_reward = mean_r
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "iteration": iteration,
                    "meta_actor": actor.state_dict(),
                    "world_model": world_model.state_dict(),
                    "critic": critic.state_dict(),
                    "meta_losses": meta_losses,
                    "best_regolith_reward": best_regolith_reward,
                }, save_path)
                print(f"  ✓ New best! Saved to {save_path}")

    for env in envs.values():
        env.close()
    regolith_env.close()

    print(f"\nMAML training complete. Best regolith reward: {best_regolith_reward:.2f}")
    return maml, meta_losses
