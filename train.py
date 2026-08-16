"""
train.py
Main training entry point for latent-terrain-locomotion.

Training phases
───────────────
Phase 1 — Warmup (random actions):
  Collect `collect_steps` transitions with random actions.
  Fills the replay buffer with diverse experience before any training.

Phase 2 — World model training:
  Alternates between:
    a) Collect one episode using current actor policy
    b) Run `train_ratio` world model training steps on replay buffer samples
    c) Run actor-critic training steps on imagined trajectories

Terrain curriculum:
  Starts with flat terrain only.
  After wm_pretrain_steps, adds sand and ice.
  After curriculum_steps, adds rock and regolith (OOD at test time).
  Training always samples uniformly from available terrains.

Logging:
  Prints loss summary every log_every steps.
  Logs linear probe accuracy every eval_every steps
  (tracks terrain latent disentanglement during training).
  Saves checkpoints every checkpoint_every steps.

Usage:
  # Local CPU (small batch, quick sanity check):
  python train.py

  # Colab T4 (set device: cuda in configs/default.yaml):
  python train.py --config configs/default.yaml

  # Custom config:
  python train.py --config configs/default.yaml --device cuda --seed 0
"""

import os
import sys
import argparse
import yaml
import time
import random
import numpy as np
import torch

from envs.a1_env import A1Env
from dreamer.world_model import WorldModel
from dreamer.agent import ReplayBuffer, ActorCriticTrainer, collect_episode, TERRAIN_NAME_TO_ID
from models.actor_critic import Actor, Critic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, world_model, actor, critic, ac_trainer, step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "world_model": world_model.state_dict(),
        "actor":       actor.state_dict(),
        "critic":      critic.state_dict(),
        "wm_opt":      world_model.optimizer.state_dict(),
        "actor_opt":   ac_trainer.actor_opt.state_dict(),
        "critic_opt":  ac_trainer.critic_opt.state_dict(),
    }, path)
    print(f"  checkpoint saved → {path}")


def load_checkpoint(path: str, world_model, actor, critic, ac_trainer):
    ckpt = torch.load(path, map_location="cpu")
    world_model.load_state_dict(ckpt["world_model"])
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    world_model.optimizer.load_state_dict(ckpt["wm_opt"])
    ac_trainer.actor_opt.load_state_dict(ckpt["actor_opt"])
    ac_trainer.critic_opt.load_state_dict(ckpt["critic_opt"])
    print(f"  resumed from step {ckpt['step']}")
    return ckpt["step"]


def get_terrain_curriculum(step: int) -> list:
    """
    Terrain curriculum — gradually introduce harder terrains.
    flat only → + sand/ice → + rock/regolith
    """
    if step < 20_000:
        return ["flat"]
    elif step < 50_000:
        return ["flat", "sand", "ice"]
    else:
        return ["flat", "sand", "ice", "rock"]
    # Note: "regolith" is held out as zero-shot test terrain


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    # ── Setup ─────────────────────────────────────────────────────────────
    seed = args.seed or cfg["training"].get("seed", 42)
    set_seed(seed)

    device_str = args.device or cfg["training"].get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("  WARNING: CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    print(f"  device: {device}")

    tr_cfg = cfg["training"]
    collect_steps    = tr_cfg.get("collect_steps", 5000)
    batch_size       = tr_cfg.get("batch_size", 16)
    batch_length     = tr_cfg.get("batch_length", 64)
    train_ratio      = tr_cfg.get("train_ratio", 512)
    replay_capacity  = tr_cfg.get("replay_capacity", 500_000)
    log_every        = tr_cfg.get("log_every", 1000)
    eval_every       = tr_cfg.get("eval_every", 10_000)
    checkpoint_every = tr_cfg.get("checkpoint_every", 25_000)

    # ── Models ────────────────────────────────────────────────────────────
    world_model = WorldModel(cfg=cfg, device=device)

    wm_cfg = cfg.get("world_model", {})
    latent_dim    = wm_cfg.get("latent_dim", 256)
    stoch_dim     = wm_cfg.get("stoch_dim", 32)
    stoch_classes = wm_cfg.get("stoch_classes", 32)
    state_dim = latent_dim + stoch_dim * stoch_classes

    ac_cfg = cfg.get("actor_critic", {})
    actor = Actor(
        state_dim=state_dim,
        action_dim=12,
        hidden_sizes=ac_cfg.get("actor_hidden", [256, 256]),
    ).to(device)

    critic = Critic(
        state_dim=state_dim,
        hidden_sizes=ac_cfg.get("critic_hidden", [256, 256]),
        gamma=ac_cfg.get("gamma", 0.99),
        lambda_=ac_cfg.get("lambda_", 0.95),
    ).to(device)

    # ── Replay buffer + trainer ───────────────────────────────────────────
    replay = ReplayBuffer(
        capacity=replay_capacity,
        obs_dim=49,
        action_dim=12,
        batch_length=batch_length,
    )

    ac_trainer = ActorCriticTrainer(
        world_model=world_model,
        actor=actor,
        critic=critic,
        cfg=cfg,
        device=device,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────
    global_step = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        global_step = load_checkpoint(
            args.checkpoint, world_model, actor, critic, ac_trainer
        )

    # ── Environment ───────────────────────────────────────────────────────
    env = A1Env(terrain_name="flat", render=False, cfg=cfg["env"])

    # ── Phase 1: Random collection warmup ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 1: Random warmup ({collect_steps} steps)")
    print(f"{'='*60}")

    warmup_steps = 0
    while warmup_steps < collect_steps:
        terrain = random.choice(["flat", "sand", "ice"])
        episode = collect_episode(
            env=env,
            world_model=world_model,
            actor=actor,
            device=device,
            random_action=True,
            terrain_name=terrain,
        )
        replay.add_episode(episode)
        warmup_steps += len(episode["rewards"])
        print(f"  warmup: {warmup_steps}/{collect_steps} steps | "
              f"episodes: {replay.num_episodes} | "
              f"terrain: {terrain} | "
              f"ep_len: {len(episode['rewards'])}", end="\r")

    print(f"\n  warmup complete — {replay.size} transitions in buffer")

    # ── Phase 2: Main training loop ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 2: Main training loop")
    print(f"{'='*60}")

    wm_losses_accum = {"total":0, "reward":0, "continue_":0, "kl":0,
                       "classify":0, "contrastive":0}
    ac_losses_accum = {"actor_loss":0, "critic_loss":0, "mean_value":0, "mean_reward":0}
    accum_count = 0
    t0 = time.time()

    # Training loop: collect 1 episode → train train_ratio steps
    max_steps = 500_000  # adjust for T4 session length

    while global_step < max_steps:

        # ── Collect one episode ────────────────────────────────────────────
        available_terrains = get_terrain_curriculum(global_step)
        terrain = random.choice(available_terrains)

        episode = collect_episode(
            env=env,
            world_model=world_model,
            actor=actor,
            device=device,
            random_action=False,
            terrain_name=terrain,
        )
        replay.add_episode(episode)
        ep_len = len(episode["rewards"])
        ep_reward = float(episode["rewards"].sum())
        global_step += ep_len

        # ── World model training ───────────────────────────────────────────
        wm_steps = max(1, min(train_ratio, replay.size // batch_size))
        for _ in range(wm_steps):
            if replay.size < batch_size * batch_length:
                break
            batch = replay.sample(batch_size)
            wm_loss = world_model.train_step(batch)

            # Accumulate for logging
            for k in wm_losses_accum:
                wm_losses_accum[k] += getattr(wm_loss, k)

        # ── Actor-critic training ──────────────────────────────────────────
        ac_steps = max(1, wm_steps // 4)
        for _ in range(ac_steps):
            if replay.size < batch_size * batch_length:
                break
            ac_loss = ac_trainer.train_step(replay, batch_size)
            for k in ac_losses_accum:
                ac_losses_accum[k] += ac_loss[k]

        accum_count += wm_steps

        # ── Logging ───────────────────────────────────────────────────────
        if global_step % log_every < ep_len:
            elapsed = time.time() - t0
            n = max(1, accum_count)
            print(
                f"\nstep {global_step:>7d} | "
                f"ep_len {ep_len:>4d} | "
                f"ep_r {ep_reward:>6.2f} | "
                f"terrain {terrain:<10s} | "
                f"buf {replay.size:>7d} | "
                f"t {elapsed:.0f}s\n"
                f"  WM  total={wm_losses_accum['total']/n:.4f}  "
                f"reward={wm_losses_accum['reward']/n:.4f}  "
                f"kl={wm_losses_accum['kl']/n:.4f}  "
                f"cls={wm_losses_accum['classify']/n:.4f}  "
                f"ctr={wm_losses_accum['contrastive']/n:.4f}\n"
                f"  AC  actor={ac_losses_accum['actor_loss']/n:.4f}  "
                f"critic={ac_losses_accum['critic_loss']/n:.4f}  "
                f"val={ac_losses_accum['mean_value']/n:.4f}"
            )
            # Reset accumulators
            wm_losses_accum = {k: 0 for k in wm_losses_accum}
            ac_losses_accum = {k: 0 for k in ac_losses_accum}
            accum_count = 0

        # ── Evaluation: linear probe accuracy ─────────────────────────────
        if global_step % eval_every < ep_len:
            # Build a small eval batch from replay buffer
            eval_batch = replay.sample(64)
            obs_eval = eval_batch["obs"][:, 0]        # (64, 49)
            labels   = eval_batch["terrain_id"]       # (64,)
            acc = world_model.linear_probe_accuracy(obs_eval, labels)

            # Epistemic uncertainty on a random state
            sample_obs = obs_eval[:1].to(device)
            sample_action = torch.zeros(1, 12, device=device)
            init_s = world_model.initial_state(1)
            _, z_t = world_model.encode(sample_obs)
            unc = world_model.epistemic_uncertainty(init_s, sample_action, z_t)

            print(f"\n  [EVAL step {global_step}]"
                  f"  terrain_probe_acc={acc:.3f}"
                  f"  epistemic_unc={unc:.6f}")

        # ── Checkpoint ────────────────────────────────────────────────────
        if global_step % checkpoint_every < ep_len:
            ckpt_path = f"checkpoints/step_{global_step:07d}.pt"
            save_checkpoint(
                ckpt_path, world_model, actor, critic, ac_trainer, global_step
            )

    print(f"\n  Training complete at step {global_step}")
    env.close()
    save_checkpoint(
        "checkpoints/final.pt", world_model, actor, critic, ac_trainer, global_step
    )


if __name__ == "__main__":
    main()