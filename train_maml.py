"""
train_maml.py
MAML meta-training entry point.
Run AFTER train.py has completed — loads the trained world model
and actor, then runs MAML meta-training on top.

Usage:
  python train_maml.py
  python train_maml.py --checkpoint checkpoints/final.pt --iterations 100
"""

import argparse
import yaml
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dreamer.world_model import WorldModel
from models.actor_critic import Actor, Critic
from meta.maml import run_maml_training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--checkpoint",  default="checkpoints/final.pt")
    p.add_argument("--iterations",  type=int, default=100)
    p.add_argument("--inner_steps", type=int, default=5)
    p.add_argument("--save",        default="checkpoints/maml_final.pt")
    p.add_argument("--episodes",    type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))

    device_str = cfg["training"].get("device", "cpu")
    device = torch.device(
        device_str if torch.cuda.is_available() or device_str == "cpu"
        else "cpu"
    )
    print(f"Device: {device}")

    # ── Build models ──────────────────────────────────────────────────────
    wm_cfg        = cfg.get("world_model", {})
    latent_dim    = wm_cfg.get("latent_dim", 256)
    stoch_dim     = wm_cfg.get("stoch_dim", 32)
    stoch_classes = wm_cfg.get("stoch_classes", 32)
    state_dim     = latent_dim + stoch_dim * stoch_classes

    ac_cfg = cfg.get("actor_critic", {})

    world_model = WorldModel(cfg=cfg, device=device)
    actor  = Actor(
        state_dim=state_dim, action_dim=12,
        hidden_sizes=ac_cfg.get("actor_hidden", [256, 256]),
    ).to(device)
    critic = Critic(
        state_dim=state_dim,
        hidden_sizes=ac_cfg.get("critic_hidden", [256, 256]),
    ).to(device)

    # ── Load pretrained checkpoint ─────────────────────────────────────────
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        world_model.load_state_dict(ckpt["world_model"])
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"WARNING: checkpoint not found at {args.checkpoint}")
        print("  Running MAML from scratch (not recommended)")

    # Override inner steps if specified
    if args.inner_steps != 5:
        cfg["maml"]["inner_steps"] = args.inner_steps

    # ── Run MAML ──────────────────────────────────────────────────────────
    maml_trainer, meta_losses = run_maml_training(
        world_model=world_model,
        actor=actor,
        critic=critic,
        cfg=cfg,
        device=device,
        n_meta_iterations=args.iterations,
        save_path=args.save,
        episodes_per_task=args.episodes, 
    )

    # ── Plot meta-loss curve ───────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0F1117')
        ax.set_facecolor('#1A1D27')
        ax.plot(meta_losses, color='#4A90D9', linewidth=1.8)
        ax.set_title("MAML meta-loss over iterations", color='white', fontsize=12)
        ax.set_xlabel("Meta-iteration", color='gray')
        ax.set_ylabel("Meta-loss", color='gray')
        ax.tick_params(colors='gray')
        for s in ax.spines.values(): s.set_color('#333')
        plt.tight_layout()
        plt.savefig("maml_loss_curve.png", dpi=150, facecolor='#0F1117')
        print("Saved: maml_loss_curve.png")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()