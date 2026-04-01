"""
tests/test_env.py
Sanity checks for the A1 environment and terrain generator.
Run with: python tests/test_env.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml


def load_cfg():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


def test_terrain_generator():
    print("\n── Terrain generator ────────────────────────────────────────────")
    import pybullet as p
    import pybullet_data
    from envs.terrain_generator import TerrainGenerator, TERRAIN_REGISTRY

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    gen = TerrainGenerator(client)

    for name in TERRAIN_REGISTRY:
        tid = gen.load(name, seed=42)
        assert isinstance(tid, int) and tid >= 0, f"Bad terrain id for {name}"
        fv = gen.as_feature_vector()
        assert fv.shape == (6,), f"Bad feature vector shape for {name}"
        print(f"  {name:12s}  id={tid}  feature_vec={fv.round(3)}")

    gen.load("rock", seed=7)
    probes = gen.sample_probe_heights(0.0, 0.0, num_probes=16)
    assert probes.shape == (16,), f"Bad probe shape: {probes.shape}"
    print(f"  rock probes (16): min={probes.min():.4f}  max={probes.max():.4f}")

    p.disconnect(client)
    print("  terrain generator: OK")


def test_env_basic():
    print("\n── A1 environment (flat terrain, no GUI) ───────────────────────")
    cfg = load_cfg()
    from envs.a1_env import A1Env

    env = A1Env(terrain_name="flat", render=False, cfg=cfg["env"])

    assert env.observation_space.shape == (49,)
    assert env.action_space.shape == (12,)
    print(f"  obs_space:    {env.observation_space.shape}  OK")
    print(f"  action_space: {env.action_space.shape}  OK")

    obs, info = env.reset(seed=0)
    assert obs.shape == (49,)
    assert not np.any(np.isnan(obs)), "NaN in initial obs"
    print(f"  reset obs:    shape={obs.shape}  mean={obs.mean():.4f}  OK")

    action = np.zeros(12, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (49,)
    assert isinstance(reward, float)
    assert not np.any(np.isnan(obs))
    print(f"  step (zeros): reward={reward:.4f}  terminated={terminated}  OK")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (49,)
    print(f"  step (random): reward={reward:.4f}  terminated={terminated}  OK")

    env.reset()
    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"  100-step episode: steps={step+1}  total_reward={total_reward:.3f}  OK")

    env.close()
    print("  A1 env (flat): OK")


def test_env_terrain_switch():
    print("\n── Terrain switching ───────────────────────────────────────────")
    cfg = load_cfg()
    from envs.a1_env import A1Env

    terrains = ["flat", "sand", "ice", "rock", "regolith"]
    env = A1Env(terrain_name="flat", render=False, cfg=cfg["env"])

    for t in terrains:
        obs, info = env.reset(terrain_name=t, seed=42)
        assert obs.shape == (49,)
        fv = env.get_terrain_feature_vector()
        assert fv.shape == (6,)
        print(f"  {t:12s}  terrain_fv={fv.round(2)}  OK")

    env.close()
    print("  terrain switching: OK")


def test_obs_sanity():
    print("\n── Observation sanity checks ───────────────────────────────────")
    cfg = load_cfg()
    from envs.a1_env import A1Env

    env = A1Env(terrain_name="flat", render=False, cfg=cfg["env"])
    obs, _ = env.reset(seed=0)

    joint_pos = obs[0:12]
    gravity   = obs[30:33]
    probes    = obs[33:49]

    from envs.a1_env import DEFAULT_JOINT_ANGLES
    assert np.allclose(joint_pos, DEFAULT_JOINT_ANGLES, atol=0.2), \
        f"Joint pos too far from default: {joint_pos}"
    print(f"  joint positions:  {joint_pos.round(3)}")

    assert abs(np.linalg.norm(gravity) - 1.0) < 0.1, \
        f"Gravity vector not unit: {gravity}"
    print(f"  gravity body:     {gravity.round(3)}  (should be ~[0, 0, -1])")

    assert np.all(np.abs(probes) < 0.5), f"Probes too large on flat: {probes}"
    print(f"  probes (flat):    max abs = {np.abs(probes).max():.4f}  OK")

    env.close()
    print("  obs sanity: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("  latent-terrain-locomotion — environment tests")
    print("=" * 60)

    try:
        test_terrain_generator()
        test_env_basic()
        test_env_terrain_switch()
        test_obs_sanity()
        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"\n[URDF ERROR] {e}")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n[ASSERTION FAILED] {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
