import os, math
import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
from typing import Optional, Tuple
from envs.terrain_generator import TerrainGenerator, TERRAIN_REGISTRY

JOINT_NAMES = [
    "FR_hip_joint","FR_thigh_joint","FR_calf_joint",
    "FL_hip_joint","FL_thigh_joint","FL_calf_joint",
    "RR_hip_joint","RR_thigh_joint","RR_calf_joint",
    "RL_hip_joint","RL_thigh_joint","RL_calf_joint",
]
DEFAULT_JOINT_ANGLES = np.array([0.0,0.7,-1.4]*4, dtype=np.float32)
JOINT_LOWER = np.array([-0.802,-1.047,-2.696]*4, dtype=np.float32)
JOINT_UPPER = np.array([0.802,4.189,-0.916]*4, dtype=np.float32)
KP, KD, MAX_TORQUE = 20.0, 0.5, 33.5


class A1Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, terrain_name="flat", render=False, cfg=None, terrain_seed=0):
        super().__init__()
        self.terrain_name = terrain_name
        self._render = render
        self._terrain_seed = terrain_seed
        cfg = cfg or {}
        self._sim_hz = cfg.get("sim_hz", 500)
        self._control_hz = cfg.get("control_hz", 50)
        self._decimation = self._sim_hz // self._control_hz
        self._episode_len = cfg.get("episode_len", 1000)
        self._num_probes = cfg.get("num_terrain_probes", 16)
        self._action_scale = cfg.get("action_scale", 20.0)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(49,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        connection = p.GUI if render else p.DIRECT
        self._client = p.connect(connection)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0/self._sim_hz, physicsClientId=self._client)

        self._terrain_gen = TerrainGenerator(self._client)
        self._robot_id = None
        self._joint_ids = []
        self._step_count = 0
        self._prev_action = np.zeros(12, dtype=np.float32)

        self._load_terrain()
        self._load_robot()

    def reset(self, *, seed=None, options=None, terrain_name=None, terrain_seed=None):
        if seed is not None:
            np.random.seed(seed)
        if terrain_name is not None:
            self.terrain_name = terrain_name
        if terrain_seed is not None:
            self._terrain_seed = terrain_seed
        self._load_terrain()
        self._reset_robot()
        self._step_count = 0
        self._prev_action = np.zeros(12, dtype=np.float32)
        return self._get_obs(), {"terrain": self.terrain_name}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target = DEFAULT_JOINT_ANGLES + action * self._action_scale * (np.pi/180)
        target = np.clip(target, JOINT_LOWER, JOINT_UPPER)
        for _ in range(self._decimation):
            self._apply_pd(target)
            p.stepSimulation(physicsClientId=self._client)
        obs = self._get_obs()
        reward, info = self._compute_reward(action)
        self._step_count += 1
        self._prev_action = action.copy()
        terminated = self._is_fallen()
        truncated = self._step_count >= self._episode_len
        info.update({"terrain": self.terrain_name, "step": self._step_count})
        return obs, reward, terminated, truncated, info

    def close(self):
        if p.isConnected(self._client):
            p.disconnect(self._client)

    def get_terrain_feature_vector(self):
        return self._terrain_gen.as_feature_vector()

    def _load_terrain(self):
        self._terrain_gen.load(self.terrain_name, seed=self._terrain_seed)

    def _find_urdf(self):
        local = os.path.join(os.path.dirname(__file__), "urdf", "a1", "a1.urdf")
        if os.path.exists(local):
            return local
        raise FileNotFoundError(
            "A1 URDF not found. Place a1.urdf in envs/urdf/a1/")

    def _load_robot(self):
        urdf = self._find_urdf()
        self._robot_id = p.loadURDF(urdf, [0,0,0.42],
            p.getQuaternionFromEuler([0,0,0]),
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self._client)
        n = p.getNumJoints(self._robot_id, physicsClientId=self._client)
        self._joint_ids = []
        for j in range(n):
            info = p.getJointInfo(self._robot_id, j, physicsClientId=self._client)
            if info[1].decode() in JOINT_NAMES:
                self._joint_ids.append(j)
        assert len(self._joint_ids) == 12, f"Expected 12 joints, got {len(self._joint_ids)}"
        for idx, jid in enumerate(self._joint_ids):
            p.resetJointState(self._robot_id, jid, DEFAULT_JOINT_ANGLES[idx], 0.0,
                              physicsClientId=self._client)

    def _reset_robot(self):
        noise = np.random.uniform(-0.02, 0.02, 3)
        pos = [noise[0], noise[1], 0.42]
        orn = p.getQuaternionFromEuler([
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.1, 0.1)])
        p.resetBasePositionAndOrientation(self._robot_id, pos, orn, physicsClientId=self._client)
        p.resetBaseVelocity(self._robot_id, [0,0,0], [0,0,0], physicsClientId=self._client)
        jn = np.random.uniform(-0.1, 0.1, 12)
        for idx, jid in enumerate(self._joint_ids):
            p.resetJointState(self._robot_id, jid,
                DEFAULT_JOINT_ANGLES[idx]+jn[idx], 0.0, physicsClientId=self._client)

    def _apply_pd(self, targets):
        for idx, jid in enumerate(self._joint_ids):
            s = p.getJointState(self._robot_id, jid, physicsClientId=self._client)
            tau = np.clip(KP*(targets[idx]-s[0]) - KD*s[1], -MAX_TORQUE, MAX_TORQUE)
            p.setJointMotorControl2(self._robot_id, jid,
                controlMode=p.TORQUE_CONTROL, force=tau, physicsClientId=self._client)

    def _get_obs(self):
        jpos = np.zeros(12, dtype=np.float32)
        jvel = np.zeros(12, dtype=np.float32)
        for idx, jid in enumerate(self._joint_ids):
            s = p.getJointState(self._robot_id, jid, physicsClientId=self._client)
            jpos[idx], jvel[idx] = s[0], s[1]
        bpos, born = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._client)
        blv, bav = p.getBaseVelocity(self._robot_id, physicsClientId=self._client)
        R = np.array(p.getMatrixFromQuaternion(born)).reshape(3,3)
        grav = R.T @ np.array([0.,0.,-1.])
        probes = self._terrain_gen.sample_probe_heights(bpos[0], bpos[1], self._num_probes)
        return np.concatenate([jpos, jvel,
            np.array(blv, dtype=np.float32),
            np.array(bav, dtype=np.float32),
            grav.astype(np.float32), probes])

    def _compute_reward(self, action):
        bpos, born = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._client)
        blv, bav = p.getBaseVelocity(self._robot_id, physicsClientId=self._client)
        R = np.array(p.getMatrixFromQuaternion(born)).reshape(3,3)
        v_fwd = float((R.T @ np.array(blv))[0])
        r = (1.5*v_fwd - 0.1*float(bav[2])**2
             - 1e-5*float(np.sum(action**2))
             - 1e-4*float(np.sum((action-self._prev_action)**2))
             + 0.2 + (-10.0 if self._is_fallen() else 0.0))
        return float(r), {"v_forward": v_fwd}

    def _is_fallen(self):
        bpos, born = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._client)
        if bpos[2] < 0.18:
            return True
        e = p.getEulerFromQuaternion(born)
        return abs(e[0]) > 0.8 or abs(e[1]) > 0.8
