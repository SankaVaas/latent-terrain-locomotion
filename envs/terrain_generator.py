import numpy as np
import pybullet as p
from dataclasses import dataclass
from typing import Optional


@dataclass
class TerrainConfig:
    name: str
    lateral_friction: float
    spinning_friction: float
    rolling_friction: float
    restitution: float
    contact_damping: float
    contact_stiffness: float
    height_scale: float
    height_noise_freq: float


TERRAIN_REGISTRY = {
    "flat": TerrainConfig("flat", 0.8, 0.01, 0.01, 0.5, 0.1, 1e4, 0.0, 1.0),
    "sand": TerrainConfig("sand", 1.2, 0.05, 0.05, 0.1, 0.8, 5e3, 0.04, 2.0),
    "ice":  TerrainConfig("ice",  0.05,0.001,0.001,0.8, 0.02,2e4, 0.01, 0.5),
    "rock": TerrainConfig("rock", 1.0, 0.03, 0.03, 0.7, 0.05,3e4, 0.12, 4.0),
    "regolith": TerrainConfig("regolith",0.6,0.08,0.08,0.05,1.5,2e3,0.06,3.0),
}


def _generate_height_field(size, height_scale, freq, seed, num_octaves=4):
    rng = np.random.default_rng(seed)
    heights = np.zeros((size, size), dtype=np.float32)
    total_amp = 0.0
    for octave in range(num_octaves):
        amp = 0.5 ** octave
        gx = max(2, int(size * freq * (2**octave) / size * 4))
        grad = rng.standard_normal((gx+1, gx+1, 2)).astype(np.float32)
        xs = np.linspace(0, gx, size, endpoint=False)
        xi = xs.astype(int); xf = xs - xi
        u = xf*xf*(3-2*xf)
        xi = np.clip(xi, 0, gx-1)
        def dg(gxi, gyi, dx, dy):
            g = grad[gxi, gyi]; return g[0]*dx + g[1]*dy
        n00=dg(xi,xi,xf,xf); n10=dg(xi+1,xi,xf-1,xf)
        n01=dg(xi,xi+1,xf,xf-1); n11=dg(xi+1,xi+1,xf-1,xf-1)
        u2=u[:,None]; v2=u[None,:]
        ix0=n00[:,None]+u2*(n10[:,None]-n00[:,None])
        ix1=n01[:,None]+u2*(n11[:,None]-n01[:,None])
        heights += amp*(ix0+v2*(ix1-ix0))
        total_amp += amp
    heights /= (total_amp+1e-8)
    heights *= height_scale
    return heights


class TerrainGenerator:
    def __init__(self, physics_client_id):
        self._client = physics_client_id
        self._terrain_id = None
        self._current_config = None
        self._field_size = 128

    def load(self, terrain_name, seed=0, position=(0,0,0)):
        if terrain_name not in TERRAIN_REGISTRY:
            raise ValueError(f"Unknown terrain '{terrain_name}'")
        config = TERRAIN_REGISTRY[terrain_name]
        self._current_config = config
        if self._terrain_id is not None:
            p.removeBody(self._terrain_id, physicsClientId=self._client)
            self._terrain_id = None
        if config.height_scale < 1e-4:
            import pybullet_data
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
            tid = p.loadURDF("plane.urdf", basePosition=position, physicsClientId=self._client)
        else:
            heights = _generate_height_field(self._field_size, config.height_scale, config.height_noise_freq, seed)
            shape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=[0.05,0.05,1.0],
                heightfieldTextureScaling=self._field_size//2,
                heightfieldData=heights.flatten().tolist(),
                numHeightfieldRows=self._field_size,
                numHeightfieldColumns=self._field_size,
                physicsClientId=self._client,
            )
            tid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                    basePosition=position, physicsClientId=self._client)
        p.changeDynamics(tid, -1,
            lateralFriction=config.lateral_friction,
            spinningFriction=config.spinning_friction,
            rollingFriction=config.rolling_friction,
            restitution=config.restitution,
            contactDamping=config.contact_damping,
            contactStiffness=config.contact_stiffness,
            physicsClientId=self._client)
        self._terrain_id = tid
        return tid

    def get_height_at(self, x, y):
        if self._current_config is None or self._current_config.height_scale < 1e-4:
            return 0.0
        result = p.rayTest([x,y,5.0],[x,y,-1.0],physicsClientId=self._client)
        if result and result[0][0]==self._terrain_id:
            return result[0][3][2]
        return 0.0

    def sample_probe_heights(self, base_x, base_y, num_probes=16, radius=0.5):
        angles = np.linspace(0, 2*np.pi, num_probes, endpoint=False)
        base_h = self.get_height_at(base_x, base_y)
        heights = np.zeros(num_probes, dtype=np.float32)
        for i, a in enumerate(angles):
            heights[i] = self.get_height_at(base_x+radius*np.cos(a), base_y+radius*np.sin(a)) - base_h
        return heights

    def as_feature_vector(self):
        if self._current_config is None:
            return np.zeros(6, dtype=np.float32)
        c = self._current_config
        return np.array([c.lateral_friction, c.restitution, c.contact_damping,
                         c.contact_stiffness/1e4, c.height_scale, c.height_noise_freq], dtype=np.float32)

    @property
    def config(self):
        return self._current_config

    @staticmethod
    def available_terrains():
        return list(TERRAIN_REGISTRY.keys())
