from typing import Tuple
import chex
import jax.numpy as jnp

from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D
from quadjax.envs.quad3d_free import Quad3D


class Quad3DHardware(Quad3D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cf = None

    def step_env(self, key: chex.PRNGKey, state: EnvState3D, action: jnp.ndarray, params: EnvParams3D) -> Tuple[jnp.ndarray, EnvState3D, float, bool, dict]:
        action = jnp.clip(action, -1, 1)
        