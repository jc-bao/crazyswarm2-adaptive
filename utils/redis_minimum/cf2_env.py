import os
from typing import Any
import jax
from jax import numpy as jp
import numpy as np
from matplotlib import pyplot as plt

from brax.io import mjcf, html
from brax import base
from brax.envs.base import PipelineEnv, State

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


class CF2Env(PipelineEnv):
    def __init__(
        self,
        ctrl_dt: float = 0.02,
        scene_file: str = "scene_mjx.xml",
        **kwargs,
    ):
        file_path = os.path.dirname(os.path.abspath(__file__))
        path = file_path + "/model/" + scene_file
        sys = mjcf.load(path)
        self._dt = ctrl_dt  # this environment is 50 fps
        sys = sys.tree_replace({"opt.timestep": 0.02})

        n_frames = kwargs.pop("n_frames", int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self._init_q = jp.array(sys.mj_model.keyframe("hover").qpos)
        self._init_u = jp.array(sys.mj_model.keyframe("hover").ctrl)
        arm_length = 0.046  # m
        arm = 0.707106781 * arm_length
        t2t = 0.006 * 0.3  # thrust-to-torque ratio
        self.B0 = jp.array(
            [
                [1, 1, 1, 1],
                [-arm, -arm, arm, arm],
                [-arm, arm, arm, -arm],
                [-t2t, t2t, -t2t, t2t],
            ]
        )
        self.thrust_min = 0.0
        self.thrust_max = 0.2
        self.nq = sys.q_size()
        self.nv = sys.qd_size()

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)
        delta_pos = (
            jax.random.uniform(
                key,
                (3,),
                minval=jp.array([-0.5, -0.5, -0.1]),
                maxval=jp.array([0.5, 0.5, 0.5]),
            )
            * 0.0
        )
        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self.nv))
        state_info = {
            "step": 0,
            "pos_tar": jp.array([0.0, 0.0, 0.3]) + delta_pos,
            "quat_tar": jp.array([0.0, 0.0, 0.0, 1.0]),  # w, x, y, z
        }
        obs = self._get_obs(pipeline_state, state_info)
        done = 0.0
        reward = self._get_reward(pipeline_state, state_info)
        metrics = {}
        state = State(
            pipeline_state, obs, reward, done, metrics, state_info
        )  # pytype: disable=wrong-arg-types
        return state

    def thrust2act(self, thrusts: jax.Array) -> jax.Array:
        return (thrusts - self.thrust_min) / (
            self.thrust_max - self.thrust_min
        ) * 2.0 - 1.0

    def act2thrust(self, acts: jax.Array) -> jax.Array:
        return (acts + 1) * (self.thrust_max - self.thrust_min) * 0.5 + self.thrust_min

    def thrust2torque(self, thrusts: jax.Array) -> jax.Array:
        eta = jp.dot(self.B0, thrusts)
        return eta

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        thrusts = self.act2thrust(action)
        eta = self.thrust2torque(thrusts)
        pipeline_state = self.pipeline_step(state.pipeline_state, eta)
        # observation data
        obs = self._get_obs(pipeline_state, state.info)
        # reward data
        reward = self._get_reward(pipeline_state, state.info)
        done = 0.0
        state_info = {
            "step": state.info["step"] + 1,
            "pos_tar": state.info["pos_tar"],
            "quat_tar": jp.where(
                state.info["step"] % 600 < 300,
                jp.array([0.0, 0.0, 0.0, 1.0]),
                jp.array([1.0, 0.0, 0.0, 0.0]),
            ),
        }

        state = state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            info=state_info,
        )
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        obs = jp.zeros(0)
        return obs

    def _get_reward(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        pos = pipeline_state.q[:3]
        quat = pipeline_state.q[3:7]
        vel = pipeline_state.qd[:3]
        omega = pipeline_state.qd[3:6]

        reward_pos = 0.0 - jp.linalg.norm(pos - state_info["pos_tar"])
        reward_rot = 0.0 - jp.linalg.norm(quat - state_info["quat_tar"])
        reward_vel = 0.0 - jp.linalg.norm(vel)
        reward_omega = 0.0 - jp.linalg.norm(omega)

        reward = (
            1.0 * reward_pos
            + 1.0 * reward_rot
            + 0.0 * reward_vel
            + 0.0 * reward_omega
        )

        return reward


if __name__ == "__main__":
    env = CF2Env()
    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    state = reset_jit(rng)
    rollout = [state.pipeline_state]
    for _ in range(50):
        rng, _ = jax.random.split(rng)
        # simple feedback controller
        thrusts = (env._init_q[2] - state.pipeline_state.q[2]) * 0.01 + 0.06622
        act = env.thrust2act(jp.array([thrusts] * 4))
        state = step_jit(state, act)
        rollout.append(state.pipeline_state)

    webpage = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout)
    import flask

    app = flask.Flask(__name__)

    @app.route("/")
    def home():
        return webpage

    app.run(port=5000)
