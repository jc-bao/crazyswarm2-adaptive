import numpy as np
import time
from jax import numpy as jnp
import jax
from brax.mjx.pipeline import init as pipeline_init
from brax.envs.base import State
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from multiprocessing import shared_memory

from mbd_core import Args, MBDPI
from cf2_env import CF2Env


class CF2Plan:
    def __init__(self):
        self._init_q = np.zeros(7)
        self._init_q[3] = 1.0
        self.q, self.dq = self._init_q, np.zeros(6)
        self.t = 0.0
        self.ctrl_dt = 0.02
        # set up planner
        self.env = CF2Env()
        self.ctrl_hover = np.ones(4) * 0.06622
        args = Args()
        self.mbdpi = MBDPI(args, self.env)
        self.rng = jax.random.PRNGKey(0)
        self.Y = jnp.zeros([args.Hnode + 1, self.mbdpi.nu])
        self.reverse_once_jit = jax.jit(self.mbdpi.reverse_once)
        self.pipeline_init_jit = jax.jit(pipeline_init)
        self.shift_vmap = jax.jit(jax.vmap(self.shift, in_axes=(1, None), out_axes=1))
        # publisher
        self.acts_shm = shared_memory.SharedMemory(
            name="acts_shm", create=False, size=50 * 4 * 32
        )
        self.acts_shared = np.ndarray(
            (50, 4), dtype=np.float32, buffer=self.acts_shm.buf
        )
        self.acts_shared[:] = self.ctrl_hover
        self.plan_time_shm = shared_memory.SharedMemory(
            name="plan_time_shm", create=False, size=32
        )
        self.plan_time_shared = np.ndarray(
            1, dtype=np.float32, buffer=self.plan_time_shm.buf
        )
        self.plan_time_shared[0] = 0.0
        # listerner
        self.time_shm = shared_memory.SharedMemory(
            name="time_shm", create=False, size=32
        )
        self.time_shared = np.ndarray(1, dtype=np.float32, buffer=self.time_shm.buf)
        self.time_shared[0] = 0.0
        self.state_shm = shared_memory.SharedMemory(
            name="state_shm", create=False, size=13 * 32
        )
        self.state_shared = np.ndarray(
            (13,), dtype=np.float32, buffer=self.state_shm.buf
        )
        self.state_shared[:] = 0.0
        self.state_shared[3] = 1.0

    def shift(self, x, shift_time):
        spline = InterpolatedUnivariateSpline(
            self.mbdpi.step_nodes * self.mbdpi.node_dt, x, k=2
        )
        x_new = spline(self.mbdpi.step_nodes * self.mbdpi.node_dt + shift_time)
        return x_new

    def get_mjx_state(self, q, qd, t):
        pipeline_state = self.pipeline_init_jit(self.env.sys, q, qd)
        step = int(t / self.ctrl_dt)
        state_info = {
            "step": step,
            "pos_tar": jnp.array([0.0, 0.0, 1.0]),
            "quat_tar": jnp.where(
                step % 600 < 300,
                jnp.array([1.0, 0.0, 0.0, 0.0]),
                jnp.array([1.0, 0.0, 0.0, 0.0]),
            ),
        }
        metrics = {}
        state = State(
            pipeline_state=pipeline_state,
            info=state_info,
            obs=jnp.zeros(0),
            reward=0.0,
            done=0.0,
            metrics=metrics,
        )
        return state

    def main_loop(self):
        last_plan_time = self.time_shared[0]
        # self.rollout = []
        while True:
            t0 = time.time()
            # get state
            plan_time = self.time_shared[0]
            state = self.get_mjx_state(
                self.state_shared[:7].copy(), self.state_shared[7:].copy(), plan_time.copy()
            )
            # self.rollout.append(state.pipeline_state)
            # shift Y
            shift_time = plan_time - last_plan_time
            if shift_time > self.ctrl_dt + 1e-3:
                print(f"[WRAN] sim overtime {(shift_time-self.ctrl_dt)*1000:.1f} ms")
            if shift_time > self.ctrl_dt * 50:
                print(
                    f"[WARN] long time unplanned {shift_time*1000:.1f} ms, reset control"
                )
                self.Y = self.Y * 0.0 + 0.5
            else:
                self.Y = self.shift_vmap(self.Y, shift_time)
            # run planner
            self.rng, self.Y, rews = self.reverse_once_jit(
                state, self.rng, self.Y, self.mbdpi.sigma_control
            )
            # convert plan to control
            us = self.mbdpi.node2u_vmap(self.Y)
            # unnormalize control
            thrusts = self.env.act2thrust(us)
            # send control
            self.plan_time_shared[0] = plan_time
            self.acts_shared[: thrusts.shape[0], :] = thrusts
            # record time
            last_plan_time = plan_time
            if time.time() - t0 > self.ctrl_dt:
                print(f"[WRAN] real overtime {(time.time()-t0)*1000:.1f} ms")


def main():
    cf2_plan = CF2Plan()

    try:
        cf2_plan.main_loop()
    except KeyboardInterrupt:
        # from brax.io import html
        # webpage = html.render(
        #     cf2_plan.env.sys.tree_replace({"opt.timestep": cf2_plan.env.dt}),
        #     cf2_plan.rollout,
        # )
        # import flask
        # app = flask.Flask(__name__)
        # @app.route("/")
        # def index():
        #     return webpage
        # app.run(port=8080)
        pass


if __name__ == "__main__":
    main()
