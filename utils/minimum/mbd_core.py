import jax
import os
from jax import numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm
import time
import functools
from dataclasses import dataclass
import tyro
import matplotlib.pyplot as plt
from brax.io import html

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipline_states) = jax.lax.scan(step, state, us)
    return rews, pipline_states


@dataclass
class Args:
    # exp
    seed: int = 0
    disable_recommended_params: bool = False
    # env
    env_name: str = "cf2"
    # diffusion
    Nsample: int = 4096  # number of samples
    Hsample: int = 50  # horizon of samples
    Hnode: int = 25  # node number for control
    Ndiffuse: int = 50  # number of diffusion steps
    temp_sample: float = 0.1  # temperature for sampling


class MBDPI:
    def __init__(self, args: Args, env):
        self.args = args
        self.env = env
        self.nu = env.action_size

        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0
        B = jnp.log(sigma1 / sigma0) / args.Ndiffuse
        self.sigmas = A * jnp.exp(B * jnp.arange(args.Ndiffuse))
        self.sigma_control = jnp.ones(args.Hnode) * 0.3

        # node to u
        self.step_us = jnp.linspace(0, 1, args.Hsample)
        self.step_nodes = jnp.linspace(0, 1, args.Hnode)
        self.ctrl_dt = 0.02
        self.node_dt = self.ctrl_dt * (args.Hsample - 1) / (args.Hnode - 1)

        # setup function
        self.rollout_us = jax.jit(functools.partial(rollout_us, self.env.step))
        self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))
        self.node2u_vmap = jax.jit(
            jax.vmap(self.node2u, in_axes=(1), out_axes=(1))
        )  # process (horizon, node)
        self.u2node_vmap = jax.jit(jax.vmap(self.u2node, in_axes=(1), out_axes=(1)))
        self.node2u_vvmap = jax.jit(
            jax.vmap(self.node2u_vmap, in_axes=(0))
        )  # process (batch, horizon, node)
        self.u2node_vvmap = jax.jit(jax.vmap(self.u2node_vmap, in_axes=(0)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def node2u(self, nodes):
        spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=2)
        us = spline(self.step_us)
        return us

    @functools.partial(jax.jit, static_argnums=(0,))
    def u2node(self, us):
        spline = InterpolatedUnivariateSpline(self.step_us, us, k=2)
        nodes = spline(self.step_nodes)
        return nodes

    @functools.partial(jax.jit, static_argnums=(0,))
    def reverse_once(self, state, rng, Ybar_i, noise_scale):
        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_Y = jax.random.normal(
            Y0s_rng, (self.args.Nsample, self.args.Hnode, self.nu)
        )
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i
        Y0s = jnp.clip(Y0s, -1.0, 1.0)
        # convert Y0s to us
        us = self.node2u_vvmap(Y0s)

        # esitimate mu_0tm1
        rewss, _ = self.rollout_us_vmap(state, us)
        rews = rewss.mean(axis=-1)
        logp0 = (rews - rews.mean(axis=-1)) / rews.std(axis=-1) / self.args.temp_sample

        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)  # NOTE: update only with reward
        # ubar = jnp.einsum("n,nij->ij", weights, us)

        return rng, Ybar, rews

    def reverse(self, state, YN, rng):
        Yi = YN
        with tqdm(range(self.args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                t0 = time.time()
                rng, Yi, rews = self.reverse_once(
                    state, rng, Yi, self.sigmas[i] * jnp.ones(self.args.Hnode)
                )
                Yi.block_until_ready()
                freq = 1 / (time.time() - t0)
                pbar.set_postfix({"rew": f"{rews.mean():.2e}", "freq": f"{freq:.2f}"})
        return Yi

    @functools.partial(jax.jit, static_argnums=(0,))
    def shift(self, Y):
        u = self.node2u_vmap(Y)
        u = jnp.roll(u, -1, axis=0)
        u = u.at[-1].set(jnp.zeros(self.nu))
        Y = self.u2node_vmap(u)
        return Y

    def shift_Y_from_u(self, u, n_step):
        u = jnp.roll(u, -n_step, axis=0)
        u = u.at[-n_step:].set(jnp.zeros_like(u[-n_step:]))
        Y = self.u2node_vmap(u)
        return Y


def main(args: Args):
    rng = jax.random.PRNGKey(seed=args.seed)
    from gr1_env import GR1Env

    env = GR1Env()
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    mbdpi = MBDPI(args, env)

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)

    YN = jnp.zeros([args.Hnode, mbdpi.nu])

    rng_exp, rng = jax.random.split(rng)
    # Y0 = mbdpi.reverse(state_init, YN, rng_exp)
    Y0 = YN

    Nstep = 100
    rews = []
    rollout = []
    zs_feet = []
    zs_feet_ref = []
    state = state_init
    with tqdm(range(Nstep), desc="Rollout") as pbar:
        for t in pbar:
            # forward single step
            state = step_env(state, Y0[0])
            rollout.append(state.pipeline_state)
            rews.append(state.reward)
            zs_feet.append(env._get_feet_height(state.pipeline_state))
            zs_feet_ref.append(env._get_feet_gait({"step": t}))

            # update Y0
            Y0 = mbdpi.shift(Y0)

            t0 = time.time()
            rng, Y0, _ = mbdpi.reverse_once(state, rng, Y0, mbdpi.sigma_control)
            freq = 1 / (time.time() - t0)
            pbar.set_postfix({"rew": f"{state.reward:.2e}", "freq": f"{freq:.2f}"})

    rew = jnp.array(rews).mean()
    print(f"mean reward = {rew:.2e}")

    # plot
    zs_feet = jnp.array(zs_feet)
    zs_feet_ref = jnp.array(zs_feet_ref)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    for i in range(2):
        axes[i].plot(zs_feet[:, i], label="feet")
        axes[i].plot(zs_feet_ref[:, i], "--", label="ref")
    plt.savefig("./results/feet.png")

    # host webpage with flask
    import flask

    app = flask.Flask(__name__)
    webpage = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout)

    @app.route("/")
    def index():
        return webpage

    app.run(port=5000)


if __name__ == "__main__":
    main(tyro.cli(Args))
