import rclpy
from rclpy.node import Node
import numpy as np
import time
from std_msgs.msg import Float32MultiArray
from jax import numpy as jnp
import jax
from brax.mjx.pipeline import init as pipeline_init
from brax.envs.base import State
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from mbd_core import Args, MBDPI
from cf2_env import CF2Env


class CF2Plan(Node):
    def __init__(self):
        super().__init__("cf2_plan")
        self.state_sub = self.create_subscription(
            Float32MultiArray, "state", self.state_callback, 100
        )
        self.acts_pub = self.create_publisher(Float32MultiArray, "action", 10)
        self._init_q = np.zeros(22)
        self._init_q[2] = 0.95
        self._init_q[3] = 1.0
        self.q, self.dq = self._init_q, np.zeros(21)
        self.t = 0.0
        self.ctrl_dt = 0.02
        # set up planner
        self.env = CF2Env()
        args = Args()
        self.mbdpi = MBDPI(args, self.env)
        self.rng = jax.random.PRNGKey(0)
        self.Y = jnp.zeros([args.Hnode, self.mbdpi.nu])
        self.reverse_once_jit = jax.jit(self.mbdpi.reverse_once)
        self.pipeline_init_jit = jax.jit(pipeline_init)
        self.shift_vmap = jax.jit(jax.vmap(self.shift, in_axes=(1, None), out_axes=1))

    def shift(self, x, shift_time):
        spline = InterpolatedUnivariateSpline(
            self.mbdpi.step_nodes * self.mbdpi.node_dt, x, k=2
        )
        x_new = spline(self.mbdpi.step_nodes * self.mbdpi.node_dt + shift_time)
        return x_new

    def get_mjx_state(self, q, qd):
        pipeline_state = self.pipeline_init_jit(self.env.sys, q, qd)
        state_info = {"step": int(self.t / self.ctrl_dt)}
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

    def state_callback(self, msg):
        state = np.array(msg.data)
        self.t = state[0] * 1.0
        self.q = state[1:23]
        self.dq = state[23:]

    def main_loop(self):
        rclpy.spin_once(self, timeout_sec=0.001)
        self.last_plan_time = self.t
        # self.rollout = []
        while rclpy.ok():
            t0 = time.time()
            # get state
            rclpy.spin_once(self, timeout_sec=0.001)
            # convert state
            state = self.get_mjx_state(self.q, self.dq)
            # self.rollout.append(state.pipeline_state)
            # if len(self.rollout) > 300:
            #     self.rollout = self.rollout[-300:]
            # shift Y
            shift_time = self.t - self.last_plan_time
            if shift_time > self.ctrl_dt + 1e-3:
                self.get_logger().warn(
                    f"planner overtime {(shift_time-self.ctrl_dt)*1000:.1f} ms"
                )
            self.get_logger().info(f"planner sim time {shift_time*1000:.1f} ms")
            self.Y = self.shift_vmap(self.Y, shift_time)
            # run planner
            self.rng, self.Y, rews = self.reverse_once_jit(
                state, self.rng, self.Y, self.mbdpi.sigma_control
            )
            self.get_logger().info(f"planner reward {rews.mean():.2f} {rews.std():.2f}")
            self.Y.block_until_ready()
            # convert plan to control
            us = self.mbdpi.node2u_vmap(self.Y)
            # send control
            acts_msg = Float32MultiArray()
            acts_msg.data = [self.t] + us.flatten().tolist()
            self.acts_pub.publish(acts_msg)
            self.last_plan_time = self.t
            self.get_logger().info(
                f"planner wall clock time {(time.time()-t0)*1000:.1f} ms"
            )


def main(args=None):
    rclpy.init(args=args)

    gr1_plan = CF2Plan()

    try:
        gr1_plan.main_loop()
    except KeyboardInterrupt:
        # from brax.io import html
        # webpage = html.render(
        #     gr1_plan.env.sys.tree_replace({"opt.timestep": gr1_plan.env.dt}),
        #     gr1_plan.rollout,
        # )
        # with open("results/rollout.html", "w") as f:
        #     f.write(webpage)
        pass

    gr1_plan.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()