import numpy as np
import time
from jax import numpy as jnp
import jax
from brax.mjx.pipeline import init as pipeline_init
from brax.envs.base import State
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from multiprocessing import shared_memory
from scipy.spatial.transform import Rotation

from mbd_core import Args, MBDPI
from cf2_env import CF2Env




class CF2PID:
    def __init__(self):
        # control params
        self.ctrl_hover = np.ones(4) * 0.06622
        self.kp = 8.0
        self.kd = 4.0
        # self.ki = 0.1
        self.ki = 0.0
        self.kp_att = 50.0 
        # self.kd_att = 10.0 
        self.kd_att = 20.0 
        self.ki_att = 0.0
        self.m = 0.027
        self.I = np.array([2.3951e-5, 2.3951e-5, 3.2347e-5])
        self.g = 9.81
        self.integral = np.zeros(3)
        self.max_thrust = 0.4
        self.ctrl_dt = 0.02
        arm_length = 0.046  # m
        arm = 0.707106781 * arm_length
        t2t = 0.006  # thrust-to-torque ratio
        self.B0 = np.array(
            [
                [1, 1, 1, 1],
                [-arm, -arm, arm, arm],
                [-arm, arm, arm, -arm],
                [-t2t, t2t, -t2t, t2t],
            ]
        )
        # publisher
        self.n_acts = 50
        self.acts_shm = shared_memory.SharedMemory(
            name="acts_shm", create=False, size=self.n_acts * 4 * 32
        )
        self.acts_shared = np.ndarray(
            (self.n_acts, 4), dtype=np.float32, buffer=self.acts_shm.buf
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

    def get_control(self, x):
        r = x[:3]
        q = x[3:7]  # quaternion [w, x, y, z]
        # convert to rotation matrix
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        Q = R.as_matrix()
        v = x[7:10]
        w = x[10:13]

        r_des = np.array([0.0, 0.0, 0.1])

        # control
        # desired force
        f_d = self.m * (
            np.array([0.0, 0.0, self.g])
            - self.kp * (r - r_des)
            - self.kd * (v - 0.0)
            - self.ki * self.integral
            + 0.0  # feedforward
        )
        thrust = (Q.T @ f_d)[2]

        # desired torque
        # desired orientation
        f_d_norm = np.linalg.norm(f_d)
        f_d_norm = np.where(f_d_norm < 1e-3, 1e-3, f_d_norm)
        z_d = f_d / f_d_norm
        axis_angle = jnp.cross(np.array([0.0, 0.0, 1.0]), z_d)
        angle = jnp.linalg.norm(axis_angle)
        angle = jnp.where(angle < 1e-3, 5e-4, angle)
        axis = jnp.where((angle < 1e-3), jnp.array([0.0, 0.0, 1.0]), axis_angle / angle)
        Rd = Rotation.from_rotvec(angle * axis)
        Qd = Rd.as_matrix()
        R_e = Qd.T @ Q
        R_diff = R_e - R_e.T
        angle_err = np.array([R_diff[2, 1], R_diff[0, 2], R_diff[1, 0]])
        ang_acc = -self.kp_att * angle_err - self.kd_att * w - self.ki_att * 0.0
        torque = self.I * ang_acc

        # update integral
        self.integral += (r - r_des) * self.ctrl_dt

        # convert to force
        eta = np.array([thrust, *torque])
        thrusts = np.linalg.solve(self.B0, eta)

        # clip thrust
        over_thrust = np.maximum(thrusts.max() - self.max_thrust, 0.0)
        thrusts = np.clip(thrusts - over_thrust, 0.0, self.max_thrust)

        return thrusts

    def main_loop(self):
        last_plan_time = self.time_shared[0]
        # self.rollout = []
        while True:
            t0 = time.time()
            # get state
            plan_time = self.time_shared[0]
            # check if time is updated
            shift_time = plan_time - last_plan_time
            if shift_time > self.ctrl_dt + 1e-3:
                print(f"[WRAN] sim overtime {(shift_time-self.ctrl_dt)*1000:.1f} ms")
            if shift_time > self.ctrl_dt * 50:
                print(
                    f"[WARN] long time unplanned {shift_time*1000:.1f} ms, reset control"
                )
            # run planner
            thrusts = self.get_control(self.state_shared)[None]
            # send control
            self.plan_time_shared[0] = plan_time
            self.acts_shared[: thrusts.shape[0], :] = thrusts
            # record time
            last_plan_time = plan_time
            if time.time() - t0 > self.ctrl_dt:
                print(f"[WRAN] real overtime {(time.time()-t0)*1000:.1f} ms")

def main():
    cf2_plan = CF2PID()

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
