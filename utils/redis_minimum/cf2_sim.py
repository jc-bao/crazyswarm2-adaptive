import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from multiprocessing import shared_memory


class CF2Sim:
    def __init__(self):
        # control related
        self.ctrl_dt = 0.02
        self.real_time_factor = 1.0
        self.sim_dt = 0.01
        self.n_acts = 50
        self.n_frame = int(self.ctrl_dt / self.sim_dt)
        self.t = 0.0
        # mujoco setup
        self.mj_model = mujoco.MjModel.from_xml_path(
            f"{os.path.dirname(os.path.abspath(__file__))}/model/scene.xml"
        )
        self.mj_model.opt.timestep = self.sim_dt
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.ctrl_hover = np.ones(4) * 0.06622
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
        # communication setup
        # publisher
        self.time_shm = shared_memory.SharedMemory(
            name="time_shm", create=True, size=32
        )
        self.time_shared = np.ndarray(1, dtype=np.float32, buffer=self.time_shm.buf)
        self.time_shared[0] = 0.0 
        self.state_shm = shared_memory.SharedMemory(
            name="state_shm", create=True, size=13 * 32
        )
        self.state_shared = np.ndarray(
            (13,), dtype=np.float32, buffer=self.state_shm.buf
        )
        self.state_shared[:] = 0.0
        self.state_shared[3] = 1.0
        # listener
        self.acts_shm = shared_memory.SharedMemory(
            name="acts_shm", create=True, size=self.n_acts * self.mj_model.nu * 32
        )
        self.acts_shared = np.ndarray(
            (self.n_acts, self.mj_model.nu), dtype=np.float32, buffer=self.acts_shm.buf
        )
        self.acts_shared[:] = self.ctrl_hover
        self.plan_time_shm = shared_memory.SharedMemory(
            name="plan_time_shm", create=True, size=32
        )
        self.plan_time_shared = np.ndarray(
            1, dtype=np.float32, buffer=self.plan_time_shm.buf
        )

    def thrust2torque(self, thrust):
        return np.dot(self.B0, thrust)

    def main_loop(self):
        with mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=True
        ) as viewer:
            while True:
                t0 = time.time()
                delta_time = self.t - self.plan_time_shared[0]
                delta_step = int(delta_time / self.ctrl_dt)
                # if delta_time > 0.02:
                #     print(f"[WARN] Delayed by {delta_time*1000.0:.1f} ms")
                if delta_step >= self.n_acts or delta_step < 0:
                    delta_step = self.n_acts - 1

                self.mj_data.ctrl = self.thrust2torque(self.acts_shared[delta_step])
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.t += self.sim_dt
                q = self.mj_data.qpos
                qd = self.mj_data.qvel
                state = np.concatenate([q, qd])

                # publish new state
                self.time_shared[:] = self.t
                self.state_shared[:] = state

                viewer.sync()
                t1 = time.time()
                if t1 - t0 < self.sim_dt:
                    time.sleep((self.sim_dt - (t1 - t0)) / self.real_time_factor)
                else:
                    print("[WARN] Sim loop overruns")

    def close(self):
        self.time_shm.close()
        self.time_shm.unlink()
        self.state_shm.close()
        self.state_shm.unlink()
        self.acts_shm.close()
        self.acts_shm.unlink()
        self.plan_time_shm.close()
        self.plan_time_shm.unlink()


def main():
    mujoco_env = CF2Sim()

    try:
        mujoco_env.main_loop()
    except KeyboardInterrupt:
        pass

    mujoco_env.close()


if __name__ == "__main__":
    main()
