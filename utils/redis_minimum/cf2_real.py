import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from multiprocessing import shared_memory
import math

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper


class CF2Real:
    def __init__(self):
        # cf related
        self.uri = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")
        cflib.crtp.init_drivers()
        self.lg_stab = LogConfig(name="Stabilizer", period_in_ms=10)
        # q
        self.lg_stab.add_variable("stateEstimateZ.x", "int16_t")
        self.lg_stab.add_variable("stateEstimateZ.y", "int16_t")
        self.lg_stab.add_variable("stateEstimateZ.z", "int16_t")
        self.lg_stab.add_variable("stateEstimateZ.quat", "uint32_t")
        # qd
        self.lg_stab.add_variable("stateEstimateZ.vx", "int16_t")
        self.lg_stab.add_variable("stateEstimateZ.vy", "int16_t")
        self.lg_stab.add_variable("stateEstimateZ.vz", "int16_t")
        self.lg_stab.add_variable("stateEstimateZ.rateRoll", "int16_t")
        self.lg_stab.add_variable("stateEstimateZ.ratePitch", "int16_t")
        self.lg_stab.add_variable("stateEstimateZ.rateYaw", "int16_t")

        self.cf = Crazyflie(rw_cache="./cache")

        # control related
        self.ctrl_dt = 0.02
        self.real_dt = 0.01
        self.n_acts = 50
        self.n_frame = int(self.ctrl_dt / self.real_dt)
        self.t = 0.0
        # mujoco setup
        self.mj_model = mujoco.MjModel.from_xml_path(
            f"{os.path.dirname(os.path.abspath(__file__))}/model/scene.xml"
        )
        self.mj_model.opt.timestep = self.real_dt
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

    def thrust2pwm(self, thrust):
        def force_to_rpm(force):
            a, b, c = 2.55077341e-08, -4.92422570e-05, -1.51910248e-01
            force_in_grams = np.clip(force * 1000.0 / 9.81, 0.0, 1000)
            rpm = (-b + np.sqrt(b**2 - 4 * a * (c - force_in_grams))) / (2 * a)
            return rpm

        def rpm_to_pwm(rpm):
            a, b = 3.26535711e-01, 3.37495115e03
            pwm = 1 / a * (rpm - b)
            return pwm

        rpm = force_to_rpm(thrust)
        pwm = rpm_to_pwm(rpm)

        return pwm

    def data2state(self, data):
        def quatdecompress(comp):
            mask = (1 << 9) - 1

            i_largest = comp >> 30
            sum_squares = 0.0
            q = np.zeros(4)

            for i in range(3, -1, -1):
                if i != i_largest:
                    mag = comp & mask
                    negbit = (comp >> 9) & 0x1
                    comp = comp >> 10
                    q[i] = (np.sqrt(0.5)) * ((float)(mag)) / mask
                    if negbit == 1:
                        q[i] = -q[i]
                    sum_squares += q[i] * q[i]
            q[i_largest] = np.sqrt(1.0 - sum_squares)
            q = np.array([q[3], *q[:3]])
            return q

        x = data["stateEstimateZ.x"] / 1000.0
        y = data["stateEstimateZ.y"] / 1000.0
        z = data["stateEstimateZ.z"] / 1000.0
        quat_comp = data["stateEstimateZ.quat"]
        quat = quatdecompress(quat_comp)
        vx = data["stateEstimateZ.vx"] / 1000.0
        vy = data["stateEstimateZ.vy"] / 1000.0
        vz = data["stateEstimateZ.vz"] / 1000.0
        wx = data["stateEstimateZ.rateRoll"] / 1000.0
        wy = data["stateEstimateZ.ratePitch"] / 1000.0
        wz = data["stateEstimateZ.rateYaw"] / 1000.0

        return np.array([x, y, z, *quat, vx, vy, vz, wx, wy, wz])

    def main_loop(self):
        with mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=True
        ) as viewer:
            with SyncCrazyflie(self.uri, cf=self.cf) as scf:
                scf.cf.param.set_value("motorPowerSet.enable", "1")
                with SyncLogger(scf, self.lg_stab) as logger:
                    try:
                        for log_entry in logger:
                            print(
                                f"[INFO] Frequency: {1.0 / (time.time() - self.t):.1f} Hz"
                            )
                            delta_time = self.t - self.plan_time_shared[0]
                            delta_step = int(delta_time / self.ctrl_dt)
                            if delta_time > 0.02:
                                print(f"[WARN] Delayed by {delta_time*1000.0:.1f} ms")
                            if delta_step >= self.n_acts or delta_step < 0:
                                delta_step = self.n_acts - 1

                            # send control
                            for i in range(1, 5):
                                pwm = (self.thrust2pwm(self.acts_shared[delta_step, i - 1]))
                                scf.cf.param.set_value(f"motorPowerSet.m{i}", pwm)

                            # get state
                            data = log_entry[1]
                            state = self.data2state(data)
                            self.t = time.time()

                            # set state to mujoco
                            self.mj_data.qpos[:] = state[:7]
                            self.mj_data.qvel[:] = state[7:]
                            mujoco.mj_forward(self.mj_model, self.mj_data)

                            # publish new state
                            self.time_shared[:] = self.t
                            self.state_shared[:] = state

                            viewer.sync()
                    except KeyboardInterrupt:
                        scf.cf.param.set_value("motorPowerSet.enable", "0")
                        pwm = 0.0
                        for i in range(1, 5):
                            scf.cf.param.set_value(f"motorPowerSet.m{i}", pwm)
                        time.sleep(0.1)

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
    real_env = CF2Real()

    try:
        real_env.main_loop()
    except KeyboardInterrupt:
        pass

    real_env.close()


if __name__ == "__main__":
    main()
