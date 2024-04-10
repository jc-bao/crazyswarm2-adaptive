import numpy as np
import rclpy
import pickle
import os
import time

import doa

from .cf_env import Crazyflie, eval_tracking_performance


def main(log_dir):
    T_takeoff = 3.0
    T_hover = 1.0
    T_task = 20.0
    mode = "figure8"
    log_dir = os.path.join(log_dir, f"PD_{mode}_{time.strftime('%Y%m%d-%H%M%S')}")
    env = Crazyflie(T_takeoff, T_hover, T_task, mode="figure8", log_folder=log_dir)

    # state: p, v, q, w
    config = doa.Controller_NN_Config(
        env.dt,
        10,
        3,
        [64, 64],
        lr=0.0,
        err_thres=100,
        max_out=np.array([5.0, 5.0, 5.0]).astype(np.float32) * env.env_params.m,
        fine_tune_layer_num=-1,
        multi_lr=[0.005, 0.05, 0.5],
    )
    controller = doa.Controller_NN(config)

    try:
        state_real = env.state_real

        total_steps = env.pos_traj.shape[0] - 1
        takeoff_step = int((T_takeoff + T_hover) / env.dt)
        landing_step = int((T_takeoff + T_hover) / env.dt)
        landing_step = total_steps - landing_step
        for timestep in range(total_steps):

            err = state_real.err
            f_disturb = state_real.f_disturb
            state = np.concatenate(
                [
                    state_real.vel * 3.0,
                    state_real.quat,
                    state_real.omega * 3.0,
                ]
            )
            action_nn, _ = controller(state, err, f_disturb)
            action_nn = action_nn.squeeze(0)

            # action_applied = action_nn
            if timestep < takeoff_step or timestep > landing_step:
                action_applied = np.array([0.0, 0.0, 0.0])
            else:
                # action_applied = action_nn
                action_applied = np.array([0.0, 0.0, 0.0])

            state_real, reward_real, done_real, info_real = env.step(action_applied)

            log_info = info_real
            env.log.append(log_info)

        for _ in range(50):
            env.set_attirate(np.zeros(3), 0.0)
    except KeyboardInterrupt:
        pass
    finally:
        log, metrics = env.dump_log()
        rclpy.shutdown()


if __name__ == "__main__":
    main(enable_logging=True)
