import numpy as np
import rclpy
import pickle
import os

import doa

from .utils import Crazyflie




def eval_tracking_performance(actual, reference):
    # calculate tracking performance
    # actual: np.array, shape (n, 2)
    # reference: np.array, shape (n, 2)
    # return: float, tracking performance
    whole_ade = np.mean(np.linalg.norm(actual - reference, axis=1))
    last_quarter_ade = np.mean(
        np.linalg.norm(
            actual[-int(len(actual) / 4) :] - reference[-int(len(reference) / 4) :],
            axis=1,
        )
    )
    max_error = np.max(np.linalg.norm(actual - reference, axis=1))
    last_quarter_max_error = np.max(
        np.linalg.norm(
            actual[-int(len(actual) / 4) :] - reference[-int(len(reference) / 4) :],
            axis=1,
        )
    )
    mse = np.mean(np.linalg.norm(actual - reference, axis=1) ** 2)
    rmse = np.sqrt(mse)
    return {
        "whole_ade": whole_ade,
        "last_quarter_ade": last_quarter_ade,
        "max_error": max_error,
        "last_quarter_max_error": last_quarter_max_error,
        "mse": mse,
        "rmse": rmse,
    }



def main(enable_logging=True):  # mode  = mppi covo-online covo-offline nn
    env = Crazyflie(enable_logging=enable_logging)

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
        env.cf.setParam("usd.logging", 1)
        state_real = env.state_real

        total_steps = env.pos_traj.shape[0] - 1
        takeoff_step = int(4.0 / env.dt)
        landing_step = int(4.0 / env.dt)
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
        env.cf.setParam("usd.logging", 0)
        if not os.path.exists(os.path.dirname(env.log_path)):
            os.makedirs(os.path.dirname(env.log_path))
        with open(env.log_path, "wb") as f:
            pickle.dump(env.log, f)
        print("log saved to", env.log_path)
        rclpy.shutdown()

        log = env.log[400:]
        metrics = eval_tracking_performance(
            np.array([log[i]["p"] for i in range(len(log))]),
            np.array([log[i]["ref_p"] for i in range(len(log))]),
        )
        print(metrics)


if __name__ == "__main__":
    main(enable_logging=True)
