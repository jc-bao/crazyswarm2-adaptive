import numpy as np


def generate_traj(init_pos: np.array, dt: float, mode: str) -> np.ndarray:
    """
    generate a trajectory with max_steps steps
    """

    # generate take off trajectory
    target_pos = np.array([0.0, 0.0, 0.0])
    t_takeoff = 3.0
    pos_takeoff, vel_takeoff, acc_takeoff = line(dt, t_takeoff, init_pos, target_pos)

    # stablize for 1.0 second
    t_stablize = 1.0
    pos_stablize, vel_stablize, acc_stablize = line(
        dt, t_stablize, target_pos, target_pos
    )

    # generate test trajectory
    t_task = 20.0

    if mode == "figure8":
        # figure 8 trajectory
        W = np.array([0.4, 0.2, 0.0]) * np.pi
        A = np.array([1.0, 1.0, 0.0])
        Phi = np.array([0.0, 0.0, 0.0])
        pos_task, vel_task, acc_task = figure8(dt, t_task, A, W, Phi)

        scale = 1.0
        pos_task = pos_task * scale
        vel_task = vel_task * scale
        acc_task = acc_task * scale

    elif mode == "random":
        pass

    else:
        raise ValueError("Invalid mode")

    # generate landing trajectory by inverse the takeoff trajectory
    pos_land_stablize, vel_land_stablize, acc_land_stablize = line(
        dt, t_stablize, pos_takeoff[-1], pos_takeoff[-1]
    )
    pos_landing, vel_landing, acc_landing = line(dt, t_takeoff, pos_task[-1], init_pos)

    # concatenate all trajectories
    pos = np.concatenate(
        [
            pos_takeoff,
            pos_stablize,
            pos_task,
            pos_land_stablize,
            pos_landing,
        ],
        axis=0,
    )
    vel = np.concatenate(
        [
            vel_takeoff,
            vel_stablize,
            vel_task,
            vel_land_stablize,
            vel_landing,
        ],
        axis=0,
    )
    acc = np.concatenate(
        [
            acc_takeoff,
            acc_stablize,
            acc_task,
            acc_land_stablize,
            acc_landing,
        ],
        axis=0,
    )

    return pos, vel, acc


def figure8(dt, T, A, W, Phi):
    W = W.reshape(-1, 1)
    t = np.arange(0, T, dt).reshape(1, -1)
    Phi = Phi.reshape(-1, 1)
    A = A.reshape(-1, 1)
    pos = A * np.sin(W * t + Phi)
    pos = pos.T
    vel = np.diff(pos, axis=0) / dt
    vel = np.concatenate([vel, vel[-1:]], axis=0)
    acc = np.diff(vel, axis=0) / dt
    acc = np.concatenate([acc, acc[-1:]], axis=0)
    return pos, vel, acc


def line(dt, T, start, end):
    t = np.arange(0, T, dt).reshape(-1, 1)
    start = start.reshape(1, -1)
    end = end.reshape(1, -1)
    pos = start + t * (end - start) / T
    vel = np.diff(pos, axis=0) / dt
    vel = np.concatenate([vel, vel[-1:]], axis=0)
    acc = np.diff(vel, axis=0) / dt
    acc = np.concatenate([acc, acc[-1:]], axis=0)
    return pos, vel, acc


def test():
    import matplotlib.pyplot as plt

    dt = 0.02
    T = 20
    start = np.array([1.0, 2.0, 3.0])
    end = np.array([2.0, 4.0, 6.0])
    pos, vel, acc = line(dt, T, start, end)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)
    fig = plt.figure()
    plt.plot(vel)
    fig = plt.figure()
    plt.plot(acc)
    plt.show()


test()
