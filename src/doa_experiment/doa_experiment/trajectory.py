import numpy as np


def generate_traj(init_pos: np.array, dt: float, mode: str = "0") -> np.ndarray:
    """
    generate a trajectory with max_steps steps
    """

    # generate take off trajectory
    target_pos = np.array([0.0, 0.0, 0.0])
    t_takeoff = 3.0
    N_takeoff = int(t_takeoff / dt)
    pos_takeoff = np.linspace(init_pos, target_pos, N_takeoff)
    vel_takeoff = np.ones_like(pos_takeoff) * (target_pos - init_pos) / t_takeoff
    acc_takeoff = np.zeros_like(pos_takeoff)

    # stablize for 1.0 second
    t_stablize = 1.0
    N_stablize = int(t_stablize / dt)
    pos_stablize = np.ones((N_stablize, 3)) * target_pos
    vel_stablize = np.zeros_like(pos_stablize)
    acc_stablize = np.zeros_like(pos_stablize)

    # generate test trajectory
    t_task = 20.0

    wx = 0.4 * np.pi
    wy = 0.2 * np.pi
    wz = 0.0 * np.pi
    t = np.linspace(0, t_task, int(t_task / dt))
    # figure 8 trajectory
    # a = 1.0  # Amplitude in x-direction
    # b = 0.5  # Amplitude in y-direction
    a = 1.0
    b = 1.0
    c = 0.0
    x = a * np.sin(wx * t) + 0.0
    y = b * np.sin(wy * t) + 0.0
    z = c * np.sin(wz * t) + 0.0
    pos_task = np.stack([x, y, z], axis=-1) + target_pos
    vel_task = np.diff(pos_task, axis=0) / dt
    vel_task = np.concatenate([vel_task, vel_task[-1:]], axis=0)
    acc_task = np.diff(vel_task, axis=0) / dt
    acc_task = np.concatenate([acc_task, acc_task[-1:]], axis=0)

    scale = 1.0
    pos_task = pos_task * scale
    vel_task = vel_task * scale
    acc_task = acc_task * scale

    # generate landing trajectory by inverse the takeoff trajectory
    pos_landing = pos_takeoff[::-1]
    vel_landing = -vel_takeoff[::-1]
    acc_landing = -acc_takeoff[::-1]

    # concatenate all trajectories
    pos = np.concatenate(
        [
            pos_takeoff,
            pos_stablize,
            pos_task,
            pos_stablize,
            pos_landing,
        ],
        axis=0,
    )
    vel = np.concatenate(
        [
            vel_takeoff,
            vel_stablize,
            vel_task,
            vel_stablize,
            vel_landing,
        ],
        axis=0,
    )
    acc = np.concatenate(
        [
            acc_takeoff,
            acc_stablize,
            acc_task,
            acc_stablize,
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



def test():
    import matplotlib.pyplot as plt
    dt = 0.02
    T = 20
    A = np.array([1.0, 1.0, 1.0])
    W = np.array([0.4, 0.2, 0.0]) * np.pi
    Phi = np.array([0.0, 0.0, 0.0])
    pos, vel, acc = figure8(dt, T, A, W, Phi)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    fig = plt.figure()
    plt.plot(vel)
    fig = plt.figure()
    plt.plot(acc)
    plt.show()
    
