import numpy as np
import scipy.interpolate as si


def traj_from_pos(pos, dt):
    vel = np.diff(pos, axis=0) / dt
    vel = np.concatenate([vel, vel[-1:]], axis=0)
    acc = np.diff(vel, axis=0) / dt
    acc = np.concatenate([acc, acc[-1:]], axis=0)
    return pos, vel, acc


def figure8(dt, T, A, W, Phi):
    W = W.reshape(-1, 1)
    t = np.arange(0, T, dt).reshape(1, -1)
    Phi = Phi.reshape(-1, 1)
    A = A.reshape(-1, 1)
    pos = A * np.sin(W * t + Phi)
    return traj_from_pos(pos.T, dt)


def line(dt, T, start, end):
    t = np.arange(0, T, dt).reshape(-1, 1)
    start = start.reshape(1, -1)
    end = end.reshape(1, -1)
    pos = start + t * (end - start) / T
    return traj_from_pos(pos, dt)


def cos_interp(dt, T, start, end):
    t = np.arange(0, T, dt).reshape(-1, 1)
    start = start.reshape(1, -1)
    end = end.reshape(1, -1)
    pos = start + (1 - np.cos(t / T * np.pi)) / 2 * (end - start)
    return traj_from_pos(pos, dt)


def random_bspline(dt, T, T_sample, start, max_speed, pos_max, pos_min):
    R = T_sample * max_speed
    N = int(T / T_sample)

    last_p = start.reshape(1, -1)
    key_points = [np.copy(last_p)] * 3
    for i in range(N):
        box_max = last_p + np.array([R, R, R])
        box_min = last_p - np.array([R, R, R])
        sample_max = np.minimum(box_max, pos_max)
        sample_min = np.maximum(box_min, pos_min)
        sample = np.random.uniform(sample_min, sample_max)
        key_points.append(sample)
        last_p = np.copy(sample)

    key_points.extend([np.copy(last_p)] * 3)
    key_points = np.concatenate(key_points)

    steps = int(T / dt)
    pos = bspline(key_points, n=steps, degree=5, periodic=False)

    return traj_from_pos(pos, dt)


def bspline(cv, n=100, degree=3, periodic=False):
    """Calculate n samples on a bspline

    cv :      Array ov control vertices
    n  :      Number of samples to return
    degree:   Curve degree
    periodic: True - Curve is closed
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree, count + degree + 1)
        factor, fraction = divmod(count + degree + 1, count)
        cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)), -1, axis=0)
        degree = np.clip(degree, 1, degree)

    # Opened curve
    else:
        degree = np.clip(degree, 1, count - 1)
        kv = np.clip(np.arange(count + degree + 1) - degree, 0, count - degree)

    # Return samples
    max_param = count - (degree * (1 - periodic))
    spl = si.BSpline(kv, cv, degree)
    return spl(np.linspace(0, max_param, n))


def generate_traj(
    init_pos: np.array,
    dt: float,
    T_takeoff: float,
    T_hover: float,
    T_task: float,
    mode: str,
) -> np.ndarray:
    """
    generate a trajectory with max_steps steps
    """

    # generate take off trajectory
    target_pos = np.array([0.0, 0.0, 0.0])
    t_takeoff = T_takeoff
    pos_takeoff, vel_takeoff, acc_takeoff = cos_interp(
        dt, t_takeoff, init_pos, target_pos
    )

    # stablize for 1.0 second
    t_stablize = T_hover
    pos_stablize, vel_stablize, acc_stablize = line(
        dt, t_stablize, target_pos, target_pos
    )

    # generate test trajectory
    t_task = T_task

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
        # random trajectory
        max_speed = 2.0
        t_sample = 1.0
        pos_max = np.array([1.0, 1.0, 1.0])
        pos_min = np.array([-1.0, -1.0, -0.5])
        pos_task, vel_task, acc_task = random_bspline(
            dt, t_task, t_sample, target_pos, max_speed, pos_max, pos_min
        )

    else:
        raise ValueError("Invalid mode")

    # generate landing trajectory by inverse the takeoff trajectory
    pos_land_stablize, vel_land_stablize, acc_land_stablize = line(
        dt, t_stablize, pos_task[-1], pos_task[-1]
    )
    pos_landing, vel_landing, acc_landing = cos_interp(
        dt, t_takeoff, pos_land_stablize[-1], init_pos
    )

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


def test():
    import matplotlib.pyplot as plt

    dt = 0.02
    T = 20
    start = np.array([0.0, 0.0, -1.0])
    pos, vel, acc = generate_traj(start, dt, "random")
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)
    fig = plt.figure()
    plt.plot(pos)
    plt.title("pos")
    fig = plt.figure()
    plt.plot(vel)
    plt.title("vel")
    fig = plt.figure()
    plt.plot(acc)
    plt.title("acc")
    plt.show()
