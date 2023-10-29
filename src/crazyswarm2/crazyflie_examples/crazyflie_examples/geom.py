import numpy as np


def conjugate_quat(quat: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion (x, y, z, w)."""
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]])


def integrate_quat(quat: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate quaternion with angular velocity omega."""
    quat_dot = 0.5 * multiple_quat(quat, np.concatenate([omega, np.zeros(1)]))
    quat = quat + dt * quat_dot
    quat = quat / np.linalg.norm(quat)
    return quat


def multiple_quat(quat1: np.ndarray, quat2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (x, y, z, w)."""
    quat = np.zeros(4)
    w = quat1[3] * quat2[3] - np.dot(quat1[:3], quat2[:3])
    xyz = quat1[3] * quat2[:3] + quat2[3] * quat1[:3] + np.cross(quat1[:3], quat2[:3])
    quat[3]=w
    quat[:3]=xyz
    return quat


def rotate_with_quat(v: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotate the vector v with quaternion quat (x, y, z, w)."""
    v = np.concatenate([v, np.zeros(1)])
    v_rot = multiple_quat(multiple_quat(quat, v), conjugate_quat(quat))
    return v_rot[:3]

# Quaternion functions

def hat(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def L(q: np.ndarray) -> np.ndarray:
    '''
    L(q) = [sI + hat(v), v; -v^T, s]
    left multiplication matrix of a quaternion
    '''
    s = q[3]
    v = q[:3]
    right = np.hstack((v, s)).reshape(-1, 1)
    left_up = s * np.eye(3) + hat(v)
    left_down = -v
    left = np.vstack((left_up, left_down))
    return np.hstack((left, right))

H = np.vstack((np.eye(3), np.zeros((1, 3))))


def E(q: np.ndarray)->np.ndarray:
    '''
    reduced matrix for quadrotor state
    '''
    I3 = np.eye(3)
    I6 = np.eye(6)
    H = np.vstack((np.eye(3), np.zeros((1, 3))))
    G = L(q) @ H
    return jax.scipy.linalg.block_diag(I3, G, I6)


def qtoQ(q: np.ndarray) -> np.ndarray:
    '''
    covert a quaternion to a 3x3 rotation matrix
    '''
    T = np.diag(np.array([-1, -1, -1, 1]))
    H = np.vstack((np.eye(3), np.zeros((1, 3))))
    # H = np.vstack((np.eye(3), np.zeros((1, 3)))) # used to convert a 3d vector to 4d vector
    Lq = L(q)
    return H.T @ T @ Lq @ T @ Lq @ H


def rptoq(phi: np.ndarray) -> np.ndarray:
    return 1/np.sqrt(1+np.dot(phi, phi))*np.concatenate((phi, np.array([1])))


def qtorp(q: np.ndarray) -> np.ndarray:
    return q[:3]/q[3]

def qtorpy(q: np.ndarray) -> np.ndarray:
    # convert quaternion (x, y, z, w) to roll, pitch, yaw

    roll = np.arctan2(2*(q[3]*q[0]+q[1]*q[2]), 1-2*(q[0]**2+q[1]**2))
    pitch = np.arcsin(2*(q[3]*q[1]-q[2]*q[0]))
    yaw = np.arctan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))

    return np.array([roll, pitch, yaw])

def axisangletoR(axis: np.ndarray, angle: float) -> np.ndarray:
    # convert axis-angle to quaternion
    # axis: 3d vector
    # angle: radian
    # output: rotation matrix
    axis = axis/np.linalg.norm(axis)
    return np.eye(3) + np.sin(angle)*hat(axis) + (1-np.cos(angle))*hat(axis)@hat(axis)

def vee(R: np.ndarray):
    # convert skew-symmetric matrix to vector
    # R: 3x3 skew-symmetric matrix
    # output: 3d vector
    # P = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    # return - (P @ R @ P).diagonal()
    return np.array([R[2, 1], R[0, 2], R[1, 0]])