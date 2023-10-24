import numpy as np
from geometry_msgs.msg import Point, Quaternion, Vector3, PoseStamped
from nav_msgs.msg import Path

#numpy ros conversion
def np2point(pos:np.ndarray):
    return Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))

def np2quat(quat:np.ndarray):
    return Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))

def np2vec3(vec:np.ndarray):
    return Vector3(x=float(vec[0]), y=float(vec[1]), z=float(vec[2]))

def point2np(pos:Point):
    return np.array([pos.x, pos.y, pos.z])

def quat2np(quat:Quaternion):
    return np.array([quat.x, quat.y, quat.z, quat.w])

def vec32np(vec:Vector3):
    return np.array([vec.x, vec.y, vec.z])

#trajectory generation
def line_traj(rate:float, start:np.ndarray, end:np.ndarray, t:float, frame_id:str="map"):
    #line trajectory
    step = int(t * rate)
    traj = Path()
    traj.header.frame_id = frame_id

    delta = end - start
    for i in range(step):
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position = np2point(start + delta * i / step)
        traj.poses.append(pose)
    return traj