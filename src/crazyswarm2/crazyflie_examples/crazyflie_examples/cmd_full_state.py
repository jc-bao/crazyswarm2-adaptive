#!/usr/bin/env python

import numpy as np
from pathlib import Path
import tf2_ros
import rclpy

from crazyflie_py import *
from crazyflie_py.uav_trajectory import Trajectory


def executeTrajectory(timeHelper, cf, trajpath, rate=100, offset=np.zeros(3), tf_buffer=None):
    if tf_buffer is None:
        enable_logging = False
    else:
        enable_logging = True
        logs = []

    traj = Trajectory()
    traj.loadcsv(trajpath)

    start_time = timeHelper.time()
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        if t > traj.duration:
            break


        e = traj.eval(t)
        cf.cmdFullState(
            e.pos + np.array(cf.initialPosition) + offset,
            e.vel,
            e.acc,
            e.yaw,
            e.omega)

        timeHelper.sleepForRate(rate)

        if enable_logging:
            trans_mocap = tf_buffer.lookup_transform('world', 'cf1', rclpy.time.Time())
            pos = trans_mocap.transform.translation
            logs.append(
                {
                    'pos_tar': np.array(e.pos), 
                    'pos': np.array([pos.x, pos.y, pos.z]) - offset, 
                    'time': t
                }
            )

    if enable_logging:
        np.save('logs.npy', logs)

def main():

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    rate = 30.0
    Z = 1.75

    # logging setup
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(buffer=tf_buffer, node = swarm.allcfs)

    cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)

    print('go to initial position...')
    cf.goTo(np.array([0.0, 0.0, Z]), 0.0, 2.0)
    timeHelper.sleep(2.0)

    print('Executing trajectory...')
    cf.setParam('usd.logging', 1)
    executeTrajectory(timeHelper, cf, "/home/pcy/Research/code/crazyswarm2-adaptive/src/crazyswarm2/crazyflie_examples/crazyflie_examples/data/figure8.csv", rate, offset=np.array([0, 0.0, Z]), tf_buffer=tf_buffer)
    cf.setParam('usd.logging', 0)

    print('go to initial position...')
    cf.goTo(np.array([0.0, 0.0, 0.5]), 0.0, 2.0)
    timeHelper.sleep(2.0)
    cf.notifySetpointsStop()
    
    print('land...')
    cf.land(targetHeight=0.04, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)


if __name__ == "__main__":
    main()