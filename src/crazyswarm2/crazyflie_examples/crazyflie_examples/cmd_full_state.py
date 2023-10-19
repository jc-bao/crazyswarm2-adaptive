#!/usr/bin/env python

import numpy as np
from pathlib import Path

from crazyflie_py import *
from crazyflie_py.uav_trajectory import Trajectory


def executeTrajectory(timeHelper, cf, trajpath, rate=100, offset=np.zeros(3)):
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


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    rate = 30.0
    Z = 1.75

    cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)

    print('go to initial position...')
    cf.goTo(np.array([0.0, 1.0, Z]), 0.0, 2.0)
    timeHelper.sleep(2.0)

    print('Executing trajectory...')
    executeTrajectory(timeHelper, cf, "/home/pcy/Research/crazyswarm2-adaptive/src/crazyswarm2/crazyflie_examples/crazyflie_examples/data/figure8.csv", rate, offset=np.array([0, 1.0, 1.0]))

    print('go to initial position...')
    cf.goTo(np.array([0.0, 0.0, 0.5]), 0.0, 2.0)
    timeHelper.sleep(2.0)
    cf.notifySetpointsStop()
    
    print('land...')
    cf.land(targetHeight=0.04, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)


if __name__ == "__main__":
    main()
