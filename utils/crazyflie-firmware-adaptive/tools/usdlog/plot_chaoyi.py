import cfusdlog
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    # load data
    data = cfusdlog.decode('/home/pcy/Research/code/crazyswarm2-adaptive/utils/scripts/log')['fixedFrequency']
    # convert to numpy arrays
    # for k, v in data.items():
    #     data[k] = np.array(v)
    # set seaborn style
    sns.set()
    # prepare data
    pos_estimated = np.stack((data['stateEstimate.x'], data['stateEstimate.y'], data['stateEstimate.z']), axis=1)
    vel_estimated = np.stack((data['stateEstimate.vx'], data['stateEstimate.vy'], data['stateEstimate.vz']), axis=1)
    acc_estimated = np.stack((data['stateEstimate.ax'], data['stateEstimate.ay'], data['stateEstimate.az']), axis=1)
    rpy_estimated = np.stack((data['stateEstimate.roll'], data['stateEstimate.pitch'], data['stateEstimate.yaw']), axis=1)
    omega_estimated = np.stack((data['gyro.x'], data['gyro.y'], data['gyro.z']), axis=1)
    pos_desired = np.stack((data['ctrltarget.x'], data['ctrltarget.y'], data['ctrltarget.z']), axis=1)
    # vel_desired = np.stack(data['ctrltarget.vx'], data['ctrltarget.vy'], data['ctrltarget.vz'], axis=1)
    # plot
    time = data['timestamp'] - data['timestamp'][0]
    # plot pos_estimated and pos_desired
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time, pos_estimated[:, i], label='pos_estimated')
        plt.plot(time, pos_desired[:, i], label='pos_desired')
        plt.legend()
    plt.savefig('/home/pcy/Research/code/crazyswarm2-adaptive/utils/scripts/plot.png')

if __name__ == '__main__':
    main()