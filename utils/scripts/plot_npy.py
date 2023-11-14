import numpy as np
from matplotlib import pyplot as plt

logs = np.load('logs.npy', allow_pickle=True)

pos_tar = np.array([log['pos_tar'] for log in logs])
pos = np.array([log['pos'] for log in logs])
time = np.array([log['time'] for log in logs])

plt.figure()

for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(time, pos_tar[:, i], label='target', linestyle='--')
    plt.plot(time, pos[:, i], label='real')
    plt.xlabel('time (s)')
    plt.ylabel(['x', 'y', 'z'][i])
    plt.legend()

plt.savefig('plot.png')