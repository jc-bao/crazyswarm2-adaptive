import cfusdlog
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import seaborn as sns

def main():
    # load data
    data = cfusdlog.decode('/media/pcy/cfSD/log40')['fixedFrequency']

    # load following parameters
    # ctrlRwik.p_error_wx
    # ctrlRwik.p_error_wy
    # ctrlRwik.p_error_wz
    # ctrlRwik.i_error_wx
    # ctrlRwik.i_error_wy
    # ctrlRwik.i_error_wz
    # ctrlRwik.d_error_wx
    # ctrlRwik.d_error_wy
    # ctrlRwik.d_error_wz
    # ctrlRwik.wx
    # ctrlRwik.wy
    # ctrlRwik.wz
    # ctrlRwik.wx_des
    # ctrlRwik.wy_des
    # ctrlRwik.wz_des

    p_error_wx = data['ctrlRwik.p_error_wx']
    p_error_wy = data['ctrlRwik.p_error_wy']
    p_error_wz = data['ctrlRwik.p_error_wz']
    i_error_wx = data['ctrlRwik.i_error_wx']
    i_error_wy = data['ctrlRwik.i_error_wy']
    i_error_wz = data['ctrlRwik.i_error_wz']
    d_error_wx = data['ctrlRwik.d_error_wx']
    d_error_wy = data['ctrlRwik.d_error_wy']
    d_error_wz = data['ctrlRwik.d_error_wz']
    wx = data['ctrlRwik.wx']
    wy = data['ctrlRwik.wy']
    wz = data['ctrlRwik.wz']
    wx_des = data['ctrlRwik.wx_des']
    wy_des = data['ctrlRwik.wy_des']
    # wz_des = data['ctrlRwik.wz_des']
    time = data['timestamp'] - data['timestamp'][0]

    sns.set()
    # plot
    # plt.plot(time, p_error_wx, label='p_error_wx')
    # plt.plot(time, i_error_wx, label='i_error_wx')
    # plt.plot(time, d_error_wx, label='d_error_wx')
    plt.plot(time, wx, label='wx')
    plt.plot(time, wx_des, label='wx_des')
    plt.legend()
    plt.savefig('plot.png')

if __name__ == '__main__':
    main()