# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import h5py
import glob
import plotutil

TIMESTEP = 0.1


def read_closing_times():
    nr_ryr = np.array([5, 11, 16, 22, 27, 32, 38, 43, 49, 54])
    closing_times = np.zeros((len(nr_ryr), 50))
    closing_times.fill(np.nan)
    for i, geom in enumerate(nr_ryr):
        infiles = sorted(glob.glob("data/raw/U3_CR{}_*.h5".format(geom)))
        for j, infile in enumerate(infiles):
            data = h5py.File(infile)
            closing_times[i, j] = (len(data)-1)*TIMESTEP
    return nr_ryr, closing_times


def plot_closing_times():
    nr_ryr, closing_times = read_closing_times()
    mean_ctimes = np.nanmean(closing_times, axis=1)
    std_ctimes = np.nanstd(closing_times, axis=1)

    plt.plot(nr_ryr, mean_ctimes, color='C0')
    plt.errorbar(nr_ryr, mean_ctimes, fmt='o', yerr=std_ctimes, color='C0')
    plotutil.simpleax(plt.gca())
    plt.xlabel('Number of RyR')
    plt.ylabel('Final Closing Time [ms]')
    plt.axis((0, 60, 0, 32))


if __name__ == '__main__':
    import matplotlib as mpl
    mpl.rcParams['errorbar.capsize'] = 3
    plot_closing_times()
    plt.show()