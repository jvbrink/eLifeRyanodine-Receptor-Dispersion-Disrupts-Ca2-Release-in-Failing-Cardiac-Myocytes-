"""
This module reads through simulation data and plots Fluo traces for all runs.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import filereader
import plotutil


def fluo_traces():
    """Plot Fluo traces from all runs.

    Create one plot for each geometry, where the F/F0 over time in each
    run in that geometry is drawn in as a single trace. Also draw in the
    threshold for spark detection.
    """
    axis = (0, 20, 0, 0.85)
    xlabel = 'Time [ms]'
    xticks = (0, 10, 20)
    ylabel = 'Average Cytosolic Fluo [$F/F0$]'
    yticks = (0.0, 0.4, 0.8)

    fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(12, 3))
    plotutil.configure_subplots(fig, ax, axis, xticks, yticks, xlabel, ylabel)

    geometries = ('G1', 'G3', 'G7', 'GX')
    titles = ('1 Solid', '3 Sub', '7 Sub', '10 Sub')

    for i, (geom, title) in enumerate(zip(geometries, titles)):
        for infile in glob.glob("data/raw/U1_{}_*.h5".format(geom)):
            f = filereader.Simulation(infile)
            ax[i].plot(f.t, f.fluo)
        ax[i].set_title(title)
        ax[i].axhline(0.4, linestyle='--', color='gray')


if __name__ == '__main__':
    fluo_traces()
    plt.show()    