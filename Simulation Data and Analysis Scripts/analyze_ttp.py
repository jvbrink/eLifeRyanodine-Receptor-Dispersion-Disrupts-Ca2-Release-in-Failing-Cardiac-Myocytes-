# -*- encoding: utf-8 -*-

import numpy as np
import glob
import matplotlib.pyplot as plt
import filereader
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

import plotutil
import scipy.stats


def ttp_histogram():
    geometries = ("G1", "G3", "G7", "GX")
    ttp = []
    fluomax = []
    for i, pattern in enumerate(geometries):
        ttp.append([])
        fluomax.append([])
        for infile in glob.glob("data/raw/U1_{}_*.h5".format(pattern)):
            data = filereader.Simulation(infile)
            ttp[i].append(data.ttp)
            fluomax[i].append(data.fluomax)

        ttp[i] = np.array(ttp[i])
        fluomax[i] = np.array(fluomax[i])

    for i in range(4):
        ttp[i] = ttp[i][fluomax[i] > 0.4]

    nr_sparks = [len(data) for data in ttp]
    nr_runs = [len(data) for data in fluomax]

    print("  Case | Runs | Sparks | Fidelity")
    fidstring = "{:>6} | {:>4} | {:>6} | {:>7.1%}"
    print("-----------------------------------")
    for geom, sparks, runs in zip(geometries, nr_sparks, nr_runs):
        print fidstring.format(geom, runs, sparks, sparks/float(runs))
    print("-----------------------------------\n")

    print("  Case | Mean TTP ± std. err [ms]")
    print("-----------------------------------")
    for geom, data, n in zip(geometries, ttp, nr_sparks):
        print("{:>6} |    {:.3f} ± {:.3f}".format(geom, np.mean(data), 
                                                  np.std(data)/np.sqrt(n)))
    print("-----------------------------------\n")
    
    bins = np.arange(0, 25, 2.5)
    weights = [np.ones(len(t))/len(t) for t in ttp]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes((0.2, 0.2, 0.7, 0.7))
    color = ('blue', (1, 0.28, 0.1), (0.9, 0.18, 0), (0.7, 0.14, 0))

    ax.hist(ttp, bins, weights=weights, histtype='bar', 
            color=color)

    ax.set_xticks((5, 10, 15, 20, 25))
    ax.set_xlabel('Time to Peak [ms]')
    ax.set_ylabel('Fraction')
    plotutil.simpleax(ax)
    
    ax = fig.add_axes((0.6, 0.6, 0.3, 0.3))
    for i, t in enumerate(ttp):
        ax.bar(i+1, np.mean(t), yerr=np.std(t)/np.sqrt(len(t)), color=color[i], 
               ecolor='black')

    ax.axis((0.5, 5.3, 0, 12))
    ax.set_ylabel('TTP [ms]')
    ax.set_xticks(())
    ax.set_yticks((0, 4, 8, 12))

    print(scipy.stats.f_oneway(*ttp))

    grouplabels = []
    grouplabels.extend(['1']*nr_sparks[0])
    grouplabels.extend(['3']*nr_sparks[1])
    grouplabels.extend(['7']*nr_sparks[2])
    grouplabels.extend(['X']*nr_sparks[3])
    endog = []
    endog.extend(ttp[0])
    endog.extend(ttp[1])
    endog.extend(ttp[2])
    endog.extend(ttp[3])

    mc = MultiComparison(np.array(endog), np.array(grouplabels))
    result = mc.tukeyhsd(alpha=0.05)
    print(result)
    print(mc.groupsunique)


if __name__ == '__main__':
    import matplotlib as mpl
    mpl.rcParams['errorbar.capsize'] = 3

    ttp_histogram()
    plt.show()
