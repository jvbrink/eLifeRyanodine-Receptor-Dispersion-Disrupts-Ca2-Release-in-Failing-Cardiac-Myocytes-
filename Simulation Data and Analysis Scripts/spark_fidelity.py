"""
Analyze data from the simulator to find spark fidelities.
The outputs of this program were used to create Figure 4.

Note that this module uses the pickled objects produced by
the convert_to_dataframe.py script.
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas
import glob
import h5py
import filereader

IDEALIZED = ['R4', 'R8', 'R12', 'R16', 'R20', 'R24']
DSTORM = ['G1', 'G3', 'G7', 'GX']

def spark_fidelity_idealized():
    """Find spark fidelity across all runs in idealized geometries.

    Sparks are defined by a 40% above baseline F/F0 threshold.
    """
    print(''.join(["{:15}".format(geom) for geom in IDEALIZED]))
    
    for i, geom in enumerate(IDEALIZED):
        data = pandas.read_pickle("data/pickles/U1_{}".format(geom))
        sparks = [f > 0.4 for f in data.fluomax]
        nr_runs = len(sparks)
        fidelity = np.average(sparks)
        print("{:<5.1%} ({:3})    ".format(fidelity, nr_runs), end='')
    print()
   

def spark_fidelity_dSTORM():
    """Find spark fidelity across all runs in dSTORM based geometries.

    Sparks are defined by a 40% above baseline F/F0 threshold.
    """
    print(''.join(["{:15}".format(geom) for geom in DSTORM]))
    
    for i, geom in enumerate(DSTORM):
        data = pandas.read_pickle("data/pickles/U1_{}".format(geom))
        sparks = [f > 0.4 for f in data.fluomax]
        nr_runs = len(sparks)
        fidelity = np.average(sparks)
        print("{:<5.1%} ({:3})    ".format(fidelity, nr_runs), end='')
    print()

def spark_fidelity_map(pattern, geometry=None, cap=None, vmax=0.25):
    """Create a scatter plot showing the channel fidelity of RyRs.

    Take in a pattern of runs and analyze how often channels are
    activated across runs. Plot the results in the geometric
    distribution of the ryrs. The optional geometry argument 
    can be used if the datafile pattern does not match the 
    geometry file pattern. The optiona cap argument can set
    an upper limit on the number of runs to include.
    The optional vmax argument sets the range of the colormaps,
    defaults to 25% to give a good dynamic range in the regime
    transitions.
    """
    # Find relevant runs
    infiles = "data/raw/{}_*.h5".format(pattern)
    print("Pattern given: {}".format(pattern))
    print("Looking for files: {}".format(infiles))
    infiles = sorted(glob.glob(infiles))
    print("{} files found.".format(len(infiles)))
    if len(infiles) == 0:
        print("Aborting.")
        return
    if cap is not None:
        files = sorted(files)[:cap]    

    # Read in RyR distribution
    if geometry is None:
        geometry = pattern
    mesh = h5py.File("geometries/{}.h5".format(geometry))
    xy = mesh['boundaries']['ryr'].value[:,:2]
    x = -xy[:,0];
    y =  xy[:,1];

    # Loop over all runs and compile how often individual channels open
    fidelity = np.zeros(len(x))
    for infile in infiles:
        f = filereader.Simulation(infile)
        fidelity += np.isfinite(f.activation_map)
    fidelity /= len(infiles)
    
    # Plot results
    plt.figure(figsize=((max(x)-min(x))*0.11, (max(y)-min(y))*0.11))
    plt.axis('off')
    plt.axis('equal')
    plt.scatter(x, y, c=fidelity, s=120, linewidth=2.0, vmin=0, 
                vmax=vmax, cmap='Oranges', edgecolor='black')
    plt.colorbar()

if __name__ == '__main__':
    # Print out spark fidelity data
    spark_fidelity_idealized()
    spark_fidelity_dSTORM()
    print()
    
    # Create spark fidelity maps
    for geom in ("G1", "G3", "G7", "GX"):
        spark_fidelity_map("U1_{}".format(geom))
        plt.show()

