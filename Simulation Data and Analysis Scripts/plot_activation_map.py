"""
Analyze data from a given single run, and plot
an activation map, showing which RyR's opened during
the simulation, and when the first opening occured.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import filereader


def spark_fidelity_map(datafile, geometry, vmax=12):
    """Create a scatter plot showing the first opening of RyRs.
    
    Take in a given single run hdf5 file and a geometry file
    corresponding to that run, and plot the activation map.
    RyR that didn't open throughout the simulation are marked
    by a black X, otherwhise the time of the first opening
    is marked by a colormap.
    The vmax optional argument sets an upper limit on the 
    color scheme, which defaults to 12 ms.
    """
    if not datafile.endswith(".h5"):
        datafile += ".h5"
    data = filereader.Simulation("data/raw/{}".format(datafile))
    amap = data.activation_map

    # Read in RyR distribution
    if not geometry.endswith(".h5"):
        geometry += ".h5"

    mesh = h5py.File("geometries/{}".format(geometry))
    xy = mesh["boundaries"]["ryr"].value[:,:2]
    x = -xy[:,0];
    y =  xy[:,1];

    plt.figure(figsize=((max(x)-min(x))*0.11, (max(y)-min(y))*0.11))
    plt.axis('off')
    plt.axis('equal')
    plt.scatter(x, y, c=amap, s=120, linewidth=2.0, vmin=0, 
                vmax=vmax, cmap='Oranges', edgecolor='black')
    plt.colorbar(label='Time until first opening [ms]')

    # Plot X's on non-activated RyR
    na = np.isnan(amap)
    plt.scatter(x[na], y[na], s=80, linewidth=2.0, marker='x',
                color='black')
    

if __name__ == "__main__":
    spark_fidelity_map("U1_G7_S80_122", "U1_G7")
    plt.show()