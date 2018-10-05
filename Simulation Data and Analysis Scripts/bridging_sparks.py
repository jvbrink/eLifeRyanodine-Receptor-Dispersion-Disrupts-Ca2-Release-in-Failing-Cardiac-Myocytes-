"""
Analyze data from simulations used to explore bridging sparks
across gaps between subclusters.

Both an idealized 2x16 cluster is explored, in addition to the 
7 subclusterts geometry case with variable jsr-padding.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import filereader


def bridging_fidelity_idealized():
    """Calculate the bridging fidelity of each geometry.

    Each simulation is carried out by opening 3 random RyR's 
    in the left cluster, and allowing the simulation to proceed 
    stochastically. If a RyR in the right cluster is triggered
    to open, that is a bridged event.
    """

    PADDING = ["U1", "U1C"]
    GEOMS = ["{}_R2x16".format(d) for d in ('D2', 'D3', 'D4')]
    
    print("{:9}{:10}{:9}".format("Padding", "Distance", "Fidelity"))
    for pad in ("U1", "U1C"):
        for geom in ("D2", "D3", "D4"):
            pattern = "data/raw/{}_{}_R2x16_*.h5".format(pad, geom)
            infiles = sorted(glob.glob(pattern))
            
            sparks = 0
            bridges = 0
            for infile in infiles:
                data = filereader.Simulation(infile)
                amap = data.activation_map
                sparks += np.nansum(amap.astype(np.bool)) > 10
                bridges += np.nansum(amap[16:]) != 0
            print(" {:9}{:10}{:>3.0%}".format(pad, geom,
                                            float(bridges)/len(infiles)))


def bridging_fidelity_G7(vmax=0.3):
    """ Exploring Bridging in G7

    The Geometry G7 has a sizeable gap in the middle of the cluster,
    here we analyze and plot how often sparks transfer across
    this gap as a function of the jsr padding used.
    """
    paddings = ("U1", "U2", "U3")

    mesh = h5py.File("geometries/U1_G7.h5")
    xy = mesh["boundaries"]["ryr"].value[:,:2]
    x = -xy[:,0];
    y =  xy[:,1];
    nr_ryr = len(x)

    # Define the two "super sub clusters"
    c1 = np.array(range(0, 8) + range(10, 13) + range(17, 19) + range(24, 29)
                  + range(33, 37) + range(40, 42) + range(43, nr_ryr))
    c2 = np.array([i for i in range(nr_ryr) if i not in c1])

    for padding in paddings:
        infiles = glob.glob("data/raw/{}_G7_S80_*.h5".format(padding))
        n_simulations = len(infiles)

        c1_sims = 0
        c2_sims = 0
        c1_sparks = 0
        c2_sparks = 0
        c1_to_c2 = 0
        c2_to_c1 = 0

        c1_fidmap = np.zeros(nr_ryr)
        c2_fidmap = np.zeros(nr_ryr)

        for infile in infiles:
            data = filereader.Simulation(infile)
            amap = data.activation_map
            trigger = np.argmax(amap == 0)

            if trigger in c1:
                c1_sims += 1
                c1_fidmap += np.isfinite(amap)
                if data.fluomax > 0.4:
                    c1_sparks += 1
                c1_to_c2 += np.isfinite(amap)[c2].sum() != 0

            else:
                c2_sims += 1
                c2_fidmap += np.isfinite(amap)
                if data.fluomax > 0.4:
                    c2_sparks += 1
                c2_to_c1 += np.isfinite(amap)[c1].sum() != 0

        c1_fidmap /= c1_sims
        c2_fidmap /= c2_sims

        print("Padding: {}".format(padding))
        print("Top to Bottom: {:.1%}".format(c1_to_c2/float(c1_sims)))
        print("Bottom to top: {:.1%}".format(c2_to_c1/float(c2_sims)))


        fig = plt.figure(figsize=(1.5*(max(x)-min(x))*0.11, (max(y)-min(y))*0.11))

        ax = fig.add_axes((0, 0, 0.4, 1.0))
        ax.scatter(x, y, c=c1_fidmap, s=100, linewidth=2.0, cmap='Oranges',
                   vmin=0.0, vmax=vmax, edgecolor='black')
        ax.axis('off')
        ax.set_aspect('equal')

        ax = fig.add_axes((0.4, 0.0, 0.4, 1.0))
        ax.scatter(x, y, c=c2_fidmap, s=100, linewidth=2.0, cmap='Oranges',
                   vmin=0.0, vmax=vmax, edgecolor='black')
        ax.axis('off')
        ax.set_aspect('equal')
        
        


if __name__ == '__main__':
    bridging_fidelity_idealized()
    bridging_fidelity_G7()
    plt.show()