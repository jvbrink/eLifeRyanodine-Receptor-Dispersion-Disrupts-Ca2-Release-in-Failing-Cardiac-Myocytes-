"""
Parse raw data and convert to slim pandas dataframes.

Each stochastic simulation of the model gives a new .hdf5 data file
containing all the outputs of the model. To make searching, exploring
and plotting the data easier, we parse the data once and pull out 
interesting values such as time to peak and max F/F0 signal. These data
are then stored as a pickled pandas DataFrame.
"""

import pandas
import os
import glob
import filereader


def create_dataframe(pattern, cap=None):
    """Create dataframe from all runs matching pattern.

    Given a pattern, all runs matching the pattern are
    found and data collected. The resulting data are 
    saved to a pickle with the same pattern name.
    The optional keyword cap can be used to set an upper
    limit on the number of runs analyzed.
    """
    infiles = "data/raw/{}_S80_*.h5".format(pattern)
    outfile = "data/pickles/{}".format(pattern)

    print("Pattern given: {}".format(pattern))
    print("Looking for files: {}".format(infiles))

    infiles = sorted(glob.glob(infiles))

    print("{} files found.".format(len(infiles)))
    if len(infiles) == 0:
        print("Aborting.")
        return

    if cap is not None:
        files = files[:cap]        

    data = []
    for infile in infiles:
        f = filereader.Simulation(infile)
        data.append({'ttp': f.ttp,
                     'fluomax': f.fluomax,
                     'openryr': f.openryr})

    df = pandas.DataFrame(data)
    df.to_pickle(outfile)


if __name__ == '__main__':
    # dSTORM geometries
    for geom in ('G1', 'G3', 'G7', 'GX'):
        create_dataframe("U1_{}".format(geom))
        print("Finished.")
        print("---")

    # Idealized geometries
    for geom in ('R4', 'R8', 'R12', 'R16', 'R20', 'R24'):
        create_dataframe("U1_{}".format(geom))
        print("Finished.")
        print("---")