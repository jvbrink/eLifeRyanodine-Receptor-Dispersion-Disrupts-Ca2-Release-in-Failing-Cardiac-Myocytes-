"""
This module is used to access and read the data output by the simulator.
Each stochastic simulation produces a hdf5-file with data for each
time point of the simulation. Here a class is provided which reads this
data, and makes it easily accesible through class attributes.
"""

import h5py
import numpy as np
import glob


class Simulation(object):
    """Read a simulation output file and make data accessible.

    This class reads a given hdf5 file output by the simulator.
    The data can then be accessed through properties of the
    FileReader object.
    
    The class attributes are variables set by the solver, and should
    be adjusted if needed. The fluoscale parameter is used to scale
    the F/F0 measured in spark dynamics to the same used in the 
    experimental setup, to make comparison easier, it is found comparing
    the biggest sparks seen computationally and experimentally.
    """
    TIMESTEP = 0.1
    FLUO = u'Fluo3-Fluo5'
    RYR = u'discrete_ryr' 
    FLUOSCALE = 0.8/1.3336

    def __init__(self, filename):
        """Open a given hdf5 run file."""
        self.filename = filename
        self.data = h5py.File(filename, 'r')
        self.nr_timesteps = len(self.data.keys())
        self.t = self.TIMESTEP*np.arange(self.nr_timesteps)
        
        self._ca = None
        self._fluo = None
        self._ryr = None
        self._mesh = None
        self._activation_map = None

    def read_ca(self):
        """Read out Ca data from h5 file"""
        self._ca = np.empty(self.nr_timesteps)
        for i, timestep in enumerate(self.data.keys()):
            ca = self.data[timestep]['cyt']['Ca']
            self._ca[i] = ca.attrs['average'][0]

    def read_fluo(self):
        """Read out Fluo data from h5 file and scale by F0."""
        self._fluo = np.empty(self.nr_timesteps)
        for i, timestep in enumerate(self.data.keys()):
            fluo = self.data[timestep]['cyt'][self.FLUO]
            self._fluo[i] = fluo.attrs['average'][0]
        self._fluo = self._fluo/self._fluo[0] 
        self._fluo -= 1
        self._fluo *= self.FLUOSCALE

    def read_ryr(self):
        """Read out data on state of RyRs."""
        self.nr_ryr = len(self.data.values()[0][self.RYR].value)
        self._ryr = np.zeros((self.nr_ryr, self.nr_timesteps),
                              dtype=np.bool)
        for i, timestep in enumerate(self.data.keys()):
            self._ryr[:, i] = self.data[timestep][self.RYR].value

    @property
    def fluo(self):
        """Average F/F0 in the cytosol over time."""
        if self._fluo is None:
            self.read_fluo()
        return self._fluo

    @property
    def ca(self):
        """Average Ca concentration in the cytosol over time."""
        if self._ca is None:
            self.read_ca()
        return self._ca

    @property
    def ttp(self):
        """Time to peak of the average F/F0 signal in ms."""
        return self.t[np.argmax(self.fluo)]

    @property
    def fluomax(self):
        """Maximum average F/F0 value in cytosol."""
        return self.fluo.max()    

    @property
    def ryr(self):
        """State of all RyR over time"""
        if self._ryr is None:
            self.read_ryr()
        return self._ryr

    @property
    def openryr(self):
        """Number of activated RyR throughout spark"""
        if self._ryr is None:
            self.read_ryr()
        return np.sum(self._ryr, axis=1).astype(np.bool).sum()

    @property
    def activation_map(self):
        """Compute the activation map of RyR.

        The activation map is the earliest time in the simulation
        each RyR opened. If a RyR did not open, the value for
        that RyR is set to np.nan.
        """
        ryr = self.ryr
        amap = np.empty(self.nr_ryr)
        amap.fill(np.nan)
        for i in range(self.nr_ryr):
            open_times, = np.where( ryr[i, :] > 0)
            if len(open_times) > 0:
                amap[i] = open_times.min()*self.TIMESTEP
        return amap
