# Simulation Data and Analysis Scripts

This folder contains the data output by the simulator, and python scripts used to analyze and plot those data.
The outputs from these scripts were combined in Inkscape to produce Figures.

The folder /geometries/ contains the geometries on which simulations were performed,
/data/ contains the output from the simulator. Each stochastic simulation produces a 
given .hdf5 binary file.

A short description of each script follows:
__analyze_ttp.py__
Analyze the time to peak F/F0 signal across many runs in different geometries.
The ttp data was used to generate Figure 6.
Also carry out statistical tests on the data (a one-way Anova test with a 
Tukey HSD post-hoc). 

__bridging_sparks.py__
This script looks at the runs in idealized geometries to understand
how sparks can bridge between subclusters. Data is used in supplemental
Figure 2.

__convert_to_dataframe.py__
Utility script for pulling out relevant data from the .hdf5 simulation results
and creating a pickled pandas dataframe for easy access.

__filereader.py__
Utility script for reading out relevant data from the .hdf5 simulation results.

__plot_activation_map.py__
Analyze data from a given single run, and plot an activation map, showing which
 RyR's opened during the simulation, and when the first opening occured.
Used to generate Figure 6.

__plot_traces.py__
Plot F/F0 traces for a large number of runs in different geometries.
Used to generate Figure 6.

__plotutil.py__
Utility script with some small plotting functionality

__spark_fidelity.py__
Analyze data from runs in the different geometries to compute
spark fidelities. Used to generate Figure 4.

__var_ryr.py__
Analyze data from a cluster with a varied number of RyR to explore
spark termination.

__visualize_geometry.py__
Used to visualize the geometry files. Used to create Figure 4 and 
supplementary Figure 2. Used as source data for Figure 6.


Figure 4
* spark_fidelity.py
* visualize_geometry.py

Figure 6
* analyze_ttp.py
* plot_activation_map.py
* plot_traces
* var_ryr.py

Figure S2:
* visualize_geometry.py
* bridging_sparks.py