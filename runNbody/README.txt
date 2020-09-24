We have collected the scripts used to generate our initial conditions here. 
If you would like to reproduce them, you may need to change the paths in the scripts accordingly

## Generating Systems
### K23_make_sims.pbs
A PBS job script to call make_systems.py for Kepler 23.
### make_systems.py
Generating systems and testing for 10^5 orbit stability (with some parallelization).
A PBS job script to call test_stability.py is created for any systems that last 10^5 orbits.
Simulation archives are saved for systems lasting >= 10^4 orbits.

## Testing N-body 10^9 Orbit Stability
### test_stability.py
Trying to integrate systems for 10^9 orbits for the systems with saved simulation archives.

## Utilities
### stability_functions.py
Holds functions that create REBOUND simulations of our desired systems and various (mostly deprecated/unused) helper functions.
### submit_jobs.py
Submits up to max_submissions PBS jobs to the CyberLAMP computing cluster

*NOTE: The PBS jobs scripts are written to be used on Penn State's CyberLAMP computing cluster
