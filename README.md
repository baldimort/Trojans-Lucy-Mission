# Trojans: The *Lucy* Mission Rendezvous

These scripts were used to produce the results in the report "Determining the Position of Patroclus for the Lucy Mission Rendezvous"

All .dat files are data files to be used in conjunction with other scripts

/Astrometry\_compare.py - opens Anchises\_180124.csv and compares the Astrometry of UCAC2, UCAC4 and Gaia

/get\_error\_jacknife.py - takes in jacnkife sample (such as jacknifing\_example.dat) as an argument and outputs for each parameter the mean and its error fitting to a gaussian

/JPL\_Find\_orb\_convergence.py - takes in either covergence\_2yr.dat or covergence\_3months.dat (ephemerides computed by JPL HORIZONS) as argument, compares the convergence of find\_orb solution to JPL orbital parameters as a function of the number of observations provided.

/plot\_RA\_DECs\_observed.py - takes in MPC\_elements\_all\_asteroids.dat and produces a plot of all observations along with a line of the expected ephemeris given by the Minor Plant Centre orbital parameters.

/CSV_positions contains the RA/DEC data for all asteroids and the archival data used

/Priamus contains the orbital parameters calculated for priamus, and copys of the csv's used.

/Patroclus contains the orbital parameters calculated, excluding and including archival data, 
