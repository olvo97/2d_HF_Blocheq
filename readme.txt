Pump-probe and single pulse optoelectronic dynamics in (stacks of) 2 dimensional semiconductors based on the Hartree-Fock level Bloch equations in the electron-hole picture

functions.py files contain the core functions and mathematical expressions
main.py (pump-probe) and single_pulse_MQW.py (MQW) utilize these functions for simulations that are specified in their respective arguments and parameter.py files
The .txt files are shell scripts that can be used to run simulations on a local Desktop or on the ITP Cluster.
Jupyter notebooks are for plotting and analyzing the results and can mostly be ignored

Linear_Spectra includes analytical expressions for the linear absorption of a single semiconductor layer, in both exciton and e-h picture

pump-probe includes pump-probe dynamics in a single semiconductor layer. The pulses and the material properties can be adjusted in the parameter files. The main.py file is an exemplary use of the functions in functions.py, leading to a single pump-probe spectrum. Intraband thermalization is included via phenomenological decay of excited carriers towards a Fermi distribution. 

MQW is used to compute the carrier dynamics and absorption in a multi-quantum well heterostructure inside a dielectric environment provided by the semiconductor substrate. Radiative coupling and partial reflections within the substrate are included.
